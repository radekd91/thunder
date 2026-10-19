import os, sys 
from pathlib import Path 
from omegaconf import OmegaConf, DictConfig

from inferno.sandbox.thunder.models.MotionLatentDiffusion import MotionLatentDiffusion
from inferno.sandbox.thunder.callbacks.TalkingHeadDiffusionRenderingCallback import TalkingHeadDiffusionRenderingCallback
from inferno.sandbox.thunder.models.DiffusionTalkingHead import DiffusionTalkingHead, prediction_dict, ConditionType
from inferno.models.IO import get_checkpoint_with_kwargs
from inferno.utils.other import get_path_to_assets
from inferno.utils.other import class_from_str
from inferno.utils.batch import dict_to_device
from inferno.utils.video import concatenate_videos

import datetime
import torch
from inferno.sandbox.thunder.utils.batch import expand_batch
from wandb import Video
import copy
import pytorch_lightning as pl

from inferno.sandbox.thunder.utils.batch import temporal_trim_dict
import numpy as np
from torch.utils.data import DataLoader
import pickle as pkl
from collections import OrderedDict
import argparse
from thunder_demo_args import get_arg_parser


def get_imagebind_encoder(cfg):
    from inferno.sandbox.thunder.utils.embeddings.imagebind import ImageBindWrapper, ModalityType
    if cfg.data.clip_feature_type == "imagebind": 
        # encoder = ImageBindWrapper(modalities=[ModalityType.TEXT])
        encoder = ImageBindWrapper(modalities=[ModalityType.TEXT, ModalityType.VISION, ModalityType.AUDIO])
        return encoder
    else: 
        raise NotImplementedError(f"Clip feature type {cfg.data.clip_feature_type} not implemented")


def get_all_subsets(given_list):
    from itertools import combinations
    subset_list = []
    for i in range(len(given_list) + 1):
        for subset in combinations(given_list, i):
            subset_list.append(list(subset))
    return subset_list


class TestDataModule(pl.LightningDataModule): 

    def __init__(self, cfg, audio_path): 
        super().__init__()
        self.cfg = cfg 
        self.audio_path = Path(audio_path)
        self.audio_file_list = []

    def prepare_data(self):
        if self.audio_path.is_file(): 
            self.audio_file_list = [self.audio_path]
        else:  #self.audio_path.is_dir():
            self.audio_file_list = list(self.audio_path.glob("*.wav"))
            self.audio_file_list += list(self.audio_path.glob("*.mp3"))
            self.audio_file_list = sorted(self.audio_file_list)

    def setup(self, stage=None):
        self.test_dataset = TestDataset(
            self.audio_file_list, 
            self.cfg.learning.batching.sequence_length_test,
            return_neck_pose = self.cfg.data.get('return_neck_pose', False),
            return_global_trans =  self.cfg.data.get('return_global_trans', False),
            return_global_pose = 'global_pose' in self.cfg.model.prior.sequence_components,
            ) 
            
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


class TestDataset(pl.LightningDataModule): 

    def __init__(self, audio_list, 
                 sequence_length, 
                 return_neck_pose = False,
                 return_global_trans =  False,
                 return_global_pose = False,
                 ) -> None:
        super().__init__()
        self.audio_list = audio_list 
        self.sequence_length = sequence_length
        self.expected_fps = 25
        self.return_neck_pose = return_neck_pose
        self.return_global_trans = return_global_trans
        self.return_global_pose = return_global_pose

    def __len__(self):
        return len(self.audio_list)

    def _get_audio_path(self, index):
        audio_path = self.audio_list[index]
        return audio_path

    def _read_audio(self, index):
        audio_path = self._get_audio_path(index)
            
        # load the audio 
        # if self.include_raw_audio:
        import librosa

        sampling_rate = 16000
        wavdata, sampling_rate = librosa.load(audio_path, sr=sampling_rate)

        if wavdata.ndim > 1:
            wavdata = librosa.to_mono(wavdata)
        wavdata = (wavdata.astype(np.float64) * 32768.0).astype(np.int16)
        
        return wavdata, sampling_rate, audio_path
    
    def _get_audio(self, index, start_frame, num_read_frames): #, video_fps, num_frames, sample):
        sample = {}

        wavdata, sampling_rate, audio_path = self._read_audio(index)
        assert sampling_rate % self.expected_fps == 0 

        sequence_length = self.sequence_length
        if sequence_length == "all":
            sequence_length = wavdata.shape[0] // (sampling_rate // self.expected_fps)

        wav_per_frame = sampling_rate // self.expected_fps 
        # wavdata_ = np.zeros((num_frames, wav_per_frame), dtype=wavdata.dtype) 
        wavdata_ = np.zeros((sequence_length, wav_per_frame), dtype=wavdata.dtype) 
        wavdata_ = wavdata_.reshape(-1)
        if wavdata.size > wavdata_.size:
            wavdata_[...] = wavdata[:wavdata_.size]
        else: 
            wavdata_[:wavdata.size] = wavdata
        wavdata_ = wavdata_.reshape((sequence_length, wav_per_frame))
        # wavdata_ = wavdata_.reshape((num_frames, wav_per_frame))
        # wavdata_ = wavdata_[start_frame:(start_frame + num_read_frames)] 
        wavdata_ = wavdata_[start_frame:(start_frame + sequence_length)] 
        if wavdata_.shape[0] < sequence_length:
            # concatente with zeros
            wavdata_ = np.concatenate([wavdata_, 
                np.zeros((sequence_length - wavdata_.shape[0], wavdata_.shape[1]),
                dtype=wavdata_.dtype)], axis=0)
        wavdata_ = wavdata_.astype(np.float64) / np.int16(np.iinfo(np.int16).max)

        sample["raw_audio"] = wavdata_ 
        sample["samplerate"] = sampling_rate

        sample["filename"] = str(audio_path)
        sample["fps"] = self.expected_fps # include the fps in the sample

        return sample


    def _get_reconstructions(self, index, start_frame, num_read_frames, sample): 
        T = sample["raw_audio"].shape[0]
        sample["gt_shape"] = np.zeros((300,), dtype=np.float32)
        sample["gt_exp"] = np.zeros((T, 100,), dtype=np.float32)
        sample["gt_jaw"] = np.zeros((T, 3,), dtype=np.float32)
        if self.return_neck_pose:
            sample["gt_neck"] = np.zeros((T, 3,), dtype=np.float32)
        if self.return_global_trans:
            sample["gt_trans"] = np.zeros((T, 3,), dtype=np.float32)
        if self.return_global_pose and not self.return_global_trans:
            sample["gt_cam"] = np.zeros((T, 3,), dtype=np.float32) 
        if self.return_global_pose:
            sample["gt_global_pose"] = np.zeros((T, 3,), dtype=np.float32) 
        return sample

    def __getitem__(self, index):
        start_frame = 0
        num_read_frames = self.sequence_length
        if self.sequence_length == "all":
            num_read_frames = None

        # 1) audio
        sample = self._get_audio(index, start_frame, num_read_frames) #, video_fps, num_frames, sample)

        # 2) dummy validity masks in absence of GT
        T = sample["raw_audio"].shape[0]
        sample["landmarks_validity"] = {} 
        sample["landmarks_validity"]["mediapipe"] = np.ones((T, 1), dtype=np.float32)

        # 3) dummy reconstructions in absence of GT
        sample = self._get_reconstructions(index, start_frame, num_read_frames, sample)

        # 4) dummy clip features in absence of GT 
        sample["clip_features"] = {}
        sample["clip_features"]["imagebind"] = torch.zeros((T, 1024), dtype=torch.float32)

        return sample
        

def prepare_data(cfg, audio_path,): 
    return TestDataModule(cfg, audio_path,), "test"


def load_model(config_path, checkpoint_mode=None):
    checkpoint_mode = checkpoint_mode or 'latest'
    cfg_file = Path(config_path) / "cfg.yaml"
    cfg = OmegaConf.load(cfg_file)

    if Path(cfg.inout.full_run_dir) != Path(config_path).parent:
        print(f"Warning: cfg.inout.full_run_dir {cfg.inout.full_run_dir} does not match the config_path {config_path}. Using config_path.")
        cfg.inout.full_run_dir = str(Path(config_path).parent) 
        cfg.inout.checkpoint_dir = str(Path(config_path) / "checkpoints")

    model_class = class_from_str(cfg.model.pl_module_class, sys.modules[__name__])

    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg, prefix="", checkpoint_mode=checkpoint_mode)

    model = model_class.instantiate(cfg, 1, "", checkpoint, checkpoint_kwargs)
    model = model.to(device='cuda')
    model.eval()

    return model, cfg


def load_data(cfg, audio_path, sequence_length): 
    cfg.data.preload_videos = False
    cfg.data.num_workers = 0 
    if sequence_length is not None:
        cfg.learning.batching.sequence_length_test = sequence_length
    dm, sequence_name = prepare_data(cfg, audio_path)
    dm.prepare_data()


    dm.setup(stage="test")

    dl = dm.test_dataloader()

    dl_name = sequence_name

    return dm, dl, dl_name


def setup_conditions(model, args, cfg):
    available_conditions = model.input_conditions
    condition_subset_list = get_all_subsets(available_conditions)
    # condition_subset_list = condition_subset_list[1:] # remove the empty subset
    
    ## remove all that don't have audio (in this demo we condition using audio (and perhaps CLIP features but only jointly with audio))
    non_audio_indices = []
    for cli, clist in enumerate(condition_subset_list):
        if ConditionType.AUDIO not in clist:
            non_audio_indices += [cli]

    for i in reversed(non_audio_indices):
        del condition_subset_list[i]

    condition_subset_dict = []
    ## remove the condition elements that have zero weight (saves unnecessary computation)
    for condition_subset_list_i in condition_subset_list:
        ## populate the condition_subset_dict with the weights
        condition_subset_dict_i = {} 
        if ConditionType.AUDIO in condition_subset_list_i:
            condition_subset_dict_i[tuple(condition_subset_list_i)] = args.cf_guidance_weight_audio
        elif ConditionType.CLIP in condition_subset_list_i:
            if args.cf_guidance_weight_clip is not None:
                condition_subset_dict_i[tuple(condition_subset_list_i)] = args.cf_guidance_weight_clip
            else:
                condition_subset_dict_i[tuple(condition_subset_list_i)] = 1.0
        else: 
            raise NotImplementedError(f"Other conditions not implemented in the demo script.")
        # for c in condition_subset_list_i:
        #     if c == ConditionType.AUDIO:
        #         condition_subset_dict_i[tuple([c])] = args.cf_guidance_weight_audio
        #     elif c == ConditionType.CLIP:
        #         if args.cf_guidance_weight_clip is not None:
        #             condition_subset_dict_i[tuple([c])] = args.cf_guidance_weight_clip
        #         else:
        #             condition_subset_dict_i[tuple([c])] = 1.0
        #     else: 
        #         raise NotImplementedError(f"Condition {c} not implemented in the demo script.")
        if args.cf_guidance_weight_uncond > 0.0:
            condition_subset_dict_i[tuple([])] = args.cf_guidance_weight_uncond ## unconditioned case
        condition_subset_dict += [condition_subset_dict_i]

    clip_encoder = None
    clip_strings = None
    clip_condition_features = None

    if ConditionType.CLIP in available_conditions:
        clip_encoder = get_imagebind_encoder(cfg)
        clip_input_dict = {}

        clip_strings = []

        if args.clip_image_conditions is None:
            image_root = get_path_to_assets() / "DiffusionReferenceImages/Ravdess"
            image_paths = [
                image_root / "Actor_23" / "angry_35.png", 
                image_root / "Actor_23" / "calm_35.png", 
                image_root / "Actor_23" / "disgust_35.png", 
                image_root / "Actor_23" / "fearful_70.png", 
                image_root / "Actor_24" / "surprised_70.png", 
                image_root / "Actor_24" / "happy_70.png", 
                image_root / "Actor_24" / "sad_70.png", 
            ]
        else:
            image_paths = [Path(x.strip()) for x in args.clip_image_conditions.split(",")]

        if len(image_paths) == 0:
            raise ValueError("No image paths provided for CLIP conditions. Please provide valid image paths.")
            
        for i in range(len(image_paths)):
            clip_strings += [image_paths[i].name]
            image_paths[i] = str(image_paths[i])
        clip_input_dict["images"] = image_paths


        clip_out = clip_encoder(**clip_input_dict)
        
        clip_condition_features = []

        if "vision" in clip_out:
            clip_condition_features += [clip_out['vision']] 

        clip_condition_features = torch.cat( clip_condition_features, dim=0)
        # del clip_strings[1:1+len(clip_strings_)]

    # return condition_subset_list, clip_strings, clip_condition_features
    return condition_subset_dict, clip_strings, clip_condition_features


def evaluate(args, config_path, audio_path, sequence_length, num_variations=None, checkpoint_mode = None, rename_old_results=False): 
    ## 1) LOAD THE MODEL AND CONFIG
    model, cfg = load_model(config_path, checkpoint_mode=checkpoint_mode)

    ## 2) LOAD THE DATA
    dm, dl, dl_name = load_data(cfg, audio_path, sequence_length)

    ## 3) SETUP THE POSSIBLE CONDITIONING INPUTS    
    condition_subset_list, clip_strings, clip_condition_features = setup_conditions(model, args, cfg)

    ## 4) SETUP OUTPUT RENDERER
    video_subfolder = args.output_path or "videos"
    num_variations = num_variations or 6

    flame_template_path = Path(cfg.model.prior.flame.flame_lmk_embedding_path).parent / "head_template.obj"
    # callback =  get_rendering_callback(cfg, flame_template_path, subfolder=video_subfolder)
    callback = TalkingHeadDiffusionRenderingCallback(
        flame_template_path, 
        path_chunks_to_cat=2, 
        predicted_vertex_key=None, 
        save_meshes=args.save_meshes, 
        image_size=400,
        subfolder=video_subfolder,
        training=False,
    )
    callback.concat_axis = 0 ## side-by-side concatenation

    ## 5) ITERATE OVER THE TEST DATA AND RUN THE REVERSE DIFFUSION
    all_output_videos = {}
    with torch.no_grad():
        for bi, batch in enumerate(dl):

            ## 5.1) PREPARE THE BATCH
            B = batch['raw_audio'].shape[0]
            T = batch['raw_audio'].shape[1]

            ## If the sequence length is specified, trim the batch to that length
            if args.max_seq_len is not None and T > args.max_seq_len:
                keys_to_ignore = ['samplerate', 'filename']
                batch = temporal_trim_dict(batch, 0, args.max_seq_len, is_batched=True, keys_to_ignore=keys_to_ignore)
                T = batch['raw_audio'].shape[1]
    
            ## move to device
            batch = dict_to_device(batch, device=model.device)
            assert B == 1, "Batch size must be 1"

            ## expand the batch to the number of variations to be generated
            batch = expand_batch(batch, num_variations)
            batch_backup = copy.deepcopy(batch)
            
            ## 5.2) SAMPLE THE NOISE TO BE DENOISED
            pl.seed_everything(args.seed)
            
            noise_shape = model.get_input_tensor_shape(T, round_up=True)
            noise_shape = [num_variations] + noise_shape
            noise = model.diffuser.get_training_init_noise(shape=noise_shape, device=model.device)  

            ## 5.3) for each condition subset, run the reverse diffusion and output the results
            for condition_subset in condition_subset_list:
                output_videos = reverse_diffusion(
                    args,
                    config_path, model,
                    num_variations,
                    noise,
                    condition_subset,
                    batch_backup,
                    bi, 
                    dl_name,
                    clip_strings,
                    clip_condition_features, video_subfolder,
                    rendering_callback=callback,
                )
                if tuple(condition_subset) not in all_output_videos:
                    all_output_videos[tuple(condition_subset)] = []
                all_output_videos[tuple(condition_subset)] += output_videos
    

    return all_output_videos


def reverse_diffusion(args, 
                      config_path, model, 
                      num_variations,
                      noise,
                      condition_subset,
                      batch_backup, 
                      bi, 
                      dl_name,
                      clip_strings, 
                      clip_condition_features, video_subfolder, 
                      rendering_callback, 
                      ):
    video_folders = []
    diff_type = []
    prediction_history = [] 
    next_prediction_history = [] 
    gt_history = [] 
    timesteps = []

    output_videos = []

    if Path(video_subfolder).is_absolute():
        output_folder = Path(video_subfolder)
    else:
        output_folder = Path(config_path) / video_subfolder

    if isinstance(condition_subset, dict):
        pass 
    else:
        condition_subset = {c : None for c in condition_subset}

    clip_condition_present = ConditionType.CLIP in condition_subset 
    if isinstance(condition_subset, dict): 
        for key in condition_subset.keys():
            if ConditionType.CLIP in key:
                clip_condition_present = True
                break

    if clip_condition_present:
        clip_conditions = clip_strings 
    else: 
        clip_conditions = [None]
    
    for condition_index, clip_condition in enumerate(clip_conditions):
        batch = copy.deepcopy(batch_backup) 

        condition_str = ""
        if len(condition_subset) == 0:
            condition_str = "uncond_"
        for c in condition_subset.keys():
            if c == ConditionType.CLIP:
                if clip_condition is not None:
                    condition_str += f"clip-{clip_condition}_"
                else: 
                    condition_str += f"clip-orig_"
            elif isinstance(c, (list, tuple)):
                c_list = [x.value for x in c]
                if len(c_list) == 0: 
                    subset_name = "uncond"
                else:
                    c_str = ""
                    for c_ in c:
                        if c_ == ConditionType.CLIP:
                            c_str += f"clip_{clip_condition}-"
                        else:
                            c_str += f"{c_.value}-" 
                    subset_name = c_str[:-1]
                condition_str += f"{subset_name}-{condition_subset[c]:0.02f}__"
            else:
                condition_str += f"{c.value}__"
        condition_str = condition_str[:-1]
        condition_str += "_seed-" + str(args.seed)

        if clip_condition_present:
            batch["clip_features"]["imagebind"] = clip_condition_features[condition_index].unsqueeze(0).unsqueeze(0).repeat(
                batch["raw_audio"].shape[0], batch["raw_audio"].shape[1], 1)
            # if condition_index > 0:
            # condition_subset[ConditionType.CLIP] = clip_condition_features[condition_index] ## the -1 accounts for the original image condition embedding
            # else:
            #     condition_subset[ConditionType.CLIP] = None

        if isinstance(list(condition_subset.keys())[0], (list, tuple)):
            guidance_dict = condition_subset 
            condition_list = None
        else: 
            condition_list = condition_subset 
            guidance_dict = None

        if args.diff_steps_to_visualize == "final":
            diff_steps_to_visualize = [model.diffuser.num_training_diffusion_steps()-1] ## the final prediction step
        else: 
            diff_steps_to_visualize = sorted([int(d_) for d_ in args.diff_steps_to_visualize.split(",")])


        batch, prediction_history_, next_prediction_history_, gt_history_, timesteps_ = model.reverse_diffusion(
            batch, bi, 
            train=False, validation=False, 
            diff_steps_to_visualize=diff_steps_to_visualize,
            condition_list=condition_list, 
            guidance_dict=guidance_dict, 
            noise=noise,
            )
        gt_history_ = None # no GT at test time

        prediction_history_ = prediction_history_[1:]

        next_prediction_history_ = None

        for di, diff_step in enumerate(diff_steps_to_visualize):

            diff_type_ = [f'reverse_diffusion/diversity/{dl_name}/{condition_str}/{i:03d}/diff_step_{diff_step:03d}' for i in range(num_variations)]
            if 'filename' in batch:
                subject = Path(batch['filename'][0]).parts[-2]
                name = Path(batch['filename'][0]).stem
            else: 
                subject = "unknown"
                name = "unknown"
            paths_ = [f'diversity/{dl_name}/{condition_str}/{subject}/{name}/sample_{i:03d}/diff_step_{diff_step:03d}' for i in range(num_variations)]

            video_folders += [paths_]

            timesteps = []

            dl_idx = 0
            rendering_callback.on_test_batch_end(
                trainer=None, # trainer
                pl_module=model, 
                # outputs=[prediction_history_, next_prediction_history_, gt_history_, timesteps_, diff_type_, paths_],
                outputs=[prediction_history_[di:di+1], None, None, [diff_step], diff_type_, paths_],
                batch=batch, 
                batch_idx=bi,
                dataloader_idx=0,
            )

            rendering_callback.on_test_end(
                trainer=None, 
                pl_module=model,
            )
            
            video_list = []
            for vf in paths_: 
                video_list += [output_folder/ f"{dl_idx:02d}" / vf / "output.mp4"]
                full_path = video_list[0].parents[2]

            if len(video_list) == 1:
                output_videos += [video_list[0]]
            else:
                concatenated_vid_path = full_path / f"variations_diff_step_{diff_step:03d}.mp4"
                concatenate_videos(video_list, concatenated_vid_path, horizontal=True, overwrite=False)
                output_videos += [concatenated_vid_path]

            ## Dump the output to a pickle file
            if args.save_predicted_flame:
                output_pickle_file = full_path / "output.pkl"
                output_dict = {} 
                for key in batch.keys(): 
                    if key == "predicted_vertices":
                        continue
                    if key.startswith("predicted_"):
                        output_dict[key] = batch[key].cpu().numpy()
                with open(output_pickle_file, "wb") as f: 
                    pkl.dump(output_dict, f)
            
    return output_videos


def split_prediction_history(history, num_variations, diffusion_step_indices=None):
    diffusion_step_indices = diffusion_step_indices or [-1]  # Default to the last step if not provided

    if isinstance(history[-1], dict):
        vertices = list(history[-1]["vertices"].split(1, dim=0))
        vertices_posed = None
        cam = None
        if "vertices_posed" in history[-1]:
            vertices_posed = list(history[-1]["vertices_posed"].split(1, dim=0))
        if "cam" in history[-1]:
            cam = list(history[-1]["cam"].split(1, dim=0))
        return [
            prediction_dict(
                vertices=vertices[i],
                vertices_posed=vertices_posed[i] if vertices_posed is not None else None,
                cam=cam[i] if cam is not None else None,
            )
            for i in range(num_variations)
        ]
    else:
        return list(history[-1].split(1, dim=0))


def main(): 
    parser = get_arg_parser()
    args = parser.parse_args()

    path_to_models = Path(args.model_folder) or get_path_to_assets() / "DiffusionTalkingHead/models" 
    model_name = args.model_name
    audio_path = args.audio_path

    sequence_length = args.max_seq_len or "all"
    if sequence_length != "all":
        sequence_length = int(sequence_length)

    num_variations = args.num_variations
    checkpoint_mode = "latest"

    config_path = path_to_models / model_name

    all_output_videos = evaluate(args, config_path, audio_path, sequence_length, num_variations, checkpoint_mode)

    for conditions_keys, video_paths in all_output_videos.items():
        for i, video_path in enumerate(video_paths):
            print(f"Saved results to videos: {str(video_path)}")


if __name__ == "__main__": 
    main()
