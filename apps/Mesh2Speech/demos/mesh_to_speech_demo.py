import os, sys 
from pathlib import Path 
from omegaconf import OmegaConf, DictConfig
import datetime
import torch
import pytorch_lightning as pl
from inferno.sandbox.thunder.utils.checkpoints import get_checkpoint_with_kwargs
from inferno.sandbox.thunder.models.MeshToSpeech import MeshToSpeech
from inferno.sandbox.thunder.models.MeshToSpeechGan import MeshToSpeechGan
from inferno.utils.other import get_path_to_assets
from inferno.utils.other import class_from_str
from inferno.utils.batch import dict_to_device
from inferno.datasets.IO import load_reconstruction_list, load_reconstruction_list_v2 
from inferno.sandbox.thunder.utils.asr.io import load_speaker_embedding
from munch import Munch, munchify
from inferno.sandbox.thunder.utils.batch import temporal_trim_dict
from inferno.sandbox.thunder.models.mesh2speech.Lip2SpeechVocoder import Lip2SpeechVocoder
import soundfile as sf
from inferno.utils.video import combine_video_audio
from mesh_to_speech_demo_args import get_arg_parser
from torch.utils.data import DataLoader
import numpy as np
from inferno.callbacks.TalkingHeadPytorchRenderingCallback import TalkingHeadTestPytorchRenderingCallback


class MeshToSpeechTestDataModule(pl.LightningDataModule): 

    def __init__(self, cfg, audio_path, reconstruction_path, speaker_embedding_path) -> None: 
        super().__init__()
        self.cfg = cfg 
        self.audio_path = Path(audio_path)
        self.reconstruction_path = Path(reconstruction_path)
        self.speaker_embedding_path = Path(speaker_embedding_path)
        self.audio_file_list = []
        self.reconstruction_file_list = []

    def prepare_data(self):
        if self.audio_path.is_file(): 
            self.audio_file_list = [self.audio_path]
            self.reconstruction_file_list = [self.reconstruction_path]
            self.speaker_embedding_file_list = [self.speaker_embedding_path]
        else:  #self.audio_path.is_dir():
            self.audio_file_list = list(self.audio_path.glob("*.wav"))
            self.audio_file_list += list(self.audio_path.glob("*.mp3"))
            self.audio_file_list = sorted(self.audio_file_list)

            audios_to_remove = []
            for ai, audio_file in enumerate(self.audio_file_list):
                reconstruction_file = self.reconstruction_path / audio_file.stem 
                if reconstruction_file.exists():
                    self.reconstruction_file_list.append(reconstruction_file)
                else:
                    print(f"Reconstruction file {reconstruction_file} does not exist, skipping.")
                    audios_to_remove.append(ai)
            for ai in reversed(audios_to_remove):
                del self.audio_file_list[ai]

    def setup(self, stage=None):
        self.test_dataset = MeshToSpeechTestDataset(
            self.audio_file_list, 
            self.reconstruction_file_list,
            self.speaker_embedding_file_list,
            self.cfg.learning.batching.sequence_length_test,
            return_neck_pose = self.cfg.data.get('return_neck_pose', False),
            return_global_trans =  self.cfg.data.get('return_global_trans', False),
            return_global_pose = self.cfg.data.get('return_global_pose', False),
            needs_video = any(
                preprocessor.get("type") == "VideoMouthCropPreprocessor"
                for preprocessor in self.cfg.model.preprocessors.values()
            ),
            ) 
        self.test_set_names = ["test"]
            
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


class MeshToSpeechTestDataset(pl.LightningDataModule): 

    def __init__(self, 
                 audio_list,
                 reconstruction_list, 
                 speaker_embedding_list,
                 sequence_length, 
                 return_neck_pose = False,
                 return_global_trans =  False,
                 return_global_pose = False,
                 needs_video = False,
                 ) -> None:
        super().__init__()
        self.audio_list = audio_list 
        self.reconstruction_list = reconstruction_list
        self.speaker_embedding_list = speaker_embedding_list
        self.sequence_length = sequence_length
        self.expected_fps = 25
        self.return_neck_pose = return_neck_pose
        self.return_global_trans = return_global_trans
        self.return_global_pose = return_global_pose
        self.needs_video = needs_video

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
            samples_per_frame = sampling_rate // self.expected_fps
            # Pad the final partial frame so the complete input audio is retained.
            sequence_length = (wavdata.shape[0] + samples_per_frame - 1) // samples_per_frame

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
        ## load the reconstruction file 
        reconstruction_path = self.reconstruction_list[index]
        assert reconstruction_path.exists() 

        end_frame = start_frame + sample["raw_audio"].shape[0]

        # shape_pose_cam = load_reconstruction_list(reconstruction_path , 
        #                                                 start_frame=start_frame, end_frame=end_frame)

        shape_pose_cam = load_reconstruction_list_v2(reconstruction_path, 
                                                        start_frame=start_frame, end_frame=end_frame)

        T = sample["raw_audio"].shape[0]
        sample["gt_shape"] = shape_pose_cam["shape"][0]#.reshape(T, -1).astype(np.float32)
        sample["gt_exp"] = shape_pose_cam["exp"][0]#.reshape(T, -1).astype(np.float32)
        sample["gt_jaw"] = shape_pose_cam["jaw"][0]#.reshape(T, -1).astype(np.float32)
        if self.return_neck_pose:
            sample["gt_neck"] = shape_pose_cam["neck"][0]#.reshape(T, -1).astype(np.float32)
        if self.return_global_trans:
            sample["gt_trans"] = shape_pose_cam["trans"][0]#.reshape(T, -1).astype(np.float32)
        if self.return_global_pose and not self.return_global_trans:
            sample["gt_cam"] = shape_pose_cam["cam"][0]#.reshape(T, -1).astype(np.float32)
        if self.return_global_pose:
            sample["gt_global_pose"] = shape_pose_cam["global_pose"][0]#.reshape(T, -1).astype(np.float32)
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

        if sample["gt_exp"].shape[0] < T:
            raise ValueError(
                f"Reconstruction contains {sample['gt_exp'].shape[0]} frames, but the audio requires {T}. "
                "Provide reconstruction for the complete audio or explicitly set --max_seq_len to a supported length."
            )
        
        sample["speaker_embedding"] = load_speaker_embedding(self.speaker_embedding_list[index])

        if self.needs_video:
            # The public smoke-test sample ships reconstruction/audio but no paired video.
            sample["video"] = np.full((T, 3, 224, 224), 0.5, dtype=np.float32)
            sample["landmarks"] = {"fan3d": np.zeros((T, 68, 2), dtype=np.float32)}

        return sample


def load_vocoder(cfg, model, trainable=False):
    if hasattr(cfg.model, "vocoder") and cfg.model.vocoder is not None: 
        vocoder = model.vocoder
    else:
        import json
        vocoder_config_file = get_path_to_assets() / "Lip2SpeechUnit/vocoder/multi_input_aug.json"
        with open(vocoder_config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        vocoder_cfg = munchify({ 
            "trainable": False, 
            "pretrained_checkpoint": get_path_to_assets() / 'Lip2SpeechUnit/vocoder/vocoder_lrs3_aug_multi.pt',
            "generator": json_config,
        })
        vocoder = Lip2SpeechVocoder(vocoder_cfg)
        vocoder.cuda()
    return vocoder


def prepare_data(cfg, audio_path, reconstruction_path, speaker_embedding_path): 
    return MeshToSpeechTestDataModule(cfg, audio_path, reconstruction_path, speaker_embedding_path), "test"


def load_data(cfg, audio_path, reconstruction_path, speaker_embedding_path, sequence_length): 
    cfg.data.preload_videos = False
    cfg.data.num_workers = 0 
    cfg.learning.batching.sequence_length_test = sequence_length if sequence_length is not None else "all"
    dm, sequence_name = prepare_data(cfg, audio_path, reconstruction_path, speaker_embedding_path)
    dm.prepare_data()

    dm.setup(stage="test")

    dl = dm.test_dataloader()

    dl_name = sequence_name
    return dm, dl, dl_name


def load_model(config_path, checkpoint_mode=None):
    checkpoint_mode = checkpoint_mode or 'best_val'
    cfg_file = Path(config_path) / "cfg.yaml"
    cfg = OmegaConf.load(cfg_file)

    if cfg.model.preprocessors.get('flame', None) is  None:
        ## if FLAME preprocessor is not defined, add it (FlameToSpeech otherwise does not compute the vertices which we need for visualization)
        flame_cfg = {
            'flame': 
                {
                'type': 'FlamePreprocessor',
                'flame': {
                    'flame_model_path': 'FLAME/geometry/generic_model.pkl',
                    'n_shape': 300,
                    'n_exp': 100,
                    'flame_lmk_embedding_path': 'FLAME/geometry/landmark_embedding.npy',

                    },
                'use_texture': False,
                }
            }
        cfg.model.preprocessors.update(flame_cfg)
    

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


def evaluate(args, config_path, checkpoint_mode = None): 
    ## 1) LOAD THE MODEL AND CONFIG
    model, cfg = load_model(config_path, checkpoint_mode=checkpoint_mode)
    model.cuda()
    vocoder = load_vocoder(cfg, model, trainable=False)


    ## 2) LOAD THE DATA
    dm, dl, dl_name = load_data(cfg, args.audio_path, args.reconstruction_path, args.speaker_embedding_path, args.max_seq_len)

    ## 3) INITIALIZE RENDERER (FOR VISUALIZATION)
    output_subfolder =  args.output_dir or "test_results" 
    output_subfolder =  Path(output_subfolder)
    output_folder = output_subfolder
    if not output_folder.is_absolute():
        output_folder = Path(config_path) / output_folder 
    rendering_callback = TalkingHeadTestPytorchRenderingCallback(
                template_mesh_path=None, 
                 path_chunks_to_cat=None, 
                 save_meshes=False, 
                 image_size=800, 
                 subfolder=output_subfolder, 
                 predicted_vertex_key="gt_vertices",
                 output_dir = config_path,
                 audio_key="predicted_wav",
    )

    ## 4) EVALUATE THE MODEL
    gt_audios = []
    predicted_audios = []
    videos = []
    with torch.no_grad():        
        for bi, batch in enumerate(dl):
            B = batch['raw_audio'].shape[0]
            T = batch['raw_audio'].shape[1]
            batch = dict_to_device(batch, 'cuda')

            ## 4a) evaluate the model
            batch = model(batch)

            rel_folder = Path(batch["filename"][0]).stem

            ## 4b) pass the predicted representation to the vocoder (if not done inside the model)
            if "predicted_audio" not in batch:
                predicted_unit_classification_vectors = torch.nn.functional.softmax(batch['predicted_unit_classification_vectors'], dim=-1)
                batch["predicted_units"] = torch.argmax(predicted_unit_classification_vectors, dim=-1)
                num_unit_classes = vocoder.generator.dict.weight.shape[0]
                predicted_units_max = batch["predicted_units"].max()
                if predicted_units_max >= num_unit_classes: 
                    print(f"[WARNING]: predicted units max is {predicted_units_max} which is higher than vocoder supported {num_unit_classes-1}." 
                            "This should not happen. Setting to zero to prevent crash.")
                    batch["predicted_units"][batch["predicted_units"] >= num_unit_classes] = 0
                batch = vocoder(batch, cfg)
                batch = vocoder(batch, cfg, input_mel_key="gt_mel", input_unit_key="gt_units", output_key="vocoded_gt_wav")

            ## 4c) save the predicted audio, ground truth audio and also the vocoded ground truth audio (for comparison)
            predicted_wav = batch["predicted_wav"][0,0].cpu().numpy()
            ## float to int16
            predicted_wav = (predicted_wav * 32767).astype('int16')

            vocoded_gt_wav = batch["vocoded_gt_wav"][0,0].cpu().numpy()
            ## float to int16
            vocoded_gt_wav = (vocoded_gt_wav * 32767).astype('int16')

            gt_wav = batch["raw_audio"][0].reshape(-1).cpu().numpy()
            ## float to int16
            gt_wav = (gt_wav * 32767).astype('int16')

            output_path = output_folder  / f"{dl_name}" / rel_folder / "predicted.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predicted_audios.append(output_path)
            sf.write(output_path, predicted_wav, 16000)
            
            output_path = output_folder  / f"{dl_name}" / rel_folder / "vocoded_gt.wav"
            sf.write(output_path, vocoded_gt_wav, 16000)

            output_path = output_folder / f"{dl_name}" / rel_folder / "gt.wav"
            gt_audios.append(output_path)
            sf.write(output_path, gt_wav, 16000)



            ## 4d) render video using the talking head callback 
            trainer_dummy_struct = munchify({
                "datamodule": dm, 
                "logger": None, 
                "global_step": 0,})
            batch['predicted_wav'] = batch['predicted_wav'].view(batch["raw_audio"].shape)

            rendering_callback.on_test_batch_end(
                trainer=trainer_dummy_struct, 
                pl_module=model, 
                outputs=None,
                batch=batch,
                batch_idx = bi,
                dataloader_idx=0,
            )
            rendering_callback.on_test_epoch_end(trainer=trainer_dummy_struct, pl_module=model)

            video_path = output_path.with_name("output.mp4")
            videos += [video_path]
        
    print(f"Evaluation completed. Results saved to '{output_folder / output_subfolder}'.")
    return gt_audios, predicted_audios, videos


def main(): 
    parser = get_arg_parser()
    args = parser.parse_args()

    path_to_models = Path(args.model_folder) or get_path_to_assets() / "DiffusionTalkingHead/models" 
    model_name = args.model_name

    config_path = path_to_models / model_name

    gt_audios, predicted_audios, videos = evaluate(args, config_path)

    num_outputs = min(len(gt_audios), len(predicted_audios), len(videos)) 
    for i in range(num_outputs):
        print(f"Output {i+1}:")
        print(f"  GT Audio: {gt_audios[i]}")
        print(f"  Predicted Audio: {predicted_audios[i]} ")
        print(f"  Video: {videos[i]}")
        print()


if __name__ == "__main__": 
    main()
