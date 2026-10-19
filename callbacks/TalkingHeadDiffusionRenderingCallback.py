"""
Author: Radek Danecek
Copyright (c) 2023, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emote@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
"""
import pytorch_lightning as pl 
from pathlib import Path
from skimage.io import imsave
import os, sys
from wandb import Video
import pickle as pkl
import numpy as np
import trimesh
from psbody.mesh.mesh import Mesh
import soundfile as sf
from inferno.models.Renderer import SRenderY
from inferno.utils.other import get_path_to_assets
import torch 
import inferno.utils.DecaUtils as util

from inferno.callbacks.TalkingHeadPytorchRenderingCallback import TalkingHeadTestPytorchRenderingCallback


class TalkingHeadDiffusionRenderingCallback(TalkingHeadTestPytorchRenderingCallback):


    def __init__(self, template_mesh_path, path_chunks_to_cat=None, predicted_vertex_key=None, 
                 save_meshes=False, image_size=400,
                 subfolder="videos",
                 background_color=None,
                 training=True,
                 ):
        super().__init__(template_mesh_path, path_chunks_to_cat, 
                         predicted_vertex_key, save_meshes, image_size, 
                         subfolder=subfolder, 
                         background_color=background_color)
        self.training = training
                         

    def on_test_batch_end(self, trainer, pl_module, 
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
        output_path = None,
    ):
        # super().on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        paths = None
        if len(outputs) == 3:
            denoised_meshes, noisy_meshes, diffusion_steps = outputs
            diffusion_type = None
        elif len(outputs) == 4:
            denoised_meshes, noisy_meshes, diffusion_steps, diffusion_type = outputs
            step_denoised_meshes = None
        elif len(outputs) == 5: 
            denoised_meshes, step_denoised_meshes, noisy_meshes, diffusion_steps, diffusion_type = outputs
        elif len(outputs) == 6: 
            denoised_meshes, step_denoised_meshes, noisy_meshes, diffusion_steps, diffusion_type, paths = outputs

        full_run_dir = output_path or pl_module.cfg.inout.full_run_dir 
        full_run_dir = Path(full_run_dir) 
        if not full_run_dir.is_absolute():
            full_run_dir = get_path_to_assets() / full_run_dir


        # for each mesh in batch, render the talking head 
        # and save the image to the output directory
        # print("Rendering talking head")
        logger = pl_module.logger
        # predicted_vertices = batch[self.predicted_vertex_key]

        has_prediction_dicts = False
        if isinstance(denoised_meshes[0], dict):
            has_prediction_dicts = True
            # dev = denoised_meshes[0]["vertices"].device
        else: 
            # dev = denoised_meshes[0].device
            pass
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.renderer = self.renderer.to(device=dev)

        # if "gt_vertices" in batch.keys():
        #     gt_vertices = batch["gt_vertices"]  
        # else: 
        #     rec_type = list(batch["reconstruction"].keys())[0]
        #     if len(batch["reconstruction"]) > 1:
        #         print("[WARNING]: More than one reconstruction type in batch. Using first one.")
        #     gt_vertices = batch["reconstruction"][rec_type]["gt_vertices"]        
        # denoised_meshes = [gt_vertices] + denoised_meshes
        # noisy_meshes = [predicted_vertices] + noisy_meshes
        # diffusion_steps = [""] + diffusion_steps
        # if diffusion_type is not None:
        #     diffusion_type = [""] + diffusion_type

        for di, d in enumerate(diffusion_steps):

            if not has_prediction_dicts:
                predicted_vertices = noisy_meshes[di].to(device=pl_module.device)
                gt_vertices = denoised_meshes[di].to(device=pl_module.device)
                step_denoised_vertices = None
                if step_denoised_meshes is not None:
                    step_denoised_vertices = step_denoised_meshes[di].to(device=pl_module.device)
                predicted_vertices_posed = None
                gt_vertices_posed = None
                step_denoised_vertices_posed = None
                predicted_vertices_cam = None
                gt_vertices_cam = None
                step_denoised_vertices_cam = None
            else: 
                predicted_vertices = None
                if noisy_meshes is not None:
                    predicted_vertices = noisy_meshes[di]["vertices"].to(device=pl_module.device)
                gt_vertices = denoised_meshes[di]["vertices"].to(device=pl_module.device)
                step_denoised_vertices = None
                if step_denoised_meshes is not None:
                    step_denoised_vertices = step_denoised_meshes[di]["vertices"].to(device=pl_module.device)
                
                predicted_vertices_posed = None
                if noisy_meshes is not None:
                    predicted_vertices_posed = noisy_meshes[di].get("vertices_posed", None) 
                    if predicted_vertices_posed is not None:
                        predicted_vertices_posed = predicted_vertices_posed.to(device=pl_module.device)
                
                gt_vertices_posed = denoised_meshes[di].get("vertices_posed", None)
                if gt_vertices_posed is not None:
                    gt_vertices_posed = gt_vertices_posed.to(device=pl_module.device)
                
                step_denoised_vertices_posed = None
                if step_denoised_meshes is not None:
                    step_denoised_vertices_posed = step_denoised_meshes[di].get("vertices_posed", None)
                    if step_denoised_vertices_posed is not None:
                        step_denoised_vertices_posed = step_denoised_vertices_posed.to(device=pl_module.device)
                
                predicted_vertices_cam = None
                if noisy_meshes is not None:
                    predicted_vertices_cam = noisy_meshes[di].get("cam", None)
                if predicted_vertices_cam is not None:
                    predicted_vertices_cam = predicted_vertices_cam.to(device=pl_module.device)
                gt_vertices_cam = denoised_meshes[di].get("cam", None)
                if gt_vertices_cam is not None:
                    gt_vertices_cam = gt_vertices_cam.to(device=pl_module.device)
                step_denoised_vertices_cam = None
                if step_denoised_meshes is not None:
                    step_denoised_vertices_cam = step_denoised_meshes[di].get("cam", None)
                    if step_denoised_vertices_cam is not None:
                        step_denoised_vertices_cam = step_denoised_vertices_cam.to(device=pl_module.device)
                        
            # B, T = gt_vertices.shape[:2]
            # B, T = predicted_vertices.shape[:2]
            if predicted_vertices is not None:
                B = predicted_vertices.shape[0]
            else: 
                B = gt_vertices.shape[0]

            if predicted_vertices is not None and gt_vertices is not None:
                T = min(gt_vertices.shape[1], predicted_vertices.shape[1])
            elif predicted_vertices is not None:
                T = predicted_vertices.shape[1]
            else:
                T = gt_vertices.shape[1]

            with torch.no_grad():
                pred_image = None
                if predicted_vertices is not None:
                    pred_image = self.render_no_pose(predicted_vertices)
                    
                gt_image = None
                if gt_vertices is not None:
                    gt_image = self.render_no_pose(gt_vertices)
                if step_denoised_vertices is not None:
                    step_denoised_image = self.render_no_pose(step_denoised_vertices)

                if predicted_vertices_posed is not None:
                    if predicted_vertices_cam is None:
                        predicted_vertices_cam = torch.tensor([[[9.,0.,0.]]], device=predicted_vertices.device).expand(B, T, -1).to(predicted_vertices.device)
                        gt_vertices_cam = predicted_vertices_cam.clone()
                    pred_image_posed = self.render_with_pose(predicted_vertices_posed, predicted_vertices_cam)
                    gt_image_posed = self.render_with_pose(gt_vertices_posed, gt_vertices_cam)
                    if step_denoised_vertices_posed is not None:
                        if step_denoised_vertices_cam is None:
                            step_denoised_vertices_cam = torch.tensor([[[9.,0.,0.]]], device=predicted_vertices.device).expand(B, T, -1).to(predicted_vertices.device)
                        step_denoised_image_posed = self.render_with_pose(step_denoised_vertices_posed, step_denoised_vertices_cam)
                    pred_image = np.concatenate([pred_image, pred_image_posed], axis=self.concat_axis+2)
                    gt_image = np.concatenate([gt_image, gt_image_posed], axis=self.concat_axis+2)
                    if step_denoised_vertices is not None:
                        step_denoised_image = np.concatenate([step_denoised_image, step_denoised_image_posed], axis=self.concat_axis+2)
                
                elif gt_vertices_posed is not None:
                    if gt_vertices_cam is None:
                        gt_vertices_cam = torch.tensor([[[9.,0.,0.]]], device=gt_vertices.device).expand(B, T, -1).to(gt_vertices.device)
                    gt_image_posed = self.render_with_pose(gt_vertices_posed, gt_vertices_cam)
                    gt_image = np.concatenate([gt_image, gt_image_posed], axis=self.concat_axis+2)
                    if step_denoised_vertices is not None and step_denoised_vertices_posed is None:
                        if step_denoised_vertices_cam is None:
                            step_denoised_vertices_cam = torch.tensor([[[9.,0.,0.]]], device=gt_vertices.device).expand(B, T, -1).to(gt_vertices.device)
                        step_denoised_image_posed = self.render_with_pose(step_denoised_vertices_posed, step_denoised_vertices_cam)
                        step_denoised_image = np.concatenate([step_denoised_image, step_denoised_image_posed], axis=self.concat_axis+2)
                

            for b in range(B):
                video_name = batch["filename"][b]
                condition_name = None
                if "condition_name" in batch.keys():
                    condition_name = batch["condition_name"][b][0]
                
                if trainer is not None and hasattr(trainer.datamodule, "test_set_names"):
                    dl_name = trainer.datamodule.test_set_names[dataloader_idx]
                else: 
                    dl_name = f"{dataloader_idx:02d}"

                diff_name = diffusion_steps[di]
                # if diff_name != "":
                if isinstance(diff_name, int):
                    diff_name = f"{diff_name:04d}"

                if diffusion_type is not None:
                    diff_type = diffusion_type[di]
                    if len(diff_name) == 0:
                        diff_name = diff_type
                    elif len(diff_type) > 0:
                        diff_name = f"{diff_type}/{diff_name}"
            
                    
                if paths is not None:
                    if self.training:
                        path = Path(full_run_dir) / self.subfolder / dl_name / paths[di]
                    else:
                        path = Path(full_run_dir) / self.subfolder / dl_name / paths[b]
                else:
                    if condition_name is None:
                        path = Path(full_run_dir) /  self.subfolder  / dl_name / self._path_chunk(video_name) / diff_name
                    else: 
                        path = Path(full_run_dir) /  self.subfolder / dl_name / self._path_chunk(video_name) / condition_name / diff_name

                if not path.is_absolute():
                    path = Path(get_path_to_assets()) / path


                condition_name = path.relative_to( Path(full_run_dir) /  self.subfolder  / dl_name)
                self.video_conditions[path] = str(condition_name)
                # if condition_name is None:
                #     self.video_conditions[path] = diff_name
                # else:
                #     self.video_conditions[path] = condition_name + "/" + diff_name
                self.dl_names[path] = dl_name

                path.mkdir(parents=True, exist_ok=True)


                # self.video_names_to_process[path] = audio_path
                if "framerate" in batch:
                    self.video_framerates_to_process[path] = batch["framerate"][b]
                else:
                    self.video_framerates_to_process[path] = self.framerate

                if 'raw_audio' in batch.keys():
                    self.audio_samplerates_to_process[path] = batch["samplerate"][b]
                
                if (path / "output.mp4").is_file() and not self.overwrite:
                    print(f"Skipping {path}. Video already exists.")
                    continue

                # audio_path = Path(video_name)
                # prev_frame_index = -1
                for t in range(T):
                    if "frame_indices" not in batch.keys():
                        frame_index = t
                    else:
                        try:
                            frame_index = batch["frame_indices"][b, t].item()
                        except IndexError: 
                            # frame_index = prev_frame_index + 1
                            continue
                    # prev_frame_index = frame_index
                    image_path = path / (self.image_format % frame_index + ".png")
                    if image_path.is_file() and not self.overwrite:
                        continue
                    if t == 0:
                        print("Writing:", image_path)

                    try:
                        valid = True
                        if "landmarks_validity" in batch.keys() and "mediapipe" in batch["landmarks_validity"]:
                            valid = bool(batch["landmarks_validity"]["mediapipe"][b, t].item())
                    except IndexError: 
                        valid = False
                        continue
                    
                    # pred_vertices = predicted_vertices[b, t].detach().cpu().view(-1,3).numpy()
                    # pred_image = self.renderer.render(pred_vertices)

                    # ref_vertices = gt_vertices[b, t].detach().cpu().view(-1,3).numpy()
                    # gt_image = self.renderer.render(ref_vertices, valid=valid)

                    # concatenate the images
                    if step_denoised_vertices is not None:
                        if step_denoised_image is not None and gt_image is not None:
                            image = np.concatenate([gt_image[b, t],  step_denoised_image[b, t], pred_image[b, t]], axis=self.concat_axis)
                        elif step_denoised_image is not None:
                            image = step_denoised_image[b, t]
                        elif gt_image is not None:
                            image = gt_image[b, t]
                        else: 
                            raise ValueError("No image to save")
                    else:
                        if gt_image is not None and pred_image is not None:
                            image = np.concatenate([gt_image[b, t], pred_image[b, t]], axis=self.concat_axis)
                        elif gt_image is not None:
                            image = gt_image[b, t]
                        elif pred_image is not None:
                            image = pred_image[b, t]
                        else:
                            raise ValueError("No image to save")
                    image = (image * 255.).astype(np.uint8)

                    # if trainer.datamodule.sequence_length_test != "all" or not self.save_images_to_disk:
                    # if trainer.datamodule.sequence_length_test != "all" or not self.save_images_to_disk:
                    if self.save_frames_to_disk:
                        imsave(image_path, image)
                    else: 
                        if not hasattr(self, "_image_cache"):
                            self._image_cache = []
                        self._image_cache += [image]

                    if self.save_meshes: 
                        pred_vertices = predicted_vertices[b, t].detach().cpu().view(-1,3).numpy()
                        # mesh = trimesh.base.Trimesh(pred_vertices, self.template.faces)
                        mesh = trimesh.base.Trimesh(pred_vertices, self.template.f, process=False)
                        # mesh_path = path / (self.image_format % frame_index + ".obj")
                        mesh_path = path / (self.image_format % frame_index + ".ply")
                        o = mesh.export(str(mesh_path))

                    if 'raw_audio' in batch.keys():
                        raw_audio_chunk = batch["raw_audio"][b,t].detach().cpu().numpy()
                    else: 
                        raw_audio_chunk = None
                    if self.save_frames_to_disk:
                        if 'raw_audio' in batch.keys():
                            audio_chunk_path = path / (self.image_format % frame_index + ".pkl")
                            with open(audio_chunk_path, "wb") as f:
                                pkl.dump(raw_audio_chunk, f)
                    else: 
                        if not hasattr(self, "_audio_cache"):
                            self._audio_cache = []
                        if raw_audio_chunk is not None:
                            self._audio_cache += [raw_audio_chunk]

                if not self.save_frames_to_disk and len(self._image_cache) > 0:
                    # write the video
                    # self._compose_chunked_audio(path, samplerate=self.audio_samplerates_to_process[path])
                    if len(self._audio_cache) > 0:
                        self._compose_chunked_audio(path, samplerate=self.audio_samplerates_to_process[path], audio_chunks=self._audio_cache)
                    self._save_video_from_tensor(self._image_cache, self.video_framerates_to_process[path], path)

                    self._image_cache = []
                    self._audio_cache = []

            # self._create_video(path, logger, trainer.global_step)

    # def on_test_epoch_begin(self, trainer, pl_module):
    #     super().on_test_epoch_begin(trainer, pl_module)
    #     self._create_videos(trainer, pl_module))

    def _save_video_from_tensor(self, image_list, framerate, path): 
        import skvideo
        image_tensor = np.stack(image_list)
        audio_path = path / "audio.wav"
        video_path = path / "output_no_audio.mp4"
        # write the video including audio 

        skvideo.io.vwrite(video_path, image_tensor, 
                          inputdict={'-r': str(framerate)}, 
                          outputdict={'-r': str(framerate)})
        
        # combine the audio and video using ffmpeg
        if audio_path.is_file():
            # ffmpeg_cmd = f"ffmpeg -y -i {str(video_path)} -i {str(audio_path)} -c:v copy -c:a aac -strict experimental {str(path / 'output.mp4')}"
            ffmpeg_cmd = f"ffmpeg -y -i {str(video_path)} -i {str(audio_path)} -c:v copy -c:a mp3 -strict experimental {str(path / 'output.mp4')}"
            os.system(ffmpeg_cmd)

            # remove the video without audio
            video_path.unlink()
            # remove the audio file
            audio_path.unlink()
        else: 
            video_path.rename(path / "output.mp4")

        

    def on_test_epoch_end(self, trainer, pl_module):
        super().on_test_epoch_end(trainer, pl_module)
        self._create_videos(trainer, pl_module)

    def _create_videos(self, trainer, pl_module):
        logger = pl_module.logger
        
        # for subfolder in self.video_names_to_process.keys():
        for subfolder in self.video_framerates_to_process.keys():
            if trainer is not None:
                step = trainer.global_step 
            else: 
                step = 0
            self._create_video(subfolder, logger, step)

        self.video_names_to_process = {}
        self.audio_samplerates_to_process = {}
        self.video_conditions = {}
        self.dl_names = {}

    def _compose_chunked_audio(self, subfolder, samplerate, audio_chunks=None):
        # create audio 
        audio_chunks_files = None
        if audio_chunks is None:
            audio_chunks_files = sorted([f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".pkl"])
            audio_chunks = [pkl.load(open(f, "rb")) for f in audio_chunks_files]

        if len(audio_chunks) > 0:
            audio_chunks = np.concatenate(audio_chunks, axis=0)
            audio_path = subfolder / "audio.wav"
            sf.write(audio_path, audio_chunks, samplerate)
        else:
            audio_path = None

        if audio_path.is_file() and audio_chunks_files is not None:
            # delete the audio chungs 
            for audio_chunk in audio_chunks_files:
                os.remove(audio_chunk)
        return audio_path

    def _create_video(self, subfolder, logger, epoch):
        # find audio
        # audio = self.video_names_to_process[subfolder]
        framerate = self.video_framerates_to_process[subfolder]
        samplerate = self.audio_samplerates_to_process[subfolder] if subfolder in self.audio_samplerates_to_process else 16000
        video_path = subfolder / ("output.mp4")

        if not video_path.is_file():

            # find all renderings in the subfolder
            renderings = sorted([f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".png"])    

            # create audio 
            audio_path = self._compose_chunked_audio(subfolder, samplerate)
            audio_chunks_files = sorted([f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".pkl"])
            # if len(audio_chunks_files) > 0:
            #     audio_chunks = [pkl.load(open(f, "rb")) for f in audio_chunks_files]
            #     audio_chunks = np.concatenate(audio_chunks, axis=0)
            #     audio_path = subfolder / "audio.wav"
            #     sf.write(audio_path, audio_chunks, samplerate)
            # else:
            #     audio_path = None

            # create video with audio
            if audio_path is not None:
                ffmpeg_cmd = f"ffmpeg -y -framerate {framerate} -start_number {renderings[0].stem} -i {str(subfolder)}/" + self.image_format + ".png"\
                    f" -i {str(audio_path)} -c:v libx264 -c:a aac -strict experimental -b:a 192k -pix_fmt yuv420p {str(video_path)}"
            else:
                ffmpeg_cmd = f"ffmpeg -y -framerate {framerate} -start_number {renderings[0].stem} -i {str(subfolder)}/" + self.image_format + ".png"\
                    f" -c:v libx264 -pix_fmt yuv420p {str(video_path)}"
            # ffmpeg_cmd = f"ffmpeg -y -framerate {framerate}  -start_number {renderings[0].stem} -i {str(subfolder)}/*.png"\
            #     f" -i {str(audio_path)} -c:v libx264 -c:a aac -strict experimental -b:a 192k -pix_fmt yuv420p {str(video_path)}"
            # ffmpeg_cmd = f"ffmpeg -framerate 30 -i {str(subfolder)}/" + self.image_format + f".png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {str(video_path)}"

            # create video from renderings
            os.system(ffmpeg_cmd)

            # check if the video was created
            if not video_path.exists():
                print(f"Video {video_path} was not created. The renderings are in {subfolder}")
                return
            
            print(f"Video {video_path} was created. Deleting renderings.")
            for rendering in renderings:
                os.remove(rendering)
            

            if audio_path is not None:
                os.remove(audio_path)

        # log the video
        self._log_video(video_path, logger, epoch)
                    
    def _log_video(self, video_path, logger, epoch):
        if logger is not None: 
            if isinstance(logger, pl.loggers.WandbLogger):
                name = "test_video" 
                dl_name = self.dl_names[video_path.parent]
                if dl_name is not None:
                    name += f"/{dl_name}"
                condition = self.video_conditions[video_path.parent]
                if condition is not None and len(condition) > 0:
                    name += "/" + condition 
                
                # try:
                #     test_ = int(video_path.parent.stem) ## a very ugly way to know if the video has a step number
                #     step_str = video_path.parent.stem
                #     sample_str = video_path.parents[2].stem
                # except ValueError: 
                #     step_str = video_path.parent.stem
                #     sample_str = video_path.parents[1].stem
                # name += "/" + sample_str + "_" + step_str

                logger.experiment.log({name: Video(str(video_path), 
                    fps=self.framerate, format="mp4", caption=str(video_path))}) #, step=epoch)