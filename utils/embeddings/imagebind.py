from pathlib import Path
from inferno.utils.other import get_path_to_externals
import os, sys 
sys.path.append(str(get_path_to_externals() / "ImageBind"))

# ImageBind uses the newer timm.layers name; older environments expose the
# same layer implementations under timm.models.layers.
import importlib
try:
    importlib.import_module("timm.layers")
except ModuleNotFoundError as exc:
    if exc.name != "timm.layers":
        raise
    sys.modules["timm.layers"] = importlib.import_module("timm.models.layers")

import imagebind
from imagebind import data
import torch
import numpy as np
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType, ImageBindModel
from imagebind import data
from pathlib import Path
from inferno.utils.other import get_path_to_assets
import torchaudio
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler, UniformClipSampler

from torchvision import transforms
image_bind_root = Path(imagebind.__file__).parent


def load_imagebind_huge(folder): 
    model = ImageBindModel(
        vision_embed_dim=1280,
        vision_num_blocks=32,
        vision_num_heads=16,
        text_embed_dim=1024,
        text_num_blocks=24,
        text_num_heads=16,
        out_embed_dim=1024,
        audio_drop_path=0.1,
        imu_drop_path=0.7,
    )
    folder = Path(folder)
    model_file = folder / "imagebind_huge.pth"
    if not model_file.exists():
        print(
            f"Downloading imagebind weights to '{str(model_file)}' ..."
        )
        folder.mkdir(exist_ok=True)
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
            str(model_file),
            progress=True,
        )

    model.load_state_dict(torch.load(str(model_file)))

    return model



def image_transformation_data(video_tensors, device):
    video_tensors = np.concatenate(video_tensors, axis=0)
    # # B, H, W, C -> B, C, H, W
    video_tensors = np.transpose(video_tensors, (0, 3, 1, 2))
    video_tensors = torch.from_numpy(video_tensors).float() / 255.0
    data_transform = transforms.Compose(
        [
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            # transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    
    # for image_path in image_paths:
    #     with open(image_path, "rb") as fopen:
    #         image = Image.open(fopen).convert("RGB")

    video_tensors = data_transform(video_tensors).to(device)
    return video_tensors
    # image_outputs.append(video_tensor)
    # return torch.stack(image_outputs, dim=0)


class ImageBindWrapper(torch.nn.Module):
    
    def __init__(self, modalities, max_batch_size=32, model=None) -> None:
        super().__init__()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # model = imagebind_model.imagebind_huge(pretrained=True)
        image_bind_folder = get_path_to_assets() / "ImageBind"
        model = model or load_imagebind_huge(folder=image_bind_folder)
        model.eval()
        model.to(device)
        self.device = device
        self.model = model
        self.modalities = modalities
        self.max_batch_size = max_batch_size

    def forward_vision(self, video=None, images=None): 
        assert not (video is not None and images is not None), "Can't specify two vision modalities but both images and videos are specified."
        embeddings = self.forward(video=video, images=images)
        vision_embeddings = embeddings[ModalityType.VISION]
        return vision_embeddings
        # assert vision_embeddings[0] == 1, "Expected only one vision embedding but got more."
        # return vision_embeddings[0]
        # return embeddings[ModalityType.VISION]
    
    
    def forward_audio(self, audio): 
        embeddings = self.forward(audio=audio)
        audio_embeddings = embeddings[ModalityType.AUDIO]
        return audio_embeddings


    def forward(self, video=None, images=None, audio=None, audios=None, text=None): 
        assert not (video is not None and images is not None), "Can't specify two vision modalities but both images and videos are specified."
        input_dict = {} 
        if video is not None: 
            if isinstance(video, Path):
                video = str(video)
            input_dict[ModalityType.VISION] = video
        if images is not None:
            input_dict[ModalityType.VISION] = images
        if audio is not None:
            input_dict[ModalityType.AUDIO] = audio
            assert audios is None, "Can't specify two audio modalities but both audio and audios are specified."
        if audios is not None:
            input_dict[ModalityType.AUDIO] = audios
            assert audio is None, "Can't specify two audio modalities but both audio and audios are specified."
            
        if text is not None:
            input_dict[ModalityType.TEXT] = text
            

        # for key in self.modalities: 
        for key in input_dict.keys():
            if key == ModalityType.TEXT: 
                fn = data.load_and_transform_text
                input_data = input_dict[key]
            elif key == ModalityType.VISION:
                if video is not None:
                    fn = data.load_and_transform_video_data
                    input_data = input_dict[key]
                elif images is not None:
                    if isinstance(images[0], str):
                        fn = data.load_and_transform_vision_data 
                        input_data = input_dict[key]
                    elif isinstance(images[0], np.ndarray):
                        fn = image_transformation_data
                        input_data = [input_dict[key]]
            elif key == ModalityType.AUDIO:
                # if isinstance(audio[0], str):
                    # fn = data.load_and_transform_audio_data 
                if audio is not None:
                    fn = load_and_transform_audio_data
                elif audios is not None:
                    fn = data.load_and_transform_audio_data

                # elif isinstance(audio[0], np.ndarray):
                input_data = input_dict[key]
            
            input_dict[key] = fn(input_data, device = self.device)
        
        with torch.no_grad():
            B = None
            if ModalityType.VISION in input_dict:
                B = input_dict[ModalityType.VISION].shape[0]
            elif ModalityType.AUDIO in input_dict:
                B = input_dict[ModalityType.AUDIO].shape[0]
            if B is None:
                embeddings = self.model(input_dict)
            else: 
                embeddings = []
                for i in range(0, B, self.max_batch_size):
                    batch = {k: v[i:i+self.max_batch_size] for k, v in input_dict.items()}
                    embeddings.append(self.model(batch))
                embeddings = {k: torch.cat([e[k] for e in embeddings], dim=0) for k in embeddings[0]}
        return embeddings
    


def load_and_transform_audio_data(
    # audio_paths,
    audio_path,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    # clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    
    if audio_path is None:
        return None

    audio_outputs = []

    # for audio_path in audio_paths:
    waveform, sr = torchaudio.load(audio_path)
    ## to mono 
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sample_rate
        )

    whole_clip_duration = waveform.size(1) / sample_rate
    # clip_sampler = ConstantClipsPerVideoSampler(
    #     clip_duration=clip_duration, clips_per_video=clips_per_video
    # )
    clip_sampler = UniformClipSampler(
        clip_duration=clip_duration, 
        # num_clips=clips_per_video
    )

    all_clips_timepoints = data.get_clip_timepoints(
        clip_sampler, waveform.size(1) / sample_rate
    )
    all_clips = []
    for clip_timepoints in all_clips_timepoints:
        waveform_clip = waveform[
            :,
            int(clip_timepoints[0] * sample_rate) : int(
                clip_timepoints[1] * sample_rate
            ),
        ]
        waveform_melspec = data.waveform2melspec(
            waveform_clip, sample_rate, num_mel_bins, target_length
        )
        all_clips.append(waveform_melspec)

    normalize = transforms.Normalize(mean=mean, std=std)
    all_clips = [normalize(ac).to(device) for ac in all_clips]
    all_clips = torch.stack(all_clips, dim=0)
    
    # audio_outputs.append(all_clips)

    return all_clips
    # return torch.stack(audio_outputs, dim=0)