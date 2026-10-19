from .mesh2speech.PreprocessorBase import PreprocessorBase 
import sys
from typing import Any, Optional, Dict, List
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import torchaudio
import warnings
from inferno.utils.other import get_path_to_externals
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

# Import the original SpeechMeshTransformer
try:
    get_path_to_ext = get_path_to_externals()
    if str(get_path_to_ext) not in sys.path:
        sys.path.append(str(get_path_to_ext))
    from Perceptual_3D_Talking_Head.src.model.modeling import SpeechMeshTransformer
    from Perceptual_3D_Talking_Head.src import utils
    SPEECH_MESH_TRANSFORMER_AVAILABLE = True
except ImportError as e:
    SPEECH_MESH_TRANSFORMER_AVAILABLE = False
    warnings.warn(f"SpeechMeshTransformer not available: {e}. Using fallback implementation.")


def load_audio_mel(fname):
    """Load audio and convert to mel-spectrogram features for perceptual loss computation"""
    audio, sample_rate = torchaudio.load(fname)
    if not sample_rate == 16000:
        audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)  # vocaset [1, 54400]
    audio_len = audio.shape[1]
    audios = []
    i = 0
    while i*640+3200 <= audio_len:
        audio_start_idx = i *640
        waveform = audio[:, audio_start_idx:audio_start_idx + 3200]
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=16000, frame_length=8,
                                                  use_energy=False,
                                                  window_type='hanning', num_mel_bins=128, dither=0.0,
                                                  frame_shift=10)
        fbank = (fbank +4.2677393) / (4.5689974 * 2)
        audios.append(fbank)  # (Time, Freq), i.e., (seq_len, num_mel_bin)
        i += 1
    return audios


def process_audio_tensor_to_mel(audio_tensor, sample_rate=16000):
    """
    Convert audio tensor to mel-spectrogram features for perceptual loss computation.
    
    Args:
        audio_tensor: Tensor of shape [B, T] containing audio waveforms
        sample_rate: Sample rate of the audio (default: 16000)
        
    Returns:
        List of lists: Each batch element contains a list of mel-spectrogram tensors
    """
    batch_size = audio_tensor.shape[0]
    batch_audios = []
    
    for b in range(batch_size):
        audio = audio_tensor[b:b+1, :]  # Keep batch dimension [1, T]
        
        # Resample if needed
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        
        audio_len = audio.shape[1]
        audios = []
        i = 0
        
        while i*640+3200 <= audio_len:
            audio_start_idx = i * 640
            waveform = audio[:, audio_start_idx:audio_start_idx + 3200]
            waveform = waveform - waveform.mean()
            
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform, 
                htk_compat=True, 
                sample_frequency=16000, 
                frame_length=8,
                use_energy=False,
                window_type='hanning', 
                num_mel_bins=128, 
                dither=0.0,
                frame_shift=10
            )
            fbank = (fbank + 4.2677393) / (4.5689974 * 2)
            audios.append(fbank)  # (Time, Freq), i.e., (seq_len, num_mel_bin)
            i += 1
            
        batch_audios.append(audios)
    
    return batch_audios


def process_audio_tensor_to_mel_vectorized(audio_tensor, sample_rate=16000):
    """
    Vectorized version: Convert audio tensor to mel-spectrogram features for perceptual loss computation.
    
    Args:
        audio_tensor: Tensor of shape [B, T] containing audio waveforms
        sample_rate: Sample rate of the audio (default: 16000)
        
    Returns:
        Tensor of shape [B, N, seq_len, 128] where N is max number of clips per batch
        Also returns a mask tensor [B, N] indicating valid clips
    """
    batch_size, audio_len = audio_tensor.shape
    
    # # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio_tensor = resampler(audio_tensor)
        audio_len = audio_tensor.shape[1]
    
    # Calculate maximum number of clips possible
    window_size = 3200
    stride = 640
    max_clips = max(1, (audio_len - window_size) // stride + 1)
    
    # Create all possible windowed segments
    # Use unfold to create sliding windows: [B, T] -> [B, max_clips, window_size]
    if audio_len >= window_size:
        # Pad audio to ensure we can extract max_clips windows
        pad_size = max(0, stride * (max_clips - 1) + window_size - audio_len)
        if pad_size > 0:
            audio_tensor = F.pad(audio_tensor, (0, pad_size))
        
        # Extract sliding windows: [B, max_clips, window_size]
        windowed_audio = audio_tensor.unfold(1, window_size, stride)[:, :max_clips, :]
        
        # Mean normalize each window: [B, max_clips, window_size]
        window_means = windowed_audio.mean(dim=2, keepdim=True)
        windowed_audio = windowed_audio - window_means
        
        # Reshape for batch processing: [B*max_clips, 1, window_size]
        windowed_audio_flat = windowed_audio.reshape(-1, 1, window_size)
        
        # Process all windows using Kaldi fbank
        # Note: Kaldi fbank doesn't support true batching, process samples individually
        fbank_features = []
        
        # Process each sample individually (Kaldi limitation)
        for i in range(windowed_audio_flat.shape[0]):
            sample = windowed_audio_flat[i]  # Shape: [1, window_size] - remove batch dimension
            
            fbank_sample = torchaudio.compliance.kaldi.fbank(
                sample,
                htk_compat=True,
                sample_frequency=16000,  # Note: original uses 16000 even though audio is resampled to 16000
                frame_length=8,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
            )
            fbank_features.append(fbank_sample)
        
        # Concatenate all chunks: [total_frames, mel_bins]
        fbank_all = torch.cat(fbank_features, dim=0)
        
        # Apply normalization
        fbank_all = (fbank_all + 4.2677393) / (4.5689974 * 2)
        
        # Reshape back to batch format: [B, max_clips, seq_len, mel_bins]
        # Each sample has the same seq_len since they all use the same window_size
        seq_len_per_sample = fbank_features[0].shape[0]  # frames per sample
        mel_bins = fbank_all.shape[1]
        fbank_batch = fbank_all.reshape(batch_size, max_clips, seq_len_per_sample, mel_bins)
        
        # Create validity mask (all clips are valid since we calculated max_clips properly)
        valid_mask = torch.ones(batch_size, max_clips, dtype=torch.bool, device=audio_tensor.device)
        
    else:
        # Audio too short, return empty features
        fbank_batch = torch.zeros(batch_size, 1, 1, 128, device=audio_tensor.device)
        valid_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=audio_tensor.device)
    
    return fbank_batch, valid_mask


def convert_vectorized_to_list_format(fbank_batch, valid_mask):
    """
    Convert vectorized output back to list format for compatibility.
    
    Args:
        fbank_batch: Tensor of shape [B, N, seq_len, 128]
        valid_mask: Tensor of shape [B, N]
        
    Returns:
        List of lists: Each batch element contains a list of mel-spectrogram tensors
    """
    batch_audios = []
    batch_size = fbank_batch.shape[0]
    
    for b in range(batch_size):
        audios = []
        valid_clips = valid_mask[b]
        for n in range(valid_clips.sum().item()):
            if valid_clips[n]:
                audios.append(fbank_batch[b, n])
        batch_audios.append(audios)
    
    return batch_audios


import kaldifeat

def process_audio_tensor_to_mel_torchaudio_batched(audio_tensor: torch.Tensor, sample_rate: int = 16000, return_list: bool = False):
    """
    Batched version using exact same torchaudio.compliance.kaldi.fbank as original.
    This produces IDENTICAL results to the original implementation.
    """
    assert audio_tensor.dim() == 2, "audio_tensor must be [B, T]"
    device = audio_tensor.device
    dtype = audio_tensor.dtype

    # ---- Resample to 16000 Hz to match original implementation ----
    target_sr = 16000
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr).to(device=device, dtype=dtype)
        x = resampler(audio_tensor)
    else:
        x = audio_tensor

    # ---- 200 ms windows every 40 ms (3200 / 640 samples), like the original loop ----
    win_len = 3200
    hop = 640
    if x.shape[1] <= win_len:
        # Handle edge case: exactly win_len samples should produce 1 clip
        if x.shape[1] == win_len:
            chunks = x.unsqueeze(1)  # [B, 1, 3200]
        else:
            if return_list:
                return [[] for _ in range(x.shape[0])]
            else:
                B = x.shape[0]
                return torch.zeros(B, 1, 1, 128, device=device), torch.zeros(B, 1, dtype=torch.bool, device=device)
    else:
        chunks = x.unfold(dimension=1, size=win_len, step=hop)   # [B, N, 3200]

    # Per-chunk mean subtraction (your: waveform = waveform - waveform.mean())
    chunks = chunks - chunks.mean(dim=-1, keepdim=True)

    # Flatten [B, N, 3200] -> [B*N, 3200]
    B, N, L = chunks.shape
    flat = chunks.reshape(B * N, L)

    # ---- Use EXACT same torchaudio.compliance.kaldi.fbank as original ----
    # Process each sample individually to match original exactly
    fbank_features = []
    
    for i in range(flat.shape[0]):
        sample = flat[i:i+1]  # Keep batch dimension: [1, 3200]
        
        # Use EXACT same parameters as original
        fbank_sample = torchaudio.compliance.kaldi.fbank(
            sample,
            htk_compat=True,
            sample_frequency=16000,  # Note: original uses 16000 even though audio is resampled to 16000
            frame_length=8,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10
        )
        fbank_features.append(fbank_sample)

    # Stack all features: [B*N, seq_len, mel_bins]
    if fbank_features:
        fbank_all = torch.cat(fbank_features, dim=0)
        
        # Apply normalization (EXACT same as original)
        fbank_all = (fbank_all + 4.2677393) / (4.5689974 * 2)
        
        # Reshape back to batch format: [B, N, seq_len, mel_bins]
        seq_len_per_sample = fbank_features[0].shape[0]  # frames per sample
        mel_bins = fbank_all.shape[1]
        fbank_batch = fbank_all.reshape(B, N, seq_len_per_sample, mel_bins)
        
        # Create validity mask (all clips are valid)
        valid_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    else:
        fbank_batch = torch.zeros(B, N, 1, 128, device=device)
        valid_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

    # Return based on flag
    if return_list:
        # Convert to list-of-lists format
        batch_audios = []
        for b in range(B):
            audios = []
            valid_clips = valid_mask[b]
            for n in range(valid_clips.sum().item()):
                if valid_clips[n]:
                    audios.append(fbank_batch[b, n])
            batch_audios.append(audios)
        return batch_audios
    else:
        return fbank_batch, valid_mask


def process_audio_tensor_to_mel_kaldi_batched(fbank, audio_tensor: torch.Tensor, sample_rate: int = 16000, return_list: bool = False):
    """
    Batched Kaldi-compliant fbank with your exact hyperparameters.
    Input:  audio_tensor [B, T] at `sample_rate`
    Output: 
        - If return_list=True: list of length B; each item is a list of [time_frames, 128] tensors (original format)
        - If return_list=False: tuple of (fbank_batch, valid_mask) where:
            - fbank_batch: [B, N, seq_len, 128] tensor
            - valid_mask: [B, N] tensor indicating valid clips
    """
    assert audio_tensor.dim() == 2, "audio_tensor must be [B, T]"
    device = audio_tensor.device
    dtype = audio_tensor.dtype

    # ---- Resample to 16000 Hz to match original implementation ----
    target_sr = 16000
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr).to(device=device, dtype=dtype)
        x = resampler(audio_tensor)
    else:
        x = audio_tensor

    # ---- 200 ms windows every 40 ms (3200 / 640 samples @19.2 kHz), like your loop ----
    win_len = 3200
    hop = 640
    if x.shape[1] <= win_len:
        # Handle edge case: exactly win_len samples should produce 1 clip
        if x.shape[1] == win_len:
            chunks = x.unsqueeze(1)  # [B, 1, 3200]
        else:
            if return_list:
                return [[] for _ in range(x.shape[0])]
            else:
                B = x.shape[0]
                return torch.zeros(B, 1, 1, 128, device=device), torch.zeros(B, 1, dtype=torch.bool, device=device)
    else:
        chunks = x.unfold(dimension=1, size=win_len, step=hop)   # [B, N, 3200]

    # Per-chunk mean subtraction (your: waveform = waveform - waveform.mean())
    chunks = chunks - chunks.mean(dim=-1, keepdim=True)

    # Flatten [B, N, 3200] -> [B*N, 3200]
    B, N, L = chunks.shape
    flat = chunks.reshape(B * N, L)

    # ---- Kaldi fbank with your exact hyperparameters ----
    # Note: Original uses sample_frequency=16000 even though audio is at 16000 Hz
    if fbank is None:
        opts = kaldifeat.FbankOptions()
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_length_ms = 8.0
        opts.frame_opts.frame_shift_ms = 10.0
        opts.frame_opts.window_type = "hanning"
        opts.frame_opts.dither = 0.0
        # your while-loop only accepted full frames, which matches Kaldi's default snip_edges=True
        opts.frame_opts.snip_edges = True
        opts.use_energy = False
        opts.mel_opts.num_bins = 128
        opts.mel_opts.htk_mode = True  # htk_compat=True

        fbank = kaldifeat.Fbank(opts)
        
        # Move to device if needed
        if device != torch.device('cpu'):
            fbank = fbank.to(device)

    # kaldifeat expects a list of 1D tensors, one for each audio sample
    flat_list = [flat[i] for i in range(flat.shape[0])]  # Convert to list of 1D tensors
    feats = fbank(flat_list)  # -> List[Tensor] of shape [time_frames, num_mel_bins]

    # Return based on flag
    if return_list:
        # ---- Repack to list-of-lists by batch and chunk ----
        feats = [(f + 4.2677393) / (4.5689974 * 2) for f in feats]
        out = []
        idx = 0
        for b in range(B):
            per_b = []
            for _ in range(N):
                per_b.append(feats[idx])  # each [time_frames, 128]
                idx += 1
            out.append(per_b)
        return out
    else:
        # ---- Return as tensor directly ----
        # Stack all features into a tensor and reshape
        if feats:
            fbank_tensor = torch.stack(feats, dim=0)  # [B*N, seq_len, mel_bins]
            fbank_tensor = (fbank_tensor + 4.2677393) / (4.5689974 * 2)  # Normalize as per your original code
            seq_len, mel_bins = fbank_tensor.shape[1], fbank_tensor.shape[2]
            fbank_batch = fbank_tensor.view(B, N, seq_len, mel_bins)  # [B, N, seq_len, mel_bins]
            valid_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            fbank_batch = torch.zeros(B, N, 1, 128, device=device)
            valid_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        
        return fbank_batch, valid_mask



class FallbackSpeechMeshTransformer(nn.Module):
    """Fallback implementation when original SpeechMeshTransformer is not available"""
    
    def __init__(self, **kwargs):
        super().__init__()
        warnings.warn("Using fallback SpeechMeshTransformer. Original model not available.")
        
    def compute_percp_loss(self, mesh, audio, save_feature=False):
        # Return zero loss if fallback is used
        return torch.tensor(0.0, device=mesh.device if hasattr(mesh, 'device') else 'cpu')


class ContrastiveSpeechMeshProcessor(PreprocessorBase):
    """
    Processor that computes contrastive perceptual loss between mesh motion and audio.
    Inspired by the Perceptual-3D-Talking-Head approach.
    """

    def __init__(self, 
                 guidance_model_path: str,
                 guidance_weight: float = 1e-7,
                 guidance_batch_size: int = 80,
                 vertex_size: int = 5023 * 3,
                 max_length: int = 5,
                 input_prefix: str = "predicted_",
                 mel_processing_method: str = "kaldi_batched",
                 max_feature_batch_size = 512,
                 **kwargs):
        
        super().__init__()
        
        self.guidance_weight = guidance_weight
        self.guidance_batch_size = guidance_batch_size
        self.max_length = max_length
        self.input_prefix = input_prefix
        self.mel_processing_method = mel_processing_method 
        self.fbank = None
        self.max_feature_batch_size = max_feature_batch_size
        # The filterbank is only needed when computing the perceptual loss.
        # Keep checkpoint loading and inference independent of its initialization.

        # Initialize the guidance model (SpeechMeshTransformer)
        if SPEECH_MESH_TRANSFORMER_AVAILABLE:
            self.guidance_model = SpeechMeshTransformer(
                vertex_size=vertex_size,
                patch_size=vertex_size,
                embed_dim=512,
                num_heads=8,
                depth=10,
                # audio
                img_size_audio=(64, 128),  # (T, F)
                patch_size_audio=16,
                embed_dim_audio=512,
                num_heads_audio=8,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                max_length=max_length,
                depth_audio=10,
            )
        else:
            self.guidance_model = FallbackSpeechMeshTransformer()
        
        # Load pretrained weights

        if not Path(guidance_model_path).is_absolute(): 
            guidance_model_path = get_path_to_externals() / guidance_model_path 

        assert guidance_model_path.exists(), "guidance_model_path must be provided"

        if guidance_model_path and Path(guidance_model_path).exists() and SPEECH_MESH_TRANSFORMER_AVAILABLE:
            print(f"Loading guidance model from {guidance_model_path}")
            checkpoint = torch.load(guidance_model_path, map_location='cpu')
            
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
                
            # Load state dict using utils
            utils.load_state_dict(self.guidance_model, checkpoint_model)
        
        # Freeze the guidance model
        for param in self.guidance_model.parameters():
            param.requires_grad = False
        
        self.guidance_model.eval()
        
    def _get_fbank(self, device):
        if self.fbank is None:
            opts = kaldifeat.FbankOptions()
            opts.frame_opts.samp_freq = 16000
            opts.frame_opts.frame_length_ms = 8.0
            opts.frame_opts.frame_shift_ms = 10.0
            opts.frame_opts.window_type = "hanning"
            opts.frame_opts.dither = 0.0
            # your while-loop only accepted full frames, which matches Kaldi's default snip_edges=True
            opts.frame_opts.snip_edges = True
            opts.use_energy = False
            opts.mel_opts.num_bins = 128
            opts.mel_opts.htk_mode = True  # htk_compat=True

            self.fbank = kaldifeat.Fbank(opts)
        self.fbank = self.fbank.to(device)
        return self.fbank

    def _load_state_dict_with_mismatch(self, model, state_dict):
        """Load state dict with handling for mismatched keys"""
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        
        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
        # Load only matching keys
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        model.load_state_dict(filtered_state_dict, strict=False)
        
    def to(self, device):
        self.guidance_model = self.guidance_model.to(device)
        if self.fbank is not None:
            self.fbank = self.fbank.to(device)
        return self
    
    @property
    def device(self):
        return next(self.guidance_model.parameters()).device

    def _compute_features_batched(self, motion_batch: torch.Tensor, fbank_batch: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, Any]:
        B, T, V3 = motion_batch.shape
        B, N, seq_len, mel_bins = fbank_batch.shape
        device = motion_batch.device
        
        # Assume uniform structure: all samples have same valid clip count
        # Take minimum clips across all samples to ensure uniformity
        valid_clips_per_sample = valid_mask.sum(dim=1)  # [B]
        min_valid_clips = valid_clips_per_sample.min().item()
        
        # Calculate uniform clip count for all samples
        clip_num = min(min_valid_clips, T - 4)
        
        if clip_num < self.max_length:
            return torch.tensor(0.0, device=device)
        
        # Extract first clip_num valid clips for all samples: [B, clip_num, seq_len, 128]
        uniform_fbank = fbank_batch[:, :clip_num, :, :]  # [B, clip_num, seq_len, 128]
        
        # Vectorized motion sequence extraction for ALL samples at once
        # Use unfold to extract sliding windows of motion sequences
        # motion_batch: [B, T, V*3] -> [B, clip_num, max_length, V*3]
        
        if T >= self.max_length and clip_num > 0:
            # Use unfold to create sliding windows: [B, T, V*3] -> [B, clips, V*3, max_length]
            motion_unfolded = motion_batch.unfold(1, self.max_length, 1)
            # Transpose to get [B, clips, max_length, V*3]
            motion_sequences = motion_unfolded.transpose(-2, -1)[:, :clip_num, :, :]
        else:
            # Handle edge case
            motion_sequences = torch.zeros(B, clip_num, self.max_length, V3, device=device, dtype=motion_batch.dtype)
        
        # Reshape for single batch processing: [B*clip_num, max_length, V*3]
        motion_flat = motion_sequences.reshape(B * clip_num, self.max_length, V3)
        
        # Reshape audio for single batch processing: [B*clip_num, seq_len, 128]
        audio_flat = uniform_fbank.reshape(B * clip_num, seq_len, mel_bins)
        
        # Add channel dimension: [B*clip_num, 1, seq_len, 128]
        audio_batch_tensor = audio_flat.unsqueeze(1)
        
        # Extract features for ALL motion and audio clips in one forward pass!
        max_feature_bach_size = self.max_feature_batch_size  
        vertex_features = [] 
        audio_features = []
        num_feature_batches = motion_flat.shape[0] // max_feature_bach_size
        if motion_flat.shape[0] % max_feature_bach_size != 0:
            num_feature_batches += 1
        for fb in range(0, num_feature_batches):
            start_idx = fb * max_feature_bach_size
            end_idx = min((fb + 1) * max_feature_bach_size, motion_flat.shape[0])
            motion_batch_part = motion_flat[start_idx:end_idx]
            audio_batch_part = audio_batch_tensor[start_idx:end_idx]
            
            # Forward pass through the guidance model
            # if fb == 0:
            #     vertex_features, audio_features = self.guidance_model.forward_features(motion_batch_part, audio_batch_part)
            # else:
            v_features, a_features = self.guidance_model.forward_features(motion_batch_part, audio_batch_part)
            vertex_features.append(v_features)
            audio_features.append(a_features)
        vertex_features = torch.cat(vertex_features, dim=0)
        audio_features = torch.cat(audio_features, dim=0)

        # vertex_features, audio_features = self.guidance_model.forward_features(motion_flat, audio_batch_tensor)

        return vertex_features, audio_features, clip_num
    

    def _compute_perceptual_loss_iterative(self, vertex_feature: torch.Tensor, audio_feature: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:

        losses = []
        
        # ## split the first dimension into batch_size, -1 and the rest 
        # vertex_feature = vertex_feature.reshape(batch_size, -1, *vertex_feature.shape[1:])  # [B, clips, feature_dim]
        # audio_feature = audio_feature.reshape(batch_size, -1, *audio_feature.shape[1:])  # [B, clips, feature_dim]

        total_batch_size = vertex_feature.shape[0] // self.guidance_batch_size
        if vertex_feature.shape[0] % self.guidance_batch_size != 0:
            total_batch_size += 1

        for b in range(total_batch_size):
            
            # vertex_feature_b = vertex_feature[b]  # [clips, feature_dim]
            # audio_feature_b = audio_feature[b]  # [clips, feature_dim]

            vertex_feature_b = vertex_feature[b*self.guidance_batch_size: (b+1)*self.guidance_batch_size]  # [clips, feature_dim]
            audio_feature_b = audio_feature[b*self.guidance_batch_size: (b+1)*self.guidance_batch_size]  # [clips, feature_dim]

            # cosine similarity as logits
            contrastive_labels = torch.arange(vertex_feature_b.shape[0], dtype=torch.long).to(audio_feature_b.device,
                                                                                            non_blocking=True)
            logits_per_video = vertex_feature_b @ audio_feature_b.t() / 0.07
            logits_per_audio = logits_per_video.t()
            loss = 0.5 * (
                    self.guidance_model.loss_func(logits_per_video, contrastive_labels) +
                    self.guidance_model.loss_func(logits_per_audio, contrastive_labels)
            )
            losses += [loss]
        return torch.stack(losses)


    def _compute_perceptual_loss_iterative_v2(self, batch_size, vertex_feature: torch.Tensor, audio_feature: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:

        losses = []
        
        # ## split the first dimension into batch_size, -1 and the rest 
        vertex_feature = vertex_feature.reshape(batch_size, -1, *vertex_feature.shape[1:])  # [B, clips, feature_dim]
        audio_feature = audio_feature.reshape(batch_size, -1, *audio_feature.shape[1:])  # [B, clips, feature_dim]

        for b in range(batch_size):
            loss = self._compute_perceptual_loss_iterative(
                vertex_feature[b], audio_feature[b], valid_mask[b]
            )
            losses.append(loss.mean())
        return torch.stack(losses)


    def _compute_perceptual_loss_fully_batched(self, motion_batch: torch.Tensor, fbank_batch: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using truly fully batched processing - NO FOR LOOPS!
        Uses forward_features directly for maximum batching efficiency.
        IMPORTANT: Maintains sample-wise independence for contrastive loss computation.
        
        Args:
            motion_batch: Tensor of shape [B, T, V*3] representing batch of vertex motions
            fbank_batch: Tensor of shape [B, N, seq_len, 128] containing batch of mel-spectrogram features
            valid_mask: Tensor of shape [B, N] indicating valid clips for each sample
            
        Returns:
            Perceptual loss scalar
        """
        B, T, V3 = motion_batch.shape
        B, N, seq_len, mel_bins = fbank_batch.shape
        device = motion_batch.device

        vertex_features, audio_features, clip_num = self._compute_features_batched(motion_batch, fbank_batch, valid_mask)

        # loss = self._compute_perceptual_loss_iterative(B, vertex_features, audio_features, valid_mask)
        loss = self._compute_perceptual_loss_iterative_v2(B, vertex_features, audio_features, valid_mask)
        
        # return loss.sum()
        return loss.mean()

    def _compute_perceptual_loss_batched(self, motion: torch.Tensor, fbank_batch: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss using batched tensor audio features with vectorized motion processing.
        
        Args:
            motion: Tensor of shape [T, V*3] representing vertex motion
            fbank_batch: Tensor of shape [N, seq_len, 128] containing mel-spectrogram features
            valid_mask: Tensor of shape [N] indicating valid clips
            
        Returns:
            Perceptual loss scalar
        """
        # Count valid audio clips
        valid_clips = valid_mask.sum().item()
        clip_num = min(valid_clips, motion.shape[0] - 4)
        
        if clip_num < self.max_length:
            return torch.tensor(0.0, device=motion.device)
        
        # Use only valid clips up to clip_num
        valid_indices = torch.where(valid_mask)[0][:clip_num]
        valid_fbank = fbank_batch[valid_indices]  # [clip_num, seq_len, 128]
        
        # Vectorized motion sequence extraction
        # Create indices for all motion sequences: [clip_num, max_length]
        start_indices = torch.arange(clip_num, device=motion.device).unsqueeze(1)  # [clip_num, 1]
        sequence_offsets = torch.arange(self.max_length, device=motion.device).unsqueeze(0)  # [1, max_length]
        motion_indices = start_indices + sequence_offsets  # [clip_num, max_length]
        
        # Extract all motion sequences at once: [clip_num, max_length, V*3]
        motion_sequences = motion[motion_indices]
        
        batch_num = clip_num // self.guidance_batch_size
        g_loss_sum = 0
        
        # Process full batches
        for b in range(batch_num):
            batch_start = b * self.guidance_batch_size
            batch_end = batch_start + self.guidance_batch_size
            
            # Get motion and audio batches
            motion_batch_tensor = motion_sequences[batch_start:batch_end]  # [batch_size, max_length, V*3]
            audio_batch_features = valid_fbank[batch_start:batch_end]  # [batch_size, seq_len, 128]
            
            # Convert to expected format [B, C=1, T, F]
            audio_batch_tensor = audio_batch_features.unsqueeze(1)  # [batch_size, 1, seq_len, 128]
            
            g_loss = self.guidance_model.compute_percp_loss(
                motion_batch_tensor, audio_batch_tensor
            )
            g_loss_sum += g_loss
        
        # Process remaining samples
        remain_num = clip_num % self.guidance_batch_size
        if remain_num >= 2:
            batch_start = batch_num * self.guidance_batch_size
            batch_end = batch_start + remain_num
            
            # Get remaining motion and audio batches
            motion_batch_tensor = motion_sequences[batch_start:batch_end]  # [remain_num, max_length, V*3]
            audio_batch_features = valid_fbank[batch_start:batch_end]  # [remain_num, seq_len, 128]
            
            # Convert to expected format [B, C=1, T, F]
            audio_batch_tensor = audio_batch_features.unsqueeze(1)  # [remain_num, 1, seq_len, 128]
            
            g_loss = self.guidance_model.compute_percp_loss(
                motion_batch_tensor, audio_batch_tensor
            )
            g_loss_sum += g_loss
            batch_num += 1
        
        if batch_num > 0:
            return g_loss_sum / batch_num
        else:
            return torch.tensor(0.0, device=motion.device)

    def _compute_perceptual_loss(self, motion: torch.Tensor, audios: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute perceptual loss using the rep_output logic from the original implementation.
        
        Args:
            motion: Tensor of shape [T, V*3] representing vertex motion
            audios: List of mel-spectrogram tensors
            
        Returns:
            Perceptual loss scalar
        """
        audio_length = len(audios)
        clip_num = min(audio_length, motion.shape[0] - 4)
        
        if clip_num < self.max_length:
            return torch.tensor(0.0, device=motion.device)
            
        audios = audios[:clip_num]
        batch_num = clip_num // self.guidance_batch_size
        g_loss_sum = 0
        
        # Process full batches
        for b in range(batch_num):
            motion_batch = []
            for i in range(self.guidance_batch_size):
                start_idx = b * self.guidance_batch_size + i
                end_idx = start_idx + self.max_length
                motion_batch.append(motion[start_idx:end_idx])
            
            motion_batch_tensor = torch.stack(motion_batch, dim=0).to(motion.device)  # [batch_size, 5, V*3]
            
            # Stack audio tensors
            audio_start = b * self.guidance_batch_size
            audio_end = audio_start + self.guidance_batch_size
            audio_batch = audios[audio_start:audio_end]
            
            # Convert list of tensors to batch tensor
            max_audio_len = max(a.shape[0] for a in audio_batch)
            audio_batch_tensor = torch.zeros(len(audio_batch), 1, max_audio_len, audio_batch[0].shape[1]).to(motion.device)  # [B, C=1, T, F]
            for i, audio_tensor in enumerate(audio_batch):
                audio_len = audio_tensor.shape[0]
                audio_batch_tensor[i, 0, :audio_len, :] = audio_tensor
            
            g_loss = self.guidance_model.compute_percp_loss(
                motion_batch_tensor, audio_batch_tensor
            )
            g_loss_sum += g_loss
        
        # Process remaining samples
        remain_num = clip_num % self.guidance_batch_size
        if remain_num >= 2:
            motion_batch = []
            for i in range(remain_num):
                start_idx = batch_num * self.guidance_batch_size + i
                end_idx = start_idx + self.max_length
                motion_batch.append(motion[start_idx:end_idx])
            
            motion_batch_tensor = torch.stack(motion_batch, dim=0).to(motion.device)
            
            audio_start = batch_num * self.guidance_batch_size
            audio_end = audio_start + remain_num
            audio_batch = audios[audio_start:audio_end]
            
            # Convert list of tensors to batch tensor
            max_audio_len = max(a.shape[0] for a in audio_batch)
            audio_batch_tensor = torch.zeros(len(audio_batch), 1, max_audio_len, audio_batch[0].shape[1]).to(motion.device)  # [B, C=1, T, F]
            for i, audio_tensor in enumerate(audio_batch):
                audio_len = audio_tensor.shape[0]
                audio_batch_tensor[i, 0, :audio_len, :] = audio_tensor
            
            g_loss = self.guidance_model.compute_percp_loss(
                motion_batch_tensor, audio_batch_tensor
            )
            g_loss_sum += g_loss
            batch_num += 1
        
        if batch_num > 0:
            return g_loss_sum / batch_num
        else:
            return torch.tensor(0.0, device=motion.device)
    
    def _extract_audio_features(self, audio_path: str) -> List[torch.Tensor]:
        """Extract mel-spectrogram features from audio file"""
        try:
            audio_features = load_audio_mel(audio_path)
            # Convert to tensors
            audio_tensors = [torch.tensor(feat, dtype=torch.float32) for feat in audio_features]
            return audio_tensors
        except Exception as e:
            print(f"Error extracting audio features from {audio_path}: {e}")
            return []
    
    def _forward(self, batch: Dict, train=False, validation=False, input_prefix=None, **kwargs: Any):
        """Forward pass computing perceptual loss"""
        input_prefix = input_prefix or self.input_prefix
        
        # Extract motion data
        assert input_prefix + "vertices" in batch
        motion = batch[input_prefix + "vertices"]  # [B, T, V*3]

        # Extract audio features if available
        perceptual_loss = torch.tensor(0.0, device=motion.device)
        
        if "raw_audio" in batch:
            # Use raw audio tensor [B, T]
            audio_tensor = batch["raw_audio"].view(batch["raw_audio"].shape[0], -1)  # Ensure 2D tensor [B, T]
            sample_rate = batch.get("samplerate", 16000)[0]
            
            # Choose processing method
            if self.mel_processing_method == "torchaudio_batched":
                # Use exact torchaudio batched processing (IDENTICAL to original)
                fbank_batch, valid_mask = process_audio_tensor_to_mel_torchaudio_batched(audio_tensor, sample_rate, return_list=False)
                
                # Use fully batched processing
                perceptual_loss = self._compute_perceptual_loss_fully_batched(motion, fbank_batch, valid_mask)
                    
            elif self.mel_processing_method == "kaldi_batched":
                # Use optimized kaldi batched processing with tensor output
                # Process entire batch at once
                fbank_batch, valid_mask = process_audio_tensor_to_mel_kaldi_batched(self._get_fbank(audio_tensor.device), audio_tensor, sample_rate, return_list=False)
                
                # Use fully batched processing
                perceptual_loss = self._compute_perceptual_loss_fully_batched(motion, fbank_batch, valid_mask)
                    
            elif self.mel_processing_method == "vectorized":
                # Use vectorized approach
                fbank_batch, valid_mask = process_audio_tensor_to_mel_vectorized(audio_tensor, sample_rate)
                batch_audio_features = convert_vectorized_to_list_format(fbank_batch, valid_mask)
                
                total_loss = 0.0
                valid_samples = 0
                
                for i, audio_features in enumerate(batch_audio_features):
                    if i >= motion.shape[0]:
                        break
                        
                    if len(audio_features) > 0:
                        # Move audio features to device
                        audio_features = [feat.to(motion.device) for feat in audio_features]
                        
                        # Compute perceptual loss for this sample
                        sample_motion = motion[i]  # [T, V*3]
                        loss = self._compute_perceptual_loss(sample_motion, audio_features)
                        total_loss += loss
                        valid_samples += 1
                
                if valid_samples > 0:
                    perceptual_loss = total_loss / valid_samples
                    
            else:  # "original"
                # Use original approach
                batch_audio_features = process_audio_tensor_to_mel(audio_tensor, sample_rate)
                
                total_loss = 0.0
                valid_samples = 0
                
                for i, audio_features in enumerate(batch_audio_features):
                    if i >= motion.shape[0]:
                        break
                        
                    if len(audio_features) > 0:
                        # Move audio features to device
                        audio_features = [feat.to(motion.device) for feat in audio_features]
                        
                        # Compute perceptual loss for this sample
                        sample_motion = motion[i]  # [T, V*3]
                        loss = self._compute_perceptual_loss(sample_motion, audio_features)
                        total_loss += loss
                        valid_samples += 1
                
                if valid_samples > 0:
                    perceptual_loss = total_loss / valid_samples
        
        elif "audio_path" in batch:
            # Extract audio features for each sample in batch
            audio_paths = batch["audio_path"] if isinstance(batch["audio_path"], list) else [batch["audio_path"]]
            
            total_loss = 0.0
            valid_samples = 0
            
            for i, audio_path in enumerate(audio_paths):
                if i >= motion.shape[0]:
                    break
                    
                audio_features = self._extract_audio_features(audio_path)
                if len(audio_features) > 0:
                    # Move audio features to device
                    audio_features = [feat.to(motion.device) for feat in audio_features]
                    
                    # Compute perceptual loss for this sample
                    sample_motion = motion[i]  # [T, V*3]
                    loss = self._compute_perceptual_loss(sample_motion, audio_features)
                    total_loss += loss
                    valid_samples += 1
            
            if valid_samples > 0:
                perceptual_loss = total_loss / valid_samples
        
        elif "rep_audio_mel" in batch:
            # Use precomputed mel features
            rep_audio_mel = batch["rep_audio_mel"]
            if isinstance(rep_audio_mel, list) and len(rep_audio_mel) > 0:
                # Handle case where rep_audio_mel is a list of features
                if motion.shape[0] == 1:  # Single sample batch
                    audio_features = [feat.clone().detach().to(motion.device) if torch.is_tensor(feat) 
                                    else torch.tensor(feat, dtype=torch.float32).to(motion.device) 
                                    for feat in rep_audio_mel]
                    sample_motion = motion[0]  # [T, V*3]
                    perceptual_loss = self._compute_perceptual_loss(sample_motion, audio_features)
        
        # Add perceptual loss to batch
        batch["contrastive_perceptual_loss"] = perceptual_loss
        
        return batch

    def forward(self, batch, train=False, validation=False, 
                input_key=None, output_key=None, with_grad=False, **kwargs):
        """Main forward method"""
        if with_grad:
            batch = self._forward(batch, train=train, validation=validation, **kwargs)
        else:
            with torch.no_grad():
                batch = self._forward(batch, train=train, validation=validation, **kwargs)
        
        return batch
