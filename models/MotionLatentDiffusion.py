


import pytorch_lightning as pl 
from typing import Any, Optional
from inferno.models.temporal.AudioEncoders import TemporalAudioEncoder
from inferno.models.temporal.Bases import *
from inferno.sandbox.thunder.models.DiffusionTalkingHead import DiffusionTalkingHead, truncate_sequence_batch
from inferno.sandbox.thunder.models.DiffuserBase import DenoiserBase
from inferno.sandbox.thunder.models.DiffuserHF import DiffuserHF
from inferno.sandbox.thunder.models.MldDenoiser import MldOriginalDenoiser
from .text.TextBase import TextEncoderBase
from .audio.AudioEncoder import PrecomputedAudioFeatureEncoder
# from .text.TextEmbeddingEncoder import TextEmbeddingEncoder
# from inferno.models.temporal.BlockFactory import norm_from_cfg
import random
import omegaconf
from inferno.utils.batch import check_nan, detach_dict
import torch.nn as nn
from inferno.models.temporal.BlockFactory import audio_model_from_cfg
from .BlockFactory import transcript_model_from_cfg, fusion_model_from_cfg

import pytorch_lightning as pl


def audio_encoder_from_cfg(cfg): 
    if cfg.type == "precomputed": 
        model = PrecomputedAudioFeatureEncoder(cfg.dim, cfg.silence_token_init)
    else:
        model = audio_model_from_cfg(cfg)
    return model


def denoiser_from_cfg(cfg, latent_frame_ratio, bottleneck_dim):
    if cfg.type == "mld_original":
        with omegaconf.open_dict(cfg.params):
            cfg.params['cross_attention_frame_ratio'] = latent_frame_ratio
            if not isinstance(cfg.params.latent_dim, int):
                if cfg.params.latent_dim[-1] == 'todo': 
                    assert bottleneck_dim is not None, "Bottleneck dimension needs to be provided for MLD denoiser"
                    cfg.params.latent_dim[-1] = bottleneck_dim
                else:
                    cfg.params['latent_dim'] = cfg.params.latent_dim
                    cfg.params['input_dim'] = bottleneck_dim
                    cfg.params['output_dim'] = bottleneck_dim
            else: 
                if cfg.params.latent_dim == 'todo': 
                    assert bottleneck_dim is not None, "Bottleneck dimension needs to be provided for MLD denoiser"
                    cfg.params.latent_dim = bottleneck_dim
                else: 
                    cfg.params['input_dim'] = bottleneck_dim
                    cfg.params['output_dim'] = bottleneck_dim

        model = MldOriginalDenoiser(cfg)
    else: 
        raise NotImplementedError(f"Denoiser type '{cfg.type}' not implemented")
    return model


def diffuser_from_cfg(cfg):
    if cfg.type == "huggingface": 
        model = DiffuserHF(cfg)
    else: 
        raise NotImplementedError(f"Diffuser type '{cfg.type}' not implemented")
    return model


def language_encoder_from(cfg):
    model = transcript_model_from_cfg(cfg)
    return model


def prior_from_cfg(cfg):
    if cfg.type == 'motion_prior': 
        from inferno.models.talkinghead.FaceFormerDecoder import load_motion_prior_net, MotionPrior
        motion_prior : MotionPrior = load_motion_prior_net(cfg.motion_prior.path, cfg.motion_prior.get('trainable', False))
    elif cfg.type == 'flame_prior': 
        from inferno.sandbox.thunder.models.FlamePriorWrapper import FlameMotionPrior
        motion_prior = FlameMotionPrior(cfg)
    else: 
        raise NotImplementedError(f"Prior type {cfg.type} not implemented")
    return motion_prior


def preprocessor_from_cfg(cfg): 
    from omegaconf import OmegaConf
    params = OmegaConf.to_container(cfg)
    params.pop('type')
    if cfg.type == "MelPreprocessor":
        from .mesh2speech.MelPreprocessor import MelPreprocessor
        preprocessor = MelPreprocessor(**params)
    elif cfg.type == "ClusteringHuBERTPreprocessor":
        from .mesh2speech.HuBERTPreprocessor import ClusteringHuBERTPreprocessor
        preprocessor = ClusteringHuBERTPreprocessor(**params)
    elif cfg.type == "MeshToSpeechPreprocessor":
        from .MeshToSpeechPreprocessor import MeshToSpeechPreprocessor
        # .MeshToSpeechPreprocessor import MeshToSpeechPreprocessor
        preprocessor = MeshToSpeechPreprocessor(**params)
    elif cfg.type == "MotionPriorPreprocessor":
        from .MeshToSpeechPreprocessor import MotionPriorPreprocessor
        preprocessor = MotionPriorPreprocessor(**params)
    elif cfg.type == "OrigDiffPoseTalkStylePreprocessor": 
        from .DiffPoseTalkStylePreprocessor import OrigDiffPoseTalkStylePreprocessor
        preprocessor = OrigDiffPoseTalkStylePreprocessor(**params)
    elif cfg.type == "ContrastiveSpeechMeshProcessor":
        from inferno.sandbox.thunder.models.ContrastiveSpeechMeshProcessor import ContrastiveSpeechMeshProcessor
        preprocessor = ContrastiveSpeechMeshProcessor(**params)
    else: 
        raise NotImplementedError(f"Preprocessor type '{cfg.type}' not implemented")
    return preprocessor



def preprocessors_from_cfg(cfg):
    preprocessors = {}
    for key, preprocessor_cfg in cfg.items():
        preprocessor = preprocessor_from_cfg(preprocessor_cfg)
        preprocessors[key] = preprocessor 
    return preprocessors


class OneHotEncoder(nn.Module): 

    def __init__(self, num_classes, output_dimension, input_key, output_key): 
        super().__init__()
        self.num_classes = num_classes
        self.output_dimension = output_dimension
        self.input_key = input_key
        self.output_key = output_key

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def forward(self, batch, train=False, validation=False,): 
        raise NotImplementedError("OneHotEncoder forward not implemented")


class LinearOneHotEncoder(OneHotEncoder): 

    def __init__(self, num_classes, output_dimension, input_key, output_key): 
        super().__init__( num_classes, output_dimension, input_key, output_key)
        self.linear = nn.Linear(num_classes, output_dimension)

    def forward(self, batch, train=False, validation=False,): 
        one_hot = batch[self.input_key]
        one_hot_vec = torch.nn.functional.one_hot(one_hot, self.num_classes).float()
        one_hot_feature = self.linear(one_hot_vec)
        batch[self.output_key] = one_hot_feature
        return batch
    
class LinearEncoder(OneHotEncoder): 

    def __init__(self, input_dim, output_dimension, input_key, output_key): 
        super().__init__( input_dim, output_dimension, input_key, output_key)
        if output_dimension == 'todo':
            output_dimension = input_dim ## shouldn't be the case, but this fixes the issue (for now)
        self.linear = nn.Linear(input_dim, output_dimension)

    def forward(self, batch, train=False, validation=False,): 
        input_feat = batch[self.input_key]
        output_feat = self.linear(input_feat)
        batch[self.output_key] = output_feat
        return batch


def one_hot_encoders_from_cfg(cfg):
    one_hot_encoders = {}
    for key, one_hot_cfg in cfg.items():
        if one_hot_cfg.type == "LinearEncoder":
            one_hot_encoders[key] = LinearEncoder(one_hot_cfg.input_dim, one_hot_cfg.output_dimension, one_hot_cfg.input_key, one_hot_cfg.output_key)
        elif one_hot_cfg.type == "LinearOneHotEncoder":
            one_hot_encoders[key] = LinearOneHotEncoder(one_hot_cfg.num_classes, one_hot_cfg.output_dimension, one_hot_cfg.input_key, one_hot_cfg.output_key)
        else: 
            raise NotImplementedError(f"One hot encoder type '{one_hot_cfg.type}' not implemented")
    return one_hot_encoders


class MotionLatentDiffusion(DiffusionTalkingHead): 

    def __init__(self,
                 cfg,
                 **kwargs):
        if 'audio' in cfg.model:
            audio_encoder = audio_encoder_from_cfg(cfg.model.audio)
        else: 
            audio_encoder = None
        if 'prior' in cfg.model:
            prior = prior_from_cfg(cfg.model.prior)

            latent_frame_ratio = self._latent_fram_size(prior)
            bottleneck_dim = prior.bottleneck_dim()
        else: 
            prior = None
            latent_frame_ratio = 1
            bottleneck_dim = None
        denoiser = denoiser_from_cfg(cfg.model.denoiser, latent_frame_ratio=latent_frame_ratio, bottleneck_dim=bottleneck_dim)
        diffuser = diffuser_from_cfg(cfg.model.diffuser)
        if 'transcript' in cfg.model:
            transcript_encoder = language_encoder_from(cfg.model.transcript)
        else: 
            transcript_encoder = None
        
        audio_transcript_fusion = None
        # if audio_encoder is not None and transcript_encoder is not None:
        if 'audio_transcript_fusion' in cfg.model:
            audio_transcript_fusion = fusion_model_from_cfg(cfg.model.audio_transcript_fusion)
            assert audio_transcript_fusion is not None, "If both audio and transcript encoder present, there needs to be a fusion module."
            # input_feature_dim = audio_transcript_fusion.output_feature_dim(
            #     audio_encoder.output_feature_dim(), 
            #     transcript_encoder.output_feature_dim())
        preprocessors = None
        if 'preprocessor' in cfg.model:
            preprocessors = preprocessors_from_cfg(cfg.model.preprocessor)
        postprocessors = None
        if 'postprocessor' in cfg.model:
            postprocessors = preprocessors_from_cfg(cfg.model.postprocessor)

        onehot_encoders = None
        if 'one_hot' in cfg.model:
            onehot_encoders = one_hot_encoders_from_cfg(cfg.model.one_hot)

        super().__init__(cfg, audio_encoder, denoiser, diffuser, transcript_encoder, prior,
                         audio_transcript_fusion=audio_transcript_fusion,
                         preprocessors=preprocessors,
                         postprocessors=postprocessors,
                         one_hot_encoders = onehot_encoders,
                         **kwargs)

    def _temporal_size_from_sample(self, sample): 
        if "audio_feature" in sample.keys():
            T = sample["audio_feature"].shape[1]
            return T
        T = sample["processed_audio"].shape[1] if "processed_audio" in sample.keys() else sample["raw_audio"].shape[1]
        return T

    def _truncate_sequence_batch(self, sample):
        T = self._temporal_size_from_sample(sample)
        if self.max_seq_length < T: # truncate
            print("[WARNING] Truncating audio sequence from {} to {}".format(T, self.max_seq_length))
            sample = truncate_sequence_batch(sample, self.max_seq_length)

        lt = self._latent_fram_size()
        if T % lt != 0:
            print(f"[WARNING] Truncating the entire sequence from {T} to {T - (T % lt)} to allow for decoding by the motion prior. "\
                  f"Avoid doing this in training and set the sequence length to a multiple of {lt}, which is the numbef of real frames in a latent frame "\
                  "for the prior used.")
            sample = truncate_sequence_batch(sample, T - (T % lt))

        return sample

    def _latent_fram_size(self, prior=None):
        prior = prior or self.prior
        q = prior.quant_factor() 
        if q == 0 and prior.__class__.__name__ == 'L2lVqVae':  
            q += 1
        return 2 ** q


    def _compute_latent_mask(self, mask):
        T = mask.shape[1]
        latent_frame_size = self._latent_fram_size()
        latent_T = T // latent_frame_size

        # if one real frame is masked, the entire latent frame is masked
        latent_mask = mask.view(-1, latent_T, latent_frame_size).any(dim=-1)
        latent_mask = latent_mask.float() 
        # latent_mask = latent_mask.unsqueeze(-1).expand(-1, -1, latent_T).reshape(-1, T)
        return latent_mask


    def encode(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict: 
        motion_prior_batch = {} 
        for key in batch:
            if key.startswith('gt_'):
                motion_prior_batch[key] = batch[key]

        motion_prior_batch = self.prior.preprocess(motion_prior_batch, training_or_validation=train or validation)
        motion_prior_batch = self.prior.compose_sequential_input(motion_prior_batch)
        motion_prior_batch = self.prior.encode(motion_prior_batch)
        motion_prior_batch = self.prior.quantize(motion_prior_batch, training_or_validation=train or validation)

        for key in motion_prior_batch:
            if key.startswith('gt_') and key not in batch.keys():
                batch[key] = motion_prior_batch[key]

        batch[self.diffuser.input_key] = motion_prior_batch["encoded_features"]
        return batch 

    def decode(self, batch: Dict, train=False, validation=False, 
               key_to_decode=None, 
               decoded_prefix=None,
               **kwargs: Any) -> Dict: 
        key_to_decode = key_to_decode or self.denoiser.denoised_key
        decoded_prefix = decoded_prefix or "predicted_"
        motion_prior_batch = {} 
        # for key in batch:
        #     if key.startswith('gt_'): ## don't be worried by this, it's not clean but no gt_ is being used in there
        #         motion_prior_batch[key] = batch[key] 

        ## what we don't predict we replace by GT
        if not self.cfg.model.output.get('predict_shapecode', False):
            motion_prior_batch["gt_shape"] = batch["gt_shape"]
        if not self.cfg.model.output.get('predict_expcode', True) or (self.cfg.model.prior.get('replace_gt_exp_posed', False) and (train or validation)): ## predict expression code true by default (backwards compatiblity)
            motion_prior_batch["gt_exp"] = batch["gt_exp"]
        if not self.cfg.model.output.get('predict_jawpose', True) or (self.cfg.model.prior.get('replace_gt_jaw_posed', False) and (train or validation)): ## predict jaw code true by default 
            motion_prior_batch["gt_jaw"] = batch["gt_jaw"]
        motion_prior_batch["encoded_features"] = batch[key_to_decode]

        ## this is not important for the motion diffusion but if it's passed in, the texture will be decoded
        if "gt_tex" in batch.keys():
            motion_prior_batch["gt_tex"] = batch["gt_tex"]

        # batch = self.prior.decode(batch, assert_size=True)
        # batch = self.prior.decompose_sequential_output(batch)
        # batch = self.prior.postprocess(batch)
        motion_prior_batch = self.prior.decode(motion_prior_batch, assert_size=False)
        motion_prior_batch = self.prior.decompose_sequential_output(motion_prior_batch)
        motion_prior_batch = self.prior.postprocess(motion_prior_batch, training_or_validation=train or validation)

        for key in motion_prior_batch:
            if key.startswith('reconstructed_'):
                new_key = key.replace('reconstructed_', decoded_prefix)
                batch[new_key] = motion_prior_batch[key]
        return batch


    def get_input_tensor_shape(self, T, round_up=False): 
        dim = self.prior.bottleneck_dim()
        latent_frame_size = self._latent_fram_size()
        latent_T =  T // latent_frame_size
        if round_up and T % latent_frame_size != 0:
            latent_T += 1
        return [latent_T, dim]

    @property
    def max_seq_length(self):
        return 5000

    def get_trainable_parameters(self):
        trainable_params = super().get_trainable_parameters()
        return trainable_params


    @classmethod
    def instantiate(cls, cfg, stage=None, prefix=None, checkpoint=None, checkpoint_kwargs=None) -> 'MotionLatentDiffusion':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = MotionLatentDiffusion(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = MotionLatentDiffusion.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=False, 
                **checkpoint_kwargs
            )
        return model
