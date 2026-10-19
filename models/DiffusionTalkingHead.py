


import pytorch_lightning as pl 
from typing import Any, Optional
from inferno.models.temporal.AudioEncoders import TemporalAudioEncoder
from inferno.models.temporal.Bases import *
from inferno.sandbox.thunder.models.TalkingHeadWithLanguageBase import TalkingHeadWithLanguageBase, truncate_sequence_batch
from inferno.sandbox.thunder.models.DiffuserBase import DenoiserBase, DiffuserBase, DenoiserObjective
from .text.TextBase import TextEncoderBase
# from .text.TextEmbeddingEncoder import TextEmbeddingEncoder
# from inferno.models.temporal.BlockFactory import norm_from_cfg
import random
import omegaconf
from inferno.utils.batch import check_nan, detach_dict
import torch.nn as nn

from inferno.layers.losses.Masked import MaskedTemporalMSELoss, MaskedTemporalMAELoss
from inferno.layers.losses.RotationLosses import compute_rotation_loss
from inferno.models.talkinghead.FaceFormer import rotation_velocity_loss, velocity_loss
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from munch import Munch
from enum import Enum


class ConditionType(Enum): 
    AUDIO =         "audio"
    TRANSCRIPT =    "transcript"
    CLIP =          "clip"
    SUJECT_ID =     "subject_id"
    EMOTION_ID =    "emotion_id"
    INTENSITY_ID =  "intensity_id"
    STYLE_FEAT =    "style_feature"
    NONE =          "none"
    GRADIENT_GUIDANCE = "grad"


# class DiffusionTalkingHead(TalkingHeadWithLanguageBase): 
class DiffusionTalkingHead(pl.LightningModule): 

    def __init__(self,
                 cfg,
                 audio_encoder: TemporalAudioEncoder,
                 denoiser: DenoiserBase = None,
                 diffuser: DiffuserBase = None, 
                 transcript_encoder: Optional[nn.Module] = None,
                 prior: Optional[nn.Module] = None,
                 renderer: Optional[Renderer] = None,
                 audio_transcript_fusion: Optional[torch.nn.Module] = None, 
                 preprocessors: Optional[Dict] = None,
                 postprocessors: Optional[Dict] = None,
                 one_hot_encoders: Optional[Dict] = None,
                 **kwargs):
        # super().__init__(text_encoder, audio_encoder, language_encoder, **kwargs)
        super().__init__()
        self.cfg = cfg
        # self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.denoiser = denoiser
        self.diffuser = diffuser
        self.transcript_encoder = transcript_encoder
        self.prior = prior
        self.renderer = renderer
        self.audio_transcript_fusion = audio_transcript_fusion
        self.one_hot_encoders = torch.nn.ModuleDict(one_hot_encoders)

        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

        if audio_encoder is not None and self.training_audio_dropout_probability is not None:
            ## register a dropout feature as a parameter 
            audio_dropout_feature = torch.nn.Parameter(torch.zeros(self.audio_encoder.output_feature_dim()), requires_grad=True)
            self.register_parameter("audio_dropout_feature", audio_dropout_feature)
        else: 
            self.audio_dropout_feature = None

        if transcript_encoder is not None and self.training_transcript_dropout_probability is not None:
            ## register a dropout feature as a parameter 
            transcript_dropout_feature = torch.nn.Parameter(torch.zeros(self.transcript_encoder.output_feature_dim()), requires_grad=True)
            self.register_parameter("transcript_dropout_feature", transcript_dropout_feature)
        else: 
            self.transcript_dropout_feature = None
        
        if self.training_clip_dropout_probability is not None:
            ## register a dropout feature as a parameter 
            clip_dropout_feature = torch.nn.Parameter(torch.zeros(self.clip_dim), requires_grad=True)
            self.register_parameter("clip_dropout_feature", clip_dropout_feature)
        else:
            self.clip_dropout_feature = None

        if self.training_id_dropout_probability is not None:
            ## register a dropout feature as a parameter 
            id_dropout_feature = torch.nn.Parameter(torch.zeros(self.subject_feature_dim), requires_grad=True)
            self.register_parameter("id_dropout_feature", id_dropout_feature)
        else:
            self.id_dropout_feature = None

        if self.training_emo_dropout_probability is not None:
            ## register a dropout feature as a parameter 
            emo_dropout_feature = torch.nn.Parameter(torch.zeros(self.emotion_feature_dim), requires_grad=True)
            self.register_parameter("emo_dropout_feature", emo_dropout_feature)
        else:
            self.emo_dropout_feature = None
        
        if self.training_int_dropout_probability is not None:
            ## register a dropout feature as a parameter 
            int_dropout_feature = torch.nn.Parameter(torch.zeros(self.intensity_feature_dim), requires_grad=True)
            self.register_parameter("int_dropout_feature", int_dropout_feature)
        else:
            self.int_dropout_feature = None

        if self.training_style_dropout_probability is not None:
            ## register a dropout feature as a parameter 
            style_dropout_feature = torch.nn.Parameter(torch.zeros(self.style_feature_dim), requires_grad=True)
            self.register_parameter("style_dropout_feature", style_dropout_feature)
        else:
            self.style_dropout_feature = None


    def to(self, *args: Any, **kwargs: Any): 
        if self.denoiser is not None:
            ## don't know why this isn't getting called if not explicit
            self.denoiser.to(*args, **kwargs)
        if self.diffuser is not None:
            ## don't know why this isn't getting called if not explicit
            self.diffuser.to(*args, **kwargs)
        if self.prior is not None:
            ## don't know why this isn't getting called if not explicit
            self.prior.to(*args, **kwargs)

        if self.preprocessors is not None:
            for key, preprocessor in self.preprocessors.items():
                preprocessor.to(*args, **kwargs)
        if self.postprocessors is not None:
            for key, postprocessor in self.postprocessors.items():
                postprocessor.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    @property
    def max_seq_length(self):
        return 5000

    @property
    def training_audio_dropout_probability(self):
        return self.cfg.model.get('training_audio_dropout_probability', None)
    

    @property
    def training_transcript_dropout_probability(self):
        return self.cfg.model.get('training_transcript_dropout_probability', None)

    @property
    def training_clip_dropout_probability(self):
        return self.cfg.model.get('training_clip_dropout_probability', None)
    
    @property
    def training_id_dropout_probability(self):
        return self.cfg.model.get('training_id_dropout_probability', None)
    
    @property
    def training_emo_dropout_probability(self):
        return self.cfg.model.get('training_emo_dropout_probability', None)

    @property
    def training_int_dropout_probability(self):
        return self.cfg.model.get('training_int_dropout_probability', None)
    
    @property
    def training_style_dropout_probability(self):
        return self.cfg.model.get('training_style_dropout_probability', None)
    

    @property
    def per_frame_subject_id(self):
        return self.cfg.model.get('per_frame_subject_id', None)
    
    @property
    def per_frame_emo_id(self):
        return self.cfg.model.get('per_frame_emo_id', None)

    @property
    def per_frame_int_id(self):
        return self.cfg.model.get('per_frame_int_id', None)
    

    @property
    def per_frame_style_feat(self):
        return self.cfg.model.get('per_frame_style_feat', None)

    @property
    def shape_feat_dim(self):
        return self.denoiser.denoiser.shape_encoded_dim

    @property
    def clip_dim(self):
        return self.denoiser.denoiser.clip_encoded_dim
    
    @property
    def subject_feature_dim(self):
        return self.one_hot_encoders["subject_id"].output_dimension
    
    @property
    def emotion_feature_dim(self):
        return self.one_hot_encoders["emotion_id"].output_dimension
        
    @property
    def intensity_feature_dim(self):
        return self.one_hot_encoders["intensity_id"].output_dimension
    
    @property
    def style_feature_dim(self):
        dim = self.one_hot_encoders["style_feature"].output_dimension
        if dim == "todo": ## hack to fix poorly saved configs
            return 128 
        return dim
    @property
    def training_image_clip_selection_strategy(self): 
        return self.cfg.model.get('training_image_clip_selection_strategy', None)

    @property 
    def audio_clip_swap_probability(self): 
        return self.cfg.model.get('audio_clip_swap_probability', None)

    @property
    def text_clip_swap_probability(self):
        return self.cfg.model.get('text_clip_swap_probability', None)

    @property 
    def clean_noised_rotations(self): 
        return self.cfg.model.get('clean_noised_rotations', False)
    
    @property 
    def clean_denoised_rotations(self): 
        return self.cfg.model.get('clean_denoised_rotations', False)

    @property
    def denoise_without_noise_bias(self): 
        return self.cfg.model.diffuser.get('denoise_without_bias', False)

    @property 
    def input_conditions(self):
        condition_list = [] 
        if self.training_audio_dropout_probability is not None and self.training_audio_dropout_probability < 1.0:
            condition_list.append(ConditionType.AUDIO)
        if self.training_transcript_dropout_probability is not None and self.training_transcript_dropout_probability < 1.0:
            condition_list.append(ConditionType.TRANSCRIPT)
        if self.training_clip_dropout_probability is not None and self.training_clip_dropout_probability < 1.0:
            condition_list.append(ConditionType.CLIP)
        if self.training_id_dropout_probability is not None and self.training_id_dropout_probability < 1.0:
            condition_list.append(ConditionType.SUJECT_ID)
        if self.training_emo_dropout_probability is not None and self.training_emo_dropout_probability < 1.0:
            condition_list.append(ConditionType.EMOTION_ID)
        if self.training_int_dropout_probability is not None and self.training_int_dropout_probability < 1.0:
            condition_list.append(ConditionType.INTENSITY_ID)
        if self.training_style_dropout_probability is not None and self.training_style_dropout_probability < 1.0:
            condition_list.append(ConditionType.STYLE_FEAT)
        return condition_list
            
    @property
    def zero_shape(self): 
        # assert not self.random_shape, "zero_shape is not supported when random_shape is True"
        return self.cfg.model.get('zero_shape', False)

    @property
    def random_shape(self):
        assert not self.zero_shape, "random_shape is not supported when zero_shape is True"
        return bool(self.cfg.model.get('random_shape_sigma', False))

    @property
    def random_shape_sigma(self):
        return self.cfg.model.get('random_shape_sigma', 0.01)

    def get_trainable_parameters(self):
        # trainable_params = super().get_trainable_parameters()
        trainable_params = []
        if self.transcript_encoder is not None:
            trainable_params += self.transcript_encoder.get_trainable_parameters()
        if self.audio_encoder is not None:
            trainable_params += self.audio_encoder.get_trainable_parameters() 
        if self.one_hot_encoders is not None:
            for key, one_hot_encoder in self.one_hot_encoders.items():
                trainable_params += one_hot_encoder.get_trainable_parameters()
        if self.denoiser is not None:
            trainable_params += self.denoiser.get_trainable_parameters()
        if self.diffuser is not None:
            trainable_params += self.diffuser.get_trainable_parameters()
        if self.prior is not None: 
            trainable_params += self.prior.get_trainable_parameters()
        if self.audio_transcript_fusion is not None: 
            trainable_params += [p for p in self.audio_transcript_fusion.parameters() if p.requires_grad]
        if self.audio_dropout_feature is not None and self.audio_dropout_feature.requires_grad:
            trainable_params.append(self.audio_dropout_feature) 
        if self.transcript_dropout_feature is not None and self.transcript_dropout_feature.requires_grad:
            trainable_params.append(self.transcript_dropout_feature)
        if self.clip_dropout_feature is not None and self.clip_dropout_feature.requires_grad:
            trainable_params.append(self.clip_dropout_feature)
        if self.id_dropout_feature is not None and self.id_dropout_feature.requires_grad:
            trainable_params.append(self.id_dropout_feature)
        if self.emo_dropout_feature is not None and self.emo_dropout_feature.requires_grad:
            trainable_params.append(self.emo_dropout_feature)
        if self.int_dropout_feature is not None and self.int_dropout_feature.requires_grad:
            trainable_params.append(self.int_dropout_feature)
        if self.style_dropout_feature is not None and self.style_dropout_feature.requires_grad:
            trainable_params.append(self.style_dropout_feature)
        return trainable_params


    def _choose_primary_3D_rec_method(self, batch, train=False, validation=False): 
        method = self.cfg.data.get('reconstruction_type', None)
        # if method is None:
        #     return batch
        if method is not None:    
            if isinstance(method, (list, omegaconf.listconfig.ListConfig)):
                # method = random.choice(method)
                method = method[0]
            if "reconstruction" in batch:
                for k in batch["reconstruction"][method].keys():
                    assert k not in batch.keys(), f"Key '{k}' already exists in batch. We don't want to overwrite it."
                    batch[k] = batch["reconstruction"][method][k]
        
        if "gt_shape" in batch.keys():
            batch["shape_feature"] = batch["gt_shape"][..., :self.shape_feat_dim]

        if "gt_shape" not in batch.keys():
            B, T = batch["gt_exp"].shape[:2]
            # batch["gt_shape"] = torch.zeros(B, T, 300, device=batch["gt_exp"].device)
            batch["gt_shape"] = torch.zeros(B, 300, device=batch["gt_exp"].device)
        elif self.zero_shape and (train or validation):
            batch["gt_shape"] = torch.zeros_like(batch["gt_shape"])
        elif not self.zero_shape and self.random_shape and (train or validation): 
            B, T = batch["gt_exp"].shape[:2] 
            sigma = self.random_shape_sigma
            batch["gt_shape"] = torch.randn(B, 300, device=batch["gt_exp"].device) * sigma
        return batch

    def render_predicted_sequence(self, batch: Dict, train=False, **kwargs):
        if self.renderer is None:
            return batch 
        batch = self.renderer(batch, train=train, input_key_prefix='predicted_', output_prefix='predicted_', **kwargs)
        return batch
    

    def render_gt_sequence(self, batch: Dict, train=False, **kwargs): 
        if self.renderer is None:
            return batch  
        with torch.no_grad():
            if "reconstruction" in batch.keys():
                for method in self.cfg.data.reconstruction_type:
                    batch["reconstruction"][method] = self.renderer(batch["reconstruction"][method], train=train, input_key_prefix='gt_', output_prefix='gt_', **kwargs)
            else:
                batch = self.renderer(batch, train=train, input_key_prefix='gt_', output_prefix='gt_', **kwargs)
        return batch

    @torch.no_grad()
    def preprocess_input(self, batch: Dict, train=False, **kwargs: Any) -> Dict:
        if self.preprocessor is not None:
            if self.device != self.preprocessor.device:
                self.preprocessor.to(self.device)
            batch = self.preprocessor(batch, input_key="video", train=train, test_time=not train, **kwargs)
        # batch = detach_dict(batch)
        return batch 
    
    def forward_prepocessors(self, batch: Dict, train=False, validation=False, with_grad=False, **kwargs: Any) -> Dict:
        if self.preprocessors is not None:
            for key, preprocessor in self.preprocessors.items():
                if with_grad:
                    batch = preprocessor(batch, train=train, validation=validation, input_key=None, with_grad=with_grad, **kwargs)
                else:
                    with torch.no_grad():
                        batch = preprocessor(batch, train=train, validation=validation, input_key=None, **kwargs)
        return batch
        
    def render_predicted_sequence(self, batch: Dict, train=False, **kwargs):
        if self.renderer is None:
            return batch 
        batch = self.renderer(batch, train=train, input_key_prefix='predicted_', output_prefix='predicted_', **kwargs)
        return batch


    def forward_audio(self, batch: Dict, train=False, desired_output_length=None, **kwargs: Any) -> Dict:
        if train or self.training_audio_dropout_probability == 1.0: 
            if self.training_audio_dropout_probability is not None:
                roll = random.random() 
                if roll < self.training_audio_dropout_probability:
                    # print("[INFO]: Dropping audio sequence")
                    # batch["audio_feature"] = torch.zeros_like(batch["audio_feature"])
                    if "raw_audio" in batch.keys():
                        B, T = batch["raw_audio"].shape[:2]
                    else:
                        B, T = batch["audio_feature"].shape[:2]
                    batch["audio_feature"] = self.audio_dropout_feature.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
                    return batch

        return self.audio_encoder(batch, train=train, desired_output_length=desired_output_length, **kwargs)


    def get_dropped_audio_feature(self, B, T):
        return self.audio_dropout_feature.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

    def dropped_audio_feature_to_batch(self, batch): 
        B, T = batch["audio_feature"].shape[:2]
        batch["audio_feature"] = self.get_dropped_audio_feature(B, T)
        return batch

    def forward_transcript(self, batch: Dict, train=False, desired_output_length=None, **kwargs: Any) -> Dict:
        if train or self.training_transcript_dropout_probability == 1.0:
            if self.training_transcript_dropout_probability is not None:
                roll = random.random() 
                if roll < self.training_transcript_dropout_probability:
                    # print("[INFO]: Dropping transcript sequence")
                    # batch["transcript_word_embeddings"] = torch.zeros_like(batch["transcript_word_embeddings"]) 
                    B, T = batch["transcript_word_embeddings_time_synced"].shape[:2]
                    batch["transcript_word_embeddings_time_synced"] = self.transcript_dropout_feature.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
                    return batch
        return self.transcript_encoder(batch, train=train, desired_output_length=desired_output_length, **kwargs)


    def forward_one_hot_encoders(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        if self.one_hot_encoders is not None:
            for key, one_hot_encoder in self.one_hot_encoders.items():
                batch = one_hot_encoder(batch, train=train, validation=validation, **kwargs)
        return batch


    def get_dropped_transcript_feature(self, B, T):
        return self.transcript_dropout_feature.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

    def dropped_transcript_feature_to_batch(self, batch):
        B, T = batch["transcript_word_embeddings_time_synced"].shape[:2]
        batch["transcript_word_embeddings_time_synced"] = self.get_dropped_transcript_feature(B, T)
        return batch 


    def forward_audio_transcript_fusion(self, batch: Dict, train=False, **kwargs: Any) -> Dict:
        return self.audio_transcript_fusion(batch, train=train, **kwargs)


    @property 
    def per_frame_clip(self): 
        split = self.training_image_clip_selection_strategy.split("_")
        return len(split) != 1

    def forward_clip_dropout(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        clip_batch = batch["clip_features"]
        clip_features_name = list(clip_batch.keys())[0]
        B, T = batch["clip_features"][clip_features_name].shape[:2]

        if train or validation: 
            if self.text_clip_swap_probability is not None:
                ## swap the text and clip features based on the probability 
                roll = random.random()
                if roll < self.text_clip_swap_probability:
                    
                    ## vision text features 
                    vision_text_clip_features = batch["vision_text_clip_features"][clip_features_name] 
                    if vision_text_clip_features.ndim == 2:
                        vision_text_clip_features = vision_text_clip_features.unsqueeze(1)
                    vision_clip_features = batch["clip_features"][clip_features_name]

                    ## make sure the clip feature is of the same length as the audio feature (repeat along time dimension if not)
                    if vision_clip_features.shape[1] != vision_text_clip_features.shape[1]:
                        T_clip = vision_clip_features.shape[1]
                        T_text = vision_text_clip_features.shape[1]
                        if T_clip > T_text:
                            ## repeat the text clip feature along time dimension
                            vision_text_clip_features = vision_text_clip_features.repeat(1, T_clip//T_text, 1)
                        elif T_text > T_clip:
                            ## truncate the text clip feature along time dimension
                            vision_text_clip_features = vision_text_clip_features[:, :T_clip]
                    # batch["clip_features"][clip_features_name] = vision_text_clip_features


                    ## audio text features 
                    audio_text_clip_features = batch["audio_text_clip_features"][clip_features_name] 
                    if audio_text_clip_features.ndim == 2:
                        audio_text_clip_features = audio_text_clip_features.unsqueeze(1)
                    audio_clip_features = batch["clip_features"][clip_features_name] 

                    ## make sure the clip feature is of the same length as the audio feature (repeat along time dimension if not)
                    if audio_clip_features.shape[1] != audio_text_clip_features.shape[1]:
                        T_clip = audio_clip_features.shape[1]
                        T_text = audio_text_clip_features.shape[1]
                        if T_clip > T_text:
                            ## repeat the text clip feature along time dimension
                            audio_text_clip_features = audio_text_clip_features.repeat(1, T_clip//T_text, 1)
                        elif T_text > T_clip:
                            ## truncate the text clip feature along time dimension
                            audio_text_clip_features = audio_text_clip_features[:, :T_clip]
                    # batch["clip_features"][clip_features_name] = vision_text_clip_features

                    # text_clip_feature = batch["text_clip_features"][clip_features_name]
                    # ## make sure the clip feature is of the same length as the audio feature (repeat along time dimension if not)
                    # if batch["clip_feature"].shape[1] != text_clip_feature.shape[1]:
                    #     T_clip = batch["clip_feature"].shape[1]
                    #     T_text = text_clip_feature.shape[1]
                    #     if T_clip > T_text:
                    #         ## repeat the text clip feature along time dimension
                    #         text_clip_feature = text_clip_feature.repeat(1, T_clip//T_text, 1)
                    #     elif T_text > T_clip:
                    #         ## truncate the text clip feature along time dimension
                    #         text_clip_feature = text_clip_feature[:, :T_clip]
                    
                    # batch["clip_feature"] = text_clip_feature


        if self.training_image_clip_selection_strategy == "random":
            ## sample a random index from 0 to T for each batch element
            indices_to_keep = torch.randint(0, T, (B,))
            batch["clip_feature"] = clip_batch[clip_features_name][:, indices_to_keep]
        elif self.training_image_clip_selection_strategy.startswith("mean"):
            split = self.training_image_clip_selection_strategy.split("_")
            if len(split) == 1: ## mean
                mask = batch["landmarks_validity"]["mediapipe"].to(dtype=torch.bool) ## account for potential padding 
                # valid_clip_features = clip_batch[clip_features_name][valid_frames]
                ## the invalid frames are zeros, in order to take the mean we need the counts of valid frames 
                valid_frames = mask.sum(dim=1).unsqueeze(-1) 
                batch["clip_feature"] = (clip_batch[clip_features_name] * mask).sum(dim=1, keepdim=True) / valid_frames
                # batch["clip_feature"] = clip_batch[clip_features_name].mean(dim=1, keepdim=True)
            else: 
                interval = int(split[1])
                mask = batch["landmarks_validity"]["mediapipe"].to(dtype=torch.bool)
                valid_frames = mask.sum(dim=1).unsqueeze(-1)
                clip_feats = clip_batch[clip_features_name]
                ## clip features are of size B, T, F
                ## reshape to B, T//interval, interval, F
                T_new = T
                if T % interval != 0:
                    ## pad the sequence to make it divisible by interval
                    pad = interval - (T % interval)
                    clip_feats = torch.cat([clip_feats, torch.zeros(B, pad, clip_feats.shape[-1], device=clip_feats.device)], dim=1)
                    mask = torch.cat([mask, torch.zeros(B, pad, 1, device=mask.device)], dim=1)
                    T_new = T + pad

                clip_feats = clip_feats.view(B, T_new//interval, interval, -1)
                mask = mask.view(B, T_new//interval, interval, 1) 
                valid_frames = mask.sum(dim=2, keepdim=True)
                
                interval_mean_features = (clip_feats * mask).sum(dim=2, keepdim=True) / valid_frames
                ## repeat the mean features to the original sequence length
                interval_mean_features = interval_mean_features.expand(B, T_new//interval, interval, -1)
                ## reshape back to B, T+pad, F
                clip_features = interval_mean_features.reshape(B, T_new, -1)
                ## truncate the sequence to the original length
                batch["clip_feature"] = clip_features[:, :T]

            ## division by zero could have caused NaNs 
            ## replace Nans by zeros
            batch["clip_feature"][batch["clip_feature"].isnan()] = 0.


        if train or validation:
            if self.audio_clip_swap_probability is not None: 
                ## swap the audio and clip features based on the probability 
                roll = random.random()
                if roll < self.audio_clip_swap_probability:
                    audio_clip_feature = batch["audio_clip_features"][clip_features_name] 
                    if audio_clip_feature.ndim == 2:
                        audio_clip_feature = audio_clip_feature.unsqueeze(1)
                    ## make sure the clip feature is of the same length as the audio feature (repeat along time dimension if not)
                    if batch["audio_feature"].shape[1] != audio_clip_feature.shape[1]:
                        T_audio = batch["audio_feature"].shape[1]
                        T_clip = audio_clip_feature.shape[1]
                        if T_audio > T_clip:
                            ## repeat the clip feature along time dimension
                            audio_clip_feature = audio_clip_feature.repeat(1, T_audio//T_clip, 1)
                        elif T_clip > T_audio:
                            ## truncate the clip feature along time dimension
                            audio_clip_feature = audio_clip_feature[:, :T_audio]
                    batch["clip_feature"] = audio_clip_feature


        if train or self.training_clip_dropout_probability == 1.0:
            if self.training_clip_dropout_probability is not None:
                roll = random.random() 
                if roll < self.training_clip_dropout_probability:
                    T_clip = batch["clip_feature"].shape[1] 
                    batch["clip_feature"] = self.clip_dropout_feature.unsqueeze(0).unsqueeze(0).expand(B, T_clip, -1)
                    return batch
        return batch


    def forward_subject_id_dropout(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        B, T = batch["raw_audio"].shape[:2]
        if train or validation:
            if self.training_id_dropout_probability is not None:
                roll = random.random() 
                if roll < self.training_id_dropout_probability:
                    batch["subject_feature"] = self.id_dropout_feature.unsqueeze(0).expand(B, -1)

        if batch["subject_feature"].ndim == 2:
            batch["subject_feature"] = batch["subject_feature"].unsqueeze(1)
            
        if self.per_frame_subject_id:
            if batch["subject_feature"].shape[1] == 1:
                batch["subject_feature"] = batch["subject_feature"].unsqueeze(1).expand(B, T, -1)

        return batch
    
    def forward_emotion_id_dropout(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        B, T = batch["raw_audio"].shape[:2]
        if train or validation:
            if self.training_emo_dropout_probability is not None:
                roll = random.random() 
                if roll < self.training_emo_dropout_probability:
                    batch["emotion_feature"] = self.emo_dropout_feature.unsqueeze(0).expand(B, -1)
        
        if batch["emotion_feature"].ndim == 2:
            batch["emotion_feature"] = batch["emotion_feature"].unsqueeze(1) 
            
        if self.per_frame_emo_id:
            if batch["emotion_feature"].shape[1] == 1:
                batch["emotion_feature"] = batch["emotion_feature"].unsqueeze(1).expand(B, T, -1)

        return batch

    def forward_intensity_id_dropout(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        B, T = batch["raw_audio"].shape[:2]
        if train or validation:
            if self.training_int_dropout_probability is not None:
                roll = random.random() 
                if roll < self.training_int_dropout_probability:
                    batch["intensity_feature"] = self.int_dropout_feature.unsqueeze(0).expand(B, -1)

        if batch["intensity_feature"].ndim == 2:
            batch["intensity_feature"] = batch["intensity_feature"].unsqueeze(1) 
            
        if self.per_frame_int_id:
            if batch["intensity_feature"].shape[1] == 1:
                batch["intensity_feature"] = batch["intensity_feature"].unsqueeze(1).expand(B, T, -1)

        return batch

    def forward_style_feat_dropout(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        B, T = batch["raw_audio"].shape[:2]
        if train or validation:
            if self.training_style_dropout_probability is not None:
                roll = random.random() 
                if roll < self.training_style_dropout_probability:
                    batch["style_feature"] = self.style_dropout_feature.unsqueeze(0).expand(B, -1)

        if batch["style_feature"].ndim == 2:
            batch["style_feature"] = batch["style_feature"].unsqueeze(1)

        if self.per_frame_style_feat: 
            if batch["style_feature"].shape[1] == 1:
                batch["style_feature"] = batch["style_feature"].unsqueeze(1).expand(B, T, -1)

        return batch

    def get_dropped_clip_feature(self, B, T):
        if not self.per_frame_clip: 
            T = 1 
        return self.clip_dropout_feature.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

    def resize_input_clip_feature(self, B, T, clip_feature):
        if not self.per_frame_clip: 
            T = 1 
        return clip_feature.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
    
    def dropped_clip_feature_to_batch(self, batch, clip_feature=None):
        B, T = batch["raw_audio"].shape[:2]
        if clip_feature is not None:
            batch["clip_feature"] = self.resize_input_clip_feature(B, T, clip_feature)
        else:
            batch["clip_feature"] = self.get_dropped_clip_feature(B, T)
        return batch

    def dropped_subject_id_feature_to_batch(self, batch, subject_feature=None):
        B, T = batch["raw_audio"].shape[:2]
        if subject_feature is not None:
            batch["subject_feature"] = self.resize_input_clip_feature(B, T, subject_feature)
        else:
            batch["subject_feature"] = self.id_dropout_feature.unsqueeze(0).expand(B, -1)
        return batch

    def dropped_emotion_id_feature_to_batch(self, batch, emotion_feature=None):
        B, T = batch["raw_audio"].shape[:2]
        if emotion_feature is not None:
            batch["emotion_feature"] = self.resize_input_clip_feature(B, T, emotion_feature)
        else:
            batch["emotion_feature"] = self.emo_dropout_feature.unsqueeze(0).expand(B, -1)
        return batch
    
    def dropped_intensity_id_feature_to_batch(self, batch, intensity_feature=None):
        B, T = batch["raw_audio"].shape[:2]
        if intensity_feature is not None:
            batch["intensity_feature"] = self.resize_input_clip_feature(B, T, intensity_feature)
        else:
            batch["intensity_feature"] = self.int_dropout_feaçture.unsqueeze(0).expand(B, -1)
        return batch


    def denoise(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict: 
        if self.denoise_without_noise_bias:
            ## subtract the noise bias 
            batch[self.denoiser.noise_key] = batch[self.denoiser.noise_key] - self.diffuser.noise_bias
        batch = self.denoiser(batch, **kwargs)
        if self.denoise_without_noise_bias: 
            ## add the noise back
            batch[self.denoiser.noise_key] = batch[self.denoiser.noise_key] + self.diffuser.noise_bias
            batch[self.denoiser.denoised_key] = batch[self.denoiser.denoised_key] + self.diffuser.noise_bias
        if self.clean_denoised_rotations:
            if hasattr(self.prior, 'clean_up_rotation_in_sequence'): ## hacky if to avoid modifying inferno files
                if self.prior._rotation_representation() == "6d":
                    batch[self.denoiser.denoised_key] = self.prior.clean_up_rotation_in_sequence(batch[self.denoiser.denoised_key])
        return batch

    def diffuse(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict: 
        batch = self.diffuser(batch, **kwargs) 
        if self.clean_noised_rotations: 
            if hasattr(self.prior, 'clean_up_rotation_in_sequence'): ## hacky if to avoid modifying inferno files
                if self.prior._rotation_representation() == "6d":
                    batch[self.diffuser.output_key] = self.prior.clean_up_rotation_in_sequence(batch[self.diffuser.output_key])
            
        return batch

    def decode(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict: 
        if self.prior is not None: 
            raise NotImplementedError("Decoding is not yet implemented")
            batch = self.prior.decode(batch, train=train, **kwargs)
        return batch
    
    def encode(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict: 
        return batch 

    def decode(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict: 
        return batch

    def decode_denoised(self, batch: Dict, next_denoised=None, train=False, validation=False, **kwargs: Any) -> Dict: 
        next_denoised = next_denoised if next_denoised is not None else batch[self.denoiser.denoised_key]
        decode_batch = {} 
        decode_batch["gt_shape"] = batch["gt_shape"]
        if not self.cfg.model.output.get('predict_expcode', True): ## predict expression code true by default (backwards compatiblity)
            decode_batch["gt_exp"] = batch["gt_exp"]
        if not self.cfg.model.output.get('predict_jawpose', True): ## predict jaw code true by default 
            decode_batch["gt_jaw"] = batch["gt_jaw"]
        if "gt_tex" in batch.keys():
            decode_batch["gt_tex"] = batch["gt_tex"]
        decode_batch[self.denoiser.denoised_key] = next_denoised
        decoded_batch = self.decode(decode_batch, train=False, validation=False)
        return decoded_batch

    def forward_postpocessors(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        if self.postprocessors is not None:
            for key, postprocessor in self.postprocessors.items():
                batch = postprocessor(batch, train=train, validation=validation, with_grad=True, input_key=None, **kwargs)
        return batch

    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        """
        batch: Dict[str, torch.Tensor]
            - audio: (B, T, F)
            # - masked_audio: (B, T, F)
        """
        # # T = batch["raw_audio"].shape[1]
        # T = batch["processed_audio"].shape[1] if "processed_audio" in batch.keys() else batch["raw_audio"].shape[1]
        # if self.max_seq_length < T: # truncate
        #     print("[WARNING] Truncating audio sequence from {} to {}".format(T, self.max_seq_length))
        #     batch = truncate_sequence_batch(batch, self.max_seq_length)

        # # preprocess input (for instance get 3D pseudo-GT )
        # batch = self.preprocess_input(batch, train=train, **kwargs)
        # check_nan(batch)
        
        # ## render the GT (useful if we want to run perceptual features on GT and condition on them) 
        # batch = self.render_gt_sequence(batch, train=train, **kwargs)

        # # batch = self.extract_gt_features(batch, train=train, **kwargs)
        # batch = self._choose_primary_3D_rec_method(batch)

        # desired_output_length = batch["gt_vertices"].shape[1] if "gt_vertices" in batch.keys() else None
        
        # if self.audio_encoder is not None:
        #     batch = self.forward_audio(batch, train=train, desired_output_length=desired_output_length, **kwargs)
        
        # if self.transcript_encoder is not None:
        #     batch = self.forward_transcript(batch, train=train, desired_output_length=desired_output_length, **kwargs)
        
        
        # if self.audio_transcript_fusion is not None: 
        #     batch = self.forward_audio_transcript_fusion(batch, train=train, **kwargs)
        
        batch = self.forward_input_encoders(batch, train=train, validation=validation, **kwargs)

        if train or validation: 
            if self.prior is not None:
                batch = self.encode(batch, train=train, validation=validation, **kwargs)
        else: 
            assert self.diffuser.output_key not in batch.keys(), \
                f"'{self.diffuser.output_key}' must be in batch dict at test time"

        if train or validation: 
            batch = self.diffuse(batch, train=train, validation=validation, **kwargs)
        else: 
            assert self.diffuser.output_key not in batch.keys(), \
                f"'{self.diffuser.output_key}' must be in batch dict at test time"

        check_nan(batch)
        # # encode the sequence
        # batch = self.encode_sequence(batch, train=train, **kwargs)
        # check_nan(batch)

        # batch = self.disentangle(batch, train=train, validation=validation, **kwargs)
        # check_nan(batch)

        ## denoise the sequence (useful if we want to run perceptual features on GT and condition on them)

        batch = self.forward_denoise(batch, train=train, validation=validation, **kwargs)

        # batch = self.denoise(batch, train=train, **kwargs)
        # check_nan(batch)


        # if self.prior is not None:
        #     batch = self.decode(batch, train=train, **kwargs)

        # # # decode the sequence
        # # batch = self.decode_sequence(batch, train=train, **kwargs)
        # # check_nan(batch)

        # ## render the sequence
        # ## batch = self.render_sequence(batch, train=train, **kwargs)
        # # render only the predicted sequence (the GT is already rendered)
        # batch = self.render_predicted_sequence(batch, train=train, **kwargs)

        # check_nan(batch)



        return batch
    

    def configure_optimizers(self):
        trainable_params = []
        trainable_params += list(self.get_trainable_parameters())

        if self.cfg.learning.optimizer == 'Adam':
            opt = torch.optim.Adam(
                trainable_params,
                lr=self.cfg.learning.learning_rate,
                amsgrad=False)
        elif self.cfg.learning.optimizer == 'SGD':
            opt = torch.optim.SGD(
                trainable_params,
                lr=self.cfg.learning.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: '{self.cfg.learning.optimizer}'")

        optimizers = [opt]
        schedulers = []

        opt_dict = {}
        opt_dict['optimizer'] = opt
        if 'learning_rate_patience' in self.cfg.learning.keys():
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                                   patience=self.cfg.learning.learning_rate_patience,
                                                                   factor=self.cfg.learning.learning_rate_decay,
                                                                   mode=self.cfg.learning.lr_sched_mode)
            schedulers += [scheduler]
            opt_dict['lr_scheduler'] = scheduler
            opt_dict['monitor'] = 'val_loss_total'
        elif 'learning_rate_decay' in self.cfg.learning.keys():
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.cfg.learning.learning_rate_decay)
            opt_dict['lr_scheduler'] = scheduler
            schedulers += [scheduler]
        elif 'learning_rate_warmup' in self.cfg.learning.keys():
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, 
                lr_lambda=lambda epoch: min(1.0, (epoch + 1) / self.cfg.learning.learning_rate_warmup.warmup_steps))
            opt_dict['lr_scheduler'] = {} 
            opt_dict['lr_scheduler']['scheduler'] = scheduler
            opt_dict['lr_scheduler']['interval'] = self.cfg.learning.learning_rate_warmup.interval
            opt_dict['lr_scheduler']['frequency'] = self.cfg.learning.learning_rate_warmup.frequency
            schedulers += [scheduler]
        return opt_dict

    def _choose_primary_3D_rec_method(self, batch, train=False, validation=False): 
        method = self.cfg.data.get('reconstruction_type', None)
        # if method is None:
        #     return batch
        if method is not None:
            if isinstance(method, (list, omegaconf.listconfig.ListConfig)):
                # method = random.choice(method)
                method = method[0]
            if "reconstruction" in batch:
                for k in batch["reconstruction"][method].keys():
                    assert k not in batch.keys(), f"Key '{k}' already exists in batch. We don't want to overwrite it."
                    batch[k] = batch["reconstruction"][method][k]
                    
        if "gt_shape" in batch.keys():
            batch["shape_feature"] = batch["gt_shape"][..., :self.shape_feat_dim]

        if "gt_shape" not in batch.keys():
            B, T = batch["gt_exp"].shape[:2]
            # batch["gt_shape"] = torch.zeros(B, T, 300, device=batch["gt_exp"].device)
            batch["gt_shape"] = torch.zeros(B, 300, device=batch["gt_exp"].device)
        elif self.zero_shape and (train or validation):
            batch["gt_shape"] = torch.zeros_like(batch["gt_shape"])

        elif not self.zero_shape and self.random_shape and (train or validation): 
            B, T = batch["gt_exp"].shape[:2] 
            sigma = self.random_shape_sigma
            batch["gt_shape"] = torch.randn(B, 300, device=batch["gt_exp"].device) * sigma

        return batch

    def disentangle_expansion_factor(self, training, validation):
        return 1


    def _rotation_representation(self):
        assert self.prior is not None
        return self.prior._rotation_representation()

    def _truncate_if_necessary(self, predicted, gt, mask=None, training = True): 
        Tp = predicted.shape[1]
        Tgt = gt.shape[1]
        Tm = mask.shape[1] if mask is not None else Tgt
        if Tp != Tgt or Tp != Tm:
            if training:
                print(f"[WARNING]: predicted and GT sequences have different lengths: {Tp} vs {Tgt}. Truncating the longer sequence."\
                    "This should not happen in training")
            min_T = min(Tp, Tgt, Tm)
            predicted = predicted[:,:min_T]
            gt = gt[:,:min_T]
            if mask is not None:
                mask = mask[:,:min_T]
        if mask is not None:
            return predicted, gt, mask
        return predicted, gt

    def _compute_latent_mask(self, mask):
        return mask

    def compute_loss_term(self, sample, training, validation, loss_name, loss_cfg): 
        loss_type = loss_name if 'loss_type' not in loss_cfg.keys() else loss_cfg['loss_type']
        
        mask_invalid = loss_cfg.get('mask_invalid', False) # mask invalid frames 
        if mask_invalid:
            if mask_invalid == "mediapipe_landmarks": 
                # frames with invalid mediapipe landmarks will be masked for loss computation
                mask = sample["landmarks_validity"]["mediapipe"].to(dtype=torch.bool)
            else: 
                raise ValueError(f"mask_invalid value of '{mask_invalid}' not supported")
        else:
            mask = torch.ones((*sample["predicted_vertices"].shape[:2], 1), device=sample["predicted_vertices"].device, dtype=torch.bool)

        # mask = torch.zeros_like(mask)

        # if "disentangled" in loss_type and not validation and not training:
        #     # disentangled losses are only computed during training
        #     return 0.

        B = sample["predicted_vertices"].shape[0] # total batch size
        T = sample["predicted_vertices"].shape[1] # sequence size
        B_orig = B // self.disentangle_expansion_factor(training, validation) # original batch size before disentanglement expansion
        # effective batch size for computation of this particular loss 
        # some loss functions should be computed on only the original part of the batch (vertex error)
        # and some on everything (lip reading loss)
        B_eff = B if loss_cfg.get('apply_on_disentangled', False) else B_orig 

        mask_sum = mask[:B_eff].sum().item()

        if mask_invalid and mask_sum == 0:
            return 0.

        if loss_type in ["jawpose_loss", "jaw_loss", "globalpose_loss", "global_loss", "neck_loss", "neckpose_loss"]:
            # gt_key = loss_cfg.get('gt_key', 'gt_jaw')
            # pred_key = loss_cfg.get('predicted_key', 'predicted_jaw') 
            gt_key = 'gt_global_pose'
            if 'jaw' in loss_type:
                gt_key = 'gt_jaw'
            elif 'neck' in loss_type:
                gt_key = 'gt_neck'
            # gt_key = sample.get('gt_key', 'gt_jaw' if 'jaw' in loss_type else 'gt_global_pose')
            pred_key = 'predicted_global_pose'
            if 'jaw' in loss_type:
                pred_key = 'predicted_jaw'
            elif 'neck' in loss_type:
                pred_key = 'predicted_neck'
            # pred_key = sample.get('predicted_key', 'predicted_jaw' if 'jaw' in loss_type else 'predicted_global_pose')
            pred_rot_ = sample[pred_key][:B_eff]
            gt_rot_ = sample[gt_key][:B_eff]
            mask_ = mask[:B_eff, ..., 0]
            pred_rot_, gt_rot_, mask_= self._truncate_if_necessary(pred_rot_, gt_rot_, mask_, 
                                                                            #   training=training or validation,
                                                                              training=False, ## just to silence the warning
                                                                              )
            rot_rep = self._rotation_representation() 
            if "jaw" not in loss_type:  ## ugly diffposetalk hack - only jaw pose is allowed to have 1D "axis-angle" representation
                if rot_rep == "aa_1": 
                    rot_rep = "aa" # axis-angle representation
            loss_value = compute_rotation_loss(
                pred_rot_, 
                gt_rot_,  
                r1_input_rep=rot_rep, 
                r2_input_rep="aa", # gt is in axis-angle
                output_rep=loss_cfg.get('rotation_rep', 'quat'), 
                metric=loss_cfg.get('metric', 'l2'), 
                mask = mask_
                )

        elif loss_type in ["expression_loss", "exp_loss"]:
            gt_key = loss_cfg.get('gt_key', 'gt_exp')
            pred_key = loss_cfg.get('predicted_key', 'predicted_exp')
            min_dim = min(sample[pred_key].shape[-1], sample[gt_key].shape[-1])
            pred_exp_ = sample[pred_key][:B_eff][...,:min_dim]
            gt_exp_ = sample[gt_key][:B_eff][...,:min_dim]
            mask_ = mask[:B_eff, ..., 0]
            pred_exp_, gt_exp_, mask_ = self._truncate_if_necessary(pred_exp_, gt_exp_, mask_, 
                                                                    #   training=training or validation,
                                                                    training=False ## just to silence the warning
                                                                    )
            loss_value = MaskedTemporalMSELoss()(pred_exp_, gt_exp_, mask=mask_)

        elif loss_type == "vertex_loss":
            gt_key = loss_cfg.get('gt_key', 'gt_vertices')
            pred_key = loss_cfg.get('predicted_key', 'predicted_vertices')
            pred_vertices_ = sample[pred_key][:B_eff]
            gt_vertices_ = sample[gt_key][:B_eff]
            mask_ = mask[:B_eff, ..., 0]
            pred_vertices_, gt_vertices_, mask_ = self._truncate_if_necessary(pred_vertices_, gt_vertices_, mask_, 
                                                                            #   training=training or validation,
                                                                              training=False ## just to silence the warning
                                                                              )
            loss_value = MaskedTemporalMSELoss()(pred_vertices_, gt_vertices_, mask=mask_)

        elif loss_type in ["global_vertex_loss", "posed_vertex_loss"]: 
            import inferno.utils.DecaUtils as util

            gt_key = loss_cfg.get('gt_key', 'gt_vertices_posed')
            pred_key = loss_cfg.get('predicted_key', 'predicted_vertices_posed')
            pred_vertices_ = sample[pred_key][:B_eff]
            gt_vertices_ = sample[gt_key][:B_eff]
            mask_ = mask[:B_eff, ..., 0] 

            trans_type = loss_cfg.get('translation_type', 'orth_cam')
            if trans_type == "orth_cam":
                pred_cam = sample["predicted_cam"][:B_eff]
                gt_cam = sample["gt_cam"][:B_eff]

                pred_vertices_cam = util.batch_orth_proj(pred_vertices_.reshape(B*T, -1, 3), 
                                                                pred_cam.reshape(B*T, *pred_cam.shape[2:])).view(B,T, -1)
                gt_vertices_cam =  util.batch_orth_proj(gt_vertices_.reshape(B*T, -1, 3,), 
                                                        gt_cam.reshape(B*T, *gt_cam.shape[2:])).view(B,T, -1)
                # assert "gt_cam" in sample.keys() and "gt_trans" in sample.keys(), "Only one, gt_cam or gt_trans, must be in the sample dict"
            elif trans_type == "trans":
                pred_trans = sample["predicted_trans"][:B_eff]
                gt_trans = sample["gt_trans"][:B_eff]

                # pred_vertices_cam = pred_vertices_ + pred_trans 
                pred_vertices_cam = pred_vertices_ 
                # gt_vertices_cam = gt_vertices_ + gt_trans
                gt_vertices_cam = gt_vertices_ 
                # assert "gt_cam" in sample.keys() and "gt_trans" in sample.keys(), "Only one, gt_cam or gt_trans, must be in the sample dict"
            elif trans_type == "none":
                pred_vertices_cam = pred_vertices_
                gt_vertices_cam = gt_vertices_     
            else: 
                raise ValueError("Invalid translation type. Must be one of 'orth_cam', 'trans' or 'none'")

            pred_vertices_cam, gt_vertices_cam, mask_ = self._truncate_if_necessary(pred_vertices_cam, gt_vertices_cam, mask_, 
                                                                                    #   training=training or validation,
                                                                                    training=False ## just to silence the warning
                                                                                    )
            loss_value = MaskedTemporalMSELoss()(pred_vertices_cam, gt_vertices_cam, mask=mask_)
            return loss_value

        elif loss_type in ["cam_loss", "trans_loss"]:
            gt_key = loss_cfg.get('gt_key', 'gt_cam')
            pred_key = loss_cfg.get('predicted_key', 'predicted_cam')
            pred_cam_ = sample[pred_key][:B_eff]
            gt_cam_ = sample[gt_key][:B_eff]
            mask_ = mask[:B_eff, ..., 0]
            pred_cam_, gt_cam_, mask_ = self._truncate_if_necessary(pred_cam_, gt_cam_, mask_, training or validation)
            loss_value = MaskedTemporalMSELoss()(pred_cam_, gt_cam_, mask=mask_)

        elif loss_type == "motion_latent_loss": 
            pred_key = loss_cfg.get('predicted_key', 'denoised_latent')
            gt_key = loss_cfg.get('gt_key', 'gt_latent')
            gt_latent_ = sample[gt_key][:B_eff]
            predicted_latent_ = sample[pred_key][:B_eff]
            mask_ = mask[:B_eff, ..., 0]
            latent_mask_ = self._compute_latent_mask(mask_)
            loss_value = MaskedTemporalMSELoss()(predicted_latent_, gt_latent_, mask=latent_mask_)

        elif loss_type == "latent_noise_loss": 
            gt_noise_ = sample["noise"][:B_eff]
            predicted_noise_ = sample["predicted_noise"][:B_eff]
            mask_ = mask[:B_eff, ..., 0]
            noise_mask_ = self._compute_latent_mask(mask_)
            loss_value = MaskedTemporalMSELoss()(predicted_noise_, gt_noise_, mask=noise_mask_)

        # velocity losses
        elif loss_type == "vertex_velocity_loss":
            gt_key = loss_cfg.get('gt_key', 'gt_vertices')
            pred_key = loss_cfg.get('predicted_key', 'predicted_vertices')
            pred_vertices_ = sample[pred_key][:B_eff] 
            gt_vertices_ = sample[gt_key][:B_eff]
            mask_ = mask[:B_eff, ..., 0]
            pred_vertices_, gt_vertices_, mask_ = self._truncate_if_necessary(pred_vertices_, gt_vertices_, mask_, 
                                                                            #   training or validation ,
                                                                            #   training=training or validation,
                                                                            training=False ## just to silence the warning
                                                                            )
                                                                              
            loss_value = velocity_loss(pred_vertices_, gt_vertices_, MaskedTemporalMSELoss(), mask=mask_)
        
        elif loss_type in ["global_vertex_velocity_loss", "posed_vertex_velocity_loss"]:
            import inferno.utils.DecaUtils as util

            gt_key = loss_cfg.get('gt_key', 'gt_vertices_posed')
            pred_key = loss_cfg.get('predicted_key', 'predicted_vertices_posed')
            pred_vertices_ = sample[pred_key][:B_eff]
            gt_vertices_ = sample[gt_key][:B_eff]
            mask_ = mask[:B_eff, ..., 0]

            trans_type = loss_cfg.get('translation_type', 'orth_cam')
            if trans_type == "orth_cam":
                pred_cam = sample["predicted_cam"][:B_eff]
                gt_cam = sample["gt_cam"][:B_eff]

                pred_vertices_cam = util.batch_orth_proj(pred_vertices_.reshape(B*T, -1, 3), 
                                                                pred_cam.reshape(B*T, *pred_cam.shape[2:])).view(B,T, -1)
                gt_vertices_cam =  util.batch_orth_proj(gt_vertices_.reshape(B*T, -1, 3,), 
                                                        gt_cam.reshape(B*T, *gt_cam.shape[2:])).view(B,T, -1)
                # assert "gt_cam" in sample.keys() and "gt_trans" in sample.keys(), "Only one, gt_cam or gt_trans, must be in the sample dict"
            elif trans_type == "trans":
                pred_trans = sample["predicted_trans"][:B_eff]
                gt_trans = sample["gt_trans"][:B_eff]

                # pred_vertices_cam = pred_vertices_ + pred_trans 
                pred_vertices_cam = pred_vertices_  
                # gt_vertices_cam = gt_vertices_ + gt_trans
                gt_vertices_cam = gt_vertices_ 
                # assert "gt_cam" in sample.keys() and "gt_trans" in sample.keys(), "Only one, gt_cam or gt_trans, must be in the sample dict"
            elif trans_type == "none":
                pred_vertices_cam = pred_vertices_
                gt_vertices_cam = gt_vertices_     
            else: 
                raise ValueError("Invalid translation type. Must be one of 'orth_cam', 'trans' or 'none'")

            pred_vertices_cam, gt_vertices_cam, mask_ = self._truncate_if_necessary(pred_vertices_cam, gt_vertices_cam, mask_, 
                                                                                    # training or validation
                                                                                    training=False ## just to silence the warning
                                                                                    )
            loss_value = velocity_loss(pred_vertices_cam, gt_vertices_cam, MaskedTemporalMSELoss(), mask=mask_)

        elif loss_type in ["cam_velocity_loss", "trans_velocity_loss"]:
            gt_key = loss_cfg.get('gt_key', 'gt_vertices')
            pred_key = loss_cfg.get('predicted_key', 'predicted_vertices')
            pred_cam_ = sample[pred_key][:B_eff] 
            gt_cam_ = sample[gt_key][:B_eff]
            mask_ = mask[:B_eff, ..., 0]
            pred_cam_, gt_cam_, mask_ = self._truncate_if_necessary(pred_cam_, gt_cam_, mask_, 
                                                                                    # training or validation
                                                                                    training=False ## just to silence the warning
            )
            loss_value = velocity_loss(pred_cam_, gt_cam_, MaskedTemporalMSELoss(), mask=mask_)

        elif loss_type in ["expression_velocity_loss", "exp_velocity_loss"]:
            gt_key = loss_cfg.get('gt_key', 'gt_exp')
            pred_key = loss_cfg.get('predicted_key', 'predicted_exp')
            min_dim = min(sample[pred_key].shape[-1], sample[gt_key].shape[-1])
            pred_exp_ = sample[pred_key][:B_eff][...,:min_dim]
            gt_exp_ = sample[gt_key][:B_eff][...,:min_dim]
            mask_ = mask[:B_eff, ..., 0]
            pred_exp_, gt_exp_, mask_ = self._truncate_if_necessary(pred_exp_, gt_exp_, mask_, 
                                                                                    # training or validation
                                                                                    training=False ## just to silence the warning
                                                                    )
            loss_value = velocity_loss(pred_exp_, gt_exp_, MaskedTemporalMSELoss(), mask=mask_)

        elif loss_type in ["jawpose_velocity_loss", "jaw_velocity_loss", "globalpose_velocity_loss", "global_pose_velocity_loss", "neck_velocity_loss", "neckpose_velocity_loss"]:
            mask_ = mask[:B_eff, ..., 0]
            # gt_key = sample.get('gt_key', 'gt_jaw' if 'jaw' in loss_type else 'gt_global_pose')
            gt_key = 'gt_global_pose'
            if 'jaw' in loss_type:
                gt_key = 'gt_jaw'
            elif 'neck' in loss_type:
                gt_key = 'gt_neck'
            # pred_key = sample.get('predicted_key', 'predicted_jaw' if 'jaw' in loss_type else 'predicted_global_pose')
            pred_key = 'predicted_global_pose'
            if 'jaw' in loss_type:
                pred_key = 'predicted_jaw'
            elif 'neck' in loss_type:
                pred_key = 'predicted_neck'
            pred_rot = sample[pred_key][:B_eff]
            gt_rot = sample[gt_key][:B_eff]
            pred_rot, gt_rot, mask_ = self._truncate_if_necessary(pred_rot, gt_rot, mask_,
                                                                # training or validation
                                                                training=False ## just to silence the warning 
                                                                  )
            T_ = pred_rot.shape[1]
            pred_rot = pred_rot.view(B_eff, T_, -1)
            gt_rot = gt_rot.view(B_eff, T_, -1)

            rot_rep = self._rotation_representation()
            if "jaw" not in loss_type:  ## ugly diffposetalk hack - only jaw pose is allowed to have 1D "axis-angle" representation
                if rot_rep == "aa_1": 
                    rot_rep = "aa" # axis-angle representation

            loss_value = rotation_velocity_loss(pred_rot, gt_rot,
                r1_input_rep=rot_rep, 
                r2_input_rep="aa", # gt is in axis-angle
                output_rep=loss_cfg.get('rotation_rep', 'quat'), 
                metric=loss_cfg.get('metric', 'l2'), 
                mask = mask_
            )

        elif loss_type == "expression_reg": 
            loss_value = (torch.sum(sample["predicted_exp"] ** 2, dim=-1) / 2).mean()
        elif loss_type == "motion_prior_gaussian_reg": 
            loss_value = (torch.sum(sample[loss_cfg.latent_key] ** 2, dim=-1) / 2).mean()

        elif loss_type in ["mel_reconstruction_loss", "mel_vocoded_loss"]:
            pred_key = loss_cfg.get('predicted_key', 'predicted_mel')
            gt_key = loss_cfg.get('gt_key', 'gt_mel')
            gt_mel = sample[gt_key]
            predicted_mel = sample[pred_key]

            T_mel = gt_mel.shape[1]
            # assert T_mel % T == 0, f"gt_mel.shape[1] ({T_mel}) must be divisible by T ({T})" 
            T_factor = T_mel // T
            if T_factor > 1:
                mask = mask.repeat_interleave(T_factor, dim=1)
                # mask_sum = mask.sum().item()
            T_gt = gt_mel.shape[1]
            T_pred = predicted_mel.shape[1]

            T_loss = min(T_gt, T_pred)

            if mask.ndim == 3: 
                mask = mask.squeeze(-1)

            if loss_cfg.metric in ["mse", "l2"]:
                loss_value = MaskedTemporalMSELoss()(predicted_mel[:,:T_loss,...], gt_mel[:,:T_loss,...], mask=mask[:,:T_loss])
            
            elif loss_cfg.metric in ["mae", "l1"]:
                loss_value = MaskedTemporalMAELoss()(predicted_mel[:,:T_loss,...], gt_mel[:,:T_loss,...], mask=mask[:,:T_loss])
            
            else: 
                raise ValueError(f"loss_cfg.metric '{loss_cfg.metric}' not supported")
        
        elif loss_type in ["unit_cross_entropy"]:
            pred_key = loss_cfg.get('predicted_key', 'predicted_unit_classification_vectors')
            gt_key = loss_cfg.get('gt_key', 'gt_units') 

            gt_units = sample[gt_key]
            predicted_unit_vectors = sample[pred_key]

            T_units = min(gt_units.shape[1], predicted_unit_vectors.shape[1])
            predicted_unit_vectors = predicted_unit_vectors[:, :T_units, :]
            gt_units = gt_units[:, :T_units]
            # assert T_units % T == 0 or (T_units+1) % T == 0, f"gt_units.shape[1] ({T_units}) must be divisible by T ({T})"
            T_factor = T_units // T
            if T_units % T != 0 :
                T_factor += 1
            mask = mask.repeat_interleave(T_factor, dim=1)
            mask = mask[:, :T_units, :].squeeze(-1)
            mask_sum = mask.sum().item()

            gt_units = gt_units.reshape(B * T_units)
            predicted_unit_vectors = predicted_unit_vectors.reshape(B * T_units, -1)

            loss_values = torch.nn.functional.cross_entropy(predicted_unit_vectors, gt_units, reduction='none')
            loss_values = loss_values.view(B, T_units)
            
            loss_values = loss_values * mask.squeeze(-1) 
            loss_value = loss_values.sum() / mask_sum
        elif loss_type in ["contrastive_perceptual_loss"]:
            loss_value = sample["contrastive_perceptual_loss"]

        return loss_value

    def compute_loss(self, sample, training, validation): 
        """
        Compute the loss for the given sample. 

        """
        losses = {}
        # loss_weights = {}
        metrics = {}

        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            assert loss_name not in losses.keys()
            losses["loss_" + loss_name] = self.compute_loss_term(sample, training, validation, loss_name, loss_cfg)

        for metric_name, metric_cfg in self.cfg.learning.metrics.items():
            assert metric_name not in metrics.keys()
            with torch.no_grad():
                metrics["metric_" + metric_name] = self.compute_loss_term(sample, training, validation, metric_name, metric_cfg)

        total_loss = None
        for loss_name, loss_cfg in self.cfg.learning.losses.items():
            term = losses["loss_" + loss_name] 
            if term is not None:
                if isinstance(term, torch.Tensor) and term.isnan().any():
                    print(f"[WARNING]: loss '{loss_name}' is NaN. Skipping this term.")
                    continue
                if total_loss is None: 
                    total_loss = 0.
                weighted_term =  (term * loss_cfg["weight"])
                total_loss = total_loss + weighted_term
                losses["loss_" + loss_name + "_w"] = weighted_term

        losses["loss_total"] = total_loss
        return total_loss, losses, metrics

    def training_step(self, batch, batch_idx, *args, **kwargs):
        training = True 
        # forward pass
        batch = self.forward(batch, train=training, validation=False, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(batch, training=training, validation=False, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {"train/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        # if self.logger is not None:
            # self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
        self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss


    def validation_step(self, batch, batch_idx, dataloader_idx=None, *args, **kwargs): 
        training = False 

        # forward pass
        batch = self.forward(batch, train=training, validation=True, **kwargs)
        # loss 
        total_loss, losses, metrics = self.compute_loss(batch, training=training, validation=True, **kwargs)

        losses_and_metrics_to_log = {**losses, **metrics}
        # losses_and_metrics_to_log = {"val_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        prefix = "val"
        # if dataloader_idx is not None and dataloader_idx > 0:
        #     prefix = prefix + f"_{dataloader_idx}"
        losses_and_metrics_to_log = {f"{prefix}/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}

        # if self.logger is not None:
        self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss
    

    def test_step(self, batch, batch_idx, *args, **kwargs):
        training = False 
        # forward pass
        import copy
        batch_copy = copy.deepcopy(batch)
        diff_type = []
        prediction_history = [] 
        next_prediction_history = [] 
        gt_history = [] 
        timesteps = []

        sample, prediction_history_, next_prediction_history_, gt_history_, timesteps_ = self.reverse_diffusion(batch, batch_idx, train=False, validation=False, **kwargs)
        diff_type += (['reverse_diffusion'] * len(timesteps_))
        prediction_history += prediction_history_
        next_prediction_history += next_prediction_history_
        gt_history += gt_history_
        timesteps += timesteps_

        # losses_and_metrics_to_log = {"test/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        # self.log_dict(losses_and_metrics_to_log, sync_dist=True) 

        # # return prediction_history, gt_history, timesteps
        sample, prediction_history_, next_prediction_history_, noisy_history_, timesteps_ = self.full_diffusion(batch_copy, batch_idx, train=False, validation=False, **kwargs)
        diff_type += (['diffusion'] * len(timesteps_))
        prediction_history += prediction_history_
        next_prediction_history += next_prediction_history_
        gt_history += noisy_history_
        timesteps += timesteps_

        return prediction_history, next_prediction_history, gt_history, timesteps, diff_type
        
        # # loss 
        # total_loss, losses, metrics = self.compute_loss(sample, training, validation=False, **kwargs)

        # losses_and_metrics_to_log = {**losses, **metrics}
        # # losses_and_metrics_to_log = {"train_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        # losses_and_metrics_to_log = {"test/" + k: v.item() if isinstance(v, (torch.Tensor,)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}
        
        # if self.logger is not None:
        #     self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
        
        # return total_loss

    def _truncate_sequence_batch(self, batch):
        T = batch["processed_audio"].shape[1] if "processed_audio" in batch.keys() else batch["raw_audio"].shape[1]
        if self.max_seq_length < T: # truncate
            print("[WARNING] Truncating audio sequence from {} to {}".format(T, self.max_seq_length))
            batch = truncate_sequence_batch(batch, self.max_seq_length)
        return batch

    def forward_input_encoders(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:

        batch = self._truncate_sequence_batch(batch)

        # # preprocess input (for instance get 3D pseudo-GT )
        # batch = self.preprocess_input(batch, train=train, **kwargs)
        check_nan(batch)
        
        ## render the GT (useful if we want to run perceptual features on GT and condition on them) 
        batch = self.render_gt_sequence(batch, train=train, **kwargs)

        # batch = self.extract_gt_features(batch, train=train, **kwargs)
        batch = self._choose_primary_3D_rec_method(batch, train=train, validation=validation)

        desired_output_length = batch["gt_vertices"].shape[1] if "gt_vertices" in batch.keys() else None
        
        ## 0) Preprocessors
        batch = self.forward_prepocessors(batch, train=train, validation=validation, with_grad=kwargs.pop('preprocessor_with_grad', False), **kwargs)


        if self.audio_encoder is not None:
            batch = self.forward_audio(batch, train=train, desired_output_length=desired_output_length, **kwargs)
        
        if self.transcript_encoder is not None:
            batch = self.forward_transcript(batch, train=train, desired_output_length=desired_output_length, **kwargs)
        
        if self.one_hot_encoders is not None:
            batch = self.forward_one_hot_encoders(batch, train=train, validation=validation, **kwargs)
        
        if self.training_clip_dropout_probability is not None:
            batch = self.forward_clip_dropout(batch, train=train, validation=validation, **kwargs)

        if self.training_id_dropout_probability is not None:
            batch = self.forward_subject_id_dropout(batch, train=train, validation=validation, **kwargs)

        if self.training_emo_dropout_probability is not None:
            batch = self.forward_emotion_id_dropout(batch, train=train, validation=validation, **kwargs)

        if self.training_int_dropout_probability is not None:
            batch = self.forward_intensity_id_dropout(batch, train=train, validation=validation, **kwargs)

        if self.training_style_dropout_probability is not None:
            batch = self.forward_style_feat_dropout(batch, train=train, validation=validation, **kwargs)

        if self.audio_transcript_fusion is not None: 
            batch = self.forward_audio_transcript_fusion(batch, train=train, validation=validation, **kwargs)
        
        return batch


    def forward_denoise(self, batch, 
                        train=False, validation=False, with_prior=True, 
                        guidance_dict=None, 
                        gradient_guidance_dict=None, 
                        with_postprocessors=True, 
                        *args, **kwargs): 
        if guidance_dict is None:
            batch = self.denoise(batch, train=train, **kwargs)
        else: 
            ## construct separate batches for each guidance signal
            batch_backup = batch.copy()

            for modality_key, guidance_value in guidance_dict.items():
                # if modality_key == "unconditioned":
                #     continue
                condition_to_keep = list(modality_key)
                # for modality in modality_key:
                #     condition_to_keep.remove(modality)
                conditioned_batch = batch_backup.copy()
                conditioned_batch = self._drop_condition_inputs(conditioned_batch, condition_to_keep)
                conditioned_batch = self.denoise(conditioned_batch, train=train, **kwargs)

                ## compute the linear combination
                if self.denoiser.denoised_key not in batch.keys(): 
                    batch[self.denoiser.denoised_key] = guidance_value * conditioned_batch[self.denoiser.denoised_key]

                    # the following line might not be correct, check later, but it's not really used anyway
                    batch[self.denoiser.predicted_noise_key] = guidance_value * conditioned_batch[self.denoiser.predicted_noise_key]
                else:
                    batch[self.denoiser.denoised_key] = batch[self.denoiser.denoised_key] + (guidance_value * conditioned_batch[self.denoiser.denoised_key])
                    # the following line might not be correct, check later, but it's not really used anyway
                    batch[self.denoiser.predicted_noise_key] = batch[self.denoiser.predicted_noise_key] + (guidance_value * conditioned_batch[self.denoiser.predicted_noise_key])

            batch_backup = batch.copy()

            if gradient_guidance_dict is not None:
                for grad_guide, weight in gradient_guidance_dict.items():

                    grad_guide_batch = batch_backup.copy()
                    gradient = grad_guide(batch)

                    ## get the noise schedule
                    timesteps = grad_guide_batch[self.diffuser.diffusion_steps_key]
                    sqrt_alpha = self.diffuser.get_training_sqrt_alphas_cumprod(timesteps)
                
                    batch[self.denoiser.denoised_key] -= 0.5 * sqrt_alpha * weight * gradient 
                    # batch[self.denoiser.denoised_key] -= 0.5 * weight * gradient

                    # the following line might not be correct, check later, but it's not really used anyway
                    batch[self.denoiser.predicted_noise_key] -= 0.5 * sqrt_alpha * weight * gradient
                    # batch[self.denoiser.predicted_noise_key] -= 0.5 * weight * sqrt_alpha * gradient


        check_nan(batch)

        # # if self.has_simple_objective:
        # if self.denoiser_objective == DenoiserObjective.X_0:
        #     ## simple objective means the output of the denoiser is not the noise but the final space
        #     # batch[self.diffuser.output_key] = batch[self.denoiser.denoised_key]
        #     noise = batch[self.diffuser.output_key] - batch[self.denoiser.denoised_key]
        #     noise2 = batch[self.diffuser.input_key] - batch[self.denoiser.denoised_key]
        #     noise3 = batch[self.diffuser.input_key] - batch[self.diffuser.output_key]
        #     print(f"Noise norm \t\t {noise.norm().item():.4f}")
        #     print(f"Noise2 norm \t\t {noise2.norm().item():.4f}")
        #     print(f"Noise3 norm \t\t {noise3.norm().item():.4f}")
        #     pass
            
        # elif self.denoiser_objective == DenoiserObjective.X_T_1:
        #     raise NotImplementedError("DenoiserObjective.X_T_1 not yet implemented")
        
        # elif self.denoiser_objective == DenoiserObjective.EPS_0:
        #     ## not a simple objective means the output of the denoiser is the noise, that is added using 
        #     ## the scheduler in the denoiser
        #     predicted_noise = batch[self.denoiser.predicted_noise_key]
        #     print(f"Pred noise norm \t\t {predicted_noise.norm().item():.4f}")
        #     # denoised_vectors = self.diffuser.testing_step(d, noise_pred)
        #     # diffusion_steps = batch[self.diffuser.diffusion_steps_key]
        #     noisy_vector = batch[self.denoiser.noise_key]

        #     # denoised_vectors = self.diffuser.apply_predicted_noise(diffusion_steps, noisy_vector, predicted_noise, 
        #     #                                                        training=train or validation,
        #     #                                                     #    training=True,
        #     #                                                        )
        #     denoised_vectors = noisy_vector - predicted_noise
        #     batch[self.denoiser.denoised_key] = denoised_vectors
            
        # elif self.denoiser_objective == DenoiserObjective.EPS_T_1:
        #     raise NotImplementedError("DenoiserObjective.EPS_T_1 not yet implemented")

        #     ##
        #     denoised_vectors = noisy_vector - predicted_noise
        #     batch[self.denoiser.denoised_key] = denoised_vectors
            
        # else: 
        #     raise NotImplementedError(f"DenoiserObjective '{self.denoiser_objective}' not supported")


        if self.prior is not None and with_prior:
            ## decode the noisy batch (for visualization or loss computation)
            with torch.no_grad():
                batch = self.decode(batch,
                                    key_to_decode=self.diffuser.output_key, 
                                    decoded_prefix="noisy_", 
                                    train=train, 
                                    validation=validation,
                                    **kwargs)
            
            ## decode the denoised batch
            batch = self.decode(batch, train=train, validation=validation, **kwargs)

        # # decode the sequence
        # batch = self.decode_sequence(batch, train=train, **kwargs)
        # check_nan(batch)

        ## render the sequence
        ## batch = self.render_sequence(batch, train=train, **kwargs)
        # render only the predicted sequence (the GT is already rendered)
        batch = self.render_predicted_sequence(batch, train=train, **kwargs)


        ## 5) Postprocessing
        if with_postprocessors:
            batch = self.forward_postpocessors(batch, train=train, validation=validation, **kwargs)

        return batch


    @property 
    def has_simple_objective(self):
        """
        Whether the model has a simple objective (i.e. no denoising) but predicting 
        the final space. This affect test time (and should afffect how the losses are computed, 
        though that is not controled through this variable but through loss list and the model config)
        """
        # simple_objective = self.cfg.model.get('simple_objective', True)
        simple_objective = self.denoiser.predicts_x()
        return  simple_objective

    @property 
    def denoiser_objective(self): 
        # if "latent_noise_loss" in self.cfg.learning.losses.keys() and not self.has_simple_objective:
        #     return "latent_noise_loss"
        # predicts_noise = False 

        # in this case, simple objective is a misnomer, if false, we predict the noise 
        if not self.has_simple_objective: 
            # are we predicting final noise or single-step noise? 
            if "latent_noise_loss" in self.cfg.learning.losses.keys():
                return DenoiserObjective.EPS_T_1
            else: 
                return DenoiserObjective.EPS_0
        else: 
            # we are predicting X (the final space as opposed to noise)
            # are we predicting X_t-1 or X_0?
            noisy_losses = [k for k in self.cfg.learning.losses.keys() if k.startswith("noisy_")]
            ## presence of "noisy_" loss signals that we are predicting X_t_1
            noisy_loss_any = any(noisy_losses)
            noisy_loss_all = False 
            if len(noisy_losses):
                noisy_loss_all = all(noisy_losses)
            assert noisy_loss_any == noisy_loss_all, "All noisy losses must be present if any is present"
            if not noisy_loss_any:
                return DenoiserObjective.X_0
            else:
                return DenoiserObjective.X_T_1


    def process_denoiser_output(self, batch, like_training=False) -> Tuple[Dict, torch.Tensor]:
        
        diffusion_step = batch[self.diffuser.diffusion_steps_key] 
        # if self.denoiser.denoised_key in batch.keys():
        if self.denoiser_objective == DenoiserObjective.X_0:
            ## simple objective means the output of the denoiser is not the noise but the final space
            ## in order for the next diffusion iteration to work, it must get x_t-1, not x_0 
            noisy_vector = batch[self.diffuser.output_key] ## x_t
            denoised_vector = batch[self.denoiser.denoised_key]  ## x_0 (predicted from the denoiser)
            
            if self.denoise_without_noise_bias: 
                noisy_vector = noisy_vector - self.diffuser.noise_bias
                denoised_vector = denoised_vector - self.diffuser.noise_bias

            if hasattr(self.prior, "clean_up_rotation_in_sequence"):
                if self.prior._rotation_representation() == "6d":
                    if self.clean_noised_rotations:
                        denoised_vector = self.prior.clean_up_rotation_in_sequence(denoised_vector)
                        noisy_vector = self.prior.clean_up_rotation_in_sequence(noisy_vector)
            noise_pred = batch[self.denoiser.predicted_noise_key] ## eps_0 (but in this case it was computed from as x_t - x_0)

            # print(f"Noisy norm \t\t {noisy_vector.norm().item():.4f}")
            # print(f"Denoised norm \t\t {denoised_vector.norm().item():.4f}")
            # print(f"Noise norm \t\t {noise_pred.norm().item():.4f}")

            # eps = self.diffuser.compute_eps_from_x0_xt(denoised_vector, noisy_vector, training=like_training)

            # compute x_t-1
            next_denoised = self.diffuser.compute_xt_1_next_from_x0(noisy_vector, denoised_vector,  batch[self.diffuser.diffusion_steps_key], training=like_training)
            
            if self.denoise_without_noise_bias: 
                next_denoised = next_denoised + self.diffuser.noise_bias
                noisy_vector = noisy_vector + self.diffuser.noise_bias
                denoised_vector = denoised_vector + self.diffuser.noise_bias

            if hasattr(self.prior, "clean_up_rotation_in_sequence"):
                if self.prior._rotation_representation() == "6d":
                    if self.clean_denoised_rotations:
                        next_denoised = self.prior.clean_up_rotation_in_sequence(next_denoised)
                        noisy_vector = self.prior.clean_up_rotation_in_sequence(noisy_vector)
                        denoised_vector = self.prior.clean_up_rotation_in_sequence(denoised_vector)

            
            # batch[self.diffuser.output_key] = next_denoised
        
        elif self.denoiser_objective == DenoiserObjective.EPS_0:
            ## not a simple objective means the output of the denoiser is the noise, that is added using 
            ## the scheduler in the denoiser
            noisy_vector = batch[self.diffuser.output_key] ## x_t
            noise_pred = batch[self.denoiser.predicted_noise_key] ## eps_0  (predicted from the denoiser)
            denoised_vector = batch[self.denoiser.denoised_key]  ## x_0  (computed as x_t - eps_0)
            if hasattr(self.prior, "clean_up_rotation_in_sequence"):
                if self.prior._rotation_representation() == "6d":
                    denoised_vector = self.prior.clean_up_rotation_in_sequence(denoised_vector)

            # print(f"Noisy norm \t\t {noisy_vector.norm().item():.4f}")
            # print(f"Denoised norm \t\t {denoised_vector.norm().item():.4f}")
            # print(f"Noise norm \t\t {noise_pred.norm().item():.4f}")

            ## compute x_t-1
            next_denoised = self.diffuser.compute_xt_1_next_from_x0(noisy_vector, denoised_vector,  batch[self.diffuser.diffusion_steps_key], training=like_training)
            # batch[self.diffuser.output_key] = next_denoised

            if hasattr(self.prior, "clean_up_rotation_in_sequence"):
                if self.prior._rotation_representation() == "6d":
                    next_denoised = self.prior.clean_up_rotation_in_sequence(next_denoised)
                    noisy_vector = self.prior.clean_up_rotation_in_sequence(noisy_vector)
                    denoised_vector = self.prior.clean_up_rotation_in_sequence(denoised_vector)
        
        elif self.denoiser_objective == DenoiserObjective.X_T_1:
            raise NotImplementedError("DenoiserObjective.X_T_1 not yet implemented")
            next_denoised = batch[self.denoiser.denoised_key]
        
        elif self.denoiser_objective == DenoiserObjective.EPS_T_1:
            noisy_vector = batch[self.diffuser.output_key] ## x_t
            denoised_vector = batch[self.denoiser.denoised_key]  ## x_t_1 
            next_denoised = batch[self.denoiser.denoised_key] ## x_t_1 
            noise_pred = batch[self.denoiser.predicted_noise_key] ## eps_t 
            if hasattr(self.prior, "clean_up_rotation_in_sequence"):
                if self.prior._rotation_representation() == "6d":
                    denoised_vector = self.prior.clean_up_rotation_in_sequence(denoised_vector)

            next_denoised = self.diffuser.apply_predicted_noise(diffusion_step, noisy_vector, noise_pred, training=False)
            # denoised_vectors = self.diffuser.apply_predicted_noise(d, noise_pred)
            # raise NotImplementedError("DenoiserObjective.X_T_1 not yet implemented")

        else:
            raise NotImplementedError(f"DenoiserObjective '{self.denoiser_objective}' not supported")
        return batch, next_denoised


    def get_input_tensor_shape(self, round_up=True):
        raise NotImplementedError()


    def _drop_condition_inputs(self, batch, condition_list):
        # if not conditioning on audio, replace with the no-audio feature
        if ConditionType.AUDIO in self.input_conditions and ConditionType.AUDIO not in condition_list: 
            ## replace with the no-audio feature 
            batch = self.dropped_audio_feature_to_batch(batch)

        # if not conditioning on transcript, replace with the no-transcript feature
        if ConditionType.TRANSCRIPT in self.input_conditions and ConditionType.TRANSCRIPT not in condition_list:
            ## replace with the no-transcript feature 
            batch = self.dropped_transcript_feature_to_batch(batch)

        if ConditionType.CLIP in self.input_conditions:
            # if not conditioning on clip, replace with the no-clip feature
            if ConditionType.CLIP not in condition_list:
                ## replace with the no-clip feature 
                # if condition_list[ConditionType.CLIP] is None:
                batch = self.dropped_clip_feature_to_batch(batch)
            else:
                # if we are conditioning on clip and the clip features is not None (i.e. it is specified, then use it)
                if isinstance(condition_list, dict) and condition_list[ConditionType.CLIP] is not None:
                    batch = self.dropped_clip_feature_to_batch(batch, clip_feature=condition_list[ConditionType.CLIP])

        if ConditionType.SUJECT_ID in self.input_conditions:
            # if not conditioning on subject_id, replace with the no-subject_id feature
            if ConditionType.SUJECT_ID not in condition_list:
                ## replace with the no-subject_id feature 
                batch = self.dropped_subject_id_feature_to_batch(batch)

        if ConditionType.EMOTION_ID in self.input_conditions:
            # if not conditioning on emotion_id, replace with the no-emotion_id feature
            if ConditionType.EMOTION_ID not in condition_list:
                ## replace with the no-emotion_id feature 
                batch = self.dropped_emotion_id_feature_to_batch(batch)
        
        if ConditionType.INTENSITY_ID in self.input_conditions:
            # if not conditioning on intensity_id, replace with the no-intensity_id feature
            if ConditionType.INTENSITY_ID not in condition_list:
                ## replace with the no-intensity_id feature 
                batch = self.dropped_intensity_id_feature_to_batch(batch)
        
        return batch


    def reverse_diffusion(self, batch, batch_idx, 
                          diffuser=None, 
                          condition_list=None, 
                          noise=None, 
                          guidance_dict=None, 
                          gradient_guidance_dict=None, 
                          **kwargs): 
        ## only one of condition_list or guidance_dict should be specified
        assert (condition_list is None) or (guidance_dict is None), "Only one of condition_list or guidance_dict should be specified"

        if guidance_dict is not None: 
            conditions = [list(k) for k in guidance_dict.keys()]
            condition_list = []
            for c in conditions:
                condition_list += c
            condition_list = list(set(condition_list))

        diffuser = diffuser or self.diffuser 

        batch = self.forward_input_encoders(batch, train=False, validation=False)

        compute_losses = False

        if self.prior is not None:
            batch = self.encode(batch, **kwargs)

        if condition_list is not None:
            ## the conditioning is specified 
            batch = self._drop_condition_inputs(batch, condition_list)


        condition_key = list(self.denoiser.condition_dict.values())[0]
        B, T = batch[condition_key].shape[0:2]
        # T = batch["audio_feature"].shape[1] if "processed_audio" in batch.keys() else batch["raw_audio"].shape[1]

        noise_shape = self.get_input_tensor_shape(T, round_up=True)
        noise_shape = [B] + noise_shape
        # noise = diffuser.get_init_noise(shape=noise_shape, device=self.device).to(self.device)
        if noise is None:
            noise = diffuser.get_training_init_noise(shape=noise_shape, device=self.device).to(self.device)
        
        like_training = True ##
        # like_training = False
        if like_training:
            D = diffuser.num_training_diffusion_steps()
            timesteps = diffuser.get_training_timesteps(D).to(self.device)
        else: 
            D = diffuser.num_testing_diffusion_steps() 
            timesteps = diffuser.get_testing_timesteps(D).to(self.device)
            
        # batch[self.diffuser.input_key] = noise
        batch[self.diffuser.output_key] = noise
        if not self.has_simple_objective: 
            batch[self.diffuser.noise_key] = noise
        prediction_history = []
        diffusion_steps = []

        if self.prior is not None:
            batch = self.decode(batch, 
                                key_to_decode=self.diffuser.output_key, 
                                # decoded_prefix = "gt_",
                                **kwargs)

        prediction_history = []
        prediction_dict(vertices=batch["predicted_vertices"],
                        vertices_posed=batch.get("predicted_vertices_posed", None),
                        cam=batch.get("predicted_cam", None),
                        global_rot=batch.get("predicted_global_pose", None),
                        )
        # prediction_history += [batch["predicted_vertices"]
        #                        ]
        prediction_history += [
            prediction_dict(
                vertices=batch["predicted_vertices"],
                vertices_posed=batch.get("predicted_vertices_posed", None),
                cam=batch.get("predicted_cam", None),
                global_rot=batch.get("predicted_global_pose", None),
                )
        ]

        step_prediction_history = []
        # step_prediction_history += [batch["predicted_vertices"]] 
        step_prediction_history += [
            prediction_dict(
                vertices=batch["predicted_vertices"],
                vertices_posed=batch.get("predicted_vertices_posed", None),
                cam=batch.get("predicted_cam", None),
                global_rot=batch.get("predicted_global_pose", None),
                )
        ]

        gt_history = []
        # gt_history += [batch["gt_vertices"]]
        gt_history += [
            prediction_dict(
                vertices=batch["gt_vertices"],
                vertices_posed=batch.get("gt_vertices_posed", None),
                cam=batch.get("gt_cam", None),
                global_rot=batch.get("gt_global_pose", None),
                )
        ]

        diffusion_steps = ["noisy_init"]

        di = 0

        freq = 10
        if D > 100:
            freq = 100
    
        # diff_steps_to_visualize = [0,1,2,3,4, 5, 7,] + list(range(10,D,freq)) 
        # diff_steps_to_visualize = [0,1,2,3,4, 5, 7,] + list(range(10,D,freq)) 
        
        if "diff_steps_to_visualize" in kwargs.keys():
            diff_steps_to_visualize = kwargs["diff_steps_to_visualize"]
        else:
            diff_steps_to_visualize = [0, 1, D//2] 

            if D-1 not in diff_steps_to_visualize:
                diff_steps_to_visualize += [D-2, D-1]
            
            diff_steps_to_visualize = sorted(list(set(diff_steps_to_visualize)))


        total_loss_history = []
        loss_history = []
        metric_history = []

        # for di, d in enumerate(range(1,D)): 
        for di, d in enumerate(range(0,D)): 

            ## 1) run the denoising step
            batch[self.diffuser.diffusion_steps_key] = timesteps[d:(d+1)].to(self.device)
            # batch = self.forward_denoise(batch, train=True)
            batch = self.forward_denoise(batch, train=like_training, 
                                         with_prior=False, with_postprocessors=False, 
                                         guidance_dict=guidance_dict,
                                         gradient_guidance_dict=gradient_guidance_dict,
                                         )

            ## 2) process the denoising output based on the denoiser objective
            batch, next_denoised = self.process_denoiser_output(batch, like_training=like_training)
            # if di == 998: 
            #     print("second last step")
            # if di == 999: 
            #     print("last step")

            # decode_batch = {} 
            # decode_batch["gt_shape"] = batch["gt_shape"]
            # decode_batch[self.denoiser.denoised_key] = next_denoised
            if di in diff_steps_to_visualize or compute_losses:
                if self.prior is not None:
                    # if not self.cfg.model.output.get('predict_shapecode', False):
                    #     decode_batch["gt_shape"] = batch["gt_shape"]
                    # if not self.cfg.model.output.get('predict_expcode', True): ## predict expression code true by default (backwards compatiblity)
                    #     decode_batch["gt_exp"] = batch["gt_exp"]
                    # if not self.cfg.model.output.get('predict_jawpose', True): ## predict jaw code true by default 
                    #     decode_batch["gt_jaw"] = batch["gt_jaw"]
                    batch = self.decode(batch, train=False)
                    decoded_batch = self.decode_denoised(batch, next_denoised, train=False, validation=False)
                    # decoded_batch = self.decode(decode_batch, train=False, validation=False)



            # print(f"Noise errors for step: {d}")
            # print(f"Added noise norm: \t{(batch['gt_latent'] - batch['noisy_latent']).norm() :.04f}")
            # print(f"Predicted noise norm: \t{batch[self.denoiser.predicted_noise_key].norm():.04f}")
            # # print(f"Noise diff norm: \t{(batch[self.diffuser.noise_key] - batch[self.denoiser.predicted_noise_key]).norm():.04f}")
            # # print(f"Noise diff norm2: \t{(batch[self.diffuser.noise_key] + batch[self.denoiser.predicted_noise_key]).norm():.04f}")

            # # print("Denoised batch decoded:")
            # noise2gt_error = (batch["gt_vertices"] - batch["noisy_vertices"]).norm(dim=-1).mean()
            # pred2gt_error = (batch["gt_vertices"] - batch["predicted_vertices"]).norm(dim=-1).mean()
            # pred2noise_error = (batch["predicted_vertices"] - batch["noisy_vertices"]).norm(dim=-1).mean()
            # print(f"Noisy to GT error: \t\t{noise2gt_error.item():.4f}")
            # print(f"Predicted to GT error: \t\t{pred2gt_error.item():.4f}")
            # print(f"Predicted to noise error:\t{pred2noise_error.item():.4f}")


            # batch = self.forward_denoise(batch, train=False)

            ## the following snippet has been refactored to be part of the forward_denoise method
            # if self.has_simple_objective:
            #     ## simple objective means the output of the denoiser is not the noise but the final space
            #     batch[self.diffuser.output_key] = batch[self.denoiser.denoised_key]
                
            # else: 
            #     ## not a simple objective means the output of the denoiser is the noise, that is added using 
            #     ## the scheduler in the denoiser
            #     noise_pred = batch[self.denoiser.predicted_noise_key]
            #     # denoised_vectors = self.diffuser.testing_step(d, noise_pred)
            #     denoised_vectors = self.diffuser.apply_predicted_noise(d, noise_pred)
            #     batch[self.diffuser.output_key] = denoised_vectors
            
            if compute_losses:
                total_loss, losses, metrics = self.compute_loss(batch, training=False, validation=False,)
                if "filename" in batch.keys():
                    prefix = Path(batch["filename"][0] )
                    prefix = prefix.parent.name + "/" + prefix.stem
                else: 
                    prefix = f"batch_{batch_idx}"

                # losses = {f"test/reverse_diff/{prefix}/{k}": v for k, v in losses.items()}
                # metrics = {f"test/reverse_diff/{prefix}/{k}": v for k, v in metrics.items()}


                # if isinstance(self.logger, WandbLogger):
                #     import wandb
                #     wandb.log({
                #         f"test/reverse_diff/{prefix}/diffusion_step": timesteps[d:(d+1)].item(), 
                #         f"test/reverse_diff/{prefix}/diffusion_step_idx": d, 
                #         **losses, 
                #         **metrics}
                #         )


                # total_loss_history += [total_loss]
                loss_history += [losses]
                metric_history += [metrics]


            # if di % freq == 0:
            if di in diff_steps_to_visualize:
                # prediction_history += [batch["predicted_vertices"]]
                prediction_history += [prediction_dict(
                    vertices=batch["predicted_vertices"],
                    vertices_posed=batch.get("predicted_vertices_posed", None),
                    cam=batch.get("predicted_cam", None),
                    global_rot=batch.get("predicted_global_pose", None),
                    )]
                # gt_history += [batch["gt_vertices"]]
                gt_history += [prediction_dict(
                    vertices=batch["gt_vertices"],
                    vertices_posed=batch.get("gt_vertices_posed", None),
                    cam=batch.get("gt_cam", None),
                    global_rot=batch.get("gt_global_pose", None),
                    )]
                # step_prediction_history += [decoded_batch["predicted_vertices"]]
                step_prediction_history += [prediction_dict(
                    vertices=decoded_batch["predicted_vertices"],
                    vertices_posed=decoded_batch.get("predicted_vertices_posed", None),
                    cam=decoded_batch.get("predicted_cam", None),
                    global_rot=decoded_batch.get("predicted_global_pose", None),
                    )]
                diffusion_steps += [timesteps[d:(d+1)].item()]



            ### prepare the batch for the next iteration
            if di < D-1:
                ## the denoised key is the input to the denoiser in the new iteration
                batch[self.diffuser.output_key] = next_denoised

                ## get rid of all predictions so that we don't use them by mistake in the next iteration
                batch.pop(self.denoiser.denoised_key)
                batch.pop(self.denoiser.predicted_noise_key)
                keys = list(batch.keys())

            # if di < D:
                for key in keys:
                    if key.startswith('predicted_'): 
                        batch.pop(key)
            
            # else: ## last iteration, denoising is finished 
            #     batch[self.denoiser.denoised_key] = next_denoised


        if gradient_guidance_dict is not None:
            for grad_guide, weight in gradient_guidance_dict.items():
                grad_guide.reset_targets()

        if compute_losses:
            diff_steps = timesteps.cpu().numpy()
            self._difusion_metric_plot(diff_steps, loss_history, "test/reverse_diff/" + prefix, D)
            self._difusion_metric_plot(diff_steps, metric_history, "test/reverse_diff/" + prefix, D)
        # lo

        # return batch, prediction_history, gt_history, diffusion_steps #, loss_history, metric_history
        return batch, prediction_history, step_prediction_history, gt_history, diffusion_steps #, loss_history, metric_history
    

    def full_diffusion(self, batch, batch_idx, diffuser=None, **kwargs): 
        diffuser = diffuser or self.diffuser 

        batch = self.forward_input_encoders(batch, train=False, validation=False)

        if self.prior is not None:
            batch = self.encode(batch, **kwargs)

        condition_key = list(self.denoiser.condition_dict.values())[0]
        B, T = batch[condition_key].shape[0:2]
        # T = batch["audio_feature"].shape[1] if "processed_audio" in batch.keys() else batch["raw_audio"].shape[1]

        noise_shape = self.get_input_tensor_shape(T, round_up=True)
        # noise_shape = self.get_input_tensor_shape(T, round_up=False)
        noise_shape = [B] + noise_shape
        noise = diffuser.get_init_noise(shape=noise_shape, device=self.device)
        D = diffuser.num_training_diffusion_steps() 

        batch[self.diffuser.noise_key] = noise

        prediction_history = []
        next_prediction_history = []
        noisy_history = []

        import copy
        backup_batch = copy.deepcopy(batch)

        if self.prior is not None:
            ## 
            batch[self.denoiser.denoised_key] = batch[self.diffuser.input_key]
            batch = self.decode(batch, **kwargs)

        # prediction_history += [batch["predicted_vertices"]]
        prediction_history += [prediction_dict(
            vertices=batch["predicted_vertices"],
            vertices_posed=batch.get("predicted_vertices_posed", None),
            cam=batch.get("predicted_cam", None),
            )]
        # next_prediction_history += [batch["predicted_vertices"]]
        next_prediction_history += [prediction_dict(
            vertices=batch["predicted_vertices"],
            vertices_posed=batch.get("predicted_vertices_posed", None),
            cam=batch.get("predicted_cam", None),
            )]
        # noisy_history += [batch["gt_vertices"]]
        noisy_history += [prediction_dict(
            vertices=batch["gt_vertices"],
            vertices_posed=batch.get("gt_vertices_posed", None),
            cam=batch.get("gt_cam", None),
            )]

        # print("GT cam: " )
        # print("mean:", batch["gt_cam"].view(B *T, -1).mean(dim=0))
        # print("std:", batch["gt_cam"].view(B *T, -1).std(dim=0))

        # diffusion_steps = list(range(D))
        # diffusion_steps = list(range(0, D, 100)) 
        diffusion_steps = list(range(0, D, 300)) 
        if D-1 not in diffusion_steps:
            diffusion_steps += [D-1]
        # diffusion_steps = [0]
        diffusion_steps = list(reversed(diffusion_steps))

        loss_history = []
        metric_history = []

        # for d in range(D-1,-1,-1):  
        for d in diffusion_steps: 
            batch = copy.deepcopy(backup_batch)
            batch[self.diffuser.diffusion_steps_key] = torch.zeros(B, device = self.device, dtype=torch.long) + d 
            batch = self.diffuse(batch, batch_idx)

            noisy_batch = {} 
            noisy_batch["gt_shape"] = batch["gt_shape"]
            if not self.cfg.model.output.get('predict_expcode', True): ## predict expression code true by default (backwards compatiblity)
                noisy_batch["gt_exp"] = batch["gt_exp"]
            if not self.cfg.model.output.get('predict_jawpose', True): ## predict jaw code true by default 
                noisy_batch["gt_jaw"] = batch["gt_jaw"]
            noisy_batch[self.denoiser.denoised_key] = batch[self.diffuser.output_key]
            noisy_batch = self.decode(noisy_batch, train=False, validation=False)


            # if self.prior is not None:
            #     batch = self.decode(batch, 
            #                         key_to_decode=self.diffuser.output_key, 
            #                         decoded_prefix = "noisy_",
            #                         **kwargs)

            batch = self.forward_denoise(batch, train=True) ## train=True to use the training noise scheduler
            
            batch, next_denoised = self.process_denoiser_output(batch, like_training=True)



            # decode_batch = {} 
            # decode_batch["gt_shape"] = batch["gt_shape"]
            # if not self.cfg.model.output.get('predict_expcode', True): ## predict expression code true by default (backwards compatiblity)
            #     decode_batch["gt_exp"] = batch["gt_exp"]
            # if not self.cfg.model.output.get('predict_jawpose', True): ## predict jaw code true by default 
            #     decode_batch["gt_jaw"] = batch["gt_jaw"]
            # decode_batch[self.denoiser.denoised_key] = next_denoised
            # decoded_batch = self.decode(decode_batch, train=False, validation=False)
            decoded_batch = self.decode_denoised(batch, next_denoised, train=False, validation=False)


            # print(f"Noise errors for step: {d}")
            # print(f"Added noise norm: \t{batch[self.diffuser.noise_key].norm() :.04f}")
            # print(f"Predicted noise norm: \t{batch[self.denoiser.predicted_noise_key].norm():.04f}")
            # print(f"Noise diff norm: \t{(batch[self.diffuser.noise_key] - batch[self.denoiser.predicted_noise_key]).norm():.04f}")
            # print(f"Noise diff norm2: \t{(batch[self.diffuser.noise_key] + batch[self.denoiser.predicted_noise_key]).norm():.04f}")

            total_loss, losses, metrics = self.compute_loss(batch, training=False, validation=False,)
            # if 'loss_motion_latent_loss' in losses:
            #     print(f"Motion latent loss: {losses['loss_motion_latent_loss']}")
            # if 'loss_vertex_loss' in losses:
            #     print(f"Vertex loss: {losses['loss_vertex_loss']}")
            # if 'loss_exp_loss' in losses:
            #     print(f"Expression loss: {losses['loss_exp_loss']}")
            # if 'loss_jaw_loss' in losses:
            #     print(f"Jaw loss: {losses['loss_jaw_loss']}")
            loss_history += [losses]
            metric_history += [metrics]
            if "filename" in batch.keys():
                prefix = Path(batch["filename"][0] )
                prefix = prefix.parent.name + "/" + prefix.stem
            else: 
                prefix = f"batch_{batch_idx}"

            # losses = {f"test/diff/{prefix}/{k}": v for k, v in losses.items()}
            # metrics = {f"test/diff/{prefix}/{k}": v for k, v in metrics.items()}
            # if isinstance(self.logger, WandbLogger):
            #     import wandb
            #     wandb.log({
            #         f"test/diff/{prefix}/diffusion_step": batch[self.diffuser.diffusion_steps_key].item(), 
            #         f"test/diff/{prefix}/diffusion_step_idx": d, 
            #         **losses, 
            #         **metrics
            #         }
            #             )



            # prediction_history += [batch["predicted_vertices"].cpu()] 
            prediction_history += [prediction_dict(
                vertices=batch["predicted_vertices"],
                vertices_posed=batch.get("predicted_vertices_posed", None), 
                # vertices_posed=backup_batch.get("gt_vertices_posed", None),

                cam=batch.get("predicted_cam", None),
                )]
            # next_prediction_history += [decoded_batch["predicted_vertices"].cpu()]
            next_prediction_history += [prediction_dict(
                vertices=decoded_batch["predicted_vertices"],
                vertices_posed=decoded_batch.get("predicted_vertices_posed", None),
                # vertices_posed=backup_batch.get("gt_vertices_posed", None),

                cam=decoded_batch.get("predicted_cam", None),
                )]
            # # noisy_history += [batch["gt_vertices"].cpu()]
            # noisy_history += [noisy_batch["predicted_vertices"].cpu()]
            noisy_history += [prediction_dict(
                vertices=noisy_batch["predicted_vertices"],
                vertices_posed=noisy_batch.get("predicted_vertices_posed", None), 
                # vertices_posed=backup_batch.get("gt_vertices_posed", None),

                cam=noisy_batch.get("predicted_cam", None),
                )]

            # if self.has_simple_objective:
            #     ## simple objective means the output of the denoiser is not the noise but the final space
            #     batch[self.diffuser.input_key] = batch[self.denoiser.denoised_key]
            #     batch.pop(self.denoiser.denoised_key)
            #     # prediction_history += [batch[self.diffuser.input_key]]

            #     prediction_history += [batch["predicted_vertices"]]
            #     noisy_history += [batch["gt_vertices"]]

            # else: 
            #     ## not a simple objective means the output of the denoiser is the noise, that is added using 
            #     ## the scheduler in the denoiser
            #     noise_pred = batch[self.denoiser.denoised_key]
            #     denoised_vectors = self.diffuser.testing_step(d, noise_pred)
            #     batch[self.denoiser.input_key] = denoised_vectors
            #     prediction_history += [denoised_vectors]


        # # batch[self.diffuser.input_key] = noise
        # batch[self.diffuser.output_key] = noise
        # prediction_history = []

        # for d in range(D): 
        #     batch[self.diffuser.diffusion_steps_key] = timesteps[d:(d+1)]
        #     batch = self.forward_denoise(batch)
        #     if self.has_simple_objective:
        #         ## simple objective means the output of the denoiser is not the noise but the final space
        #         batch[self.diffuser.input_key] = batch[self.denoiser.denoised_key]
        #         batch.pop(self.denoiser.denoised_key)
        #         prediction_history += [batch[self.diffuser.input_key]]
        #     else: 
        #         ## not a simple objective means the output of the denoiser is the noise, that is added using 
        #         ## the scheduler in the denoiser
        #         noise_pred = batch[self.denoiser.denoised_key]
        #         denoised_vectors = self.diffuser.testing_step(d, noise_pred)
        #         batch[self.denoiser.input_key] = denoised_vectors
        #         prediction_history += [denoised_vectors]
        

        # self._difusion_metric_plot(diffusion_steps, losses, "test/diff/" + prefix, D)


        # diff_steps = np.array(diffusion_steps)
        # self._difusion_metric_plot(diff_steps, loss_history, "test/diffusion/" + prefix, len(diff_steps))
        # self._difusion_metric_plot(diff_steps, metric_history, "test/diffusion/" + prefix, len(diff_steps))

        # if isinstance(self.logger, WandbLogger):
        #     import wandb
        #     ## create a plot of the losses and metrics with x-axis of the diffusion step 
        #     ## and the y-axis of the loss or metric value
        #     ## this is useful to see how the loss and metrics evolve over the diffusion steps
        #     ## and to see if the model is overfitting or underfitting
        #     ## and to see if the model is learning anything at all

        #     import plotly.express as px
        #     import plotly.graph_objects as go
        #     import plotly.figure_factory as ff
        #     import numpy as np
        #     import pandas as pd
            
        #     for k, v in losses.items():
        #         # if isinstance(v, torch.Tensor):
        #         #     losses[k] = v.item()
        #         values = []
        #         for d in range(D):
        #             values += [v.item()]
        #         values = np.array(values)
        #         fig = go.Figure()
        #         fig.add_trace(go.Scatter(x=list(diffusion_steps), y=values, mode='lines+markers', name=k))
        #         fig.show()

        #         # log the plot to wandb 
        #         wandb.log({f"test/diff/{prefix}/{k}": fig})
        
        diffusion_steps = ["enc_dec"] + diffusion_steps
        return batch, prediction_history, next_prediction_history, noisy_history, diffusion_steps
        # return batch, prediction_history, noisy_history, diffusion_steps


    def _difusion_metric_plot(self, diffusion_steps, losses, prefix, D):
        import plotly.graph_objects as go
        for k, v in losses[0].items():
            # if isinstance(v, torch.Tensor):
            #     losses[k] = v.item()
            values = []
            for d in range(D):
                values += [losses[d][k].item()]
            values = np.array(values)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=diffusion_steps, y=values, mode='lines+markers', name=k))
            # fig.update_layout(title='Metric Over Diffusion Steps', xaxis_title='Diffusion Step', yaxis_title='Metric Value')
            # fig.show()

            if isinstance(self.logger, WandbLogger):
                import wandb
                # log the plot to wandb 
                wandb.log({f"{prefix}/{k}_fig": wandb.Plotly(fig)})
            
            del fig

    # def full_denoise(self):
    #     texts = batch["text"]
    #     lengths = batch["length"]

    #     if self.cfg.TEST.COUNT_TIME:
    #         self.starttime = time.time()

    #     if self.stage in ['diffusion', 'vae_diffusion']:
    #         # diffusion reverse
    #         if self.do_classifier_free_guidance:
    #             uncond_tokens = [""] * len(texts)
    #             if self.condition == 'text':
    #                 uncond_tokens.extend(texts)
    #             elif self.condition == 'text_uncond':
    #                 uncond_tokens.extend(uncond_tokens)
    #             texts = uncond_tokens
    #         text_emb = self.text_encoder(texts)
    #         z = self._diffusion_reverse(text_emb, lengths)
    #     elif self.stage in ['vae']:
    #         motions = batch['motion']
    #         z, dist_m = self.vae.encode(motions, lengths)

    #     with torch.no_grad():
    #         # ToDo change mcross actor to same api
    #         if self.vae_type in ["mld","actor"]:
    #             feats_rst = self.vae.decode(z, lengths)
    #         elif self.vae_type == "no":
    #             feats_rst = z.permute(1, 0, 2)

    #     if self.cfg.TEST.COUNT_TIME:
    #         self.endtime = time.time()
    #         elapsed = self.endtime - self.starttime
    #         self.times.append(elapsed)
    #         if len(self.times) % 100 == 0:
    #             meantime = np.mean(
    #                 self.times[-100:]) / self.cfg.TEST.BATCH_SIZE
    #             print(
    #                 f'100 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
    #             )
    #         if len(self.times) % 1000 == 0:
    #             meantime = np.mean(
    #                 self.times[-1000:]) / self.cfg.TEST.BATCH_SIZE
    #             print(
    #                 f'1000 iter mean Time (batch_size: {self.cfg.TEST.BATCH_SIZE}): {meantime}',
    #             )
    #             with open(pjoin(self.cfg.FOLDER_EXP, 'times.txt'), 'w') as f:
    #                 for line in self.times:
    #                     f.write(str(line))
    #                     f.write('\n')
    #     joints = self.feats2joints(feats_rst.detach().cpu())
    #     return remove_padding(joints, lengths)


    # def _diffusion_reverse(self, encoder_hidden_states, lengths=None):
    #     # init latents
    #     bsz = encoder_hidden_states.shape[0]
    #     if self.do_classifier_free_guidance:
    #         bsz = bsz // 2
    #     if self.vae_type == "no":
    #         assert lengths is not None, "no vae (diffusion only) need lengths for diffusion"
    #         latents = torch.randn(
    #             (bsz, max(lengths), self.cfg.DATASET.NFEATS),
    #             device=encoder_hidden_states.device,
    #             dtype=torch.float,
    #         )
    #     else:
    #         latents = torch.randn(
    #             (bsz, self.latent_dim[0], self.latent_dim[-1]),
    #             device=encoder_hidden_states.device,
    #             dtype=torch.float,
    #         )

    #     # scale the initial noise by the standard deviation required by the scheduler
    #     latents = latents * self.scheduler.init_noise_sigma
    #     # set timesteps
    #     self.scheduler.set_timesteps(
    #         self.cfg.model.scheduler.num_inference_timesteps)
    #     timesteps = self.scheduler.timesteps.to(encoder_hidden_states.device)
    #     # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    #     # eta (η) is only used with the DDIMScheduler, and between [0, 1]
    #     extra_step_kwargs = {}
    #     if "eta" in set(
    #             inspect.signature(self.scheduler.step).parameters.keys()):
    #         extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

    #     # reverse
    #     for i, t in enumerate(timesteps):
    #         # expand the latents if we are doing classifier free guidance
    #         latent_model_input = (torch.cat(
    #             [latents] *
    #             2) if self.do_classifier_free_guidance else latents)
    #         lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
    #                            else lengths)
    #         # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    #         # predict the noise residual
    #         noise_pred = self.denoiser(
    #             sample=latent_model_input,
    #             timestep=t,
    #             encoder_hidden_states=encoder_hidden_states,
    #             lengths=lengths_reverse,
    #         )[0]
    #         # perform guidance
    #         if self.do_classifier_free_guidance:
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + self.guidance_scale * (
    #                 noise_pred_text - noise_pred_uncond)
    #             # text_embeddings_for_guidance = encoder_hidden_states.chunk(
    #             #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
    #         latents = self.scheduler.step(noise_pred, t, latents,
    #                                           **extra_step_kwargs).prev_sample
    #         # if self.predict_epsilon:
    #         #     latents = self.scheduler.step(noise_pred, t, latents,
    #         #                                   **extra_step_kwargs).prev_sample
    #         # else:
    #         #     # predict x for standard diffusion model
    #         #     # compute the previous noisy sample x_t -> x_t-1
    #         #     latents = self.scheduler.step(noise_pred,
    #         #                                   t,
    #         #                                   latents,
    #         #                                   **extra_step_kwargs).prev_sample

    #     # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
    #     latents = latents.permute(1, 0, 2)
    #     return latents
    



    @classmethod
    def instantiate(cls, cfg, stage=None, prefix=None, checkpoint=None, checkpoint_kwargs=None) -> 'DiffusionTalkingHead':
        """
        Function that instantiates the model from checkpoint or config
        """
        if checkpoint is None:
            model = DiffusionTalkingHead(cfg)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = DiffusionTalkingHead.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=False, 
                **checkpoint_kwargs
            )
        return model


def prediction_dict(vertices, vertices_posed=None, cam=None, global_rot=None): 
    d = Munch() 
    d.vertices = vertices.detach().cpu()
    if vertices_posed is not None:
        d.vertices_posed = vertices_posed.detach().cpu()
    if cam is not None:
        d.cam = cam.detach().cpu()
    if global_rot is not None:
        d.global_rot = global_rot.detach().cpu()
    return d