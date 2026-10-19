import pytorch_lightning as pl 
from typing import Any, Optional
from inferno.models.temporal.AudioEncoders import TemporalAudioEncoder
from inferno.models.temporal.Bases import *
from .text.TextBase import TextEncoderBase
from .mesh2speech.MeshEncoderBase import Mesh4DEncoderBase
from .mesh2speech.Mesh2SpeechDecoderBase import Mesh4DecoderBase 
from .mesh2speech.VocoderBase import Mesh2SpeechVocoderBase
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
import torch.nn.functional as F


class MeshToSpeech(pl.LightningModule): 

    def __init__(self,
                 cfg,
                 preprocessors: Optional[Dict] = None,
                 input_encoder: Mesh4DEncoderBase = None,
                 sequence_encoder: SequenceEncoder = None, 
                #  transcript_encoder: Optional[nn.Module] = None,
                #  prior: Optional[nn.Module] = None,
                #  renderer: Optional[Renderer] = None, 
                 speech_prediction_heads: Dict = None,
                 vocoder: Optional[Mesh2SpeechVocoderBase] = None,
                 postprocessors: Optional[Dict] = None,

                 **kwargs):
        
        super().__init__()
        self.cfg = cfg
        self.preprocessors = preprocessors
        self.input_encoder = input_encoder
        self.sequence_encoder = sequence_encoder
        self.prediction_heads = torch.nn.ModuleDict(speech_prediction_heads)
        self.vocoder = vocoder
        # self.prior = prior
        self.postprocessors = postprocessors
        self._initialize_losses() 


    def _initialize_losses(self): 
        pass 
        ## TODO this is where losses will be initialized 

    @property
    def max_seq_length(self):
        return 5000
    
    def to(self, device):
        if self.preprocessors is not None:
            for preprocessor in self.preprocessors.values():
                preprocessor.to(device)
        if self.postprocessors is not None:
            for postprocessor in self.postprocessors.values():
                postprocessor.to(device)
        return super().to(device)
    
    @property
    def zero_shape(self): 
        # assert not self.random_shape, "zero_shape is not supported when random_shape is True"
        return self.cfg.model.input.get('zero_shape', False)

    @property
    def random_shape(self):
        assert not self.zero_shape, "random_shape is not supported when zero_shape is True"
        return bool(self.cfg.model.input.get('random_shape_sigma', False))

    @property
    def random_shape_sigma(self):
        return self.cfg.model.input.get('random_shape_sigma', 0.01)

    def get_trainable_parameters(self):
        trainable_params = []
        if self.input_encoder is not None:
            trainable_params += self.input_encoder.get_trainable_parameters()
        if self.sequence_encoder is not None:
            trainable_params += self.sequence_encoder.get_trainable_parameters()
        if self.prediction_heads is not None:
            for key, head in self.prediction_heads.items():
                trainable_params += head.get_trainable_parameters()
        return trainable_params
    
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
        return opt_dict
    
    def _choose_primary_3D_rec_method(self, batch, train=False, validation=False): 
        method = self.cfg.data.get('reconstruction_type', None)
        if method is not None:
            if isinstance(method, (list, omegaconf.listconfig.ListConfig)):
                # method = random.choice(method)
                method = method[0]
            if "reconstruction" in batch:
                for k in batch["reconstruction"][method].keys():
                    assert k not in batch.keys(), f"Key '{k}' already exists in batch. We don't want to overwrite it."
                    batch[k] = batch["reconstruction"][method][k]
                # batch.pop("reconstruction")
            if self.cfg.model.get('input_shapecode', False):
                assert "gt_shape" in batch.keys(), "Key 'shapecode' not found in batch."
        if "gt_shape" not in batch.keys():
            B, T = batch["gt_exp"].shape[:2]
            # batch["gt_shape"] = torch.zeros(B, T, 300, device=batch["gt_exp"].device)
            batch["gt_shape"] = torch.zeros(B, 300, device=batch["gt_exp"].device)
        elif self.zero_shape:
            batch["gt_shape"] = torch.zeros_like(batch["gt_shape"])
        elif self.random_shape and (train or validation): ## we don't want to sample random shape during testing
            B, T = batch["gt_exp"].shape[:2] 
            sigma = self.random_shape_sigma
            batch["gt_shape"] = torch.randn(B, 300, device=batch["gt_exp"].device) * sigma
        return batch

    def _choose_padding_mask(self, batch):
        batch["padding_mask"] = batch["landmarks_validity"]["mediapipe"][...,0].to(dtype=torch.bool)
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


    def forward_mesh_encoder(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        batch = self.input_encoder(batch, train=train, validation=validation, **kwargs)
        return batch

    def forward_sequence_encoder(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        batch = self.sequence_encoder(batch, train=train, validation=validation, **kwargs)
        return batch

    def forward_speech_prediction_head(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        for key, speech_prediction_head in self.prediction_heads.items():
            batch = speech_prediction_head(batch, train=train, validation=validation, **kwargs)
        return batch
    
    def forward_vocoder(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        if self.vocoder is not None:
            ## forward the prediction
            batch = self.vocoder(batch, train=train, validation=validation, **kwargs) 
            with torch.no_grad():
                ## compute the predicted audio
                self.vocoder(batch, input_mel_key="gt_mel", 
                             input_unit_key="gt_units", output_key="vocoded_gt_wav", 
                             train=False, validation=False, **kwargs)
        return batch

    def forward_postpocessors(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        if self.postprocessors is not None:
            for key, postprocessor in self.postprocessors.items():
                batch = postprocessor(batch, train=train, validation=validation, input_key=None, **kwargs)
        return batch


    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        batch = self._choose_padding_mask(batch)
        batch = self._choose_primary_3D_rec_method(batch, train=train, validation=validation)

        ## 0) Preprocessors
        batch = self.forward_prepocessors(batch, train=train, validation=validation, with_grad=kwargs.pop('preprocessor_with_grad', False), **kwargs)
     
        ## 1) Mesh Encoder 
        batch = self.forward_mesh_encoder(batch, train=train, validation=validation, **kwargs)

        ## 2) Sequence Encoder 
        batch = self.forward_sequence_encoder(batch, train=train, validation=validation, **kwargs)

        ## 3) Speech prediction 
        batch = self.forward_speech_prediction_head(batch, train=train, validation=validation, **kwargs)

        ## 4) Vocoder
        batch = self.forward_vocoder(batch, train=train, validation=validation, **kwargs)

        ## 5) Postprocessing
        batch = self.forward_postpocessors(batch, train=train, validation=validation, **kwargs)

        return batch


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
        prefix = "val"
        # if dataloader_idx is not None and dataloader_idx > 0:
        #     prefix = prefix + f"_{dataloader_idx}"
        # losses_and_metrics_to_log = {"val_" + k: v.item() for k, v in losses_and_metrics_to_log.items()}
        losses_and_metrics_to_log = {f"{prefix}/" + k: v.item() if isinstance(v, (torch.Tensor)) else v if isinstance(v, float) else 0. for k, v in losses_and_metrics_to_log.items()}

        # if self.logger is not None:
        self.log_dict(losses_and_metrics_to_log, on_step=False, on_epoch=True, sync_dist=True) # log per epoch, # recommended
            # self.log_dict(losses_and_metrics_to_log, on_step=True, on_epoch=True, sync_dist=True) # log per epoch, # recommended

        return total_loss
    

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
            mask = torch.ones((*sample["gt_exp"].shape[:2], 1), device=sample["gt_exp"].device, dtype=torch.bool)

        # mask = torch.zeros_like(mask)

        # if "disentangled" in loss_type and not validation and not training:
        #     # disentangled losses are only computed during training
        #     return 0.

        B = sample["gt_exp"].shape[0] # total batch size
        T = sample["gt_exp"].shape[1] # sequence size
       
        mask_sum = mask.sum().item()

        if mask_invalid and mask_sum == 0:
            return 0.

        if loss_type in ["mel_reconstruction_loss", "mel_vocoded_loss"]:
            pred_key = loss_cfg.get('predicted_key', 'predicted_mel')
            gt_key = loss_cfg.get('gt_key', 'gt_mel')
            gt_mel = sample[gt_key]
            predicted_mel = sample[pred_key]

            T_mel = gt_mel.shape[1]
            assert T_mel % T == 0, f"gt_mel.shape[1] ({T_mel}) must be divisible by T ({T})" 
            T_factor = T_mel // T
            if T_factor > 1:
                mask = mask.repeat_interleave(T_factor, dim=1)
                mask_sum = mask.sum().item()

            if mask.ndim == 3: 
                mask = mask.squeeze(-1)

            if loss_cfg.metric in ["mse", "l2"]:
                loss_value = MaskedTemporalMSELoss()(predicted_mel, gt_mel, mask=mask)
            
            elif loss_cfg.metric in ["mae", "l1"]:
                loss_value = MaskedTemporalMAELoss()(predicted_mel, gt_mel, mask=mask)
            
            else: 
                raise ValueError(f"loss_cfg.metric '{loss_cfg.metric}' not supported")
             
        elif loss_type == "unit_cross_entropy": 
            pred_key = loss_cfg.get('predicted_key', 'predicted_unit_classification_vectors')
            gt_key = loss_cfg.get('gt_key', 'gt_units') 

            gt_units = sample[gt_key]
            predicted_unit_vectors = sample[pred_key]



            T_units = gt_units.shape[1]
            predicted_unit_vectors = predicted_unit_vectors[:, :T_units, :]
            assert T_units % T == 0 or (T_units+1) % T == 0, f"gt_units.shape[1] ({T_units}) must be divisible by T ({T})"
            T_factor = T_units // T
            if T_units % T != 0 :
                T_factor += 1
            mask = mask.repeat_interleave(T_factor, dim=1)
            mask = mask[:, :T_units, :].squeeze(-1)
            mask_sum = mask.sum().item()

            gt_units = gt_units.view(B * T_units)
            predicted_unit_vectors = predicted_unit_vectors.reshape(B * T_units, -1)

            loss_values = F.cross_entropy(predicted_unit_vectors, gt_units, reduction='none')
            loss_values = loss_values.view(B, T_units)
            
            loss_values = loss_values * mask.squeeze(-1) 
            loss_value = loss_values.sum() / mask_sum

        elif loss_type == "lip_feature_loss":
            pred_key = loss_cfg.get('predicted_key', 'input_features')
            gt_key = loss_cfg.get('gt_key', 'lip_features')
            gt_lip_features = sample[gt_key]
            predicted_lip_features = sample[pred_key]

            T_lip_features = gt_lip_features.shape[1]
            assert T_lip_features % T == 0, f"gt_lip_features.shape[1] ({T_lip_features}) must be divisible by T ({T})" 
            T_factor = T_lip_features // T
            if T_factor > 1:
                mask = mask.repeat_interleave(T_factor, dim=1)
                mask_sum = mask.sum().item()

            if mask.ndim == 3: 
                mask = mask.squeeze(-1)

            if loss_cfg.metric in ["mse", "l2"]:
                loss_value = MaskedTemporalMSELoss()(predicted_lip_features, gt_lip_features, mask=mask)
            
            elif loss_cfg.metric in ["mae", "l1"]:
                loss_value = MaskedTemporalMAELoss()(predicted_lip_features, gt_lip_features, mask=mask)
            
            else: 
                raise ValueError(f"loss_cfg.metric '{loss_cfg.metric}' not supported")
            

        elif loss_type == "wav_rec":
            pred_key = loss_cfg.get('predicted_key', 'predicted_wav')
            gt_key = loss_cfg.get('gt_key', 'raw_audio')
            gt_waveform = sample[gt_key]
            predicted_waveform = sample[pred_key]

            T = sample["gt_exp"].shape[1]
            gt_waveform = gt_waveform.view(B, T, -1)
            predicted_waveform = predicted_waveform.view(B, T, -1)


            if mask.ndim == 3: 
                mask = mask.squeeze(-1)

            if loss_cfg.metric in ["mse", "l2"]:
                loss_value = MaskedTemporalMSELoss()(predicted_waveform, gt_waveform, mask=mask)
            
            elif loss_cfg.metric in ["mae", "l1"]:
                loss_value = MaskedTemporalMAELoss()(predicted_waveform, gt_waveform, mask=mask)
            
            else: 
                raise ValueError(f"loss_cfg.metric '{loss_cfg.metric}' not supported")
 
        elif loss_type in ["softplus"]:
            pred_key = loss_cfg.input_key
            gt_key = loss_cfg.get('gt_key', 'gt_mel')
            value = sample[pred_key]
            l = -1 if loss_cfg.label else 1
            loss_value = F.softplus(l*value).mean()

        else: 
            raise ValueError(f"loss_type '{loss_type}' not supported")

        return loss_value



    @classmethod
    def instantiate(cls, cfg, stage=None, prefix=None, checkpoint=None, checkpoint_kwargs=None) -> 'MeshToSpeech':
        """
        Function that instantiates the model from checkpoint or config
        """
        kwargs = instantiate_components(cfg)
        if checkpoint is None:
            model = MeshToSpeech(cfg, **kwargs)
        else:
            checkpoint_kwargs = checkpoint_kwargs or {}
            model = MeshToSpeech.load_from_checkpoint(
                checkpoint_path=checkpoint, 
                cfg=cfg, 
                strict=False, 
                **kwargs,
                **checkpoint_kwargs
            )
        return model
    

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
    elif cfg.type == "VideoMouthCropPreprocessor": 
        from .mesh2speech.VideoMouthCropPreprocessor import VideoMouthCropPreprocessor
        preprocessor = VideoMouthCropPreprocessor(**params)
    elif cfg.type == "FlamePreprocessor":
        from inferno.models.temporal.Preprocessors import FlamePreprocessor
        from munch import munchify
        preprocessor = FlamePreprocessor(munchify(params))
    else: 
        raise NotImplementedError(f"Preprocessor type '{cfg.type}' not implemented")
    return preprocessor


def preprocessors_from_cfg(cfg):
    preprocessors = {}
    for key, preprocessor_cfg in cfg.items():
        preprocessor = preprocessor_from_cfg(preprocessor_cfg)
        preprocessors[key] = preprocessor 
    return preprocessors


def mesh_encoder_from_cfg(cfg):
    if cfg.type == "FlameParamConcatenation":
        from .mesh2speech.MeshEncoderBase import FlameParamConcatenation
        model = FlameParamConcatenation(cfg)
    elif cfg.type == "FlameParamCatLinear":
        from .mesh2speech.MeshEncoderBase import FlameParamCatLinear
        model = FlameParamCatLinear(cfg)
    elif cfg.type == "FlameParamCatMLP":
        from .mesh2speech.MeshEncoderBase import FlameParamCatMLP
        model = FlameParamCatMLP(cfg)
    elif cfg.type == "VertexLinear":
        from .mesh2speech.MeshEncoderBase import VertexLinear
        model = VertexLinear(cfg)
    elif cfg.type == "VertexMLP":
        from .mesh2speech.MeshEncoderBase import VertexMLP
        model = VertexMLP(cfg)
    elif cfg.type == "VertexTransformer":
        from .mesh2speech.MeshEncoderBase import VertexTransformer
        model = VertexTransformer(cfg)
    elif cfg.type == "Lip2SpeechEncoder": 
        from .mesh2speech.MeshEncoderBase import Lip2SpeechEncoder
        model = Lip2SpeechEncoder(cfg)
    else: 
        raise NotImplementedError(f"Mesh encoder type '{cfg.type}' not implemented")
    return model


def sequence_encoder_from_cfg(cfg):
    if cfg.type == "Lip2SpeechConformer":
        from .mesh2speech.Mesh2SpeechConformer import Lip2SpeechConformer
        model = Lip2SpeechConformer(cfg)
    else: 
        raise NotImplementedError(f"Denoiser type '{cfg.type}' not implemented")
    return model


def prediction_heads_from_cfg(cfg):
    prediction_heads = {}
    for key, head_cfg in cfg.items():
        if head_cfg.type == "Lip2SpeechUnitClassifier":
            from .mesh2speech.Lip2SpeechDecoderBase import Lip2SpeechUnitClassifier
            head = Lip2SpeechUnitClassifier(head_cfg)
        elif head_cfg.type == "Lip2SpeechMelDecoder":
            from .mesh2speech.Lip2SpeechDecoderBase import Lip2SpeechMelDecoder
            head = Lip2SpeechMelDecoder(head_cfg)
        else: 
            raise NotImplementedError(f"Prediction head type '{head_cfg.type}' not implemented")
        prediction_heads[key] = head 
    return prediction_heads


def vocoder_from_cfg(cfg):
    if cfg.type == "Lip2SpeechVocoder":
        from inferno.sandbox.thunder.models.mesh2speech.Lip2SpeechVocoder import Lip2SpeechVocoder
        vocoder = Lip2SpeechVocoder(cfg)
    else:
        raise NotImplementedError(f"Vocoder type '{cfg.type}' not implemented")
    return vocoder


def instantiate_components(cfg): 
    preprocessors = preprocessors_from_cfg(cfg.model.preprocessors)
    input_encoder = mesh_encoder_from_cfg(cfg.model.input_encoder)
    sequence_encoder = sequence_encoder_from_cfg(cfg.model.sequence_encoder)
    prediction_heads = prediction_heads_from_cfg(cfg.model.prediction_heads) 
    vocoder = None
    if "vocoder" in cfg.model.keys() and cfg.model.vocoder is not None:
        vocoder = vocoder_from_cfg(cfg.model.vocoder) 
    postprocessors = None
    if "postprocessors" in cfg.model.keys():
        postprocessors = preprocessors_from_cfg(cfg.model.postprocessors)

    kwargs = { 
        "preprocessors": preprocessors,
        "input_encoder": input_encoder,
        "sequence_encoder": sequence_encoder,
        "speech_prediction_heads": prediction_heads,
        "vocoder": vocoder, 
        "postprocessors": postprocessors
    }
    return kwargs