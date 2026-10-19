import os 
import torch
from .Mesh2SpeechDecoderBase import Mesh4DecoderBase
from torch import nn
from typing import Dict, Any
from pathlib import Path
from inferno.utils.other import get_path_to_assets

class Lip2SpeechMelDecoder(Mesh4DecoderBase):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # d = cfg.conformer_embed_dim
        d = self.input_dim
        self.mel_conv = nn.Sequential(
            nn.Conv1d(in_channels=d+self.speaker_embedding_dim,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.dropout),
            nn.GELU(),
            nn.Conv1d(in_channels=d,out_channels=d,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.dropout),
            nn.GELU(),
            nn.Conv1d(in_channels=d,out_channels=self.output_dim,kernel_size=3,stride=1,padding=1),
            nn.Dropout(cfg.dropout),
            nn.GELU(),
        )
        self.mel_proj = nn.Linear(d, 160)

        self.trainable = cfg.get('trainable', True)
        self.pretrained_checkpoint = cfg.get('pretrained_checkpoint', None)

        if self.pretrained_checkpoint is not None:
            self._load_pretrained()

        if not self.trainable:
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        mode = mode and self.trainable
        super().train(mode)

    def _load_pretrained(self):
        if self.pretrained_checkpoint is not None:
            print(f"Loading pretrained model from {self.pretrained_checkpoint}")
            if not Path(self.pretrained_checkpoint).is_absolute():
                self.pretrained_checkpoint = get_path_to_assets() / self.pretrained_checkpoint
            checkpoint = torch.load(self.pretrained_checkpoint)
            self.mel_conv.load_state_dict(checkpoint["mel_conv"])
            self.mel_proj.load_state_dict(checkpoint["mel_proj"])

    @property
    def input_dim(self):
        # d = cfg.conformer_embed_dim
        return self.cfg.input_dim

    @property
    def output_dim(self):
        return self.cfg.input_dim ## same as input_dim for now

    @property
    def speaker_embedding_dim(self):
        if self.uses_speaker_embedding:
            return self.cfg.speaker_embedding_dim
        return 0

    @property
    def input_key(self):
        return "encoded_features" 
    
    @property
    def speaker_embedding_key(self):
        return "speaker_embedding"

    @property 
    def uses_speaker_embedding(self): 
        return self.cfg.get("uses_speaker_embedding", True)

    @property
    def output_key(self):
        return "predicted_mel"

    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:

        x = batch[self.input_key]

        # if spk_emb is not None:
        if self.uses_speaker_embedding:
            spk_emb = batch[self.speaker_embedding_key]
            assert spk_emb.size(-1) == 256
            spk_x = torch.cat([spk_emb.unsqueeze(1).repeat(1,x.size(1),1), x], dim=-1)
        else:
            spk_x = x

        encoder_out_mel = self.mel_proj(self.mel_conv(spk_x.transpose(1,2)).transpose(1,2))

        B, T, D = encoder_out_mel.shape
        encoder_out_mel = encoder_out_mel.reshape(B, T, D//2, 2).transpose(-1,-2).reshape(B, T*2, D//2)

        batch[self.output_key] = encoder_out_mel
        return batch


class Lip2SpeechUnitClassifier(Mesh4DecoderBase): 
    
    def __init__(self, cfg,):
        super().__init__()
        self.cfg = cfg
        
        # d = cfg.conformer_embed_dim
        # tgt_dict = cfg.tgt_dict

        self.final_dropout = nn.Dropout(cfg.final_dropout)

        # if tgt_dict is not None:
        #     # self.proj_out = Linear(d, len(tgt_dict))
        #     self.proj_out = Lip2SpeechMLP(
        #         d, [d, d, len(tgt_dict)], cfg.final_dropout, nn.GELU
        #     )
        # elif getattr(cfg, "decoder_embed_dim", d) != d:
            # self.proj_out = nn.Linear(d, cfg.decoder_embed_dim)
        if self.input_dim != self.output_dim:
            self.proj_out = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.proj_out = None

        self.trainable = cfg.get('trainable', True)
        self.pretrained_checkpoint = cfg.get('pretrained_checkpoint', None)

        if self.pretrained_checkpoint is not None:
            self.load_pretrained()

        if not self.trainable:
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        mode = mode and self.trainable
        super().train(mode)

    def load_pretrained(self):
        if self.pretrained_checkpoint is not None:
            print(f"Loading pretrained model from {self.pretrained_checkpoint}")
            if not Path(self.pretrained_checkpoint).is_absolute():
                self.pretrained_checkpoint = get_path_to_assets() / self.pretrained_checkpoint
            checkpoint = torch.load(self.pretrained_checkpoint)
            self.proj_out.load_state_dict(checkpoint)


    @property
    def output_dim(self):
        # d = cfg.conformer_embed_dim
        return self.cfg.output_dim

    @property
    def input_dim(self):
        # d = cfg.conformer_embed_dim
        return self.cfg.input_dim

    @property
    def input_key(self):
        return "encoded_features" 
    
    @property
    def output_key(self):
        return "predicted_unit_classification_vectors"

    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        x = batch[self.input_key]
        
        tbc=False ## TODO figure out how to set this
        if tbc:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        x = self.proj_out(x)
        batch[self.output_key] = x
        return batch


import math
class Lip2SpeechMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dims,
        dropout: float = 0.1,
        nonlinearity = nn.ReLU,
        normalization = None, #nn.BatchNorm1d,  # nn.LayerNorm,
        special_bias: bool = False,
        add_bn_first: bool = False,
    ):
        super().__init__()
        projection_prev_dim = input_dim
        projection_modulelist = []
        last_dim = mlp_dims[-1]
        mlp_dims = mlp_dims[:-1]

        if add_bn_first:
            if normalization is not None:
                projection_modulelist.append(normalization(projection_prev_dim))
            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))

        for idx, mlp_dim in enumerate(mlp_dims):
            fc_layer = nn.Linear(projection_prev_dim, mlp_dim)
            nn.init.kaiming_normal_(fc_layer.weight, a=0, mode='fan_out')
            projection_modulelist.append(fc_layer)
            projection_modulelist.append(nonlinearity())

            if normalization is not None:
                projection_modulelist.append(normalization(mlp_dim))

            if dropout != 0:
                projection_modulelist.append(nn.Dropout(dropout))
            projection_prev_dim = mlp_dim

        self.projection = nn.Sequential(*projection_modulelist)
        self.last_layer = nn.Linear(projection_prev_dim, last_dim)
        nn.init.kaiming_normal_(self.last_layer.weight, a=0, mode='fan_out')
        if special_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.last_layer.bias, bias_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input_arguments:
            @x: torch.FloatTensor
        """
        x = self.projection(x)
        x = self.last_layer(x)
        return x
