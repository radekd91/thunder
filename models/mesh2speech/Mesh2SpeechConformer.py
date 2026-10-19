import torch
import os, sys 
from pathlib import Path
from .Mesh2SpeechEncoder import Mesh2SpeechEncoderBase
from typing import Dict, Any
from inferno.utils.other import get_path_to_assets

# sys.path.insert(0, str(Path(__file__).parent / "external"))
# try:
    # from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from .external.espnet.nets.pytorch_backend.transformer.encoder import Encoder
# except ImportError:
#     pass 
# sys.path.pop(0)


class Lip2SpeechConformer(Mesh2SpeechEncoderBase):

    def __init__(self, cfg, ):
        super().__init__()
        self.cfg = cfg

        # d = cfg.encoder_params.conformer_embed_dim
        # if 512 != d:
        #     self.proj_in = torch.nn.Linear(cfg.w2v_args.model.encoder_embed_dim, d)
        # else:
        #     self.proj_in = None

        self.encoder = Encoder(
            idim=-1,
            **cfg.encoder_args,
            # attention_dim=cfg.conformer_embed_dim, # adim
            # attention_heads=cfg.conformer_attention_heads, # aheads
            # linear_units=cfg.conformer_ffn_embed_dim, # eunits
            # num_blocks=cfg.conformer_layers, #elayers
            # dropout_rate=cfg.conformer_dropout, # dropout_rate
            # positional_dropout_rate=cfg.conformer_dropout, # dropout_rate
            # attention_dropout_rate=cfg.conformer_attention_dropout, # transformer_attn_dropout_rate
            # input_layer="conv3d", # transformer_input_layer
            # normalize_before=cfg.conformer_layer_norm_first,
            # macaron_style=1, # macaron_style
            # encoder_attn_layer_type="rel_mha", # transformer_encoder_attn_layer_type
            # use_cnn_module=1, # use_cnn_module
            # zero_triu=False, # zero_triu
            # cnn_module_kernel=31, # cnn_module_kernel
            # relu_type="swish", # relu_type
            # a_upsample_ratio=1, # a_upsample_ratio,
        )

        ## this is a hack to detele the "frontend" from the external code's model 
        self.encoder.frontend = None
        # d = cfg.conformer_embed_dim

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
            if not Path(self.pretrained_checkpoint).is_absolute():
                self.pretrained_checkpoint = get_path_to_assets() / self.pretrained_checkpoint
            print(f"Loading pretrained model from {self.pretrained_checkpoint}")
            checkpoint = torch.load(self.pretrained_checkpoint)
            self.encoder.load_state_dict(checkpoint, strict=True)
    

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]


    @property
    def input_key(self):
        return "input_features" 
    
    @property
    def speaker_embedding_key(self):
        return "speaker_embedding"

    @property
    def output_key(self):
        return "encoded_features"

    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:

        ## 
        # tbc=True ## TODO figure out how to set this

        x = batch[self.input_key]
        padding_mask = batch["padding_mask"]
        # spk_emb = batch["speaker_embedding"]

        # x = self.encoder.frontend(x.squeeze(1))

        x = x.repeat_interleave(2, dim=1)
        padding_mask = padding_mask.repeat_interleave(2, dim=1)

        # if self.proj_in:
        #     x = self.proj_in(x)

        x, masks = self.encoder.forward_after_frontend(
            x,
            masks = padding_mask.unsqueeze(-2), ## the mask going in needs to be True on valid positions
        )

        # padding_mask = masks.squeeze(-2)

        # if spk_emb is not None:
        #     assert spk_emb.size(-1) == 256
        #     spk_x = torch.cat([spk_emb.unsqueeze(1).repeat(1,x.size(1),1), x], dim=-1)
        # else:
        #     spk_x = x

        # encoder_out_mel = self.mel_proj(self.mel_conv(spk_x.transpose(1,2)).transpose(1,2))

        # B, T, D = encoder_out_mel.shape
        # encoder_out_mel = encoder_out_mel.reshape(B, T, D//2, 2).transpose(-1,-2).reshape(B, T*2, D//2)

        # if tbc:
        #     # B x T x C -> T x B x C
        #     x = x.transpose(0, 1)

        # x = self.final_dropout(x)

        # if self.proj_out:
        #     x = self.proj_out(x)

        batch[self.output_key] = x

        # return {
        #     "encoder_out": x,  # T x B x C
        #     "encoder_padding_mask": padding_mask,  # B x T
        #     "padding_mask": padding_mask,
        #     "encoder_out_mel": encoder_out_mel,
        # }
        return batch

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out[
                "encoder_out"
            ].index_select(1, new_order)
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        if encoder_out["padding_mask"] is not None:
            encoder_out["padding_mask"] = encoder_out[
                "padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict
    


    