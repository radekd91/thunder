import torch
import torch.nn as nn
from torch import  nn
from ...models.architectures.tools.embeddings import (TimestepEmbedding,
                                                       Timesteps)
from ...models.operator import PositionalEncoding
from ...models.operator.cross_attention import (SkipTransformerEncoder,
                                                 TransformerDecoder,
                                                 TransformerDecoderLayer,
                                                 TransformerEncoder,
                                                 TransformerEncoderLayer, 
                                                 ModulatedTransformerDecoderLayer, 
                                                 ModulatedTransformerDecoder
                                                 )
from ...models.operator.position_encoding import build_position_encoding
from ...utils.temos_utils import lengths_to_mask
from typing import List

from inferno.models.temporal.TransformerMasking import (init_alibi_biased_mask, init_alibi_biased_mask_future, 
                                                        init_faceformer_biased_mask, init_faceformer_biased_mask_future)

from .mld_denoiser_v2 import generate_cross_attention_mask_torch
from typing import List
import omegaconf


class TimeStepModulator(nn.Module): 

    def __init__(self, dim, 
                #  modulator
                 ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, 2*dim)
        # self.modulator = modulator

    def _forward_alpha_beta(self, time_embedding):
        ## given timestep embedding, compute beta and gamma
        x = self.activation(time_embedding)
        x = self.linear(x)
        beta, gamma = x.chunk(2, dim=-1)
        return beta, gamma
    

class FaceTalkModulator(TimeStepModulator): 

    def __init__(self, dim, 
                #  modulator
                 ) -> None:
        super().__init__(dim, 
                        #  modulator
                         )
        self.activation = nn.Mish()
        self.film = FiLM()

    def forward(self, x, time_embedding):
        ## time_embedding: [B, F]
        ## x: [T, B, F]
        beta, gamma = self._forward_alpha_beta(time_embedding)
        x = self.film(x, gamma, beta)
        return x


class FiLM(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, gamma, beta):
        T, B, F = x.size(0), x.size(1), x.size(2)
        beta = beta.view(1, B, F)
        gamma = gamma.view(1, B, F)
        x = gamma * x + beta
        return x
    

class MldDenoiserTimestepped(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int = 263,
                 condition: List[str] = "audio",
                 latent_dim: list = 256,
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_dec",
                 modulator_shared: bool = False,
                 modulator_type: str = "film",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 768,
                 audio_encoded_dim: int = 768,
                 nclasses: int = 10,
                 cross_attention_mask_type = None,
                 cross_attention_frame_ratio = 1,
                 mask_invalid: bool = False,
                 input_projector: bool = False,
                 output_projector: bool = False,
                 input_dim: int = None,
                 output_dim: int = None,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim or self.latent_dim
        self.output_dim = output_dim or input_dim or self.latent_dim
        self.text_encoded_dim = text_encoded_dim
        self.audio_encoded_dim = audio_encoded_dim
        if not isinstance(condition, (List, list, tuple, omegaconf.ListConfig)):
            condition = [condition]
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = ablation.SKIP_CONNECT
        self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        self.modulator_shared = modulator_shared
        self.modulator_type = modulator_type
        self.pe_type = ablation.DIFF_PE_TYPE
        self.cross_attention_mask_type = cross_attention_mask_type
        self.num_heads = num_heads
        # self.mask_invalid = mask_invalid

        if self.diffusion_only:
            # assert self.arch == "trans_enc", "only implement encoder for diffusion-only"
            self.pose_embd = nn.Linear(nfeats, self.latent_dim)
            self.pose_proj = nn.Linear(self.latent_dim, nfeats)

        # emb proj
        for cond in self.condition:
            if cond in ["text", "text_uncond"]:
                raise NotImplementedError("text condition not implemented")
                # text condition
                # project time from text_encoded_dim to latent_dim
                self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                        freq_shift)
                self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                        self.latent_dim)
                # project time+text to latent_dim
                if text_encoded_dim != self.latent_dim:
                    # todo 10.24 debug why relu
                    self.emb_proj_text = nn.Sequential(
                        nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))
            elif cond in ["audio", "audio_uncond"]: 
                self.time_proj = Timesteps(audio_encoded_dim, flip_sin_to_cos,
                                                freq_shift) 
                self.time_embedding = TimestepEmbedding(audio_encoded_dim, 
                                                        self.latent_dim)
                if audio_encoded_dim != self.latent_dim:
                    self.emb_proj_audio = nn.Sequential(
                        nn.ReLU(), nn.Linear(audio_encoded_dim, self.latent_dim))
                    
                
            # elif cond in ['action']:
            #     self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos,
            #                             freq_shift)
            #     self.time_embedding = TimestepEmbedding(self.latent_dim,
            #                                             self.latent_dim)
            #     self.emb_proj = EmbedAction(nclasses,
            #                                 self.latent_dim,
            #                                 guidance_scale=guidance_scale,
            #                                 guidance_uncodp=guidance_uncondp)
            else:
                raise TypeError(f"condition type {self.condition} not supported")

        if self.pe_type == "actor":
            self.query_pos = PositionalEncoding(self.latent_dim, dropout)
            self.self_attention_mask = None
            self.mem_pos = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
            self.self_attention_mask = None
            self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding=position_embedding)
        elif self.pe_type == "alibi": 
            self.query_pos = None
            self.mem_pos = None
            max_seq_len=2000
            self_attention_mask = init_alibi_biased_mask_future(num_heads=num_heads, max_seq_len=max_seq_len)
            # prepend a row and column with zeros 
            # self_attention_mask = torch.cat([torch.zeros((self_attention_mask.shape[0], max_seq_len, 1), 
            #                                              device=self_attention_mask.device), self_attention_mask], dim=1)
            # self_attention_mask_ = init_alibi_biased_mask(num_heads=num_heads, max_seq_len=max_seq_len)
            self.register_buffer('self_attention_mask', self_attention_mask)
        else:
            raise ValueError("Not Support PE type")

        if self.arch == "trans_enc":
            raise NotImplementedError("skip connection not implemented")
            if self.ablation_skip_connection:
                # use DETR transformer
                encoder_layer = TransformerEncoderLayer(
                    self.latent_dim,
                    num_heads,
                    ff_size,
                    dropout,
                    activation,
                    normalize_before,
                )
                encoder_norm = nn.LayerNorm(self.latent_dim)
                self.encoder = SkipTransformerEncoder(encoder_layer,
                                                      num_layers, encoder_norm)
            else:
                # use torch transformer
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation)
                self.encoder = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=num_layers)
        elif self.arch == "trans_dec":
            if self.modulator_type == "film":
                modulator = FaceTalkModulator(self.latent_dim)
            else: 
                raise ValueError(f"Not supported modulator type {self.modulator_type}!")
            if self.modulator_shared: 
                self.modulator = modulator
                modulator = None
            else: 
                self.modulator = None 
            decoder_layer = ModulatedTransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                modulator,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = ModulatedTransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )

            max_T = 2000 
            self.cross_attention_frame_ratio = cross_attention_frame_ratio
            self.cross_attention_mask_type = cross_attention_mask_type
            self._init_cross_attention_mask(max_T)
        else:
            raise ValueError(f"Not supported architechure{self.arch}!")
        
        self.input_proj = None
        if input_projector:
            self.input_proj = nn.Linear(self.input_dim, self.latent_dim )
        self.output_proj = None
        if output_projector:
            self.output_proj = nn.Linear(self.latent_dim, self.output_dim)

    def _init_cross_attention_mask(self, max_T, dev=None):
        if self.cross_attention_mask_type not in [None, 'none', 'None']: 
            if hasattr(self, 'cross_attention_mask'):
                dev = dev or self.cross_attention_mask.device
                del self.cross_attention_mask
            cross_attention_mask = generate_cross_attention_mask_torch(
                T=max_T, 
                q=self.cross_attention_frame_ratio, 
                mask_type=self.cross_attention_mask_type, 
                num_heads=self.num_heads).to(dev)
            self.register_buffer('cross_attention_mask', cross_attention_mask)
        else: 
            cross_attention_mask = None

    def to(self, *args, **kwargs): 
        return super().to(*args, **kwargs)

    def forward(self,
                sample,
                timestep,
                conditions,
                lengths=None,
                attention_mask=None,
                **kwargs):
        B = sample.shape[0]
        T = sample.shape[1]
        
        if self.input_proj is not None:
            sample = self.input_proj(sample)

        # 0.  dimension matching
        # B, T, F to T, B, F
        sample = sample.permute(1, 0, 2)

        # 0. check lengths for no vae (diffusion only)
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.device)

        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML

        timesteps = timestep.expand(B).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb) #.unsqueeze(0)

        final_condition = []

        if self.cross_attention_mask_type not in [None, 'none', 'None']:
            cross_attention_masks = []
        else: 
            cross_attention_masks = None

        for cond in self.condition:
        # for condition_name, encoder_hidden_states in conditions.keys():

            # condition_type = 

            # 2. condition + time embedding
            if cond in ["text", "text_uncond"]:
                # encoder_hidden_states = conditions['text_condition']
                encoder_hidden_states = conditions['text']

                # B, T, F to T, B, F
                encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
                text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
                # textembedding projection
                if self.text_encoded_dim != self.latent_dim:
                    # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                    modality_emb_latent = self.emb_proj_text(text_emb)
                else:
                    modality_emb_latent = text_emb
            elif cond in ["audio"]: 
                # encoder_hidden_states = conditions['audio_condition']
                encoder_hidden_states = conditions['audio']
                encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
                audio_emb = encoder_hidden_states  # [num_frames, bs, latent_dim]
                # audio embedding projection
                if self.audio_encoded_dim != self.latent_dim:
                    # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                    modality_emb_latent = self.emb_proj_audio(audio_emb)
                else:
                    modality_emb_latent = audio_emb

                if self.arch == "trans_dec":
                    Tc = audio_emb.shape[0]
                    if cross_attention_masks is not None:
                        # if cross attention mask too small, reinitialize a larger one
                        if self.cross_attention_mask.shape[-2] < T or self.cross_attention_mask.shape[-1] < Tc:
                            print("[WARNING] cross_attention_mask too small, reinitializing. This should be avoided in training.")
                            self._init_cross_attention_mask(Tc+1, self.cross_attention_mask.device)
                        cross_attention_mask = self.cross_attention_mask[..., :T, :Tc]
                        assert cross_attention_mask.shape[-2:] == (T, Tc)
                        cross_attention_masks += [cross_attention_mask]
            # elif self.condition in ['action']:
            #     action_emb = self.emb_proj(encoder_hidden_states)
            #     if self.abl_plus:
            #         emb_latent = action_emb + time_emb
            #     else:
            #         emb_latent = torch.cat((time_emb, action_emb), 0)
            else:
                raise TypeError(f"condition type {self.condition} not supported")
            
            final_condition += [modality_emb_latent]

        ## combine all conditions along the time dimension
        emb_latent = torch.cat(final_condition, 0)


        # 4. transformer
        if self.arch == "trans_enc":
            if self.diffusion_only:
                sample = self.pose_embd(sample)
                xseq = torch.cat((emb_latent, sample), axis=0)
            else:
                xseq = torch.cat((sample, emb_latent), axis=0)

            # if self.ablation_skip_connection:
            #     xseq = self.query_pos(xseq)
            #     tokens = self.encoder(xseq)
            # else:
            #     # adding the timestep embed
            #     # [seqlen+1, bs, d]
            #     # todo change to query_pos_decoder
            if self.query_pos is not None:
                xseq = self.query_pos(xseq)
            

            tokens = self.encoder(xseq, mask=attention_mask)

            if self.diffusion_only:
                sample = tokens[emb_latent.shape[0]:]
                sample = self.pose_proj(sample)

                # zero for padded area
                sample[~mask.T] = 0
            else:
                sample = tokens[:sample.shape[0]]

        elif self.arch == "trans_dec":
            if cross_attention_masks is not None:

                if self.cross_attention_mask_type not in [None, 'none', 'None']:
                    if self.cross_attention_mask.ndim == 3: 
                        cross_attn_mask = torch.cat(cross_attention_masks, dim=2)
                        cross_attn_mask = cross_attn_mask.repeat(B, 1, 1)
                    else:
                        cross_attn_mask = torch.cat(cross_attention_masks, dim=1)
            else: 
                cross_attn_mask = None

            if self.diffusion_only:
                sample = self.pose_embd(sample)

            # tgt    - [1 or 5 or 10, bs, latent_dim]
            # memory - [token_num, bs, latent_dim]
            if self.query_pos is not None:
                sample = self.query_pos(sample)
            if self.mem_pos is not None:
                emb_latent = self.mem_pos(emb_latent)
            
            self_attention_mask = None
            if self.self_attention_mask is not None: 
                self_attention_mask = self.self_attention_mask[..., :T, :T]
                # if self.mask_invalid: 
                #     self_attention_mask[..., ~mask.T, :] = -torch.inf
                #     self_attention_mask[..., :, ~mask.T] = -torch.inf
                if self_attention_mask.ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                    self_attention_mask = self_attention_mask.repeat(B, 1, 1)

            if self.modulator_shared:
                modulator = self.modulator 
            else:
                modulator = None

            sample = self.decoder(tgt=sample, 
                                  tgt_mask=self_attention_mask,
                                  memory=emb_latent, 
                                  memory_mask=cross_attn_mask, 
                                  modulator=modulator, 
                                  time_embedding=time_emb,
                                  ).squeeze(0)

            if self.diffusion_only:
                sample = self.pose_proj(sample)
                # zero for padded area
                sample[~mask.T] = 0
        else:
            raise TypeError(f"{self.arch} is not supoorted")

        # 5. T, B, F to B, T, F
        sample = sample.permute(1, 0, 2)

        if self.output_proj is not None:
            sample = self.output_proj(sample)

        return sample

   
class EmbedAction(nn.Module):

    def __init__(self,
                 num_actions,
                 latent_dim,
                 guidance_scale=7.5,
                 guidance_uncodp=0.1,
                 force_mask=False):
        super().__init__()
        self.nclasses = num_actions
        self.guidance_scale = guidance_scale
        self.action_embedding = nn.Parameter(
            torch.randn(num_actions, latent_dim))

        self.guidance_uncodp = guidance_uncodp
        self.force_mask = force_mask
        self._reset_parameters()

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        if not self.training and self.guidance_scale > 1.0:
            uncond, output = output.chunk(2)
            uncond_out = self.mask_cond(uncond, force=True)
            out = self.mask_cond(output)
            output = torch.cat((uncond_out, out))

        output = self.mask_cond(output)

        return output.unsqueeze(0)

    def mask_cond(self, output, force=False):
        bs, d = output.shape
        # classifer guidence
        if self.force_mask or force:
            return torch.zeros_like(output)
        elif self.training and self.guidance_uncodp > 0.:
            mask = torch.bernoulli(
                torch.ones(bs, device=output.device) *
                self.guidance_uncodp).view(
                    bs, 1)  # 1-> use null_cond, 0-> use real cond
            return output * (1. - mask)
        else:
            return output

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
