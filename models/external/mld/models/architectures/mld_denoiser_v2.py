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
                                                 TransformerEncoderLayer)
from ...models.operator.position_encoding import build_position_encoding
from ...utils.temos_utils import lengths_to_mask
from typing import List
import omegaconf

from inferno.models.temporal.TransformerMasking import (init_alibi_biased_mask, init_alibi_biased_mask_future, 
                                                        init_faceformer_biased_mask, init_faceformer_biased_mask_future)

def generate_cross_attention_mask_torch(T, q, mask_type='binary', num_heads=None):
    """
    Generate a mask for cross-attention where the key sequence has length T
    and the query/memory sequence has length T/q, for use with PyTorch transformers.
    Supports binary masks and ALiBi (Attention with Linear Biases) masks.
    
    Parameters:
    - T: int, the length of the key sequence.
    - q: int, the factor by which the query/memory sequence is shorter than the key sequence.
    - mask_type: str, the type of mask to generate ('binary' or 'alibi').
    
    Returns:
    - mask: torch.Tensor, a mask of shape (1, T/q, T) for binary mask, or a linear bias
            mask for ALiBi with the same shape. The leading dimension of 1 is added to match
            the expected input shape for PyTorch's attention mechanisms.
    """
    if mask_type == 'binary':
        # Create a tensor filled with zeros for binary mask
        # mask = torch.zeros(T // q, T)
        
        # for i in range(T // q):
        #     start_idx = i * q
        #     end_idx = start_idx + q
        #     mask[i, start_idx:end_idx] = 1

        mask = torch.ones(T // q, T) * (-torch.inf)
        for i in range(T // q):
            start_idx = i * q
            end_idx = start_idx + q
            mask[i, start_idx:end_idx] = 0.
        
        # Add a dimension to match the expected input shape for PyTorch attention layers
        # mask = mask.T
        # mask = mask.unsqueeze(0)
    
    elif mask_type == 'alibi':
        # Create a linear bias mask for ALiBi
        assert num_heads is not None, "Number of heads must be specified for ALiBi mask"
        mask = init_faceformer_biased_mask_future(num_heads=num_heads, max_seq_len=T,period=q)
        mask = mask[..., ::q, :T]
        # mask = torch.zeros(T // q, T)
        
        # for i in range(T // q):
        #     for j in range(T):
        #         # Calculate the linear bias, which increases with the distance from the diagonal
        #         linear_bias = j - (i * q + q // 2)
        #         mask[i, j] = linear_bias
        
        # # Scale the biases to be in a suitable range for attention logits
        # # mask = mask.T
        # mask = mask.unsqueeze(0) * -1e4  # Scale factor may need adjustment based on model and dataset
    
    else:
        raise ValueError("Unsupported mask type. Please choose 'binary' or 'alibi'.")

    return mask




class MldDenoiser(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int = 263,
                 condition: List[str] = "audio",
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 768,
                 audio_encoded_dim: int = 768,
                 clip_encoded_dim: int = None,
                 shape_encoded_dim: int = None,
                 subject_id_dim: int = None,
                 emotion_id_dim: int = None,
                 intensity_id_dim: int = None,
                 style_feature_dim: int = None,
                 nclasses: int = 10, ## unused remnant from ml
                 cross_attention_mask_type = None,
                 cross_attention_frame_ratio = 1,
                 text_cross_attention_mask_type = None,
                 clip_cross_attention_mask_type = None,
                 mask_invalid: bool = False, ## unused remnant from mld
                 input_projector: bool = False,
                 output_projector: bool = False,
                 input_dim: int = None,
                 output_dim: int = None,
                 with_condition_relu: bool = True,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        self.input_dim = input_dim or self.latent_dim
        self.output_dim = output_dim or input_dim or self.latent_dim
        self.text_encoded_dim = text_encoded_dim
        self.audio_encoded_dim = audio_encoded_dim
        self.clip_encoded_dim = clip_encoded_dim
        self.shape_encoded_dim = shape_encoded_dim
        if not isinstance(condition, (List, list, tuple, omegaconf.ListConfig)):
            condition = [condition]
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = ablation.SKIP_CONNECT
        self.diffusion_only = ablation.VAE_TYPE == "no"
        self.arch = arch
        self.pe_type = ablation.DIFF_PE_TYPE
        self.cross_attention_mask_type = cross_attention_mask_type
        self.text_cross_attention_mask_type = text_cross_attention_mask_type
        self.clip_cross_attention_mask_type = clip_cross_attention_mask_type
        self.num_heads = num_heads
        self.with_condition_relu = with_condition_relu
        # self.mask_invalid = mask_invalid

        if self.diffusion_only:
            # assert self.arch == "trans_enc", "only implement encoder for diffusion-only"
            self.pose_embd = nn.Linear(nfeats, self.latent_dim)
            self.pose_proj = nn.Linear(self.latent_dim, nfeats)

        # emb proj
        for cond in self.condition:
            if cond in ["text", "text_uncond"]:
                # text condition
                # project time from text_encoded_dim to latent_dim
                self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                        freq_shift)
                self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                        self.latent_dim)
                # project time+text to latent_dim
                if text_encoded_dim != self.latent_dim:
                    text_projection = [] 
                    if self.with_condition_relu:
                        text_projection += [nn.ReLU()]
                    text_projection += [nn.Linear(text_encoded_dim, self.latent_dim)]
                    self.emb_proj_text = nn.Sequential(*text_projection)
            elif cond in ["audio", "audio_uncond"]: 
                self.time_proj = Timesteps(audio_encoded_dim, flip_sin_to_cos,
                                                freq_shift) 
                self.time_embedding = TimestepEmbedding(audio_encoded_dim, 
                                                        self.latent_dim)
                if audio_encoded_dim != self.latent_dim:
                    audio_projection = []
                    if self.with_condition_relu:
                        audio_projection += [nn.ReLU()]
                    audio_projection += [nn.Linear(audio_encoded_dim, self.latent_dim)]
                    self.emb_proj_audio = nn.Sequential(*audio_projection)
                    
            elif cond in ['clip']:
                if clip_encoded_dim != self.latent_dim:
                    clip_projection = []
                    if self.with_condition_relu:
                        clip_projection += [nn.ReLU()]
                    clip_projection += [nn.Linear(clip_encoded_dim, self.latent_dim)]
                    self.emb_proj_clip = nn.Sequential(*clip_projection)

            elif cond in ['shape']:
                if shape_encoded_dim != self.latent_dim:
                    shape_projection = []
                    if self.with_condition_relu:
                        shape_projection += [nn.ReLU()]
                    shape_projection += [nn.Linear(shape_encoded_dim, self.latent_dim)]
                    self.emb_proj_shape = nn.Sequential(*shape_projection)
                
            elif cond in ['subject']:
                pass ## no projection needed, the subject feature should already be in the latent_dim
            #     subject_id_projection = []
            #     subject_id_projection += [nn.Linear(subject_id_dim, self.latent_dim)]
            #     self.emb_proj_subject_id = nn.Sequential(*subject_id_projection)

            elif cond in ['emotion']:
                pass ## no projection needed, the emotion feature should already be in the latent_dim
            #     emotion_id_projection = []
            #     emotion_id_projection += [nn.Linear(emotion_id_dim, self.latent_dim)]
            #     self.emb_proj_emotion_id = nn.Sequential(*emotion_id_projection)

            elif cond in ['intensity']:
                pass ## no projection needed, the intensity feature should already be in the latent_dim
            #     intensity_id_projection = []
            #     intensity_id_projection += [nn.Linear(intensity_id_dim, self.latent_dim)]
            #     self.emb_proj_intensity_id = nn.Sequential(*intensity_id_projection)

            # elif cond in ['action']:
            #     self.time_proj = Timesteps(self.latent_dim, flip_sin_to_cos,
            #                             freq_shift)
            #     self.time_embedding = TimestepEmbedding(self.latent_dim,
            #                                             self.latent_dim)
            #     self.emb_proj = EmbedAction(nclasses,
            #                                 self.latent_dim,
            #                                 guidance_scale=guidance_scale,
            #                                 guidance_uncodp=guidance_uncondp)
            elif cond in ['style']:
                pass #
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
            raise NotImplementedError("Transformer encoder architecture is not supported")
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
            decoder_layer = TransformerDecoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = TransformerDecoder(
                decoder_layer,
                num_layers,
                decoder_norm,
                return_intermediate=return_intermediate_dec,
            )

            max_T = 2000 
            self.cross_attention_frame_ratio = cross_attention_frame_ratio
            self.cross_attention_mask_type = cross_attention_mask_type
            self._init_cross_attention_mask(max_T)
            # if cross_attention_mask_type not in [None, 'none', 'None']:
            #     cross_attention_mask = generate_cross_attention_mask_torch(
            #         T=max_T, q=q, mask_type='binary')
            #     self.register_buffer('cross_attention_mask', cross_attention_mask)
            # else: 
            #     cross_attention_mask = None
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
            audio_mask_dim = cross_attention_mask.ndim
        else: 
            cross_attention_mask = None
            audio_mask_dim = 0

        if self.text_cross_attention_mask_type not in [None, 'none', 'None']: 
            if hasattr(self, 'text_cross_attention_mask'):
                dev = dev or self.text_cross_attention_mask.device
                del self.text_cross_attention_mask
            text_cross_attention_mask = generate_cross_attention_mask_torch(
                T=max_T, 
                q=self.cross_attention_frame_ratio, 
                mask_type=self.text_cross_attention_mask_type, 
                num_heads=self.num_heads).to(dev)
            text_mask_dim = text_cross_attention_mask.ndim
        else: 
            text_cross_attention_mask = None
            text_mask_dim = 0

        if self.clip_cross_attention_mask_type not in [None, 'none', 'None']:
            if hasattr(self, 'clip_cross_attention_mask'):
                dev = dev or self.clip_cross_attention_mask.device
                del self.clip_cross_attention_mask
            clip_cross_attention_mask = generate_cross_attention_mask_torch(
                T=max_T, 
                q=self.cross_attention_frame_ratio, 
                mask_type=self.clip_cross_attention_mask_type, 
                num_heads=self.num_heads).to(dev)
            clip_mask_dim = clip_cross_attention_mask.ndim
        else:
            clip_cross_attention_mask = None
            clip_mask_dim = 0

        max_mask_dim = max(audio_mask_dim, text_mask_dim, clip_mask_dim)
        if cross_attention_mask is not None and cross_attention_mask.ndim < max_mask_dim:
            cross_attention_mask = cross_attention_mask.unsqueeze(0).repeat(self.num_heads, 1, 1)
        if text_cross_attention_mask is not None and text_cross_attention_mask.ndim < max_mask_dim:
            text_cross_attention_mask = text_cross_attention_mask.unsqueeze(0).repeat(self.num_heads, 1, 1)
        if clip_cross_attention_mask is not None and clip_cross_attention_mask.ndim < max_mask_dim:
            clip_cross_attention_mask = clip_cross_attention_mask.unsqueeze(0).repeat(self.num_heads, 1, 1)
        # if audio_mask_dim is not None and text_mask_dim is not None:
        #     if audio_mask_dim == 3 and text_mask_dim == 2:
        #         text_cross_attention_mask = text_cross_attention_mask.unsqueeze(0).repeat(self.num_heads, 1, 1)
        #     elif audio_mask_dim == 2 and text_mask_dim == 3:
        #         cross_attention_mask = cross_attention_mask.unsqueeze(0).repeat(self.num_heads, 1, 1)
        #     elif audio_mask_dim == text_mask_dim:
        #         pass 
        #     else: 
        #         raise ValueError("cross attention mask dimension mismatch")

        if cross_attention_mask is not None: 
            self.register_buffer('cross_attention_mask', cross_attention_mask)
        if text_cross_attention_mask is not None:
            self.register_buffer('text_cross_attention_mask', text_cross_attention_mask)
        if clip_cross_attention_mask is not None:
            self.register_buffer('clip_cross_attention_mask', clip_cross_attention_mask)

    def to(self, *args, **kwargs): 
        return super().to(*args, **kwargs)

    @property
    def cross_attention_mask_ndim(self):
        audio_mask_dim = 2
        if hasattr(self, 'cross_attention_mask'):
            audio_mask_dim = self.cross_attention_mask.ndim
        text_mask_dim = 2
        if hasattr(self, 'text_cross_attention_mask'):
            text_mask_dim = self.text_cross_attention_mask.ndim
        clip_mask_dim = 2
        if hasattr(self, 'clip_cross_attention_mask'):
            clip_mask_dim = self.clip_cross_attention_mask.ndim
        return max(audio_mask_dim, text_mask_dim, clip_mask_dim)

    @property
    def uses_cross_attention_mask(self):
        return self.cross_attention_mask is not None or self.text_cross_attention_mask is not None

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
        # sample [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
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
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        final_condition = []

        if self.uses_cross_attention_mask:
            cross_attention_masks = []
        else: 
            cross_attention_masks = None

        for cond in self.condition:
        # for condition_name, encoder_hidden_states in conditions.keys():

            # 2. condition + time embedding
            if cond in ["text", "text_uncond"]:
                # encoder_hidden_states = conditions['text_condition']
                encoder_hidden_states = conditions['text']

                # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
                encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
                text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
                # textembedding projection
                if self.text_encoded_dim != self.latent_dim:
                    # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                    modality_emb_latent = self.emb_proj_text(text_emb)
                else:
                    modality_emb_latent = text_emb
                # if self.abl_plus:
                #     emb_latent = time_emb + text_emb_latent
                # else:
                #     emb_latent = torch.cat((time_emb, text_emb_latent), 0)
                    
                if self.arch == "trans_dec":
                    Tc = text_emb.shape[0]
                    if cross_attention_masks is not None:
                        # if cross attention mask too small, reinitialize a larger one
                        if self.text_cross_attention_mask.shape[-2] < T or self.text_cross_attention_mask.shape[-1] < Tc:
                            print("[WARNING] cross_attention_mask too small, reinitializing. This should be avoided in training.")
                            self._init_cross_attention_mask(Tc+1, self.text_cross_attention_mask.device)
                        cross_attention_mask = self.text_cross_attention_mask[..., :T, :Tc]
                        assert cross_attention_mask.shape[-2:] == (T, Tc)
                        cross_attention_masks += [cross_attention_mask]

            elif cond in ["audio"]: 
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

                # if self.abl_plus:
                #     emb_latent = audio_emb + audio_emb_latent
                # else:
                #     emb_latent = torch.cat((time_emb, audio_emb_latent), 0)
            # elif self.condition in ['action']:
            #     action_emb = self.emb_proj(encoder_hidden_states)
            #     if self.abl_plus:
            #         emb_latent = action_emb + time_emb
            #     else:
            #         emb_latent = torch.cat((time_emb, action_emb), 0)
            elif cond in ['clip']:
                encoder_hidden_states = conditions['clip']
                T_clip = encoder_hidden_states.shape[1]
                encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
                clip_emb = encoder_hidden_states
                if self.clip_encoded_dim != self.latent_dim:
                    modality_emb_latent = self.emb_proj_clip(clip_emb)
                else: 
                    modality_emb_latent = clip_emb
                if self.arch == "trans_dec":
                    if cross_attention_masks is not None:
                        if T_clip == 1: ## only one global CLIP feature, no need for elaborate cross attention, each frame should pay attention
                            ## in this case the cross attention a zero vector mask (zero equates to maximum attention after softmax)
                            cross_attention_mask = torch.zeros((T, 1), device=audio_emb.device, dtype=audio_emb.dtype)
                            cross_attention_masks += [cross_attention_mask]
                        else: ## CLIP features are per-frame and need to be attended to by each timestep according to the predefined cross attention mask
                            # if cross attention mask too small, reinitialize a larger one
                            if self.clip_cross_attention_mask.shape[-2] < T or self.clip_cross_attention_mask.shape[-1] < T_clip:
                                print("[WARNING] cross_attention_mask too small, reinitializing. This should be avoided in training.")
                                self._init_cross_attention_mask(T_clip+1, self.clip_cross_attention_mask.device)
                            cross_attention_mask = self.clip_cross_attention_mask[..., :T, :T_clip]
                            assert cross_attention_mask.shape[-2:] == (T, T_clip)
                            cross_attention_masks += [cross_attention_mask]
            elif cond in ['shape']:
                encoder_hidden_states = conditions['shape'].unsqueeze(1)  # [batch_size, 1, shape_encoded_dim]
                encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
                shape_emb = encoder_hidden_states
                if self.shape_encoded_dim != self.latent_dim:
                    modality_emb_latent = self.emb_proj_shape(shape_emb)
                else: 
                    modality_emb_latent = shape_emb
                if self.arch == "trans_dec":
                    T_shape = shape_emb.shape[0]
                    if cross_attention_masks is not None:
                        if T_shape == 1: ## only one global shape feature, no need for elaborate cross attention, each frame should pay attention
                            cross_attention_mask = torch.zeros((T, 1), device=audio_emb.device, dtype=audio_emb.dtype)
                            cross_attention_masks += [cross_attention_mask]
                        else: ## shape features are per-frame and need to be attended to by each timestep according to the predefined cross attention mask
                            if self.cross_attention_mask.shape[-2] < T or self.cross_attention_mask.shape[-1] < T_shape:
                                print("[WARNING] cross_attention_mask too small, reinitializing. This should be avoided in training.")
                                self._init_cross_attention_mask(T_shape+1, self.cross_attention_mask.device)    
                            cross_attention_mask = self.cross_attention_mask[..., :T, :T_shape]
                            assert cross_attention_mask.shape[-2:] == (T, T_shape)
                            cross_attention_masks += [cross_attention_mask]


            elif cond in ['subject']:
                encoder_hidden_states = conditions['subject']
                encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
                subject_id_emb = encoder_hidden_states
                # modality_emb_latent = self.emb_proj_subject_id(subject_id_emb)
                modality_emb_latent = subject_id_emb

                if self.arch == "trans_dec": 
                    T_subject_id = subject_id_emb.shape[0]
                    if cross_attention_masks is not None:
                        if T_subject_id == 1: ## only one global subject_id feature, no need for elaborate cross attention, each frame should pay attention
                            cross_attention_mask = torch.zeros((T, 1), device=audio_emb.device, dtype=audio_emb.dtype)
                            cross_attention_masks += [cross_attention_mask]
                        else: ## subject_id features are per-frame and need to be attended to by each timestep according to the predefined cross attention mask
                            if self.cross_attention_mask.shape[-2] < T or self.cross_attention_mask.shape[-1] < T_subject_id:
                                print("[WARNING] cross_attention_mask too small, reinitializing. This should be avoided in training.")
                                self._init_cross_attention_mask(T_subject_id+1, self.cross_attention_mask.device)
                            cross_attention_mask = self.cross_attention_mask[..., :T, :T_subject_id]
                            assert cross_attention_mask.shape[-2:] == (T, T_subject_id)
                            cross_attention_masks += [cross_attention_mask]

            elif cond in ['emotion']:
                encoder_hidden_states = conditions['emotion']
                encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
                emotion_id_emb = encoder_hidden_states
                # modality_emb_latent = self.emb_proj_emotion_id(emotion_id_emb)
                modality_emb_latent = emotion_id_emb

                if self.arch == "trans_dec":
                    T_emotion_id = emotion_id_emb.shape[0]
                    if cross_attention_masks is not None:
                        if T_emotion_id == 1: ## only one global emotion_id feature, no need for elaborate cross attention, each frame should pay attention
                            cross_attention_mask = torch.zeros((T, 1), device=audio_emb.device, dtype=audio_emb.dtype)
                            cross_attention_masks += [cross_attention_mask]
                        else: ## emotion_id features are per-frame and need to be attended to by each timestep according to the predefined cross attention mask
                            if self.cross_attention_mask.shape[-2] < T or self.cross_attention_mask.shape[-1] < T_emotion_id:
                                print("[WARNING] cross_attention_mask too small, reinitializing. This should be avoided in training.")
                                self._init_cross_attention_mask(T_emotion_id+1, self.cross_attention_mask.device)
                            cross_attention_mask = self.cross_attention_mask[..., :T, :T_emotion_id]
                            assert cross_attention_mask.shape[-2:] == (T, T_emotion_id)
                            cross_attention_masks += [cross_attention_mask]
            
            elif cond in ['intensity']:
                encoder_hidden_states = conditions['intensity']
                encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
                intensity_id_emb = encoder_hidden_states
                # modality_emb_latent = self.emb_proj_intensity_id(intensity_id_emb)
                modality_emb_latent = intensity_id_emb

                if self.arch == "trans_dec": 
                    T_intensity_id = intensity_id_emb.shape[0]
                    if cross_attention_masks is not None:
                        if T_intensity_id == 1: ## only one global intensity_id feature, no need for elaborate cross attention, each frame should pay attention
                            cross_attention_mask = torch.zeros((T, 1), device=audio_emb.device, dtype=audio_emb.dtype)
                            cross_attention_masks += [cross_attention_mask]
                        else: ## intensity_id features are per-frame and need to be attended to by each timestep according to the predefined cross attention mask
                            if self.cross_attention_mask.shape[-2] < T or self.cross_attention_mask.shape[-1] < T_intensity_id:
                                print("[WARNING] cross_attention_mask too small, reinitializing. This should be avoided in training.")
                                self._init_cross_attention_mask(T_intensity_id+1, self.cross_attention_mask.device)
                            cross_attention_mask = self.cross_attention_mask[..., :T, :T_intensity_id]
                            assert cross_attention_mask.shape[-2:] == (T, T_intensity_id)
                            cross_attention_masks += [cross_attention_mask]
            
            elif cond in ['style']: 
                encoder_hidden_states = conditions['style']
                encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
                style_feature_emb = encoder_hidden_states
                modality_emb_latent = style_feature_emb

                if self.arch == "trans_dec":
                    T_style_feature = style_feature_emb.shape[0]
                    if cross_attention_masks is not None:
                        if T_style_feature == 1:
                            cross_attention_mask = torch.zeros((T, 1), device=audio_emb.device, dtype=audio_emb.dtype)
                            cross_attention_masks += [cross_attention_mask]
                        else: 
                            if self.cross_attention_mask.shape[-2] < T or self.cross_attention_mask.shape[-1] < T_style_feature:
                                print("[WARNING] cross_attention_mask too small, reinitializing. This should be avoided in training.")
                                self._init_cross_attention_mask(T_style_feature+1, self.cross_attention_mask.device)
                            cross_attention_mask = self.cross_attention_mask[..., :T, :T_style_feature]
                            assert cross_attention_mask.shape[-2:] == (T, T_style_feature)
                            cross_attention_masks += [cross_attention_mask]

            else:
                raise TypeError(f"condition type {self.condition} not supported")
            
            final_condition += [modality_emb_latent]

        ## combine all conditions along the time dimension
        emb_latent = torch.cat(final_condition, 0)

        if self.abl_plus:
            emb_latent = emb_latent + time_emb
        else:
            emb_latent = torch.cat((time_emb, emb_latent), 0)

        # 4. transformer
        if self.arch == "trans_enc":
            raise NotImplementedError("Transformer encoder architecture is not supported")
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
                # timestep_vector_mask = torch.ones((T, 1),device=cross_attention_mask.device, dtype=cross_attention_mask.dtype)
                timestep_vector_mask = torch.zeros((T, 1),device=emb_latent.device, dtype=emb_latent.dtype)

                if self.cross_attention_mask_type not in [None, 'none', 'None']:
                    if self.cross_attention_mask_ndim == 3: # the mask's first dimension needs to be num_head * batch_size
                        timestep_vector_mask = timestep_vector_mask.unsqueeze(0).repeat(self.num_heads, 1, 1) 
                    cross_attention_masks = [timestep_vector_mask] + cross_attention_masks
                    if self.cross_attention_mask_ndim == 3: 
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



            sample = self.decoder(tgt=sample, 
                                  tgt_mask=self_attention_mask,
                                  memory=emb_latent, 
                                  memory_mask=cross_attn_mask
                                  ).squeeze(0)

            if self.diffusion_only:
                sample = self.pose_proj(sample)
                # zero for padded area
                sample[~mask.T] = 0
        else:
            raise TypeError(f"{self.arch} is not supoorted")

        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        sample = sample.permute(1, 0, 2)

        if self.output_proj is not None:
            sample = self.output_proj(sample)

        return sample
        # return (sample, )

    def forward_old(self,
                sample,
                timestep,
                encoder_hidden_states,
                lengths=None,
                attention_mask=None,
                **kwargs):
        B = sample.shape[0]
        T = sample.shape[1]
        # 0.  dimension matching
        # sample [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
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
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 2. condition + time embedding
        if self.condition in ["text", "text_uncond"]:
            # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
            text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
            # textembedding projection
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_emb)
            else:
                text_emb_latent = text_emb
            if self.abl_plus:
                emb_latent = time_emb + text_emb_latent
            else:
                emb_latent = torch.cat((time_emb, text_emb_latent), 0)
        elif self.condition in ["audio"]: 
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
            audio_emb = encoder_hidden_states  # [num_frames, bs, latent_dim]
            # audio embedding projection
            if self.audio_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                audio_emb_latent = self.emb_proj(audio_emb)
            else:
                audio_emb_latent = audio_emb
            if self.abl_plus:
                emb_latent = audio_emb + audio_emb_latent
            else:
                emb_latent = torch.cat((time_emb, audio_emb_latent), 0)
        elif self.condition in ['action']:
            action_emb = self.emb_proj(encoder_hidden_states)
            if self.abl_plus:
                emb_latent = action_emb + time_emb
            else:
                emb_latent = torch.cat((time_emb, action_emb), 0)
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        # 4. transformer
        if self.arch == "trans_enc":
            raise NotImplementedError("Transformer encoder architecture is not supported")
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
            if self.diffusion_only:
                sample = self.pose_embd(sample)

            # tgt    - [1 or 5 or 10, bs, latent_dim]
            # memory - [token_num, bs, latent_dim]
            if self.query_pos is not None:
                sample = self.query_pos(sample)
            if self.mem_pos is not None:
                emb_latent = self.mem_pos(emb_latent)
            sample = self.decoder(tgt=sample, memory=emb_latent, memory_mask=attention_mask).squeeze(0)

            if self.diffusion_only:
                sample = self.pose_proj(sample)
                # zero for padded area
                sample[~mask.T] = 0
        else:
            raise TypeError("{self.arch} is not supoorted")

        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        sample = sample.permute(1, 0, 2)

        return sample
        # return (sample, )


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
