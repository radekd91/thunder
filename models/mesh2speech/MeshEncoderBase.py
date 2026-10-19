import torch
# import numpy as np
from typing import Dict, Any
from pathlib import Path
import sys, os
import math
from inferno.models.MLP import MLP
from inferno.utils.other import get_path_to_assets

class Mesh4DEncoderBase(torch.nn.Module): 

    def __init__(self) -> None:
        super().__init__()

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]
    

class FlameParamConcatenation(Mesh4DEncoderBase): 

    def __init__(self, cfg, ):
        super().__init__()
        self.cfg = cfg

    @property
    def keys_to_concat(self):
        return self.cfg.keys_to_concat

    @property
    def input_dim(self):
        return sum(self.keys_to_concat.values())
        
    @property
    def output_dim(self):
        return sum(self.keys_to_concat.values())


    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:

        # reconstruction_dict = batch["reconstruction"] if "reconstruction" in batch else \
        #     batch["reconstruction"][list(batch["reconstruction"].keys())[0]]

        tensor_list = []
        for key in self.keys_to_concat.keys():
            tensor_list.append(batch[key])

        concatenated_tensor = torch.cat(tensor_list, dim=-1)
        batch["input_features"] = concatenated_tensor
        return batch
    

class FlameParamCatLinear(FlameParamConcatenation): 
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)


    @property
    def output_dim(self):
        return self.cfg.output_dim

    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        batch = super().forward(batch, train, validation, **kwargs)

        x = batch["input_features"]
        x = self.linear(x)
        batch["input_features"] = x

        return batch
    
class FlameParamCatMLP(FlameParamConcatenation): 
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        batch_norm = None
        if cfg.batch_norm == 'batch_norm':
            batch_norm = torch.nn.BatchNorm1d
        elif cfg.batch_norm == 'layer_norm':
            batch_norm = torch.nn.LayerNorm
        elif cfg.batch_norm == 'instance_norm':
            batch_norm = torch.nn.InstanceNorm1d
        self.mlp = MLP(self.input_dim, self.output_dim, self.cfg.hidden_layer_sizes, batch_norm=batch_norm)

    @property
    def output_dim(self):
        return self.cfg.output_dim

    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        batch = super().forward(batch, train, validation, **kwargs)

        x = batch["input_features"]
        B, T, F = x.shape
        x = x.view(B*T, F)
        x = self.mlp(x)
        x = x.view(B, T, -1)
        batch["input_features"] = x

        return batch
    


class VertexEncoderBase(Mesh4DEncoderBase): 

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg 
        self.vertex_indices = cfg.get('vertex_indices', False)
        if isinstance(self.vertex_indices, str):
            # Load from file, read all lines and convert to a list of integers
            if not Path(self.vertex_indices).is_absolute():
                self.vertex_indices = Path(self.vertex_indices)
                self.vertex_indices = get_path_to_assets() / self.vertex_indices

            with open(self.vertex_indices, "r") as f:
                self.vertex_indices = [int(x) for x in f.readlines()]
        
        if self.vertex_indices:
            self.vertex_indices = torch.tensor(self.vertex_indices)

    def select_vertices(self, verts): 
        B, T, V = verts.shape
        verts = verts.view(B, T, V // 3, 3)
        verts = verts[:, :, self.vertex_indices, :]
        verts = verts.reshape(B, T, -1)
        return verts

    @property
    def num_vertices(self):
        if self.vertex_indices is False:
            return 5023
        else: 
            return len(self.vertex_indices)

    @property
    def input_dim(self):
        if self.vertex_indices is False:
            return 3 * 5023
        else: 
            return 3 * len(self.vertex_indices)

    @property
    def output_dim(self):
        return self.cfg.output_dim


class VertexLinear(VertexEncoderBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        verts = batch["gt_vertices"] 
        if self.vertex_indices is not False:
            verts = self.select_vertices(verts)
        features = self.linear(verts)
        batch["input_features"] = features
        return batch


class VertexMLP(VertexEncoderBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        batch_norm = None
        if cfg.batch_norm == 'batch_norm':
            batch_norm = torch.nn.BatchNorm1d
        elif cfg.batch_norm == 'layer_norm':
            batch_norm = torch.nn.LayerNorm
        elif cfg.batch_norm == 'instance_norm':
            batch_norm = torch.nn.InstanceNorm1d
        self.mlp = MLP(self.input_dim, self.output_dim, self.cfg.hidden_layer_sizes, batch_norm=batch_norm)

    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        verts = batch["gt_vertices"] 
        if self.vertex_indices is not False:
            verts = self.select_vertices(verts)
        B, T, V = verts.shape
        verts = verts.view(B*T, V)
        features = self.mlp(verts)
        features = features.view(B, T, -1)
        batch["input_features"] = features
        return batch



class PositionalEncodingBase(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout)
        self._init_positional_encoding()

    def _init_positional_encoding(self):
        raise NotImplementedError("Subclasses must implement this method")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class SinePositionalEncoding(PositionalEncodingBase):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__(d_model, dropout, max_len)


    def _init_positional_encoding(self):
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

class LearnedPositionalEncoding(PositionalEncodingBase):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__(d_model, dropout, max_len)

    def _init_positional_encoding(self):
        self.pe = torch.nn.Parameter(torch.randn(1, self.max_len, self.d_model))
        torch.nn.init.xavier_uniform_(self.pe)



class VertexTransformer(VertexEncoderBase): 

    def __init__(self, cfg):
        super().__init__(cfg)
        dim_factor = 1
        ## linear layer that expands the vertex coordinates to the feature dimension
        features_dim = self.cfg.feature_dim * dim_factor
        self.linear = torch.nn.Linear(3, features_dim)
        
        self._init_positional_encoding(features_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model = features_dim, 
                    nhead = self.cfg.nhead, 
                    dim_feedforward = dim_factor*self.cfg.feature_dim, 
                    activation = self.cfg.activation,
                    dropout = self.cfg.dropout, 
                    batch_first = True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.num_layers)
        
        if self.cfg.use_output_linear_layer:
            # self.out_linear = torch.nn.Linear(features_dim, self.output_dim)
            self.out_linear = torch.nn.Linear(self.output_dim, self.output_dim)


    def _init_positional_encoding(self, features_dim): 
        if self.cfg.positional_encoding_type == "learnable":
            self.PE = LearnedPositionalEncoding(features_dim, dropout=0, max_len=self.num_vertices)
        elif self.cfg.positional_encoding_type == "none":
            self.PE = None
        else: 
            raise NotImplementedError(f"Positional encoding {self.cfg.positional_encoding_type} not implemented")

        ## initialize transformer 
        
    def _forward_positional_encoding(self, x):
        if self.PE is not None:
            self.PE 
        return x

    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        verts = batch["gt_vertices"] 
        if self.vertex_indices is not False:
            verts = self.select_vertices(verts)

        B, T = verts.shape [:2]
        if verts.dim() == 3:
            verts = verts.view(B, T, self.num_vertices, 3)
        V, F = verts.shape[2:4]
        verts = verts.view(B*T, V, F)
        
        hidden_states = self.linear(verts)
        Fh = hidden_states.shape[-1]
        ## the vertex coordinates are the features space here and the vertex dimension is the sequence length
        if self.PE is not None:
            hidden_states = self.PE(hidden_states)
        ## pass through transformer
        hidden_states = self.encoder(hidden_states)

        ## from BT, V, F to B, T, 1, F  ## essentially collapsing the vertex dimension 
        if self.cfg.collapse_type == "mean":
            hidden_states = hidden_states.mean(dim=-2, keepdim=True)
        elif self.cfg.collapse_type == "first":
            hidden_states = hidden_states[..., 0, :].view(B, T, -1)
        elif self.cfg.collapse_type == "last":
            hidden_states = hidden_states[..., -1, :].view(B, T, -1)
        elif self.cfg.collapse_type == "first_n":
            n = self.output_dim // Fh 
            assert n * Fh == self.output_dim, f"Output dim {self.output_dim} must be divisible by the number of features {Fh}"
            hidden_states = hidden_states[..., :n, :].view(B, T, self.output_dim)
        else: 
            raise NotImplementedError(f"Collapse type {self.cfg.collapse_type} not implemented")

        hidden_states = hidden_states.view(B, T, -1)
        if self.cfg.use_output_linear_layer:
            hidden_states = self.out_linear(hidden_states)
        batch["input_features"] = hidden_states
        return batch


class Lip2SpeechEncoder(Mesh4DEncoderBase): 

    def __init__(self, cfg):
        super().__init__()

        # sys.path.insert(0, str(Path(__file__).parent / "external"))
        # from espnet.nets.pytorch_backend.backbones.conv3d_extractor  import Conv3dResNet
        # from espnet.nets.pytorch_backend.backbones.conv1d_extractor  import Conv1dResNet
        # sys.path.pop(0)
        from .external.espnet.nets.pytorch_backend.backbones.conv3d_extractor  import Conv3dResNet
        from .external.espnet.nets.pytorch_backend.backbones.conv1d_extractor  import Conv1dResNet
        
        self.cfg = cfg
        if cfg.input_layer == "conv1d":
            self.frontend = Conv1dResNet(
                relu_type=cfg.relu_type,
                a_upsample_ratio=cfg.a_upsample_ratio,
            )
        elif cfg.input_layer == "conv3d":
            self.frontend = Conv3dResNet(relu_type=cfg.relu_type)
        else:
            raise NotImplementedError(f"Input layer '{cfg.input_layer}' not implemented")

        self.trainable = cfg.get('trainable', True)
        self.uses_orig_image_range = cfg.get('uses_orig_image_range', False)
        self.im_transform = None
         ## the original lip2speech model expects the input to be in a particular range
        if self.uses_orig_image_range:
            from torchvision.transforms import Normalize
            self.im_transform = Normalize(cfg.get('orig_transform_mean'), 
                                          cfg.get('orig_transform_std'))

        self.pretrained_checkpoint = cfg.get('pretrained_checkpoint', None)
        if self.pretrained_checkpoint:
            assert self.uses_orig_image_range, \
                "Lip2speech unit pretrained model expects the input to be in a particular range but uses_orig_image_range is False"
            self._load_pretrained()

        if not self.trainable:
            for param in self.frontend.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        mode = mode and self.trainable
        return super().train(mode)
        

    def _load_pretrained(self):
        print(f"Loading pretrained checkpoint from {self.pretrained_checkpoint}")
        from inferno.utils.other import get_path_to_assets
        if not Path(self.pretrained_checkpoint).is_absolute():
            # self.pretrained_checkpoint = Path(self.pretrained_checkpoint)
            self.pretrained_checkpoint = get_path_to_assets() / self.pretrained_checkpoint
        checkpoint = torch.load(self.pretrained_checkpoint)
        self.frontend.load_state_dict(checkpoint, strict=True)


    def forward(self, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        x = batch["video_mouth_crop"]

        if self.uses_orig_image_range: ## the original lip2speech model expects the input to be in a particular range
            x = self.im_transform(x)

        ## remove the channel dimension, it should be 1 here and the frontend expects the input without it
        x = x.squeeze(2) 
        x = self.frontend(x)
        batch["input_features"] = x
        return batch
    