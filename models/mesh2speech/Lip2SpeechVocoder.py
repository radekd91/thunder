
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from typing import Dict, Any
from inferno.utils.other import get_path_to_assets, get_path_to_externals

from .VocoderBase import Mesh2SpeechVocoderBase


class Lip2SpeechVocoder(Mesh2SpeechVocoderBase): 

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.trainable = cfg.get("trainable", False)
        if isinstance(cfg.generator, str): 
            import json
            with open(cfg.generator) as f:
                data = f.read()
            json_config = json.loads(data)
            cfg.generator = json_config
        self.generator = MelCodeGenerator(cfg.generator)

        self.pretrained_checkpoint = cfg.get("pretrained_checkpoint", None)
        if self.pretrained_checkpoint is not None:
            self._load_pretrained(self.pretrained_checkpoint)

        if not self.trainable:
            for param in self.generator.parameters():
                param.requires_grad = False

    def _load_pretrained(self, checkpoint):
        print(f"Loading pretrained model from {checkpoint}")
        if not Path(checkpoint).is_absolute():
            checkpoint = get_path_to_assets() / checkpoint
        checkpoint = torch.load(checkpoint)
        self.generator.load_state_dict(checkpoint['generator'], strict=True)

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def train(self, mode: bool = True):
        mode = mode and self.trainable
        return super().train(mode)

    @property
    def input_mel_key(self):
        return "predicted_mel" 
    
    @property
    def input_unit_key(self):
        return "predicted_units" 
    
    @property
    def speaker_embedding_key(self):
        return "speaker_embedding"

    @property
    def output_key(self):
        return "predicted_wav"
    
    @property
    def training_unit_sampling_strategy(self): 
        return self.cfg.get('training_unit_sampling_strategy', 'straight_through')

    @property
    def testing_unit_sampling_strategy(self): 
        return self.cfg.get('testing_unit_sampling_strategy', 'argmax')
    
    @property
    def gumbel_temperature(self):
        return self.cfg.get('gumbel_temperature', 0.5)

    def _sample_units(self, unit_key, batch: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        classification_vector_key = unit_key[:-1] + '_classification_vectors'

        if train or validation:
            sampling_strategy = self.training_unit_sampling_strategy
        else:
            sampling_strategy = self.testing_unit_sampling_strategy

        if sampling_strategy == 'argmax':
            predicted_unit_classification_vectors = F.softmax(batch[classification_vector_key], dim=-1)
            # predicted_unit_classification_vectors = batch[classification_vector_key]
            batch[unit_key] = torch.argmax(predicted_unit_classification_vectors, dim=-1)
        elif sampling_strategy == 'straight_through':
            # predicted_unit_classification_vectors = F.softmax(batch[classification_vector_key], dim=-1)
            predicted_unit_classification_vectors = batch[classification_vector_key]
            indices = predicted_unit_classification_vectors.argmax(dim=-1) 
            batch[unit_key] = F.one_hot(indices, num_classes=predicted_unit_classification_vectors.shape[-1]).float() \
                - predicted_unit_classification_vectors.detach() + predicted_unit_classification_vectors
        elif sampling_strategy == "gumbel_softmax":
            predicted_unit_classification_vectors = batch[classification_vector_key]
            batch[unit_key] = F.gumbel_softmax(predicted_unit_classification_vectors, 
                                               tau=self.gumbel_temperature, 
                                               hard=False)
        else: 
            raise ValueError(f"Invalid sampling strategy {sampling_strategy}")
        
        return batch


    def forward(self, batch: Dict, train=False, validation=False, 
                input_mel_key=None, input_unit_key=None, output_key=None,
                **kwargs: Any) -> Dict:
        input_mel_key = input_mel_key or self.input_mel_key
        input_unit_key = input_unit_key or self.input_unit_key
        output_key = output_key or self.output_key
        generator_batch = {}
        T = batch[input_mel_key].shape[1] 
        if input_unit_key not in batch: 
            classification_vector_key = input_unit_key[:-1] + '_classification_vectors'
            assert classification_vector_key in batch, f"Expected {classification_vector_key} in batch if {input_unit_key} is not present."
            batch = self._sample_units(input_unit_key, batch, train=train, validation=validation, **kwargs)
            ## 
        if T // 2 > batch[input_unit_key].shape[1]:
            batch[input_unit_key] = F.pad(batch[input_unit_key], (0, T//2 - batch[input_unit_key].shape[1]))
        else: 
            batch[input_unit_key] = batch[input_unit_key][:, :T//2]
        generator_batch['mel'] = batch[input_mel_key].transpose(1, 2)
        # if batch[input_unit_key].dtype != torch.long: 
        #     batch[input_unit_key] = batch[input_unit_key].long()
        generator_batch['code'] = batch[input_unit_key]
        generator_batch['spkr'] = batch[self.speaker_embedding_key]
        output_wav = self.generator(**generator_batch)
        batch[output_key] = output_wav
        return batch


## Adapted from lip2speeech-unit: https://github.com/choijeongsoo/lip2speech-unit/tree/main/speech-resynthesis, 
## which was originally taken from: https://github.com/facebookresearch/speech-resynthesis 
# try:
import sys
from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent / "external" / "speech-resynthesis")) 
# sys.path.insert(0, str(get_path_to_externals() / "speech-resynthesis")) 
# from models import CodeGenerator 
from .external.speech_resynthesis.models import CodeGenerator
# sys.path.pop(0)
class MelCodeGenerator(CodeGenerator):
    def __init__(self, h):
        super().__init__(h)
        self.dict = nn.Embedding(h.num_embeddings, h.embedding_dim)
        self.multispkr = h.get('multispkr', None)

        embedder_dim = h.get("embedder_dim", None)
        if self.multispkr and not embedder_dim:
            self.spkr = nn.Embedding(h.get("num_speakers", 200), h["embedding_dim"])
        elif embedder_dim:
            self.spkr = nn.Linear(embedder_dim, h["embedding_dim"])

        self.layer = nn.Sequential(
            nn.ConvTranspose1d(h.embedding_dim, h.embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(h.embedding_dim, h.embedding_dim)

    def forward(self, **kwargs):
        x = kwargs['mel']

        if kwargs['code'].dtype != torch.long:
            ## do matrix multiplication instead
            ## there is a weird hack if we want to use the original pretrained model, there is a class count mismatch error 
            ## so we need to cut off the extra classes
            code = torch.matmul(kwargs['code'][...,:self.dict.weight.data.shape[0]], 
                                self.dict.weight.data)
        else:
            code = self.dict(kwargs['code'])
        code = self.layer(code.permute(0,2,1)).permute(0,2,1)
        code = self.dropout(code)
        code = self.fc(code)
        code = code.permute(0,2,1)

        x = torch.cat([x, code], dim=1)

        if self.multispkr:
            spkr = self.spkr(kwargs['spkr'])
            spkr = self._upsample(spkr, x.shape[-1])
            x = torch.cat([x, spkr], dim=1)

        for k, feat in kwargs.items():
            if k in ['spkr', 'code', 'mel']:
                continue

            feat = self._upsample(feat, x.shape[-1])
            x = torch.cat([x, feat], dim=1)

        return super(CodeGenerator, self).forward(x)


# except ImportError:
#     print("Could not import CodeGenerator from speech-resynthesis. Make sure to install the package.")
#     # raise ImportError("CodeGenerator not found. Please install the speech-resynthesis package.")
