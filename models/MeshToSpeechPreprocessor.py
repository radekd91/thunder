from .mesh2speech.PreprocessorBase import PreprocessorBase 
from omegaconf import OmegaConf, DictConfig
from inferno.sandbox.thunder.utils.checkpoints import get_checkpoint_with_kwargs
from inferno.sandbox.thunder.models.mesh2speech.Lip2SpeechVocoder import Lip2SpeechVocoder
from .MeshToSpeech import MeshToSpeech
from inferno.models.temporal.motion_prior.MotionPrior import MotionPrior
from inferno.utils.other import class_from_str, get_path_to_assets
from munch import Munch, munchify
import sys
from typing import Any, Optional, Dict
from pathlib import Path
from inferno.models.talkinghead.FaceFormerDecoder import load_motion_prior_net
import torch


def load_mesh_to_speech_model(cfg: DictConfig, mode='best') -> MeshToSpeech:
    ## locate checkpoint 

    checkpoint, checkpoint_kwargs = get_checkpoint_with_kwargs(cfg, prefix="", checkpoint_mode=mode)
    class_ = class_from_str(cfg.model.pl_module_class, sys.modules[__name__])
    model = class_.instantiate(cfg, 1, "", checkpoint, checkpoint_kwargs)
    return model


def load_and_configure_mesh_to_speech(cfg: DictConfig, 
                                    #   prior_cfg: DictConfig, 
                                      mode='best_val') -> MeshToSpeech:
    if 'preprocessors' not in cfg.model.keys():
        cfg.model.preprocessors = OmegaConf.create({})

    if 'mel_extractor' in cfg.model.preprocessors:
        cfg.model.preprocessors.pop("mel_extractor", None)
    if 'hubert_units' in cfg.model.preprocessors:
        cfg.model.preprocessors.pop("hubert_units", None)
        
    # if 'flame' not in cfg.model.preprocessors:
    #     if prior_cfg is not None:
    #         flame_path = prior_cfg.model.preprocessor.flame.flame_model_path
    #     else: 
    #         flame_path = "FLAME/geometry/generic_model_2023.pkl"
    #     preprocessor_cfg = OmegaConf.create( { 
    #         "flame": {
    #         "type": "FlamePreprocessor",
    #         "flame": {
    #             "n_exp": 100,
    #             "n_shape": 300,
    #             "flame_model_path": flame_path,
    #             "flame_lmk_embedding_path": "FLAME/geometry/landmark_embedding.npy",
    #         },
    #         "use_texture": False,
    #         },
    #     })
    #     cfg.model.preprocessors.update(preprocessor_cfg)

    ## 1) load model
    # mode = 'latest'
    # mode = 'best'
    # mode = 'best_val'
    model = load_mesh_to_speech_model(cfg, mode)
    model.eval()
    model.to(device='cuda')

    ## freeze the model
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(cfg.model, "vocoder"): 
        if cfg.model.vocoder is not None: 
            vocoder = model.vocoder
        else: 
            vocoder = None
    else:
        import json
        vocoder_config_file = get_path_to_assets() / "Lip2SpeechUnit/vocoder/multi_input_aug.json"
        vocoder_checkpoint = get_path_to_assets() / "Lip2SpeechUnit/vocoder/vocoder_lrs3_aug_multi.pt"
        with open(vocoder_config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        vocoder_cfg = munchify({ 
            "trainable": False, 
            "pretrained_checkpoint": str(vocoder_checkpoint),
            "generator": json_config,
        })
        vocoder = Lip2SpeechVocoder(vocoder_cfg)
        vocoder.cuda()

    return model, vocoder


class MeshToSpeechPreprocessor(PreprocessorBase):

    def __init__(self, 
                #  cfg,
                 model_name, 
                 path_to_models, 
                 input_prefix,
                 direct_vertices=True,
                 checkpoint_mode="best_val",
                 without_vocoder = False,
                #  *args, 
                 **kwargs):
        
        super().__init__()
        # self.cfg = cfg
        # path_to_models = cfg.model.path_to_models
        # model_name = cfg.model.model_name
        m2s_cfg_path = Path(path_to_models) / model_name / "cfg.yaml"
        if not m2s_cfg_path.is_absolute():
            m2s_cfg_path = get_path_to_assets() / m2s_cfg_path
        # checkpoint_mode = cfg.model.checkpoint_mode
        self.input_prefix = input_prefix
        self._direct_vertices = direct_vertices
        m2s_config = OmegaConf.load(m2s_cfg_path) 
        if str(m2s_cfg_path.parent) != m2s_config.inout.full_run_dir: 
            m2s_config.inout.full_run_dir = str(m2s_cfg_path.parent)
            m2s_config.inout.checkpoint_dir = str(m2s_cfg_path.parent / "checkpoints")

        if without_vocoder:
            m2s_config.model.vocoder = None
        self.model, self.vocoder  = load_and_configure_mesh_to_speech(m2s_config, checkpoint_mode)
        
    def to(self, device):
        self.model = self.model.to(device)
        if self.vocoder is not None:
            self.vocoder = self.vocoder.to(device)
        return self
    
    @property
    def device(self):
        return self.model.device
    
    @property 
    def direct_vertices(self):
        return self.model.cfg.model.input.input_vertices and self._direct_vertices
    
    @property
    def input_expcode(self):
        return self.model.cfg.model.input.input_expcode
    
    @property
    def input_jawpose(self):
        return self.model.cfg.model.input.input_jawpose

    def _m2s_forward(self, batch: Dict, train=False, validation=False, **kwargs: Any):
        batch = self.model._choose_padding_mask(batch)

        # # ## 0) Preprocessors - skip preprocessing, shouldn't be necessary
        if not self.direct_vertices:
            batch = self.model._choose_primary_3D_rec_method(batch)
            batch = self.model.forward_prepocessors(batch, train=train, validation=validation, with_grad=True, **kwargs)
     
        ## 1) Mesh Encoder 
        batch = self.model.forward_mesh_encoder(batch, train=train, validation=validation, **kwargs)

        ## 2) Sequence Encoder 
        batch = self.model.forward_sequence_encoder(batch, train=train, validation=validation, **kwargs)

        ## 3) Speech prediction 
        batch = self.model.forward_speech_prediction_head(batch, train=train, validation=validation, **kwargs)

        # ## 4) Vocoder - skip vocoder for now
        # batch = self.model.forward_vocoder(batch, train=train, validation=validation, **kwargs)

        # ## 5) Postprocessing - skip postprocessing, shouldn't be necessary
        # batch = self.model.forward_postpocessors(batch, train=train, validation=validation, **kwargs)

        return batch

    def _forward(self, batch: Dict, train=False, validation=False, input_prefix=None, **kwargs: Any):
        m2s_batch = {}

        # input_prefix = 'predicted_enc_dec_'
        input_prefix = input_prefix or self.input_prefix 
        # if self.model.cfg.model.input.input_vertices: 
        if self.direct_vertices: 
            m2s_batch["gt_vertices"] = batch[input_prefix + "vertices"]
            T = batch[input_prefix + "vertices"].shape[1]
        
        # if self.model.cfg.model.input.input_expcode:
        if self.input_expcode or not self.direct_vertices:
            F = batch[input_prefix + "exp"].shape[-1]
            T = batch[input_prefix + "exp"].shape[1]
            if not self.direct_vertices:
                F_m2s = 100
            else:
                F_m2s = self.model.cfg.model.input_encoder.keys_to_concat.gt_exp    
            m2s_batch["gt_exp"] = batch[input_prefix + "exp"]
            if F < F_m2s: ## pad with zeros
                pad_size = F_m2s - F
                pad = torch.zeros(m2s_batch["gt_exp"].shape[0], m2s_batch["gt_exp"].shape[1], pad_size, device=m2s_batch["gt_exp"].device)
                m2s_batch["gt_exp"] = torch.cat([m2s_batch["gt_exp"], pad], dim=-1)
            elif F > F_m2s:
                m2s_batch["gt_exp"] = m2s_batch["gt_exp"][:, :F_m2s, ...]
        # if self.model.cfg.model.input.input_jawpose:
        if self.input_jawpose or not self.direct_vertices:
            m2s_batch["gt_jaw"] = batch[input_prefix + "jaw"]
        m2s_batch["speaker_embedding"] = batch["speaker_embedding"]
        m2s_batch["landmarks_validity"] = batch["landmarks_validity"]
        for key in m2s_batch["landmarks_validity"].keys():
            m2s_batch["landmarks_validity"][key] = m2s_batch["landmarks_validity"][key][:, :T, ...]
 

        keys_before = set(m2s_batch.keys()) 
        m2s_batch = self._m2s_forward(m2s_batch, train=train, validation=validation, **kwargs)
        keys_after = set(m2s_batch.keys())
        new_keys = keys_after - keys_before

        for key in new_keys:
            batch[key] = m2s_batch[key]

        return batch

    def forward(self, batch, train=False, validation=False, 
            input_key=None, output_key=None, with_grad=False, **kwargs):   
        if with_grad:
            batch = self._forward(batch, train=train, validation=validation, **kwargs)
        else:
            with torch.no_grad():
                batch = self._forward(batch, train=train, validation=validation, **kwargs)
        
        return batch
        
    # def preprocess(self, data):
    #     # Preprocess data here
    #     return data

    # def postprocess(self, data):
    #     # Postprocess data here
    #     return data

    # def __call__(self, data):
    #     return self.preprocess(data) 
    


class MotionPriorPreprocessor(PreprocessorBase):

    def __init__(self, 
                #  cfg, 
                 model_name, 
                 path_to_models,
                 output_prefix,
                 #  *args, 
                 **kwargs):
        super().__init__()
        # path_to_models = cfg.model.path_to_models
        # model_name = cfg.model.model_name
        # # prior_cfg_path = Path(path_to_models) / model_name / "cfg.yaml"
        prior_cfg_path = Path(path_to_models) / model_name
        # checkpoint_mode = cfg.model.checkpoint_mode

        self.output_prefix = output_prefix

        # prior_config = OmegaConf.load(prior_cfg_path)
        # self.model = load_mesh_to_speech_model(prior_config, checkpoint_mode)
        self.model : MotionPrior = load_motion_prior_net(prior_cfg_path, trainable=False)
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    @property
    def device(self):
        return self.model.device
    
    def _mp_forward(self, batch: Dict, train=False, validation=False, **kwargs: Any):
        batch = self.model.encoding_step(batch, train=train, validation=validation, **kwargs)
        batch = self.model.decoding_step(batch, train=train, validation=validation, **kwargs)
        return batch

    def _forward(self, batch: Dict, train=False, validation=False, **kwargs: Any):
        prior_batch = {}

        input_prefix = 'predicted_'
        for key in self.model.cfg.model.sequence_components:
            prior_batch["gt_" + key] = batch[input_prefix + key]
        prior_batch["gt_shape"] = batch["gt_shape"]
        
        prior_batch = self._mp_forward(prior_batch, train=train, validation=validation, **kwargs)

        # output_prefix = 'predicted_enc_dec_'
        output_prefix = self.output_prefix
        for key in self.model.cfg.model.sequence_components:
            batch[output_prefix + key] = prior_batch["reconstructed_" + key]
        batch[output_prefix + "vertices"] = prior_batch["reconstructed_vertices"]

        return batch


    def forward(self, batch, train=False, validation=False, 
            input_key=None, output_key=None, with_grad=False, 
            **kwargs):   
        if with_grad:
            batch = self._forward(batch, train=train, validation=validation, **kwargs)
        else:
            with torch.no_grad():
                batch = self._forward(batch, train=train, validation=validation, **kwargs)
        
        return batch

    # def preprocess(self, data):
    #     # Preprocess data here
    #     return data

    # def postprocess(self, data):
    #     # Postprocess data here
    #     return data

    # def __call__(self, data):
    #     return self.preprocess(data)