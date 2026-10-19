from .mesh2speech.PreprocessorBase import PreprocessorBase 
from omegaconf import OmegaConf, DictConfig
# from inferno.sandbox.thunder.models.mesh2speech.Lip2SpeechVocoder import Lip2SpeechVocoder
# from .MeshToSpeech import MeshToSpeech
# from inferno.models.temporal.motion_prior.MotionPrior import MotionPrior
from inferno.utils.other import class_from_str, get_path_to_assets, get_path_to_externals
from inferno.sandbox.thunder.utils.batch import temporal_trim_dict
from munch import Munch, munchify
import sys
from typing import Any, Optional, Dict
from pathlib import Path
from inferno.models.talkinghead.FaceFormerDecoder import load_motion_prior_net
import torch
import copy


class OrigDiffPoseTalkStylePreprocessor(PreprocessorBase):
    
    def __init__(self, 
                #  cfg,
                model_name, 
                path_to_models, 
                data_root=None,
                #  *args, 
                **kwargs):
        super().__init__()

        path_to_dfp = get_path_to_externals() / "DiffPoseTalk" 
        # path_to_dfp = get_path_to_externals() 
        # sys.path.insert(0, str(path_to_dfp))
        sys.path.insert(0, str(get_path_to_externals()))
        # from extract_style import StyleExtractor
        from DiffPoseTalk.extract_style import StyleExtractor
        sys.path.pop(0)  # Remove the path to avoid conflicts with other imports 
        from munch import Munch, munchify

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_path = path_to_dfp / path_to_models / model_name 
        assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist."
        if data_root is not None and not Path(data_root).exists():
            local_data_root = get_path_to_assets() / "TFHP/processed"
            if local_data_root.exists():
                data_root = local_data_root
        self.style_extractor = StyleExtractor(checkpoint_path, device=device, data_root=data_root)

        for param in self.style_extractor.model.parameters():
            param.requires_grad = False

    def to(self, device):
        self.style_extractor.model = self.style_extractor.model.to(device)
        return self
    
    def max_seq_len(self):
        return self.style_extractor.n_motions

    @property
    def device(self):
        return self.style_extractor.model.device
    
    def _dfps_forward(self, batch: Dict, train=False, validation=False, **kwargs: Any):
        # feats, mask = self.model._get_input_feats(batch) 
        # style_feature = self.model._forward_single(feats, mask) 
        exp = batch['gt_exp']
        jaw = batch['gt_jaw']
        global_pose = batch.get('gt_global_pose', None)
        style_feature = self.style_extractor.extract_inferno(exp, jaw, global_pose, start_frame=0)
        return style_feature

    def _forward(self, batch: Dict, train=False, validation=False, input_prefix=None, **kwargs: Any):
        dfp_batch = {}
        input_prefix = input_prefix or "gt_"
        dfp_batch["gt_exp"] = batch[f"{input_prefix}exp"]#.clone()
        dfp_batch["gt_jaw"] = batch[f"{input_prefix}jaw"]#.clone()
        if not self.style_extractor.no_head_pose:
            dfp_batch["gt_global_pose"] = batch[f"{input_prefix}global_pose"]#.clone()

        # if self.model.config.model.get('use_global_translation', False):
        #     dfp_batch["gt_trans"] = batch[f"{input_prefix}trans"]#.clone()
        # assert  self.model.config.model.get('use_global_pose', False) is True, "DiffPoseTalk does not support global pose extraction"

        # if self.model.config.model.get('use_head_pose', False):
        #     dfp_batch["gt_global_pose"] = batch[f"{input_prefix}global_pose"]#.clone()

        # assert  self.model.config.model.get('use_neck_pose', False) is False, "DiffPoseTalk does not support neck pose extraction"
            # dfp_batch["gt_neck_pose"] = batch[f"{input_prefix}neck_pose"]#.clone()

        # dfp_batch["landmarks_validity"] = batch['landmarks_validity']
        # # dfp_batch["landmarks_validity"]["mediapipe"] = dfp_batch["landmarks_validity"]["mediapipe"].clone()

        # max accepted sequencte length 
        # dfp_batch = copy.deepcopy(dfp_batch)
        dfp_batch = temporal_trim_dict(dfp_batch, start_frame=0, end_frame=self.max_seq_len(), is_batched=True)

        style_feature = self._dfps_forward(dfp_batch, train=train, validation=validation, **kwargs)

        batch["style_feat"] = style_feature

        return batch

    def forward(self, batch, train=False, validation=False, 
            input_key=None, output_key=None, with_grad=False, **kwargs):   
        if with_grad:
            batch = self._forward(batch, train=train, validation=validation, **kwargs)
        else:
            with torch.no_grad():
                batch = self._forward(batch, train=train, validation=validation, **kwargs)
        
        return batch
