from .DiffuserBase import DenoiserBase 
from .external.mld.models.architectures.mld_denoiser import MldDenoiser as MldDenoiserOld
from .external.mld.models.architectures.mld_denoiser_v2 import MldDenoiser as MldDenoiser
from .external.mld.models.architectures.mld_denoiser_timestepped import MldDenoiserTimestepped 
from .external.mld.models.architectures.mld_denoiser_multi_modal import MldDenoiserMultimodal 

from omegaconf import open_dict

class MldOriginalDenoiser(DenoiserBase): 

    def __init__(self,
                 cfg,
                 ):
        super().__init__()
        self.cfg = cfg
        v = self.cfg.get('denoiser_version', 1)
        condition_list = list(self.cfg.condition_dict.keys())
        if 'condition' in cfg.params:
            ## backwards compatibility
            # del cfg.params['condition'] 
            for ci, cond in enumerate(condition_list):
                if cond.endswith("_condition"):
                    condition_list[ci] = cond[:len(cond) - len("_condition")]
            cfg.params['condition'] = condition_list
        else: 
            with open_dict(cfg.params):
                cfg.params['condition'] = condition_list
        if v == 1: 
            self.denoiser = MldDenoiserOld(**cfg.params)
        elif v == 2: 
            self.denoiser = MldDenoiser(**cfg.params)
        elif v == 3:
            self.denoiser = MldDenoiserTimestepped(**cfg.params)
        elif v == 4:
            self.denoiser = MldDenoiserMultimodal(**cfg.params)
        else:
            raise ValueError(f"Invalid version {v} for MLD denoiser")
        

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def _forward(self, diff_steps, inputs, conditions):
        if isinstance(self.denoiser, MldDenoiserOld): 
            condition_key = list(conditions.keys())[0]
            hidden_states = conditions[condition_key]
            return self.denoiser(sample=inputs, 
                                timestep=diff_steps, 
                                 encoder_hidden_states=hidden_states
                                # conditions=conditions
                                )
        else:
            return self.denoiser(
                    sample=inputs, 
                    timestep=diff_steps, 
                    conditions=conditions
                )