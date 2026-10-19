import torch 
from enum import Enum


class DenoiserObjective(Enum): 
    """
    Labels which objective the denoiser has.
    """
    X_0 = 1 ## predict X_0 (final denoised sample) from X_t (noisy sample)
    EPS_0 = 2 ## predict EPS_0, i.e. the noise that produces X_0 from X_t when applied
    X_T_1 = 3 ## predict X_t-1 from X_t
    EPS_T_1 = 4 ## predict EPS_t-1, i.e. the noise that produces X_t-1 from X_t when applied


class DiffuserBase(torch.nn.Module):

    def __init__(self, cfg, *args, **kwargs) -> None:
        self.cfg = cfg
        super().__init__(*args, **kwargs)

    @property
    def input_key(self):
        return self.cfg.input_key

    @property
    def output_key(self):
        return self.cfg.output_key
    
    @property
    def noise_key(self):
        # return self.cfg.noise_key
        return "noise"
    
    def forward(self, batch):
        raise NotImplementedError("This should be implemented by deriving classes")
        # Implement the forward pass of your model here
        # batch["denoised"] = ...
        # return batch

    def apply_predicted_noise(self, diffusion_step, noisy_vector, predicted_noise, training):
        raise NotImplementedError("This should be implemented by deriving classes")

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    @property
    def diffusion_steps_key(self): 
        return "diffusion_steps"

class DenoiserBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def _init_arch(self): 
        raise NotImplementedError("This should be implemented by deriving classes")
        # Define your model architecture here
        self.layer1 = torch.nn.Linear(..., ...)
        # Add more layers as needed

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def predicts_x(self):
        """
        Returns true if the denoiser predicts the denoised sample, false if it predicts the noise.
        """
        return self.cfg.get("simple_objective", True)

    @property
    def condition_dict(self):
        return self.cfg.condition_dict

    @property
    def noise_key(self):
        return self.cfg.noise_key
        # return "noise"
    
    @property
    def predicted_noise_key(self):
        return self.cfg.get('predicted_noise_key', "predicted_noise")
        # return "noise"
    

    @property
    def denoised_key(self):
        return self.cfg.denoised_key

    def forward(self, batch):
        diff_steps = batch["diffusion_steps"]
        inputs = batch[self.noise_key]
        conditions = {key: batch[self.condition_dict[key] ] for key in self.condition_dict.keys()}
        outputs = self._forward(diff_steps, inputs, conditions)
        if self.predicts_x():
            ## outputs .... x_0
            ## inputs ... x_t  = x_0 + eps_0
            ## predicted noise ... eps_0 = x_t - x_0 
            batch[self.denoised_key] = outputs
            batch[self.predicted_noise_key] = inputs - outputs
        else:
            ## outputs .... eps_0
            ## predicted noise ...  eps_0
            ## inputs ... x_t  
            ## denoised x_0 = x_t - eps_0
            batch[self.predicted_noise_key] = outputs
            batch[self.denoised_key] = inputs - outputs
        # print("Predicted noise norm: ", torch.norm(batch[self.predicted_noise_key]).item())
        return batch
    

    def _forward(self, diff_steps, inputs, **kwargs):
        raise NotImplementedError("This should be implemented by deriving classes")





