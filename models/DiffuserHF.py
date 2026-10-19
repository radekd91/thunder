from .DiffuserBase import DiffuserBase 
from inferno.utils.other import class_from_str
import sys
import torch
from diffusers import DDIMScheduler, DDPMScheduler
from .CustomDDPMScheduler import CustomDDPMScheduler
import inspect
import omegaconf


def instantiate_from_config(config):
    # if not "target" in config:
    #     if config == '__is_first_stage__':
    #         return None
    #     elif config == "__is_unconditional__":
    #         return None
    #     raise KeyError("Expected key `target` to instantiate.")
    return class_from_str(config["target"], sys.modules[__name__])(**config.get("params", dict()))


class DiffuserHF(DiffuserBase): 

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.scheduler = instantiate_from_config(cfg.scheduler)
        self.scheduler.set_timesteps(self.cfg.scheduler.num_inference_timesteps)
        self.noise_scheduler = instantiate_from_config(cfg.noise_scheduler)
        noise_std = self.cfg.noise_std if 'noise_std' in self.cfg.keys() else 1.
        if isinstance(noise_std, (dict, omegaconf.DictConfig)):
            std_vector = []
            for noise, indices in noise_std.items():
                start_i, end_i = indices 
                std_vector += [noise] * (end_i - start_i)
            noise_std = torch.tensor(std_vector).unsqueeze(0).unsqueeze(0)
            # self.register_buffer("noise_std", noise_std)
            # self.register_buffer("noise_std", noise_std)
            self.noise_std = noise_std
        elif isinstance(noise_std, (list, tuple, omegaconf.ListConfig)):
            std_vector = []
            for elem in noise_std: 
                assert isinstance(elem, ((dict, omegaconf.DictConfig))) 
                for noise, indices in elem.items():
                    start_i, end_i = indices 
                    std_vector += [noise] * (end_i - start_i)
            noise_std = torch.tensor(std_vector).unsqueeze(0).unsqueeze(0)
            # self.register_buffer("noise_std", noise_std)
            self.noise_std = noise_std
        elif isinstance(noise_std, (int, float)):
            self.noise_std = float(noise_std)
            # self.noise_std = torch.tensor(self.noise_std).unsqueeze(0).unsqueeze(0)
        else: 
            raise ValueError(f"noise_std must be a float or a dict, got {type(noise_std)}")
        
        noise_bias = self.cfg.noise_bias if 'noise_bias' in self.cfg.keys() else 0.
        if isinstance(noise_bias, (dict, omegaconf.DictConfig)):
            bias_vector = []
            for noise, indices in noise_bias.items():
                start_i, end_i = indices 
                bias_vector += [noise] * (end_i - start_i)
            noise_bias = torch.tensor(bias_vector).unsqueeze(0).unsqueeze(0)
            # self.register_buffer("noise_bias", noise_bias)
            self.noise_bias = noise_bias
        elif isinstance(noise_bias, (list, tuple, omegaconf.ListConfig)):
            bias_vector = []
            for elem in noise_bias: 
                assert isinstance(elem, ((dict, omegaconf.DictConfig))) 
                for noise, indices in elem.items():
                    start_i, end_i = indices 
                    bias_vector += [noise] * (end_i - start_i)
            noise_bias = torch.tensor(bias_vector).unsqueeze(0).unsqueeze(0)
            self.noise_bias = noise_bias
        elif isinstance(noise_bias, (int, float)):
            self.noise_bias = float(noise_bias)
            # self.noise_bias = torch.tensor(self.noise_bias).unsqueeze(0).unsqueeze(0)
        else: 
            raise ValueError(f"noise_bias must be a float or a dict, got {type(noise_bias)}")

    def to(self, device, *args, **kwargs):
        if isinstance(self.noise_std, torch.Tensor): 
            self.noise_std = self.noise_std.to(device)
        if isinstance(self.noise_bias, torch.Tensor): 
            self.noise_bias = self.noise_bias.to(device)
        return super().to(device, *args)
    

    def forward(self, batch):
        # # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        # latents = latents.permute(1, 0, 2)
        latents = batch[self.input_key]

        # # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        if self.noise_key not in batch.keys(): 
            if self.cfg.get('debug_zero_noise', False):
                noise = torch.zeros_like(latents)
            else:
                noise = torch.randn_like(latents) * self.noise_std

            if self.cfg.get('debug_zero_out_latent', False):
                latents = latents * 0
                batch[self.input_key] = latents
        else: 
            print("[WARNING] noise passed into the diffuser instead of being sampled. This should only be done at test time.")
            noise = batch[self.noise_key]
        B = latents.shape[0]
        T = latents.shape[1] 

        if self.diffusion_steps_key not in batch.keys():
            # Sample a random timestep for each motion
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B, ),
                device=latents.device,
            )
            timesteps = timesteps.long()
            batch[self.diffusion_steps_key] = timesteps
        else: 
            print("[WARNING] timesteps passed into the diffuser instead of being sampled. This should only be done at test time.")
            timesteps = batch[self.diffusion_steps_key]
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        batch[self.noise_key] = latents - noisy_latents
        # print(f"Timestep \t\t{timesteps.item()}")
        # print(f"Added noise norm \t{batch[self.noise_key].norm():4f}")
        batch[self.output_key] = noisy_latents
        return batch

    def get_init_noise(self, like_what=None, shape=None, device=None):
        assert not ((like_what is None and shape is None) or (like_what is not None and shape is not None))
        if shape is not None:
            noise = torch.randn(*shape, device=device) * self.noise_std
        else:
            noise =  torch.randn_like(like_what) * self.noise_std
        noise = noise * self.scheduler.init_noise_sigma
        noise = noise + self.noise_bias
        assert self.scheduler.init_noise_sigma == 1., f"Sanity checking the noise sigma from diffusers. Should be 1 but is {self.scheduler.init_noise_sigma}"
        return noise

    def get_training_init_noise(self, like_what=None, shape=None, device=None):
        assert not ((like_what is None and shape is None) or (like_what is not None and shape is not None))
        if shape is not None:
            noise = torch.randn(*shape, device=device)  * self.noise_std
        else:
            noise =  torch.randn_like(like_what) * self.noise_std
        noise = noise * self.noise_scheduler.init_noise_sigma
        noise = noise + self.noise_bias
        assert self.noise_scheduler.init_noise_sigma == 1., f"Sanity checking the noise sigma from diffusers Should be 1 but is {self.noise_scheduler.init_noise_sigma}"
        return noise


    def get_testing_timesteps(self, D): 
        self.scheduler.set_timesteps(D)
        timesteps = self.scheduler.timesteps
        return timesteps
    
    def get_training_timesteps(self, D): 
        self.noise_scheduler.set_timesteps(D)
        timesteps = self.noise_scheduler.timesteps
        return timesteps

    def get_num_training_timesteps(self): 
        return self.noise_scheduler.timesteps.numel()
    
    def get_training_sqrt_alphas_cumprod(self, timesteps):
        return self.noise_scheduler.alphas_cumprod.to(device=timesteps.device)[timesteps] ** 0.5


    def compute_eps_from_x0_xt(self, x0, xt, timesteps, training):
        """ 
        Given x_t and x_0, compute the noise at step t. 
        """
        scheduler = self.noise_scheduler if training else self.scheduler
        # noise = x0 - xt
        alphas_cumprod = scheduler.alphas_cumprod.to(dtype=x0.dtype, device=x0.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # scaled_noise = noise / sqrt_one_minus_alpha_prod 

        eps = (xt - (sqrt_alpha_prod * x0) ) / sqrt_one_minus_alpha_prod 
        return eps

    

    def compute_xt_1_next_from_x0(self, noisy_vector, denoised_vector, timesteps, training):
        """
        Computes x_t-1 from x_t and x_0. 
        noisy_vector ... x_t
        denoised_vector ... x_0 
        timestep ... t
        """
        scheduler = self.noise_scheduler if training else self.scheduler
        

        eps = self.compute_eps_from_x0_xt(denoised_vector, noisy_vector, timesteps, training)
        noise = noisy_vector - denoised_vector

        ## compute x_t-1 
        
        ## use the scheduler to get the noise scale at step t, step 0, and step t-1
        alphas_cumprod = scheduler.alphas_cumprod.to(dtype=noisy_vector.dtype, device=noisy_vector.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # ## This is how the noise should be scaled
        # scaled_noise = noise / sqrt_one_minus_alpha_prod 
        # # scaled_noise2 = noise * sqrt_one_minus_alpha_prod
        # # scaled_noise3 = (noisy_vector - (sqrt_alpha_prod * denoised_vector)) / sqrt_one_minus_alpha_prod

        ## 
        # print(f"Timestep {timesteps.item()}")
        # print(f"Noise norm \t {noise.norm().item():.4f}")
        # print(f"Noise norm2 \t {scaled_noise.norm().item():.4f}")
        # print(f"Noise norm3 \t {scaled_noise2.norm().item():.4f}")
        # print(f"Noise norm4 \t {scaled_noise3.norm().item():.4f}")

        next_timesteps = timesteps - 1
        # current_timestep_indices = (scheduler.timesteps.cuda() == timesteps).nonzero()
        # next_timesteps_indices = current_timestep_indices + 1 

        scheduler_steps = scheduler.timesteps.to(eps.device)

        num_timesteps = scheduler_steps.shape[0]

        matches = (scheduler_steps == timesteps.unsqueeze(1))
        next_timesteps_indices = matches.int().argmax(dim=1) + 1

        next_in_bounds = next_timesteps_indices <  num_timesteps

        next_timesteps[next_in_bounds] = scheduler_steps[next_timesteps_indices.clip(0, num_timesteps - 1)] 
        next_timesteps[~next_in_bounds] = 0

        # next_noisy_vector2 = scheduler.add_noise(denoised_vector, scaled_noise,
        #                                                next_timesteps)

        next_alpha_prod = torch.zeros(size = timesteps.shape, dtype=sqrt_one_minus_alpha_prod.dtype, device=sqrt_one_minus_alpha_prod.device )
        final_timesteps = next_timesteps < 0
        # if next_timesteps < 0: 
        if hasattr(scheduler, 'final_alpha_cumprod'):
            next_alpha_prod[final_timesteps] = scheduler.final_alpha_cumprod 
        else: 
            next_alpha_prod[final_timesteps] = scheduler.one
        # else:
            next_alpha_prod[~final_timesteps] = alphas_cumprod[next_timesteps] 
        next_sqrt_alpha_prod = next_alpha_prod ** 0.5
        next_sqrt_alpha_prod = next_sqrt_alpha_prod.flatten()

        next_sqrt_one_minus_alpha_prod = (1 - next_alpha_prod) ** 0.5

        # next_noisy_vector_ = next_sqrt_alpha_prod * denoised_vector + next_sqrt_one_minus_alpha_prod * scaled_noise 
        # next_noisy_vector_sanity = sqrt_alpha_prod * denoised_vector + sqrt_one_minus_alpha_prod * scaled_noise 
        next_noisy_vector = next_sqrt_alpha_prod * denoised_vector + next_sqrt_one_minus_alpha_prod * eps 
        ## sanity check next_noisy_vector and noisy_vector_sanity
        # noisy_vector_sanity = sqrt_alpha_prod * denoised_vector + sqrt_one_minus_alpha_prod * eps 
        return next_noisy_vector


    def apply_predicted_noise(self, diffusion_step, noisy_vector, predicted_noise, training): 
        """
        Apply the predicted noise to the noisy vector, either during training or testing.
        """
        scheduler = self.noise_scheduler if training else self.scheduler
        # scheduler = self.noise_scheduler 
        
        if training:
            ## here we don't need to use the scheduler to step the noise
            denoised_vector = noisy_vector - predicted_noise
        else:
            ## use the scheduler to step the noise
            extra_step_kwargs = {}
            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, and between [0, 1]
            if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
                extra_step_kwargs["eta"] = self.cfg.scheduler.eta
            denoised_vector = scheduler.step(predicted_noise, diffusion_step.item(), noisy_vector,
                                        **extra_step_kwargs).prev_sample
        return denoised_vector


    # def testing_step(self, d, noisy_vector, predicted_noise): 
    #     extra_step_kwargs = {}
    #     # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    #     # eta (η) is only used with the DDIMScheduler, and between [0, 1]
    #     if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
    #         extra_step_kwargs["eta"] = self.cfg.scheduler.eta
    #     denoised_vector = self.scheduler.step(predicted_noise, d, noisy_vector,
    #                                  **extra_step_kwargs).prev_sample
    #     return denoised_vector

    def num_training_diffusion_steps(self):
        return self.cfg.noise_scheduler.params.num_train_timesteps
    
    def num_testing_diffusion_steps(self):
        return self.cfg.scheduler.num_inference_timesteps

        # # Predict the noise residual
        # noise_pred = self.denoiser(
        #     sample=noisy_latents,
        #     timestep=timesteps,
        #     encoder_hidden_states=encoder_hidden_states,
        #     lengths=lengths,
        #     return_dict=False,
        # )[0]
        # # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
        # if self.cfg.LOSS.LAMBDA_PRIOR != 0.0:
        #     noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
        #     noise, noise_prior = torch.chunk(noise, 2, dim=0)
        # else:
        #     noise_pred_prior = 0
        #     noise_prior = 0
        # n_set = {
        #     "noise": noise,
        #     "noise_prior": noise_prior,
        #     "noise_pred": noise_pred,
        #     "noise_pred_prior": noise_pred_prior,
        # }
        # if not self.predict_epsilon:
        #     n_set["pred"] = noise_pred
        #     n_set["latent"] = latents

