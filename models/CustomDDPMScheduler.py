import torch
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler, rescale_zero_terminal_snr

class CustomDDPMScheduler(DDPMScheduler):
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "cosine",
        s: float = 0.008,
        rescale_betas_zero_snr: bool = False,
        **kwargs
    ):
        # Call the base constructor
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear",  # Placeholder, we override below
            rescale_betas_zero_snr=False,  # Also handled below
            **kwargs
        )

        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'quadratic':
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        elif beta_schedule == 'sigmoid':
            betas = torch.sigmoid(torch.linspace(-5, 5, num_train_timesteps)) * (beta_end - beta_start) + beta_start
        elif beta_schedule == 'cosine':
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alpha_bars = torch.cos(((x / num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clamp(betas, 0.0001, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        # Optionally pad for compatibility with custom schedulers (e.g., other project's implementation)
        betas = torch.cat([torch.zeros(1), betas], dim=0)

        if rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)

        # Overwrite internal attributes
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.init_noise_sigma = 1.0
