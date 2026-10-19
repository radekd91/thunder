import torch

class Mesh2SpeechDiscriminatorBase(torch.nn.Module): 

    def __init__(self) -> None:
        super().__init__()


    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]


# try:
from .external.visual_context_attentional_gan.src.models.generator import Discriminator, Avgpool, ResBlk #, sync_Discriminator

class VcaDiscriminator(Mesh2SpeechDiscriminatorBase):
    
    def __init__(self, cfg): 
        super().__init__()
        self.cfg = cfg
        self.discriminator = Discriminator(num_class=cfg.num_classes, max_conv_dim=cfg.max_conv_dim, phase=cfg.phase)

    @property
    def out_keys(self):
        return f"discrimintor_{self.cfg.phase}" 
    
    def forward(self, batch, train=False, validation=False, input_prefix="", output_prefix="", detach_inputs=False, **kwargs):
        mel = batch[f"{input_prefix}_mel"]
        if detach_inputs:
            mel = mel.detach()

        if self.cfg.phase == '2':
            mel = torch.nn.functional.interpolate(mel, scale_factor=0.5, mode='bilinear')
        elif self.cfg.phase == '3':
            mel = torch.nn.functional.interpolate(mel, scale_factor=0.25, mode='bilinear')
        
        ## B,T,F -> B,F,T 
        x = mel.permute(0,2,1)

        raise NotImplementedError("Need to implement condition")
        uout, cout = self.discriminator(x, condition, vid_max_length)

        batch[f"{output_prefix}_{self.out_keys}_uncond"] = uout
        batch[f"{output_prefix}_{self.out_keys}_cond"] = cout
        return uout, cout


# class VcaSyncDiscriminator(Mesh2SpeechDiscriminatorBase):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.discriminator = sync_Discriminator(temp=cfg.temp)


#     def forward(self, batch, train=False, validation=False, **kwargs):

#         v_feat = batch["video_features"]
#         aud = batch["audio_features"]

#         gen = False

#         loss = self.discriminator(v_feat, aud, gen)

#         return loss


class UnconditionedDiscriminator(torch.nn.Module):

    def __init__(self, num_class=1, max_conv_dim=512, phase='1'):
        super().__init__()
        dim_in = 32
        blocks = []
        blocks += [torch.nn.Conv2d(1, dim_in, 5, 1, 2)]

        if phase == '1':
            repeat_num = 2
        elif phase == '2':
            repeat_num = 3
        else:
            repeat_num = 4

        for _ in range(repeat_num): # 80,4T --> 40,2T --> 20,T --> 10,T/2 --> 5,T/4
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        self.main = torch.nn.Sequential(*blocks)

        uncond = []
        uncond += [torch.nn.LeakyReLU(0.2)]
        uncond += [torch.nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        uncond += [torch.nn.LeakyReLU(0.2)]
        uncond += [Avgpool()]
        uncond += [torch.nn.Linear(dim_out, num_class)]
        self.uncond = torch.nn.Sequential(*uncond)

    def forward(self, x):
        out = self.main(x).clone()
        uout = self.uncond(out)
        uout = uout.view(uout.size(0), -1)  # (batch, num_domains)
        return uout


class VcaUnconditionedDiscriminator(Mesh2SpeechDiscriminatorBase):
    
    def __init__(self, cfg): 
        super().__init__()
        self.cfg = cfg
        self.discriminator = UnconditionedDiscriminator(num_class=cfg.num_classes, max_conv_dim=cfg.max_conv_dim, phase=cfg.phase)

    @property
    def out_keys(self):
        return f"p{self.cfg.phase}" 

    @property
    def spectral_downsample(self):
        return self.cfg.get('spectral_downsample', True)

    def forward(self, batch, train=False, validation=False, input_prefix="", output_prefix="", detach_inputs=False, **kwargs):
        mel = batch[f"{input_prefix}_mel"]
        if detach_inputs:
            mel = mel.detach()

        ## B,T,F -> B,F,T 
        mel = mel.permute(0,2,1).unsqueeze(1)
        
        if self.cfg.phase == '2':
            mel = torch.nn.functional.interpolate(mel, scale_factor=(1. if not self.spectral_downsample else 0.5, 0.5), mode='bilinear')
        elif self.cfg.phase == '3':
            mel = torch.nn.functional.interpolate(mel, scale_factor=(1. if not self.spectral_downsample else 0.25, 0.25), mode='bilinear')

        uout = self.discriminator(mel)
        batch[f"{output_prefix}_{self.out_keys}"] = uout
        return batch
        

# except ImportError:
#     print('[WARNING] VCA-GAN repo not found. Will not be imported.')




