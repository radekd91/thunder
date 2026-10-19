import os 
import torch

class Mesh4DecoderBase(torch.nn.Module): 

    def __init__(self) -> None:
        super().__init__()


    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]
    