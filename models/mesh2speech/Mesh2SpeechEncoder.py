import torch

class Mesh2SpeechEncoderBase(torch.nn.Module):  

    def __init__(self) -> None:
        super().__init__()


    def get_trainable_parameters(self):
        raise NotImplementedError("get_trainable_parameters not implemented")
    


