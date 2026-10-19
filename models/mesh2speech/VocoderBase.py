import torch


class Mesh2SpeechVocoderBase(torch.nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, batch, train=False, validation=False, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")


