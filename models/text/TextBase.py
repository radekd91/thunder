import torch


class TextEncoderBase(torch.nn.Module): 

    def __init__(self):
        super().__init__() 

    def forward(self, sample, train=False, desired_output_length=None, **kwargs): 
        raise NotImplementedError("Subclasses must implement this method")

    def get_trainable_parameters(self): 
        raise NotImplementedError()

    def output_feature_dim(self): 
        raise NotImplementedError()


