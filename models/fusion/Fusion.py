import torch 


class FusionBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch, *args, **kwargs):
        raise NotImplementedError("Sub-classes must implement this")
    
    def output_feature_dim(self, *args, **kwargs):
        raise NotImplementedError("Sub-classes must implement this")


class ConcatFusion(FusionBase):
    def __init__(self, input_keys, output_key):
        super().__init__()
        self.input_keys = input_keys
        self.output_key = output_key

    def forward(self, batch, *args, **kwargs):
        tensors_to_concat = [batch[key] for key in self.input_keys]
        batch[self.output_key] = torch.cat(tensors_to_concat, dim=-1)
        return batch
    
    def output_feature_dim(self, modality_dim_1, modality_dim_2):
        return modality_dim_1 + modality_dim_2