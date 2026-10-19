import torch
from inferno.models.temporal.Bases import TemporalAudioEncoder


class PrecomputedAudioFeatureEncoder(TemporalAudioEncoder): 

    def __init__(self, dim, silence_token_init):
        super().__init__()
        self.dim = dim 
        # self.trainable = trainable
        self.silence_token_init = silence_token_init
        ## add learnable tensor for that will be added to the audio frames that are masked 
        ## (accounting for when the audio loaded is shorter than the sequence of the input)
        self.silence_embedding = torch.nn.Parameter(torch.randn(dim)) 
        if self.silence_token_init == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.silence_embedding)
        elif self.silence_token_init in ["zeros", "zero"]:
            torch.nn.init.zeros_(self.silence_embedding)
        else: 
            raise NotImplementedError(f"Silence token init {self.silence_token_init} not supported")

    def forward(self, sample, train=False, desired_output_length=None, **kwargs):
        audio_features = sample["audio_feature"]
        audio_mask = sample["audio_mask"] 
       
        B, T = audio_features.size(0), audio_features.size(1)
        silent_embedding = self.silence_embedding.unsqueeze(0).repeat(B, T, 1)
        silent_embedding = audio_mask.float()[:,:T, None] * silent_embedding
        # silent_embedding = audio_mask.float()[..., None] * silent_embedding
        sample["audio_feature"] = audio_features + silent_embedding
        return sample

    def output_feature_dim(self):
        return self.dim

    def get_trainable_parameters(self): 
        return [self.silence_embedding]