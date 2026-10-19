from .PreprocessorBase import PreprocessorBase
import numpy as np
import torch
import joblib
from transformers import AutoModel, AutoConfig, Wav2Vec2FeatureExtractor, AutoFeatureExtractor
from pathlib import Path
from inferno.utils.other import get_path_to_assets


class ClusteringHuBERTPreprocessor(PreprocessorBase):

    def __init__(self, model_name, cluster_model_path, clustering_layer=6, use_preprocessor=False, **kwargs):
        super().__init__()
        self.model_name = model_name or "facebook/hubert-base-ls960"
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # model = LlavaPreTrainedModel.from_pretrained(model_name)
        # model = AutoModel.from_pretrained(model_name).to(device)
        # processor = Wav2Vec2Processor.from_pretrained(model_name)
        # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        self.cluster_model_path = cluster_model_path

        if not Path(cluster_model_path).is_absolute(): 
            self.cluster_model_path = get_path_to_assets() / cluster_model_path

        self.clustering_model = joblib.load(open(self.cluster_model_path, "rb"))
        self.clustering_model.verbose = False

        self.clustering_layer = clustering_layer
        self.use_preprocessor = use_preprocessor


    def forward(self, batch, train=False, validation=False, **kwargs):
        waveform = batch["raw_audio"] 
        B, T, F = waveform.shape[:3]
        waveform = waveform.view(B, T * F)
        if self.use_preprocessor:
            waveform = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt").input_values.to(waveform.device)[0]
        features = self.model(waveform, output_hidden_states=True)
        speech2unit_feats = features.hidden_states[self.clustering_layer] 
        T_feat = speech2unit_feats.shape[1]
        # feat_np = speech2unit_feats[0].detach().cpu().numpy().astype(np.float32)
        feat_np = speech2unit_feats.detach().view(B*T_feat, -1).cpu().numpy().astype(np.float32)
        pred = self.clustering_model.predict(feat_np)
        pred = pred.reshape(B, T_feat)
        batch["gt_units"] = torch.from_numpy(pred).to(torch.long).to(self.model.device)
        return batch

    def to(self, device):
        self.model = self.model.to(device)
        return self

    @property
    def device(self):
        return self.model.device
    
    
