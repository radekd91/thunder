from .PreprocessorBase import PreprocessorBase
import numpy as np
import torch
from pathlib import Path
from inferno.utils.other import get_path_to_assets
from inferno.models.temporal.Renderers import cut_mouth_vectorized
from inferno.layers.losses.MediaPipeLandmarkLosses import *

class VideoMouthCropPreprocessor(PreprocessorBase):

    def __init__(self, 
                mouth_window_margin, 
                landmark_source,
                mouth_crop_height, 
                mouth_crop_width,
                convert_grayscale=True,
    ):
        self.mouth_window_margin = mouth_window_margin
        self.landmark_source = landmark_source
        if self.landmark_source in ["fan3d", "fan"]:
            self.mouth_landmark_start_idx = 48
            self.mouth_landmark_stop_idx = 68
            self.mouth_indices = np.arange(self.mouth_landmark_start_idx, self.mouth_landmark_stop_idx, dtype=np.int32)
        elif self.landmark_source == "mediapipe":
            self.mouth_landmark_start_idx = None
            self.mouth_landmark_stop_idx = None
            self.mouth_indices = np.array(sorted(UPPER_INNER_LIP_LINE + LOWER_INNER_LIP_LINE + UPPER_OUTTER_LIP_LINE + LOWER_OUTTER_LIP_LINE))
        else:
            raise ValueError(f"landmark_source should be either 'fan3d' or 'fan', but got {self.landmark_source}")
        self.mouth_crop_height = mouth_crop_height
        self.mouth_crop_width = mouth_crop_width
        self.convert_grayscale = convert_grayscale


    def forward(self, batch, train=False, validation=False, **kwargs):
        images = batch["video"]
        if self.landmark_source in batch["landmarks"]:
            landmarks = batch["landmarks"][self.landmark_source]
        else: 
            raise ValueError(f"FAN landmarks not found in batch. Instead we have only {list(batch['landmarks'].keys())}")

        mouth_crops = cut_mouth_vectorized(
            images, 
            landmarks, 
            mouth_window_margin = self.mouth_window_margin, 
            mouth_landmark_start_idx = self.mouth_landmark_start_idx, 
            mouth_landmark_stop_idx = self.mouth_landmark_stop_idx,
            mouth_landmark_indices = self.mouth_indices,
            mouth_crop_height = self.mouth_crop_height, 
            mouth_crop_width = self.mouth_crop_width,
            convert_grayscale = self.convert_grayscale
            )
        del images 
        del batch["video"]
        batch["video_mouth_crop"] = mouth_crops

        # ## debug visualizations
        # import plotly.graph_objects as go
        # # plot the original images with landmarks 
        # im = images[0][0].cpu().numpy()
        # im = (im * 255.).astype(np.uint8)        
        # # CHW -> HWC
        # im = im.transpose(1,2,0)
        # landmarksd = (landmarks[0][0].cpu().numpy() + 1. ) * 0.5 * im.shape[0]

        # fig = go.Figure(data=go.Image(z=im))
        # # fig.add_trace()
        # fig.add_trace(go.Scatter(x=landmarksd[..., 0], y=landmarksd[..., 1], mode='markers', marker=dict(size=5)))
        # fig.show()

        # # ## plot the cropped images
        # mouth_crop = mouth_crops[0].cpu().numpy()
        # ## T,C,H,W -> T,H,W,C
        # mouth_crop = mouth_crop.transpose(0,2,3,1)
        # ## T,H,W,C -> T*H,W,C
        # T = mouth_crop.shape[0]
        # HC = mouth_crop.shape[1]
        # mouth_crop = mouth_crop.reshape(-1, mouth_crop.shape[2], mouth_crop.shape[3])
        # mouth_crop = (mouth_crop * 255.).astype(np.uint8) 
        # # if grayscale, make rgb 
        # if mouth_crop.shape[-1] == 1:
        #     mouth_crop = np.repeat(mouth_crop, 3, axis=-1)
        # fig = go.Figure(go.Image(z=mouth_crop[:10*HC, ...]))
        # fig.show()
        return batch