"""
# -----------------------------------------------------------------------------
# Copyright (c) 2024 Max Planck Institute for Intelligent Systems
# Author: Radek Danecek
#
# All rights reserved.
#
# This file is part of project THUNDER and is distributed under the terms and 
# conditions defined in the LICENSE file, which is part of this source code package.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions, and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the Max Planck Institute for Intelligent Systems nor the
#    names of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------
This file was copied over from TalkingHeadBase.py from the public inferno library and then modified
"""



import pytorch_lightning as pl 
from typing import Any, Optional
from inferno.models.temporal.AudioEncoders import TemporalAudioEncoder
from inferno.models.temporal.Bases import *
from inferno.models.talkinghead.TalkingHeadBase import TalkingHeadBase, truncate_sequence_batch
from .text.TextBase import TextEncoderBase
# from .text.TextEmbeddingEncoder import TextEmbeddingEncoder
# from inferno.models.temporal.BlockFactory import norm_from_cfg
import random
import omegaconf
from inferno.utils.batch import check_nan, detach_dict


class TalkingHeadWithLanguageBase(TalkingHeadBase): 

    def __init__(self, 
                cfg,
                audio_model: Optional[TemporalAudioEncoder] = None, 
                transcript_model: Optional[TextEncoderBase] = None, 
                audio_transcript_fusion: Optional[torch.nn.Module] = None, 
                sequence_encoder : SequenceEncoder = None,
                sequence_decoder : Optional[SequenceDecoder] = None,
                shape_model: Optional[ShapeModel] = None,
                preprocessor: Optional[Preprocessor] = None,
                renderer: Optional[Renderer] = None,
                # post_fusion_norm: Optional = None,
                *args: Any, **kwargs: Any) -> None:
        super().__init__(cfg, 
                         audio_model, 
                         sequence_encoder,
                         sequence_decoder,
                         shape_model,
                         preprocessor,
                         renderer,
                         *args, 
                         **kwargs
                         )  
        self.transcript_model = transcript_model
        self.audio_transcript_fusion = audio_transcript_fusion

    @property
    def max_seq_length(self):
        return 5000

    def get_trainable_parameters(self):
        trainable_params = super().get_trainable_parameters()
        if self.transcript_model is not None:
            trainable_params += self.transcript_model.get_trainable_parameters()
        return trainable_params

    def forward_transcript(self, sample: Dict, train=False, desired_output_length=None, **kwargs: Any) -> Dict:
        return self.transcript_model(sample, train=train, desired_output_length=desired_output_length, **kwargs)

    def audio_transcript_fusion_forward(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        return self.audio_transcript_fusion(sample, train=train, **kwargs)


    def encode_sequence(self, sample: Dict, train=False, **kwargs: Any) -> Dict:
        input_key = "audio_transcript_fused"
        if self.audio_transcript_fusion is not None: 
            input_key = "audio_transcript_fused"
        elif self.audio_model is not None:
            input_key = "audio_feature"
        elif self.transcript_model is not None:
            input_key = "transcript_word_embeddings_time_synced"
        else: 
            raise RuntimeError("This should not happen")
        return self.sequence_encoder(sample, input_key=input_key, **kwargs)

    def forward(self, sample: Dict, train=False, validation=False, **kwargs: Any) -> Dict:
        """
        sample: Dict[str, torch.Tensor]
            - audio: (B, T, F)
            # - masked_audio: (B, T, F)
        """
        # T = sample["raw_audio"].shape[1]
        if "audio_feature" in sample.keys():
            T = sample["audio_feature"].shape[1]
        else:
            T = sample["processed_audio"].shape[1] if "processed_audio" in sample.keys() else sample["raw_audio"].shape[1]
        if self.max_seq_length < T: # truncate
            print("[WARNING] Truncating audio sequence from {} to {}".format(T, self.max_seq_length))
            sample = truncate_sequence_batch(sample, self.max_seq_length)

        # preprocess input (for instance get 3D pseudo-GT )
        sample = self.preprocess_input(sample, train=train, **kwargs)
        check_nan(sample)
        
        ## render the GT (useful if we want to run perceptual features on GT and condition on them) 
        sample = self.render_gt_sequence(sample, train=train, **kwargs)

        sample = self.extract_gt_features(sample, train=train, **kwargs)

        sample = self._choose_primary_3D_rec_method(sample)

        teacher_forcing = kwargs.pop("teacher_forcing", False)
        desired_output_length = sample["gt_vertices"].shape[1] if "gt_vertices" in sample.keys() else None
        
        if self.audio_model is not None:
            sample = self.forward_audio(sample, train=train, desired_output_length=desired_output_length, **kwargs)
        
        if self.transcript_model is not None:
            sample = self.forward_transcript(sample, train=train, desired_output_length=desired_output_length, **kwargs)
        
        if self.audio_transcript_fusion is not None: 
            sample = self.audio_transcript_fusion_forward(sample, train=train, **kwargs)
        
        check_nan(sample)
        # encode the sequence
        sample = self.encode_sequence(sample, train=train, **kwargs)
        check_nan(sample)

        sample = self.disentangle(sample, train=train, validation=validation, **kwargs)
        check_nan(sample)

        # decode the sequence
        sample = self.decode_sequence(sample, train=train, teacher_forcing=teacher_forcing, **kwargs)
        check_nan(sample)

        ## render the sequence
        ## sample = self.render_sequence(sample, train=train, **kwargs)
        # render only the predicted sequence (the GT is already rendered)
        sample = self.render_predicted_sequence(sample, train=train, **kwargs)

        check_nan(sample)
        return sample

    def _compute_loss(self, sample, training, validation, loss_name, loss_cfg): 
        raise NotImplementedError("Please implement this method in your child class")

