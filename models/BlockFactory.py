from pathlib import Path 
from .text.TextEmbeddingEncoder import (TextEmbeddingEncoder, FrameSyncedTextEmbeddingEncoder,
                                        PrecomputedTextEmbeddingEncoder)
from .text.TextEmbeddingEncoderV2 import FrameSyncedTextEmbeddingMapper


def transcript_model_from_cfg(cfg):
    if cfg.type == "none":
        return None
    if cfg.type == "bert_frame_synced": 
        encoder = FrameSyncedTextEmbeddingEncoder(cfg.silence_token_init, cfg.model_specifier, cfg.trainable)
    elif cfg.type == "precomputed_embeddings_enc": 
        encoder = PrecomputedTextEmbeddingEncoder(cfg.dim, cfg.silence_token_init)
    elif cfg.type == "precomputed_embeddings_not_synced": 
        encoder = FrameSyncedTextEmbeddingMapper(cfg.silence_token_init, cfg.model_specifier, cfg.trainable)
    elif cfg.type == "bert": 
        encoder = TextEmbeddingEncoder(cfg.model_specifier, cfg.trainable)
    else: 
        raise ValueError(f"Unknown audio model type '{cfg.type}'")
    return encoder


def fusion_model_from_cfg(cfg): 
    if cfg.type == "none":
        return None
    elif cfg.type == "concat": 
        from .fusion.Fusion import ConcatFusion
        return ConcatFusion(cfg.input_keys, cfg.output_key)
    else: 
        raise ValueError(f"Unknown fusion model type '{cfg.type}'")