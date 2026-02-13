# Use open_clip adapter (supports OpenCLIP and SigLIP2).
from .open_clip_adapter import load, tokenize, CLIPAdapter, available_models

__all__ = ["load", "tokenize", "CLIPAdapter", "available_models"]
