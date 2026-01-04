# Use open_clip adapter instead of original CLIP
from .open_clip_adapter import load, tokenize, CLIPAdapter

__all__ = ['load', 'tokenize', 'CLIPAdapter']
