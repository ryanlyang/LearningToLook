"""
CLIP backend switcher.

Set CLIP_BACKEND to "openai" to use the original OpenAI CLIP implementation.
Default is OpenCLIP ("openclip").
"""
import os

_backend = os.environ.get("CLIP_BACKEND", "openclip").strip().lower()

if _backend in {"openai", "clip"}:
    from .clip import load, tokenize, available_models
    CLIPAdapter = None
    __all__ = ["load", "tokenize", "available_models"]
else:
    # Default to OpenCLIP adapter
    from .open_clip_adapter import load, tokenize, CLIPAdapter
    __all__ = ["load", "tokenize", "CLIPAdapter"]
