"""
CLIP backend switcher.

Set CLIP_BACKEND to "openai" to use the original OpenAI CLIP implementation.
Set CLIP_BACKEND to "openclip" for OpenCLIP models.
Set CLIP_BACKEND to "siglip2" for SigLIP2 models via open_clip.
Default is OpenCLIP ("openclip").
"""
import os

_backend = os.environ.get("CLIP_BACKEND", "openclip").strip().lower()

if _backend in {"openai", "clip"}:
    from .clip import load, tokenize, available_models
    CLIPAdapter = None
    __all__ = ["load", "tokenize", "available_models"]
elif _backend in {"openclip", "siglip2"}:
    from .open_clip_adapter import load, tokenize, CLIPAdapter
    __all__ = ["load", "tokenize", "CLIPAdapter"]
else:
    raise ValueError(
        f"Unknown CLIP_BACKEND='{_backend}'. "
        "Expected one of: openai, openclip, siglip2."
    )
