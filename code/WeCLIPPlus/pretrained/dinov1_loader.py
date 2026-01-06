"""
DINOv1 Model Loader
===================
This module provides loading functions for the original DINO (v1) models
from Facebook AI Research.

DINOv1 was the original self-supervised vision transformer model.
DINOv2 is the improved version with better performance.
"""

import torch
import torch.hub
import os
import shutil
import time

# Available DINOv1 models from Facebook Research
DINOV1_MODELS = {
    'dino_vits16': 'dino_vits16',
    'dino_vits8': 'dino_vits8',
    'dino_vitb16': 'dino_vitb16',
    'dino_vitb8': 'dino_vitb8',
    'dino_resnet50': 'dino_resnet50',
}

# Feature dimensions for each model
DINOV1_DIMS = {
    'dino_vits16': 384,   # ViT-Small with patch size 16
    'dino_vits8': 384,    # ViT-Small with patch size 8
    'dino_vitb16': 768,   # ViT-Base with patch size 16
    'dino_vitb8': 768,    # ViT-Base with patch size 8
    'dino_resnet50': 2048, # ResNet50
}

# Patch sizes for each model
DINOV1_PATCH_SIZES = {
    'dino_vits16': 16,
    'dino_vits8': 8,
    'dino_vitb16': 16,
    'dino_vitb8': 8,
    'dino_resnet50': None,  # ResNet doesn't use patches
}


def load_dinov1_model(model_name, pretrained=True):
    """
    Load a DINOv1 model using torch.hub (original Facebook implementation)

    The original DINO implementation handles variable image sizes natively
    through positional embedding interpolation, unlike timm's implementation.

    Args:
        model_name: Name of the DINO model (e.g., 'dino_vits16', 'dino_vitb16')
        pretrained: Whether to load pretrained weights

    Returns:
        DINOv1 model
    """
    if model_name not in DINOV1_MODELS:
        available = ', '.join(DINOV1_MODELS.keys())
        raise ValueError(f"Unknown DINOv1 model: {model_name}. Available models: {available}")

    import sys
    print(f"Loading DINOv1 model: {model_name}", file=sys.stderr, flush=True)
    print(f"Using torch.hub to load original DINO implementation", file=sys.stderr, flush=True)

    # Use torch.hub to load the original Facebook DINO implementation
    # The cache should already be patched by fix_dino_utils.py
    print("Loading model with torch.hub...", file=sys.stderr, flush=True)

    try:
        model = torch.hub.load('facebookresearch/dino:main', model_name,
                              pretrained=pretrained,
                              trust_repo=True,
                              force_reload=False,  # Don't force reload if cache is already patched
                              skip_validation=True)
        print(f"Successfully loaded {model_name} via torch.hub", file=sys.stderr, flush=True)
        return model
    except Exception as e:
        print(f"Error loading via torch.hub: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

        # If it fails, try clearing cache and reloading
        cache_dir = os.path.join(torch.hub.get_dir(), 'facebookresearch_dino_main')
        if os.path.exists(cache_dir):
            print(f"Clearing cached DINO repository at {cache_dir} and retrying...", file=sys.stderr, flush=True)
            try:
                shutil.rmtree(cache_dir)
                print("Cache cleared, reloading...", file=sys.stderr, flush=True)
                time.sleep(0.5)
            except Exception as clear_error:
                print(f"Warning: Could not clear cache: {clear_error}", file=sys.stderr, flush=True)

        # Retry with force_reload
        print("Retrying with force_reload=True...", file=sys.stderr, flush=True)
        model = torch.hub.load('facebookresearch/dino:main', model_name,
                              pretrained=pretrained,
                              trust_repo=True,
                              force_reload=True,
                              skip_validation=True)
        print(f"Successfully loaded {model_name} on retry", file=sys.stderr, flush=True)
        return model


def dino_vits16(pretrained=True):
    """Load DINO ViT-Small with patch size 16"""
    return load_dinov1_model('dino_vits16', pretrained=pretrained)


def dino_vits8(pretrained=True):
    """Load DINO ViT-Small with patch size 8"""
    return load_dinov1_model('dino_vits8', pretrained=pretrained)


def dino_vitb16(pretrained=True):
    """Load DINO ViT-Base with patch size 16"""
    return load_dinov1_model('dino_vitb16', pretrained=pretrained)


def dino_vitb8(pretrained=True):
    """Load DINO ViT-Base with patch size 8"""
    return load_dinov1_model('dino_vitb8', pretrained=pretrained)


def dino_resnet50(pretrained=True):
    """Load DINO ResNet50"""
    return load_dinov1_model('dino_resnet50', pretrained=pretrained)


def get_dinov1_feature_dim(model_name):
    """Get the feature dimension for a DINOv1 model"""
    if model_name not in DINOV1_DIMS:
        raise ValueError(f"Unknown DINOv1 model: {model_name}")
    return DINOV1_DIMS[model_name]


def get_dinov1_patch_size(model_name):
    """Get the patch size for a DINOv1 model"""
    if model_name not in DINOV1_PATCH_SIZES:
        raise ValueError(f"Unknown DINOv1 model: {model_name}")
    return DINOV1_PATCH_SIZES[model_name]


# For backward compatibility
__all__ = [
    'dino_vits16',
    'dino_vits8',
    'dino_vitb16',
    'dino_vitb8',
    'dino_resnet50',
    'load_dinov1_model',
    'get_dinov1_feature_dim',
    'get_dinov1_patch_size',
    'DINOV1_MODELS',
    'DINOV1_DIMS',
    'DINOV1_PATCH_SIZES',
]
