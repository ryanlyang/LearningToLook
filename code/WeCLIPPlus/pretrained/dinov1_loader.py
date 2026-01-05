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
    Load a DINOv1 model from torch.hub

    Args:
        model_name: Name of the DINO model (e.g., 'dino_vits16', 'dino_vitb16')
        pretrained: Whether to load pretrained weights

    Returns:
        DINOv1 model
    """
    if model_name not in DINOV1_MODELS:
        available = ', '.join(DINOV1_MODELS.keys())
        raise ValueError(f"Unknown DINOv1 model: {model_name}. Available models: {available}")

    print(f"Loading DINOv1 model: {model_name}")

    # Load from torch.hub (Facebook Research DINO repository)
    # Use source='github' to avoid cached import conflicts
    model = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=pretrained,
                          source='github', trust_repo=True)

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
