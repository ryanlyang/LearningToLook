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
    Load a DINOv1 model by downloading weights directly

    This avoids torch.hub import conflicts by loading weights into a model
    we construct ourselves.

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

    # Try to import timm's vision transformer as a fallback
    try:
        import timm
        print(f"Using timm to load DINO model", file=sys.stderr, flush=True)

        # Map DINO model names to timm equivalents
        timm_name_map = {
            'dino_vits16': 'vit_small_patch16_224.dino',
            'dino_vits8': 'vit_small_patch8_224.dino',
            'dino_vitb16': 'vit_base_patch16_224.dino',
            'dino_vitb8': 'vit_base_patch8_224.dino',
        }

        if model_name in timm_name_map:
            print(f"Creating timm model: {timm_name_map[model_name]}", file=sys.stderr, flush=True)
            model = timm.create_model(
                timm_name_map[model_name],
                pretrained=pretrained,
                dynamic_img_size=True  # Allow variable input sizes
            )

            # Patch the patch_embed forward to remove strict size checking
            if hasattr(model, 'patch_embed'):
                original_forward = model.patch_embed.forward

                def patched_forward(x):
                    # Store original settings
                    B, C, H, W = x.shape
                    # Call original but catch and ignore size assertions
                    try:
                        return original_forward(x)
                    except AssertionError as e:
                        if "divisible by patch size" in str(e) or "doesn't match model" in str(e):
                            # Manually do the patching without strict checks
                            x = model.patch_embed.proj(x)
                            if model.patch_embed.flatten:
                                x = x.flatten(2).transpose(1, 2)
                            x = model.patch_embed.norm(x) if model.patch_embed.norm is not None else x
                            return x
                        else:
                            raise

                model.patch_embed.forward = patched_forward
                print(f"Patched patch_embed to allow flexible image sizes", file=sys.stderr, flush=True)

            # Disable dynamic positional embedding to use interpolation like original DINO
            # Set the model to not use strict pos embed
            if hasattr(model, 'pos_embed'):
                model.pos_embed.requires_grad = False
                print(f"Set pos_embed to non-trainable for interpolation", file=sys.stderr, flush=True)

            # Verify model has expected methods
            if not hasattr(model, 'forward_features'):
                print(f"Warning: timm model lacks forward_features method, falling back to torch.hub", file=sys.stderr, flush=True)
                raise AttributeError("Model missing forward_features")

            print(f"Successfully loaded {model_name} via timm with forward_features method", file=sys.stderr, flush=True)
            return model
        else:
            print(f"Warning: {model_name} not available in timm, falling back to torch.hub", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"timm not available: {e}, falling back to torch.hub", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"Error loading via timm: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print("Falling back to torch.hub", file=sys.stderr, flush=True)

    # Fallback to torch.hub if timm fails
    # Clear the cached DINO repo to avoid import conflicts
    cache_dir = os.path.join(torch.hub.get_dir(), 'facebookresearch_dino_main')
    if os.path.exists(cache_dir):
        print(f"Clearing cached DINO repository at {cache_dir}", file=sys.stderr, flush=True)
        try:
            shutil.rmtree(cache_dir)
            print("Cache cleared successfully", file=sys.stderr, flush=True)
            time.sleep(0.5)
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}", file=sys.stderr, flush=True)

    print("Loading model with torch.hub (force_reload=True)...", file=sys.stderr, flush=True)
    model = torch.hub.load('facebookresearch/dino:main', model_name,
                          pretrained=pretrained,
                          trust_repo=True,
                          force_reload=True,
                          skip_validation=True)

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
