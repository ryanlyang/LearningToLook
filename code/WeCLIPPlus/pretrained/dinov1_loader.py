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
    import subprocess
    print(f"Loading DINOv1 model: {model_name}", file=sys.stderr, flush=True)
    print(f"Using torch.hub to load original DINO implementation", file=sys.stderr, flush=True)

    # Use torch.hub to load the original Facebook DINO implementation
    print("Loading model with torch.hub...", file=sys.stderr, flush=True)

    try:
        model = torch.hub.load('facebookresearch/dino:main', model_name,
                              pretrained=pretrained,
                              trust_repo=True,
                              force_reload=False,
                              skip_validation=True)
        print(f"Successfully loaded {model_name} via torch.hub", file=sys.stderr, flush=True)
        return model
    except ImportError as e:
        if "trunc_normal_" in str(e):
            print(f"Detected trunc_normal_ import error - patching cache...", file=sys.stderr, flush=True)

            # Run the fix script to patch the cache
            cache_dir = os.path.join(torch.hub.get_dir(), 'facebookresearch_dino_main')
            if os.path.exists(cache_dir):
                utils_file = os.path.join(cache_dir, 'utils.py')

                # Add the missing function directly
                print(f"Patching {utils_file}...", file=sys.stderr, flush=True)
                with open(utils_file, 'r') as f:
                    content = f.read()

                if 'def trunc_normal_' not in content:
                    # Add the function at the beginning after imports
                    patch_code = '''import math
import warnings
import torch

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fills the input Tensor with values drawn from a truncated normal distribution."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

'''
                    lines = content.split('\n')
                    # Find where to insert (after imports)
                    import_end = 0
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if stripped and not stripped.startswith('import') and not stripped.startswith('from') and not stripped.startswith('#'):
                            import_end = i
                            break

                    lines.insert(import_end, patch_code)

                    with open(utils_file, 'w') as f:
                        f.write('\n'.join(lines))

                    print(f"Successfully patched utils.py", file=sys.stderr, flush=True)

                    # CRITICAL: Clear Python's import cache to pick up the patched file
                    # BUT DON'T clear torch.utils or other system modules!
                    print(f"Clearing DINO-specific Python import cache...", file=sys.stderr, flush=True)
                    import importlib
                    # Only remove modules from the DINO hub cache directory
                    # DO NOT remove torch.utils, numpy.utils, or other system utils
                    modules_to_clear = [key for key in sys.modules.keys()
                                       if ('vision_transformer' in key and 'torch' not in key) or
                                          ('dino' in key.lower() and 'torch' not in key)]
                    for mod in modules_to_clear:
                        print(f"Removing cached module: {mod}", file=sys.stderr, flush=True)
                        del sys.modules[mod]

                    # Also clear importlib cache
                    importlib.invalidate_caches()
                    print(f"DINO import cache cleared (torch.utils preserved)", file=sys.stderr, flush=True)

                    # Retry loading with force_reload to re-import everything
                    print("Retrying model load after patching...", file=sys.stderr, flush=True)
                    model = torch.hub.load('facebookresearch/dino:main', model_name,
                                          pretrained=pretrained,
                                          trust_repo=True,
                                          force_reload=True,  # Force reimport of hub modules
                                          skip_validation=True)
                    print(f"Successfully loaded {model_name} after patching", file=sys.stderr, flush=True)
                    return model
                else:
                    print(f"trunc_normal_ already exists in utils.py - skipping cache clear and reload", file=sys.stderr, flush=True)
                    # No need to clear cache or force reload - the patch already exists
                    # Just retry the original load without force_reload
                    print("Retrying model load without cache manipulation...", file=sys.stderr, flush=True)
                    model = torch.hub.load('facebookresearch/dino:main', model_name,
                                          pretrained=pretrained,
                                          trust_repo=True,
                                          force_reload=False,  # Don't force reload since patch already exists
                                          skip_validation=True)
                    print(f"Successfully loaded {model_name} with existing patch", file=sys.stderr, flush=True)
                    return model
        else:
            raise
    except Exception as e:
        print(f"Unexpected error loading via torch.hub: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise


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
