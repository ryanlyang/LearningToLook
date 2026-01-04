# DINO Model Guide

This guide explains how to switch between DINOv1 and DINOv2 models in the WeCLIP+ codebase.

## Overview

The codebase now supports **both DINOv1 and DINOv2** models:

- **DINOv1**: Original self-supervised vision transformer from Facebook AI (2021)
- **DINOv2**: Improved version with better performance (2023)

## Quick Start

To switch DINO models, edit the `dino_init` section in [configs/voc_attn_reg.yaml](configs/voc_attn_reg.yaml):

```yaml
dino_init:
  dino_model: 'dino_vits16'        # Model name
  dino_fts_fuse_dim: 384           # Feature dimension (must match model)
  decoder_layer: 3                 # Number of decoder layers
```

## Available Models

### DINOv1 Models

| Model Name | Description | Feature Dim | Patch Size | Performance |
|------------|-------------|-------------|------------|-------------|
| `dino_vits16` | ViT-Small, patch 16 | 384 | 16 | ⭐⭐⭐ |
| `dino_vits8` | ViT-Small, patch 8 | 384 | 8 | ⭐⭐⭐⭐ (slower) |
| `dino_vitb16` | ViT-Base, patch 16 | 768 | 16 | ⭐⭐⭐⭐ |
| `dino_vitb8` | ViT-Base, patch 8 | 768 | 8 | ⭐⭐⭐⭐⭐ (slowest) |
| `dino_resnet50` | ResNet50 backbone | 2048 | N/A | ⭐⭐ |

### DINOv2 Models

| Model Name | Description | Feature Dim | Patch Size | Performance |
|------------|-------------|-------------|------------|-------------|
| `dinov2_vits14` | ViT-Small, patch 14 | 384 | 14 | ⭐⭐⭐⭐ |
| `dinov2_vits14_reg` | ViT-Small with registers | 384 | 14 | ⭐⭐⭐⭐⭐ |
| `dinov2_vitb14` | ViT-Base, patch 14 | 768 | 14 | ⭐⭐⭐⭐⭐ |
| `dinov2_vitb14_reg` | ViT-Base with registers | 768 | 14 | ⭐⭐⭐⭐⭐ |
| `dinov2_vitl14` | ViT-Large, patch 14 | 1024 | 14 | ⭐⭐⭐⭐⭐ |
| `dinov2_vitl14_reg` | ViT-Large with registers | 1024 | 14 | ⭐⭐⭐⭐⭐ |

## Configuration Examples

### Example 1: DINOv1 ViT-Small (Default)

```yaml
dino_init:
  dino_model: 'dino_vits16'
  dino_fts_fuse_dim: 384
  decoder_layer: 3
```

**Use when**: You want the original DINO with good performance and reasonable speed.

### Example 2: DINOv1 ViT-Base (Higher Quality)

```yaml
dino_init:
  dino_model: 'dino_vitb16'
  dino_fts_fuse_dim: 768
  decoder_layer: 3
```

**Use when**: You need better feature quality and have more compute.

### Example 3: DINOv1 with Smaller Patches (Best Quality, Slowest)

```yaml
dino_init:
  dino_model: 'dino_vitb8'
  dino_fts_fuse_dim: 768
  decoder_layer: 3
```

**Use when**: You need the best possible features from DINOv1, regardless of speed.

### Example 4: DINOv2 ViT-Small with Registers (Recommended)

```yaml
dino_init:
  dino_model: 'dinov2_vits14_reg'
  dino_fts_fuse_dim: 384
  decoder_layer: 3
```

**Use when**: You want the best DINOv2 small model (better than DINOv1).

### Example 5: DINOv2 ViT-Large (Best Performance)

```yaml
dino_init:
  dino_model: 'dinov2_vitl14_reg'
  dino_fts_fuse_dim: 1024
  decoder_layer: 5
```

**Use when**: You have significant compute and want the absolute best performance.

## DINOv1 vs DINOv2 Comparison

| Aspect | DINOv1 | DINOv2 |
|--------|--------|--------|
| **Release** | 2021 | 2023 |
| **Training Data** | ImageNet-1K | 142M curated images |
| **Performance** | Good | Better |
| **Patch Sizes** | 8, 16 | 14 only |
| **Registers** | ❌ No | ✅ Yes (`_reg` variants) |
| **Speed** | Varies (8 slower, 16 faster) | Moderate |
| **Recommended** | Legacy projects | New projects |

## Important Notes

### Feature Dimensions Must Match

**CRITICAL**: The `dino_fts_fuse_dim` must match the model's feature dimension:

- **384** → ViT-Small models (`vits`)
- **768** → ViT-Base models (`vitb`)
- **1024** → ViT-Large models (`vitl`)
- **2048** → ResNet50

**Wrong configuration will cause dimension mismatch errors!**

### Patch Size Differences

- **DINOv1**: Supports patch sizes 8 and 16
  - Smaller patches (8) = better features, slower
  - Larger patches (16) = faster, slightly worse features
- **DINOv2**: Only patch size 14
  - Good balance between quality and speed

### What Are "Registers"?

DINOv2 models with `_reg` suffix include **register tokens** - learnable tokens that improve feature quality by acting as information sinks. They generally perform better with minimal speed impact.

**Recommendation**: Always use `_reg` variants for DINOv2 when possible.

## Model Loading

Models are loaded automatically from:

- **DINOv1**: `torch.hub` (Facebook Research DINO repository)
- **DINOv2**: Local pretrained directory

First run will download weights (may take a few minutes).

## Switching Models Mid-Project

To switch DINO models:

1. **Edit config**: Change `dino_model` and `dino_fts_fuse_dim` in config file
2. **No code changes needed**: The model loader handles everything
3. **Restart training**: New model will be loaded automatically

**Note**: Switching models mid-training will break checkpoint loading since architectures differ.

## Recommendations

### For Best Performance
```yaml
dino_init:
  dino_model: 'dinov2_vitl14_reg'
  dino_fts_fuse_dim: 1024
  decoder_layer: 5
```

### For Best Speed
```yaml
dino_init:
  dino_model: 'dino_vits16'
  dino_fts_fuse_dim: 384
  decoder_layer: 3
```

### For Balanced Quality/Speed
```yaml
dino_init:
  dino_model: 'dinov2_vitb14_reg'
  dino_fts_fuse_dim: 768
  decoder_layer: 3
```

### For Legacy Compatibility
```yaml
dino_init:
  dino_model: 'dino_vits16'  # Original DINO
  dino_fts_fuse_dim: 384
  decoder_layer: 3
```

## Troubleshooting

### Error: "Unknown DINO model"

**Cause**: Model name typo or unsupported model

**Fix**: Check spelling and use one of the supported model names from the tables above

### Error: Dimension mismatch

**Cause**: `dino_fts_fuse_dim` doesn't match model's feature dimension

**Fix**:
- ViT-Small → 384
- ViT-Base → 768
- ViT-Large → 1024
- ResNet50 → 2048

### Model download fails

**Cause**: Network issues or torch.hub problems

**Fix**:
```bash
# Clear torch hub cache
rm -rf ~/.cache/torch/hub/*

# Try again
python your_training_script.py
```

### Out of memory with large models

**Cause**: Model too large for GPU

**Fix**: Use smaller model (ViT-Small instead of ViT-Large) or reduce batch size

## Technical Details

### DINOv1 Implementation

DINOv1 models are loaded via `torch.hub` from the official Facebook Research repository:

```python
from pretrained.dinov1_loader import dino_vits16

model = dino_vits16(pretrained=True)
```

### DINOv2 Implementation

DINOv2 models are loaded from the local pretrained directory:

```python
from pretrained.facebookDinov2.hubconf import dinov2_vits14

model = dinov2_vits14(pretrained=True)
```

### Feature Extraction

Both DINOv1 and DINOv2 models extract features similarly:

```python
# Forward pass returns normalized patch tokens
dino_ftses = self.dino_encoder.forward_features(dino_img)
dino_fts = dino_ftses['x_norm_patchtokens']
```

## Summary

- **DINOv1**: Original, supports multiple patch sizes (8, 16)
- **DINOv2**: Improved, patch size 14 only, better performance
- **Switch models**: Edit config file, no code changes needed
- **Match dimensions**: `dino_fts_fuse_dim` must match model
- **Use `_reg`**: Register variants generally perform better

For most new projects, **DINOv2 with registers** is recommended!
