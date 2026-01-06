"""
DINOv1 Model Loader
===================
Loads the original DINO (v1) ViT-Small model with local weights.
This avoids torch.hub issues by loading the architecture and weights directly.
"""

import torch
import torch.nn as nn
import math
from functools import partial


class VisionTransformer(nn.Module):
    """
    Vision Transformer for DINOv1.
    Simplified version that matches the pretrained weights structure.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384,
                 depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., norm_layer=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

    def interpolate_pos_encoding(self, x, w, h):
        """Interpolate positional encoding for variable image sizes."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_size
        h0 = h // self.patch_size

        # Add small number to avoid floating point error
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        """Prepare tokens with positional encoding."""
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(self, x):
        """Forward pass."""
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_intermediate_layers(self, x, n=1):
        """
        Get intermediate layer outputs (for compatibility with DINOv1 API).
        Returns the output from the last n layers.
        """
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    """MLP block."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def dino_vits16(pretrained=True, pretrained_path=None):
    """
    Load DINOv1 ViT-Small with patch size 16.

    Args:
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained weights file (if None, uses default location)

    Returns:
        VisionTransformer model
    """
    import os

    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    if pretrained:
        if pretrained_path is None:
            # Default path
            pretrained_path = os.path.join(
                os.path.dirname(__file__),
                'dino_deitsmall16_pretrain.pth'
            )

        if os.path.exists(pretrained_path):
            print(f"Loading DINOv1 weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded DINOv1 ViT-S/16 weights")
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            print("Using randomly initialized weights")

    return model


# Model metadata
DINOV1_MODELS = {
    'dino_vits16': 'dino_vits16',
}

DINOV1_DIMS = {
    'dino_vits16': 384,
}

DINOV1_PATCH_SIZES = {
    'dino_vits16': 16,
}


def get_dinov1_feature_dim(model_name):
    """Get the feature dimension for a DINOv1 model."""
    if model_name not in DINOV1_DIMS:
        raise ValueError(f"Unknown DINOv1 model: {model_name}")
    return DINOV1_DIMS[model_name]


def get_dinov1_patch_size(model_name):
    """Get the patch size for a DINOv1 model."""
    if model_name not in DINOV1_PATCH_SIZES:
        raise ValueError(f"Unknown DINOv1 model: {model_name}")
    return DINOV1_PATCH_SIZES[model_name]


__all__ = [
    'dino_vits16',
    'get_dinov1_feature_dim',
    'get_dinov1_patch_size',
    'DINOV1_MODELS',
    'DINOV1_DIMS',
    'DINOV1_PATCH_SIZES',
]
