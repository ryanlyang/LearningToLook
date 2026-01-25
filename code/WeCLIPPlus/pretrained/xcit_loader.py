"""
XCiT (Cross-Covariance Image Transformer) Model Loader
========================================================
Loads XCiT models trained with DINO from local weights.
Based on the official XCiT implementation.

Reference: "XCiT: Cross-Covariance Image Transformers" (https://arxiv.org/abs/2106.09681)
"""

import torch
import torch.nn as nn
from functools import partial


class ConvPatchEmbed(nn.Module):
    """Convolutional patch embedding used in XCiT."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # XCiT uses a 4-stage convolutional stem
        # 3 -> 48 -> 96 -> 192 -> 384
        self.proj = nn.Sequential(
            # Stage 1: 3 -> 48
            nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 8, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 8),
                nn.GELU(),
            ),
            # MaxPool
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Stage 2: 48 -> 96
            nn.Sequential(
                nn.Conv2d(embed_dim // 8, embed_dim // 4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 4),
                nn.GELU(),
            ),
            # MaxPool
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Stage 3: 96 -> 192
            nn.Sequential(
                nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.GELU(),
            ),
            # MaxPool
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Stage 4: 192 -> 384
            nn.Sequential(
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim),
            ),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        # Flatten spatial dimensions
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        return x, (Hp, Wp)


class PositionalEncodingFourier(nn.Module):
    """Positional encoding using Fourier features (used in XCiT)."""
    def __init__(self, hidden_dim=384, dim=64, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.temperature = temperature
        self.dim = dim  # This should be 64 for xcit_small_12_p16

    def forward(self, x, H, W):
        B, N, C = x.shape
        # Create positional grid
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device).unsqueeze(0).repeat(H, 1)

        # Normalize
        eps = 1e-6
        y_embed = y_embed / (H + eps) * 2 * 3.1416
        x_embed = x_embed / (W + eps) * 2 * 3.1416

        # Fourier features with 16 frequency bands (dim=64 -> 16 bands * 2 (sin+cos) * 2 (x,y) = 64)
        num_pos_feats = self.dim // 2  # 32
        dim_t = torch.arange(num_pos_feats // 2, dtype=torch.float32, device=x.device)  # 16 bands
        dim_t = self.temperature ** (2 * dim_t / (num_pos_feats // 2))

        # Compute sin/cos for x and y coordinates
        pos_x = x_embed.unsqueeze(0).unsqueeze(0) / dim_t.view(1, -1, 1, 1)  # [1, 16, H, W]
        pos_y = y_embed.unsqueeze(0).unsqueeze(0) / dim_t.view(1, -1, 1, 1)  # [1, 16, H, W]

        # Stack sin and cos
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=2).flatten(1, 2)  # [1, 32, H, W]
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=2).flatten(1, 2)  # [1, 32, H, W]

        pos = torch.cat([pos_y, pos_x], dim=1).repeat(B, 1, 1, 1)  # [B, 64, H, W]

        # Project to embedding dimension
        pos = self.token_projection(pos)  # [B, C, H, W]
        pos = pos.flatten(2).transpose(1, 2)  # [B, N, C]

        return pos


class XCA(nn.Module):
    """Cross-Covariance Attention (XCA) module."""
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # L2 normalize
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        # Cross-covariance attention: (K^T @ Q) instead of (Q @ K^T)
        attn = (k.transpose(-2, -1) @ q) * self.temperature  # [B, heads, head_dim, head_dim]
        attn = attn.softmax(dim=-1)

        # Apply attention
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # [B, heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class Attention(nn.Module):
    """Standard multi-head attention (for CLS blocks)."""
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
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


class LPI(nn.Module):
    """Local Patch Interaction module - uses depthwise separable convolution."""
    def __init__(self, in_features, act_layer=nn.GELU, kernel_size=3):
        super().__init__()
        # Two depthwise convolutions with BN in between
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=in_features)
        self.act1 = act_layer()
        self.bn = nn.BatchNorm2d(in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=in_features)
        self.act2 = act_layer()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


class XCiTBlock(nn.Module):
    """XCiT block."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 norm_layer=nn.LayerNorm, eta=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim)

        # Layer scale parameters (gamma)
        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))
        self.gamma3 = nn.Parameter(eta * torch.ones(dim))

    def forward(self, x, H, W):
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        x = x + self.gamma3 * self.local_mp(self.norm3(x), H, W)
        return x


class ClassAttentionBlock(nn.Module):
    """Class attention block (standard transformer block for CLS token)."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 norm_layer=nn.LayerNorm, eta=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        # Layer scale parameters
        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))

    def forward(self, x):
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class XCiT(nn.Module):
    """XCiT model."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, depth=12,
                 num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0.,
                 norm_layer=None, eta=1.0, cls_attn_layers=2):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Patch embedding
        self.patch_embed = ConvPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Positional encoding
        self.pos_embeder = PositionalEncodingFourier(hidden_dim=embed_dim, dim=64)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks (XCA blocks)
        self.blocks = nn.ModuleList([
            XCiTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                     drop=drop_rate, norm_layer=norm_layer, eta=eta)
            for _ in range(depth)])

        # Class attention blocks (standard attention with CLS token)
        self.cls_attn_blocks = nn.ModuleList([
            ClassAttentionBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate, norm_layer=norm_layer, eta=eta)
            for _ in range(cls_attn_layers)])

        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x, (H, W) = self.patch_embed(x)

        # Add positional encoding
        pos = self.pos_embeder(x, H, W)
        x = x + pos

        # Apply XCA blocks
        for blk in self.blocks:
            x = blk(x, H, W)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply class attention blocks
        for blk in self.cls_attn_blocks:
            x = blk(x)

        x = self.norm(x)

        return x[:, 0]  # Return CLS token

    def get_intermediate_layers(self, x, n=1):
        """Get intermediate layer outputs (for compatibility with DINO API)."""
        B = x.shape[0]

        # Patch embedding
        x, (H, W) = self.patch_embed(x)

        # Add positional encoding
        pos = self.pos_embeder(x, H, W)
        x = x + pos

        output = []

        # Apply XCA blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)
            if len(self.blocks) - i <= n:
                # Add CLS token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x_with_cls = torch.cat((cls_tokens, x), dim=1)

                # Apply class attention blocks
                for cls_blk in self.cls_attn_blocks:
                    x_with_cls = cls_blk(x_with_cls)

                x_with_cls = self.norm(x_with_cls)
                output.append(x_with_cls)

        return output


def xcit_small_12_p16(pretrained=True, pretrained_path=None):
    """
    Load XCiT-Small-12/16 model trained with DINO.

    Args:
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained weights file

    Returns:
        XCiT model
    """
    import os

    model = XCiT(
        patch_size=16, embed_dim=384, depth=12, num_heads=8, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, cls_attn_layers=2)

    if pretrained:
        if pretrained_path is None:
            # Default path
            pretrained_path = os.path.join(
                os.path.dirname(__file__),
                'dino_xcit_small_12_p16_pretrain.pth'
            )

        if os.path.exists(pretrained_path):
            print(f"Loading XCiT weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded XCiT-Small-12/16 DINO weights")
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            print("Using randomly initialized weights")

    return model


def xcit_medium_24_p16(pretrained=True, pretrained_path=None):
    """
    Load XCiT-Medium-24/16 model trained with DINO.

    Args:
        pretrained: Whether to load pretrained weights
        pretrained_path: Path to pretrained weights file

    Returns:
        XCiT model
    """
    import os

    model = XCiT(
        patch_size=16, embed_dim=512, depth=24, num_heads=8, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), eta=1.0, cls_attn_layers=2)

    if pretrained:
        if pretrained_path is None:
            # Default path
            pretrained_path = os.path.join(
                os.path.dirname(__file__),
                'dino_xcit_medium_24_p16_pretrain.pth'
            )

        if os.path.exists(pretrained_path):
            print(f"Loading XCiT weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded XCiT-Medium-24/16 DINO weights")
        else:
            print(f"Warning: Pretrained weights not found at {pretrained_path}")
            print("Using randomly initialized weights")

    return model


# Model metadata
XCIT_MODELS = {
    'xcit_small_12_p16': 'xcit_small_12_p16',
    'xcit_medium_24_p16': 'xcit_medium_24_p16',
}

XCIT_DIMS = {
    'xcit_small_12_p16': 384,
    'xcit_medium_24_p16': 512,
}

XCIT_PATCH_SIZES = {
    'xcit_small_12_p16': 16,
    'xcit_medium_24_p16': 16,
}


def get_xcit_feature_dim(model_name):
    """Get the feature dimension for an XCiT model."""
    if model_name not in XCIT_DIMS:
        raise ValueError(f"Unknown XCiT model: {model_name}")
    return XCIT_DIMS[model_name]


def get_xcit_patch_size(model_name):
    """Get the patch size for an XCiT model."""
    if model_name not in XCIT_PATCH_SIZES:
        raise ValueError(f"Unknown XCiT model: {model_name}")
    return XCIT_PATCH_SIZES[model_name]


__all__ = [
    'xcit_small_12_p16',
    'xcit_medium_24_p16',
    'get_xcit_feature_dim',
    'get_xcit_patch_size',
    'XCIT_MODELS',
    'XCIT_DIMS',
    'XCIT_PATCH_SIZES',
]
