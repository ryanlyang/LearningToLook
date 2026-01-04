"""
OpenCLIP Adapter
================
This module provides a compatibility layer between open_clip and the original CLIP interface
used in WeCLIP+. It wraps open_clip models to provide the same API as the custom CLIP implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple
import open_clip
from PIL import Image
import numpy as np

# Import the custom attention module from the original CLIP implementation
import clip.myAtt as myAtt


def upsample_pos_emb(emb, new_size):
    """Upsample positional embeddings for higher resolution

    Args:
        emb: Position embeddings tensor of shape [N, D]
        new_size: Tuple of (height, width) for the new size

    Returns:
        Upsampled position embeddings
    """
    # Keep the class token
    first = emb[:1, :]
    emb = emb[1:, :]
    N, D = emb.size(0), emb.size(1)
    size = int(np.sqrt(N))
    assert size * size == N, f"Position embedding size mismatch: {N} != {size}^2"

    # Reshape and upsample
    emb = emb.permute(1, 0)
    emb = emb.view(1, D, size, size).contiguous()
    emb = F.interpolate(emb, size=new_size, mode='bilinear', align_corners=False)
    emb = emb.view(D, -1).contiguous()
    emb = emb.permute(1, 0)

    # Concatenate class token back
    emb = torch.cat([first, emb], 0)
    emb = nn.parameter.Parameter(emb.half())
    return emb


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Quick GELU activation function"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Transformer block with custom attention tracking"""

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # Use custom multihead attention that tracks attention weights
        self.attn = myAtt.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        attn_output, attn_weight = self.attention(self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weight


class CustomTransformer(nn.Module):
    """Transformer with attention weight tracking"""

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, require_all_fts=False):
        attn_weights = []
        x_all = []
        with torch.no_grad():
            layers = self.layers if x.shape[0] == 77 else self.layers - 1
            for i in range(layers):
                x, attn_weight = self.resblocks[i](x)
                x_all.append(x)
                attn_weights.append(attn_weight)

        if require_all_fts:
            return x_all, attn_weights
        else:
            return x, attn_weights


class VisionTransformerWrapper(nn.Module):
    """Wrapper around open_clip's vision transformer to match original CLIP interface"""

    def __init__(self, openclip_visual, patch_size=16):
        super().__init__()
        self.openclip_visual = openclip_visual
        self.patch_size = patch_size

        # Extract key components from open_clip's vision transformer
        self.conv1 = openclip_visual.conv1
        self.class_embedding = openclip_visual.class_embedding
        self.positional_embedding = openclip_visual.positional_embedding
        self.ln_pre = openclip_visual.ln_pre if hasattr(openclip_visual, 'ln_pre') else nn.Identity()
        self.ln_post = openclip_visual.ln_post if hasattr(openclip_visual, 'ln_post') else nn.Identity()
        self.proj = openclip_visual.proj if hasattr(openclip_visual, 'proj') else None

        # Create custom transformer with attention tracking
        width = self.positional_embedding.shape[-1]

        # Try to get the number of layers and heads from the original transformer
        if hasattr(openclip_visual, 'transformer'):
            layers = len(openclip_visual.transformer.resblocks)
            # Assume 64 dims per head as in original CLIP
            heads = width // 64
        else:
            # Fallback values
            layers = 12
            heads = width // 64

        self.transformer = CustomTransformer(width, layers, heads)

        # Copy weights from open_clip transformer to custom transformer
        if hasattr(openclip_visual, 'transformer') and hasattr(openclip_visual.transformer, 'resblocks'):
            self._copy_transformer_weights(openclip_visual.transformer.resblocks)

        self.output_dim = self.proj.shape[1] if self.proj is not None else width
        self.input_resolution = openclip_visual.image_size if hasattr(openclip_visual, 'image_size') else 224

    def _copy_transformer_weights(self, source_blocks):
        """Copy weights from open_clip transformer blocks to custom blocks"""
        for i, (src_block, dst_block) in enumerate(zip(source_blocks, self.transformer.resblocks)):
            # Copy attention weights
            if hasattr(src_block, 'attn'):
                # Open CLIP uses different attention structure
                # We need to adapt the weights to our custom attention module
                if hasattr(src_block.attn, 'in_proj_weight'):
                    dst_block.attn.in_proj_weight.data.copy_(src_block.attn.in_proj_weight.data)
                if hasattr(src_block.attn, 'in_proj_bias') and src_block.attn.in_proj_bias is not None:
                    dst_block.attn.in_proj_bias.data.copy_(src_block.attn.in_proj_bias.data)
                if hasattr(src_block.attn, 'out_proj'):
                    dst_block.attn.out_proj.weight.data.copy_(src_block.attn.out_proj.weight.data)
                    if src_block.attn.out_proj.bias is not None:
                        dst_block.attn.out_proj.bias.data.copy_(src_block.attn.out_proj.bias.data)

            # Copy layer norms
            if hasattr(src_block, 'ln_1'):
                dst_block.ln_1.weight.data.copy_(src_block.ln_1.weight.data)
                dst_block.ln_1.bias.data.copy_(src_block.ln_1.bias.data)
            if hasattr(src_block, 'ln_2'):
                dst_block.ln_2.weight.data.copy_(src_block.ln_2.weight.data)
                dst_block.ln_2.bias.data.copy_(src_block.ln_2.bias.data)

            # Copy MLP weights
            if hasattr(src_block, 'mlp'):
                # Open CLIP's MLP structure might differ slightly
                if hasattr(src_block.mlp, 'c_fc'):
                    dst_block.mlp[0].weight.data.copy_(src_block.mlp.c_fc.weight.data)
                    dst_block.mlp[0].bias.data.copy_(src_block.mlp.c_fc.bias.data)
                    dst_block.mlp[2].weight.data.copy_(src_block.mlp.c_proj.weight.data)
                    dst_block.mlp[2].bias.data.copy_(src_block.mlp.c_proj.bias.data)

    def forward(self, x: torch.Tensor, H: int, W: int, require_all_fts=False, clip_flag=16):
        """Forward pass matching original CLIP interface

        Args:
            x: Input image tensor
            H: Image height
            W: Image width
            require_all_fts: Whether to return features from all layers
            clip_flag: Patch size (14 or 16)

        Returns:
            Tuple of (features, attention_weights)
        """
        # Upsample positional embeddings based on input size
        if clip_flag == 16:
            self.positional_embedding_new = upsample_pos_emb(self.positional_embedding, (H // 16, W // 16))
        else:
            self.positional_embedding_new = upsample_pos_emb(self.positional_embedding, (H // 14, W // 14))

        # Patch embedding
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # Add class token
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # Add positional embedding
        x = x + self.positional_embedding_new.to(x.dtype)
        x = self.ln_pre(x)

        # Transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn_weight = self.transformer(x, require_all_fts=require_all_fts)

        return x, attn_weight


class CLIPAdapter(nn.Module):
    """Adapter to make open_clip compatible with WeCLIP+ code"""

    def __init__(self, openclip_model, patch_size=16):
        super().__init__()
        self.openclip_model = openclip_model

        # Wrap the vision transformer
        self.visual = VisionTransformerWrapper(openclip_model.visual, patch_size=patch_size)

        # Keep references to text encoder components
        self.transformer = openclip_model.transformer if hasattr(openclip_model, 'transformer') else openclip_model.text
        self.token_embedding = openclip_model.token_embedding
        self.positional_embedding = openclip_model.positional_embedding
        self.ln_final = openclip_model.ln_final
        self.text_projection = openclip_model.text_projection
        self.logit_scale = openclip_model.logit_scale
        self.context_length = openclip_model.context_length if hasattr(openclip_model, 'context_length') else 77

        # Vocab size
        self.vocab_size = openclip_model.vocab_size if hasattr(openclip_model, 'vocab_size') else self.token_embedding.num_embeddings

    @property
    def dtype(self):
        """Return the dtype of the model"""
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, H, W, require_all_fts=False, clip_flag=16):
        """Encode image and return features and attention weights

        Args:
            image: Input image tensor
            H: Image height
            W: Image width
            require_all_fts: Whether to return all layer features
            clip_flag: Patch size (14 or 16)

        Returns:
            Tuple of (features, attention_weights)
        """
        f_x, f_attn = self.visual(image.type(self.dtype), H, W, require_all_fts=require_all_fts, clip_flag=clip_flag)
        return f_x, f_attn

    def encode_text(self, text):
        """Encode text using the text encoder

        Args:
            text: Tokenized text tensor

        Returns:
            Text features
        """
        # Use open_clip's encode_text method
        return self.openclip_model.encode_text(text)

    def forward_last_layer(self, image_features, text_features):
        """Forward through last transformer layer

        Args:
            image_features: Image features from previous layers
            text_features: Text features

        Returns:
            Tuple of (logits, attention_weights)
        """
        x, attn_weight = self.visual.transformer.resblocks[self.visual.transformer.layers - 1](image_features)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x)
        x = torch.mean(x[:, 1:, :], dim=1)

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        image_features = x

        # Normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        if (self.visual.transformer.layers - 1) == 23:
            logit_scale = self.logit_scale.exp() / 4
        else:
            logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        # Softmax
        logits_per_image = logits_per_image.softmax(dim=-1)

        return logits_per_image, attn_weight

    def forward_mylast_layer(self, image_features):
        """Forward through last layer only

        Args:
            image_features: Image features

        Returns:
            Tuple of (features, attention_weights)
        """
        x, attn_weight = self.visual.transformer.resblocks[self.visual.transformer.layers - 1](image_features)
        return x, attn_weight


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit: bool = False, download_root: str = None):
    """Load a CLIP model using open_clip

    This function provides the same interface as the original clip.load() but uses open_clip underneath.

    Args:
        name: Model name (e.g., 'ViT-B-16', 'ViT-L-14') or path to checkpoint
        device: Device to load model on
        jit: Whether to use JIT (not supported with open_clip)
        download_root: Path to download models (not used with open_clip)

    Returns:
        Tuple of (model, preprocess_transform)
    """
    # Map original CLIP model names to open_clip equivalents
    # Use -quickgelu variants to match original CLIP activation function
    model_mapping = {
        'ViT-B/32': 'ViT-B-32-quickgelu',
        'ViT-B/16': 'ViT-B-16-quickgelu',
        'ViT-L/14': 'ViT-L-14-quickgelu',
        'ViT-L/14@336px': 'ViT-L-14-quickgelu',  # 336px variant
        'RN50': 'RN50',
        'RN101': 'RN101',
        'RN50x4': 'RN50x4',
        'RN50x16': 'RN50x16',
        'RN50x64': 'RN50x64',
    }

    # Extract patch size from model name
    patch_size = 16  # default
    if '14' in name or '/14' in name:
        patch_size = 14
    elif '32' in name or '/32' in name:
        patch_size = 32

    # Convert model name if needed
    openclip_name = model_mapping.get(name, name.replace('/', '-'))

    # Try to load from a checkpoint file if name is a path
    if name.endswith('.pt'):
        # For .pt files, we'll ignore them and use open_clip's pretrained weights
        # The original CLIP .pt files are compatible with open_clip weights
        # Determine architecture from filename
        if 'ViT-B-16' in name or 'ViT-B/16' in openclip_name:
            openclip_name = 'ViT-B-16-quickgelu'
        elif 'ViT-B-32' in name:
            openclip_name = 'ViT-B-32-quickgelu'
        elif 'ViT-L-14' in name:
            openclip_name = 'ViT-L-14-quickgelu'
        else:
            # Default to ViT-B-16 if can't determine
            openclip_name = 'ViT-B-16-quickgelu'

        pretrained = 'openai'  # Use OpenAI pretrained weights from open_clip
        print(f"Note: Ignoring local checkpoint {name}, using open_clip pretrained weights for {openclip_name}")
        state_dict = None
    else:
        pretrained = 'openai'  # Use OpenAI weights
        state_dict = None

    # Load the open_clip model
    model, _, preprocess = open_clip.create_model_and_transforms(
        openclip_name,
        pretrained=pretrained,
        device=device
    )

    # If we have a state dict from a file, load it
    if state_dict is not None:
        # The state dict from original CLIP needs to be adapted
        # For now, we'll use the pretrained open_clip weights
        # In production, you'd need to carefully map the keys
        pass

    # Wrap the model in our adapter
    adapted_model = CLIPAdapter(model, patch_size=patch_size)
    adapted_model.to(device)
    adapted_model.eval()

    return adapted_model, preprocess


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False):
    """Tokenize text using open_clip's tokenizer

    Args:
        texts: String or list of strings to tokenize
        context_length: Maximum context length
        truncate: Whether to truncate text that's too long

    Returns:
        Tokenized text tensor
    """
    return open_clip.tokenize(texts, context_length=context_length)
