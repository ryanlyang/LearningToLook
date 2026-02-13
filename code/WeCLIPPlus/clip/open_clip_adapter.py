"""
OpenCLIP/SigLIP2 Adapter
========================
Compatibility layer for WeCLIP+ that keeps the same interface expected by the
original CLIP implementation while loading models from open_clip.
"""

import os
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip.myAtt as myAtt


_TOKENIZER_FN = open_clip.tokenize
_TOKENIZER_CONTEXT_LENGTH = 77


def upsample_pos_emb(emb: torch.Tensor, new_size: Tuple[int, int]) -> nn.Parameter:
    """Upsample positional embeddings to match the requested patch grid size."""
    first = emb[:1, :]
    emb = emb[1:, :]
    num_tokens, dim = emb.size(0), emb.size(1)
    size = int(np.sqrt(num_tokens))
    assert size * size == num_tokens, f"Position embedding size mismatch: {num_tokens} != {size}^2"

    emb = emb.permute(1, 0).view(1, dim, size, size).contiguous()
    emb = F.interpolate(emb, size=new_size, mode="bilinear", align_corners=False)
    emb = emb.view(dim, -1).permute(1, 0).contiguous()
    emb = torch.cat([first, emb], dim=0)
    return nn.parameter.Parameter(emb.to(dtype=first.dtype))


def _copy_parameter_if_compatible(dst: torch.Tensor, src: torch.Tensor) -> bool:
    if dst is None or src is None:
        return False
    if dst.shape != src.shape:
        return False
    dst.data.copy_(src.data)
    return True


def _looks_like_local_checkpoint(name: str) -> bool:
    if not name:
        return False
    return name.endswith(".pt") or os.path.isfile(name)


def _infer_patch_size_from_name(name: str, default: int = 16) -> int:
    if not name:
        return default
    name_l = name.lower()

    match = re.search(r"(?:/|-|_|p)(8|14|16|32)(?:px)?(?:$|/|-|_)", name_l)
    if match:
        return int(match.group(1))

    for p in (8, 14, 16, 32):
        if str(p) in name_l:
            return p
    return default


def _parse_model_name_from_checkpoint(path: str) -> str:
    filename = os.path.basename(path)
    if "ViT-B-16" in filename or "ViT-B_16" in filename:
        return "ViT-B-16-quickgelu"
    if "ViT-B-32" in filename or "ViT-B_32" in filename:
        return "ViT-B-32-quickgelu"
    if "ViT-L-14" in filename or "ViT-L_14" in filename:
        return "ViT-L-14-quickgelu"
    return "ViT-B-16-quickgelu"


def _safe_list_pretrained() -> List[Tuple[str, str]]:
    if not hasattr(open_clip, "list_pretrained"):
        return []
    try:
        return list(open_clip.list_pretrained())
    except Exception:
        return []


def _pick_siglip2_pair() -> Tuple[Union[str, None], Union[str, None]]:
    candidates = [
        (model_name, pretrained_name)
        for model_name, pretrained_name in _safe_list_pretrained()
        if "siglip2" in model_name.lower() or "siglip2" in pretrained_name.lower()
    ]

    if not candidates:
        return None, None

    def _rank(pair: Tuple[str, str]) -> Tuple[int, int, int, str, str]:
        model_name, pretrained_name = pair
        model_name_l = model_name.lower()
        pretrained_name_l = pretrained_name.lower()
        return (
            0 if "vit-b" in model_name_l else 1,
            0 if "16" in model_name_l else 1,
            0 if "webli" in pretrained_name_l else 1,
            model_name_l,
            pretrained_name_l,
        )

    candidates.sort(key=_rank)
    return candidates[0]


def _resolve_openclip_model_request(
    name: str,
    pretrained: Optional[str] = None,
) -> Tuple[str, str, int, Union[str, None]]:
    backend = os.environ.get("CLIP_BACKEND", "openclip").strip().lower()

    model_override = os.environ.get("CLIP_MODEL_NAME")
    pretrained_override = os.environ.get("CLIP_PRETRAINED")
    explicit_pretrained = pretrained

    if backend == "siglip2":
        model_override = os.environ.get("SIGLIP2_MODEL_NAME", model_override)
        pretrained_override = os.environ.get("SIGLIP2_PRETRAINED", pretrained_override)

    model_mapping = {
        "ViT-B/32": "ViT-B-32-quickgelu",
        "ViT-B/16": "ViT-B-16-quickgelu",
        "ViT-L/14": "ViT-L-14-quickgelu",
        "ViT-L/14@336px": "ViT-L-14-quickgelu",
        "RN50": "RN50",
        "RN101": "RN101",
        "RN50x4": "RN50x4",
        "RN50x16": "RN50x16",
        "RN50x64": "RN50x64",
    }

    requested_name = name or ""
    is_local_checkpoint = _looks_like_local_checkpoint(requested_name)
    note = None
    discovered_pretrained = None

    if model_override:
        model_name = model_override
        if is_local_checkpoint:
            note = (
                f"Note: ignoring local checkpoint {requested_name}; "
                f"using CLIP_MODEL_NAME={model_name}"
            )
    elif backend == "siglip2":
        if requested_name and not is_local_checkpoint and "siglip2" in requested_name.lower():
            model_name = requested_name.replace("/", "-")
        else:
            model_name, discovered_pretrained = _pick_siglip2_pair()
            if model_name is None:
                raise RuntimeError(
                    "Could not auto-discover a SigLIP2 model in open_clip. "
                    "Set SIGLIP2_MODEL_NAME and SIGLIP2_PRETRAINED (or "
                    "CLIP_MODEL_NAME and CLIP_PRETRAINED) explicitly."
                )
            note = f"Using auto-discovered SigLIP2 model: {model_name}"
    else:
        if is_local_checkpoint:
            model_name = _parse_model_name_from_checkpoint(requested_name)
            note = (
                f"Note: ignoring local checkpoint {requested_name}, using "
                f"open_clip pretrained weights for {model_name}"
            )
        else:
            model_name = model_mapping.get(requested_name, requested_name.replace("/", "-"))

    if not model_name:
        model_name = "ViT-B-16-quickgelu"

    if pretrained_override:
        pretrained_name = pretrained_override
    elif explicit_pretrained:
        pretrained_name = explicit_pretrained
    elif discovered_pretrained:
        pretrained_name = discovered_pretrained
    elif backend == "siglip2":
        pretrained_name = "webli"
    else:
        pretrained_name = "openai"

    patch_size = _infer_patch_size_from_name(model_name, default=16)
    return model_name, pretrained_name, patch_size, note


def _set_tokenizer(model_name: str) -> None:
    global _TOKENIZER_FN
    try:
        _TOKENIZER_FN = open_clip.get_tokenizer(model_name)
    except Exception:
        _TOKENIZER_FN = open_clip.tokenize


def _infer_tokenizer_context_length(model: nn.Module, default: int = 77) -> int:
    for attr_path in (
        ("context_length",),
        ("text", "context_length"),
    ):
        obj = model
        ok = True
        for attr in attr_path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            try:
                value = int(obj)
                if value > 0:
                    return value
            except Exception:
                pass

    text_mod = getattr(model, "text", None)
    pos = getattr(text_mod, "positional_embedding", None)
    if hasattr(pos, "shape") and len(pos.shape) >= 1:
        try:
            value = int(pos.shape[0])
            if value > 0:
                return value
        except Exception:
            pass
    return int(default)


class LayerNorm(nn.LayerNorm):
    """LayerNorm variant that safely handles mixed precision."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Transformer block with tracked attention weights."""

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = myAtt.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model),
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
    """Transformer stack that can return per-layer features and attention maps."""

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor, require_all_fts: bool = False):
        attn_weights = []
        x_all = []
        with torch.no_grad():
            layers = self.layers if x.shape[0] == 77 else self.layers - 1
            layers = max(layers, 1)
            for i in range(layers):
                x, attn_weight = self.resblocks[i](x)
                x_all.append(x)
                attn_weights.append(attn_weight)

        if require_all_fts:
            return x_all, attn_weights
        return x, attn_weights


class VisionTransformerWrapper(nn.Module):
    """Wraps open_clip visual backbone into the WeCLIP+ expected interface."""

    def __init__(self, openclip_visual: nn.Module, patch_size: int = 16):
        super().__init__()
        self.openclip_visual = openclip_visual
        self._visual_source = openclip_visual

        def _resolve_timm_vit_source(visual: nn.Module):
            if hasattr(visual, "patch_embed") and hasattr(visual, "blocks"):
                return visual
            for attr in ("trunk", "model", "backbone"):
                nested = getattr(visual, attr, None)
                if nested is not None and hasattr(nested, "patch_embed") and hasattr(nested, "blocks"):
                    return nested
            return None

        if hasattr(openclip_visual, "conv1") and hasattr(openclip_visual, "transformer"):
            self.conv1 = openclip_visual.conv1
            self.class_embedding = openclip_visual.class_embedding
            self.positional_embedding = openclip_visual.positional_embedding
            self.ln_pre = openclip_visual.ln_pre if hasattr(openclip_visual, "ln_pre") else nn.Identity()
            self.ln_post = openclip_visual.ln_post if hasattr(openclip_visual, "ln_post") else nn.Identity()
            self.proj = openclip_visual.proj if hasattr(openclip_visual, "proj") else None
            source_blocks = openclip_visual.transformer.resblocks
        elif _resolve_timm_vit_source(openclip_visual) is not None:
            vit_source = _resolve_timm_vit_source(openclip_visual)
            self._visual_source = vit_source
            self.conv1 = vit_source.patch_embed.proj

            cls_token = getattr(vit_source, "cls_token", None)
            pos_embed = getattr(vit_source, "pos_embed", None)
            width = int(self.conv1.weight.shape[0])

            if cls_token is not None:
                self.class_embedding = nn.Parameter(cls_token.squeeze(0).squeeze(0))
            else:
                self.class_embedding = nn.Parameter(torch.zeros(width))

            if pos_embed is None:
                image_size = getattr(openclip_visual, "image_size", None)
                if image_size is None:
                    image_size = getattr(vit_source, "img_size", 224)
                if isinstance(image_size, (tuple, list)):
                    image_size = image_size[0]
                p = self.conv1.kernel_size[0] if isinstance(self.conv1.kernel_size, tuple) else self.conv1.kernel_size
                grid = max(int(image_size) // int(p), 1)
                self.positional_embedding = nn.Parameter(torch.zeros(1 + grid * grid, width))
            else:
                pos_embed = pos_embed.squeeze(0)
                num_prefix_tokens = int(getattr(vit_source, "num_prefix_tokens", 1))
                if num_prefix_tokens > 1 and pos_embed.shape[0] > num_prefix_tokens:
                    pos_embed = torch.cat([pos_embed[:1], pos_embed[num_prefix_tokens:]], dim=0)
                elif num_prefix_tokens == 0:
                    pad = torch.zeros(1, pos_embed.shape[1], dtype=pos_embed.dtype, device=pos_embed.device)
                    pos_embed = torch.cat([pad, pos_embed], dim=0)
                self.positional_embedding = nn.Parameter(pos_embed)

            self.ln_pre = getattr(vit_source, "norm_pre", nn.Identity())
            self.ln_post = getattr(vit_source, "norm", nn.Identity())
            self.proj = getattr(vit_source, "proj", None)
            source_blocks = vit_source.blocks
        else:
            attrs = sorted([name for name in dir(openclip_visual) if not name.startswith("_")])
            raise RuntimeError(
                "Unsupported open_clip visual backbone. Expected CLIP-style (conv1/transformer) "
                "or timm-style (patch_embed/blocks) ViT. "
                f"visual_type={type(openclip_visual).__name__}, attrs={attrs[:40]}"
            )

        width = self.positional_embedding.shape[-1]
        layers = len(source_blocks)
        heads = getattr(getattr(source_blocks[0], "attn", None), "num_heads", width // 64)

        self.transformer = CustomTransformer(width, layers, heads)
        self._copy_transformer_weights(source_blocks)

        if isinstance(self.proj, (torch.Tensor, nn.Parameter)):
            self.output_dim = self.proj.shape[1]
        elif isinstance(self.proj, nn.Linear):
            self.output_dim = self.proj.out_features
        else:
            self.output_dim = width

        if hasattr(self.conv1, "kernel_size"):
            kernel = self.conv1.kernel_size[0] if isinstance(self.conv1.kernel_size, tuple) else self.conv1.kernel_size
            self.patch_size = int(kernel)
        else:
            self.patch_size = int(patch_size)

        image_size = getattr(openclip_visual, "image_size", None)
        if image_size is None:
            image_size = getattr(self._visual_source, "img_size", 224)
        if isinstance(image_size, (tuple, list)):
            image_size = image_size[0]
        self.input_resolution = int(image_size)

    def _copy_transformer_weights(self, source_blocks) -> None:
        for src_block, dst_block in zip(source_blocks, self.transformer.resblocks):
            attn_src = getattr(src_block, "attn", None)
            if attn_src is not None:
                if hasattr(attn_src, "in_proj_weight"):
                    _copy_parameter_if_compatible(dst_block.attn.in_proj_weight, attn_src.in_proj_weight)
                    if getattr(attn_src, "in_proj_bias", None) is not None:
                        _copy_parameter_if_compatible(dst_block.attn.in_proj_bias, attn_src.in_proj_bias)
                elif hasattr(attn_src, "qkv"):
                    _copy_parameter_if_compatible(dst_block.attn.in_proj_weight, attn_src.qkv.weight)
                    if getattr(attn_src.qkv, "bias", None) is not None:
                        _copy_parameter_if_compatible(dst_block.attn.in_proj_bias, attn_src.qkv.bias)
                elif all(hasattr(attn_src, attr) for attr in ("q_proj", "k_proj", "v_proj")):
                    qkv_w = torch.cat(
                        [attn_src.q_proj.weight, attn_src.k_proj.weight, attn_src.v_proj.weight],
                        dim=0,
                    )
                    _copy_parameter_if_compatible(dst_block.attn.in_proj_weight, qkv_w)
                    if (
                        getattr(attn_src.q_proj, "bias", None) is not None
                        and getattr(attn_src.k_proj, "bias", None) is not None
                        and getattr(attn_src.v_proj, "bias", None) is not None
                    ):
                        qkv_b = torch.cat(
                            [attn_src.q_proj.bias, attn_src.k_proj.bias, attn_src.v_proj.bias],
                            dim=0,
                        )
                        _copy_parameter_if_compatible(dst_block.attn.in_proj_bias, qkv_b)

                out_proj_src = getattr(attn_src, "out_proj", None)
                if out_proj_src is None:
                    out_proj_src = getattr(attn_src, "proj", None)
                if out_proj_src is not None:
                    _copy_parameter_if_compatible(dst_block.attn.out_proj.weight, out_proj_src.weight)
                    if getattr(out_proj_src, "bias", None) is not None:
                        _copy_parameter_if_compatible(dst_block.attn.out_proj.bias, out_proj_src.bias)

            ln1_src = getattr(src_block, "ln_1", None) or getattr(src_block, "norm1", None)
            ln2_src = getattr(src_block, "ln_2", None) or getattr(src_block, "norm2", None)
            if ln1_src is not None:
                _copy_parameter_if_compatible(dst_block.ln_1.weight, ln1_src.weight)
                _copy_parameter_if_compatible(dst_block.ln_1.bias, ln1_src.bias)
            if ln2_src is not None:
                _copy_parameter_if_compatible(dst_block.ln_2.weight, ln2_src.weight)
                _copy_parameter_if_compatible(dst_block.ln_2.bias, ln2_src.bias)

            mlp_src = getattr(src_block, "mlp", None)
            if mlp_src is None:
                continue

            fc1_src = getattr(mlp_src, "c_fc", None) or getattr(mlp_src, "fc1", None)
            fc2_src = getattr(mlp_src, "c_proj", None) or getattr(mlp_src, "fc2", None)
            if fc1_src is None and isinstance(mlp_src, nn.Sequential) and len(mlp_src) >= 3:
                if isinstance(mlp_src[0], nn.Linear):
                    fc1_src = mlp_src[0]
                if isinstance(mlp_src[2], nn.Linear):
                    fc2_src = mlp_src[2]

            if fc1_src is not None:
                _copy_parameter_if_compatible(dst_block.mlp[0].weight, fc1_src.weight)
                if getattr(fc1_src, "bias", None) is not None:
                    _copy_parameter_if_compatible(dst_block.mlp[0].bias, fc1_src.bias)
            if fc2_src is not None:
                _copy_parameter_if_compatible(dst_block.mlp[2].weight, fc2_src.weight)
                if getattr(fc2_src, "bias", None) is not None:
                    _copy_parameter_if_compatible(dst_block.mlp[2].bias, fc2_src.bias)

    def _apply_visual_projection(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            return x
        if isinstance(self.proj, (torch.Tensor, nn.Parameter)):
            return x @ self.proj
        if isinstance(self.proj, nn.Linear):
            return self.proj(x)
        return x

    def forward(self, x: torch.Tensor, H: int, W: int, require_all_fts: bool = False, clip_flag: int = 16):
        patch = int(clip_flag) if int(clip_flag) > 0 else self.patch_size
        grid_h = max(H // patch, 1)
        grid_w = max(W // patch, 1)
        self.positional_embedding_new = upsample_pos_emb(self.positional_embedding, (grid_h, grid_w))

        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        class_embedding = self.class_embedding
        if class_embedding.ndim != 1:
            class_embedding = class_embedding.reshape(-1)
        cls_token = class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        pos = self.positional_embedding_new.to(dtype=x.dtype, device=x.device)
        if pos.shape[0] != x.shape[1]:
            raise RuntimeError(
                f"Positional embedding/token mismatch: pos={pos.shape[0]} tokens={x.shape[1]}. "
                f"Check clip_flag={clip_flag} and model patch size."
            )
        x = x + pos
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x, attn_weight = self.transformer(x, require_all_fts=require_all_fts)
        return x, attn_weight


class CLIPAdapter(nn.Module):
    """Adapter exposing the interface expected by WeCLIP+."""

    def __init__(self, openclip_model: nn.Module, patch_size: int = 16):
        super().__init__()
        self.openclip_model = openclip_model
        self.visual = VisionTransformerWrapper(openclip_model.visual, patch_size=patch_size)

        self.transformer = getattr(openclip_model, "transformer", None)
        self.token_embedding = getattr(openclip_model, "token_embedding", None)
        self.positional_embedding = getattr(openclip_model, "positional_embedding", None)
        self.ln_final = getattr(openclip_model, "ln_final", None)
        self.text_projection = getattr(openclip_model, "text_projection", None)
        self.logit_scale = getattr(openclip_model, "logit_scale", None)
        self.context_length = getattr(openclip_model, "context_length", 77)
        self.vocab_size = (
            openclip_model.vocab_size
            if hasattr(openclip_model, "vocab_size")
            else getattr(self.token_embedding, "num_embeddings", 0)
        )

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, H, W, require_all_fts: bool = False, clip_flag: int = 16):
        return self.visual(image.type(self.dtype), H, W, require_all_fts=require_all_fts, clip_flag=clip_flag)

    def encode_text(self, text):
        return self.openclip_model.encode_text(text)

    def _apply_visual_projection(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual._apply_visual_projection(x)

    def forward_last_layer(self, image_features, text_features):
        x, attn_weight = self.visual.transformer.resblocks[self.visual.transformer.layers - 1](image_features)
        x = x.permute(1, 0, 2)

        x = self.visual.ln_post(x)
        x = torch.mean(x[:, 1:, :], dim=1)
        image_features = self._apply_visual_projection(x)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if self.logit_scale is None:
            logit_scale = torch.tensor(1.0, device=image_features.device, dtype=image_features.dtype)
        elif (self.visual.transformer.layers - 1) == 23:
            logit_scale = self.logit_scale.exp() / 4
        else:
            logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_image = logits_per_image.softmax(dim=-1)
        return logits_per_image, attn_weight

    def forward_mylast_layer(self, image_features):
        x, attn_weight = self.visual.transformer.resblocks[self.visual.transformer.layers - 1](image_features)
        return x, attn_weight


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: str = None,
    pretrained: Optional[str] = None,
):
    """
    Load a model with open_clip while preserving the original clip.load() API.
    """
    del jit
    del download_root

    model_name, pretrained_name, patch_size, note = _resolve_openclip_model_request(name, pretrained=pretrained)
    if note:
        print(note)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained_name,
        device=device,
    )
    global _TOKENIZER_CONTEXT_LENGTH
    _TOKENIZER_CONTEXT_LENGTH = _infer_tokenizer_context_length(model, default=77)
    _set_tokenizer(model_name)

    adapted_model = CLIPAdapter(model, patch_size=patch_size)
    adapted_model.to(device)
    adapted_model.eval()
    return adapted_model, preprocess


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False):
    """Tokenize text for the most recently loaded open_clip model."""
    resolved_context_length = int(context_length)
    if resolved_context_length == 77 and _TOKENIZER_CONTEXT_LENGTH != 77:
        resolved_context_length = _TOKENIZER_CONTEXT_LENGTH

    try:
        return _TOKENIZER_FN(
            texts,
            context_length=resolved_context_length,
            truncate=truncate,
        )
    except TypeError:
        pass
    except Exception:
        pass

    try:
        return _TOKENIZER_FN(
            texts,
            context_length=resolved_context_length,
        )
    except TypeError:
        pass
    except Exception:
        pass

    try:
        return _TOKENIZER_FN(
            texts,
            truncate=truncate,
        )
    except TypeError:
        pass
    except Exception:
        pass

    if resolved_context_length != 77:
        try:
            return _TOKENIZER_FN(texts)
        except Exception:
            pass

    try:
        return _TOKENIZER_FN(texts)
    except Exception:
        pass

    return open_clip.tokenize(
        texts,
        context_length=resolved_context_length,
        truncate=truncate,
    )


def available_models() -> List[str]:
    """Return available model names from open_clip, best effort."""
    names = []
    if hasattr(open_clip, "list_models"):
        try:
            names.extend(open_clip.list_models())
        except Exception:
            pass
    for model_name, _ in _safe_list_pretrained():
        names.append(model_name)
    return sorted(set(names))
