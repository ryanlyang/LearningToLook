import argparse
import os
import re
from typing import Dict, List

from PIL import Image


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def _default_repo_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def _resolve_paths(repo_root):
    repo_root = os.path.abspath(repo_root)
    weclip_root = os.path.join(repo_root, "code", "WeCLIPPlus")
    voc_root = os.path.join(weclip_root, "VOCdevkit", "VOC2012")
    return {
        "weclip_root": weclip_root,
        "config": os.path.join(weclip_root, "configs", "voc_attn_reg.yaml"),
        "config_dir": os.path.join(weclip_root, "configs"),
        "voc_root": voc_root,
        "set_dir": os.path.join(voc_root, "ImageSets", "Main"),
        "dest_dir": os.path.join(voc_root, "JPEGImages"),
        "clip_pretrain_path": os.path.join(weclip_root, "pretrained", "ViT-B-16.pt"),
    }


def _write_runtime_config(
    base_config,
    output_dir,
    voc_root,
    clip_pretrain_path,
    dino_model=None,
    dino_fts_dim=None,
    dino_decoder_layers=None,
):
    os.makedirs(output_dir, exist_ok=True)
    name_list_dir = os.path.join(voc_root, "ImageSets", "Main")

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(base_config)
        cfg.dataset.root_dir = voc_root
        cfg.dataset.name_list_dir = name_list_dir
        cfg.clip_init.clip_pretrain_path = clip_pretrain_path
        if dino_model:
            cfg.dino_init.dino_model = dino_model
        if dino_fts_dim is not None:
            cfg.dino_init.dino_fts_fuse_dim = int(dino_fts_dim)
        if dino_decoder_layers is not None:
            cfg.dino_init.decoder_layer = int(dino_decoder_layers)

        output_path = os.path.join(output_dir, "voc_attn_reg_runtime.yaml")
        OmegaConf.save(cfg, output_path)
        return output_path
    except Exception:
        with open(base_config, "r") as f:
            content = f.read()

        content = re.sub(r"(root_dir:\s*')([^']*)(')", rf"\1{voc_root}\3", content)
        content = re.sub(r"(name_list_dir:\s*')([^']*)(')", rf"\1{name_list_dir}\3", content)
        content = re.sub(r"(clip_pretrain_path:\s*')([^']*)(')", rf"\1{clip_pretrain_path}\3", content)

        if dino_model:
            content = re.sub(r"(dino_model:\s*')([^']*)(')", rf"\1{dino_model}\3", content)
        if dino_fts_dim is not None:
            content = re.sub(r"(dino_fts_fuse_dim:\s*)([0-9]+)", rf"\1{int(dino_fts_dim)}", content)
        if dino_decoder_layers is not None:
            content = re.sub(r"(decoder_layer:\s*)([0-9]+)", rf"\1{int(dino_decoder_layers)}", content)

        output_path = os.path.join(output_dir, "voc_attn_reg_runtime.yaml")
        with open(output_path, "w") as f:
            f.write(content)
        return output_path


def _iter_class_images(split_root: str):
    """Yield (class_name, image_path) for split_root/class_name/*.ext."""
    if not os.path.isdir(split_root):
        return
    for class_name in sorted(os.listdir(split_root)):
        class_dir = os.path.join(split_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            src = os.path.join(class_dir, fname)
            if not os.path.isfile(src):
                continue
            if os.path.splitext(fname)[1].lower() not in _IMAGE_EXTS:
                continue
            yield class_name, src


def _sanitize_token(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


def _build_id(class_name: str, image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return f"{_sanitize_token(class_name)}_{_sanitize_token(stem)}"


def _prepare_redmeat_dataset(split_images_dir: str, class_name: str, set_dir: str, dest_dir: str) -> Dict[str, int]:
    """
    Ingest split_images/{train,val}/<class>/*.jpg and write everything as one
    pooled dataset with IDs: <source_class>_<image_stem>.
    """
    os.makedirs(set_dir, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)

    split_roots = [
        os.path.join(split_images_dir, "train"),
        os.path.join(split_images_dir, "val"),
    ]
    for split_root in split_roots:
        if not os.path.isdir(split_root):
            raise FileNotFoundError(f"Missing required split directory: {split_root}")

    basenames: List[str] = []
    seen_ids = set()
    copied = 0
    skipped = 0

    for split_root in split_roots:
        for src_class, image_path in _iter_class_images(split_root):
            base_id = _build_id(src_class, image_path)
            unique_id = base_id
            suffix = 1
            while unique_id in seen_ids:
                unique_id = f"{base_id}_{suffix}"
                suffix += 1
            seen_ids.add(unique_id)

            dst_path = os.path.join(dest_dir, unique_id + ".jpg")
            if os.path.exists(dst_path):
                skipped += 1
            else:
                Image.open(image_path).convert("RGB").save(dst_path, "JPEG", quality=95)
                copied += 1
            basenames.append(unique_id)

    basenames = sorted(basenames)
    if not basenames:
        raise RuntimeError(
            f"No images found under train/val class folders in: {split_images_dir}"
        )

    # Use all images for both train and val, so both dist_clip_voc and
    # test_msc_flip_voc process the full pooled set.
    train_path = os.path.join(set_dir, "train.txt")
    val_path = os.path.join(set_dir, "val.txt")
    cls_train_path = os.path.join(set_dir, f"{class_name}_train.txt")
    cls_val_path = os.path.join(set_dir, f"{class_name}_val.txt")

    with open(train_path, "w") as f:
        f.write("\n".join(basenames) + "\n")
    with open(val_path, "w") as f:
        f.write("\n".join(basenames) + "\n")
    with open(cls_train_path, "w") as f:
        f.writelines(f"{b} 1\n" for b in basenames)
    with open(cls_val_path, "w") as f:
        f.writelines(f"{b} 1\n" for b in basenames)

    return {
        "num_ids": len(basenames),
        "copied": copied,
        "skipped_existing": skipped,
    }


def main(
    repo_root,
    split_images_dir,
    class_name,
    setup_data,
    results_dir,
    clip_backend,
    clip_model,
    clip_pretrained,
    dino_model,
    dino_fts_dim,
    dino_decoder_layers,
):
    if class_name:
        os.environ["CLIP_TEXT_VERSION"] = class_name
    if clip_backend:
        os.environ["CLIP_BACKEND"] = clip_backend
    if clip_model:
        os.environ["CLIP_MODEL_NAME"] = clip_model
    if clip_pretrained:
        os.environ["CLIP_PRETRAINED"] = clip_pretrained

    from scripts import dist_clip_voc
    import test_msc_flip_voc

    paths = _resolve_paths(repo_root)
    clip_pretrain_path = clip_model or paths["clip_pretrain_path"]

    if setup_data:
        stats = _prepare_redmeat_dataset(
            split_images_dir=split_images_dir,
            class_name=class_name,
            set_dir=paths["set_dir"],
            dest_dir=paths["dest_dir"],
        )
        print(
            "Prepared Red Meat dataset: "
            f"{stats['num_ids']} pooled IDs, "
            f"{stats['copied']} copied, "
            f"{stats['skipped_existing']} skipped existing."
        )
    else:
        train_txt = os.path.join(paths["set_dir"], "train.txt")
        val_txt = os.path.join(paths["set_dir"], "val.txt")
        if not os.path.isfile(train_txt) or not os.path.isfile(val_txt):
            raise FileNotFoundError(
                "ImageSets/Main train.txt/val.txt missing. "
                "Run once with --setup-data."
            )

    config = _write_runtime_config(
        paths["config"],
        paths["config_dir"],
        paths["voc_root"],
        clip_pretrain_path,
        dino_model=dino_model,
        dino_fts_dim=dino_fts_dim,
        dino_decoder_layers=dino_decoder_layers,
    )

    from omegaconf import OmegaConf

    runtime_cfg = OmegaConf.load(config)
    if dino_model:
        runtime_cfg.dino_init.dino_model = dino_model
    if dino_fts_dim is not None:
        runtime_cfg.dino_init.dino_fts_fuse_dim = int(dino_fts_dim)
    if dino_decoder_layers is not None:
        runtime_cfg.dino_init.decoder_layer = int(dino_decoder_layers)
    OmegaConf.save(runtime_cfg, config)

    print(
        f"Runtime config: dino_model={runtime_cfg.dino_init.dino_model}, "
        f"dino_fts_fuse_dim={runtime_cfg.dino_init.dino_fts_fuse_dim}, "
        f"decoder_layer={runtime_cfg.dino_init.decoder_layer}"
    )

    final_path = dist_clip_voc.main(config)

    if results_dir:
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(paths["weclip_root"], results_dir)
        test_msc_flip_voc.args.work_dir = results_dir

    test_msc_flip_voc.outer_main(final_path, config, cfg_override=runtime_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate pseudo masks for Red Meat dataset in Food-101 split format. "
            "Reads split_images/train and split_images/val and pools both."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=_default_repo_root(),
        help="Absolute path to the LearningToLook repo root.",
    )
    parser.add_argument(
        "--split-images-dir",
        default="/home/ryreu/guided_cnn/Food101/data/food-101-redmeat/split_images",
        help="Path to directory containing train/ and val/ class folders.",
    )
    parser.add_argument(
        "--class-name",
        default="meat",
        help="Foreground class label used by CLIP text and ImageSets labels.",
    )
    parser.add_argument(
        "--setup-data",
        dest="setup_data",
        action="store_true",
        help="Prepare VOC-style JPEGImages/ImageSets from split_images/train+val.",
    )
    parser.add_argument(
        "--no-setup-data",
        dest="setup_data",
        action="store_false",
        help="Skip data setup and reuse current VOC-style data.",
    )
    parser.add_argument(
        "--results-dir",
        default="results_redmeat",
        help="Output directory for prediction_cmap.",
    )
    parser.add_argument(
        "--clip-backend",
        default=None,
        choices=["openai", "openclip", "siglip2"],
        help="Override CLIP backend for this run.",
    )
    parser.add_argument(
        "--clip-model",
        default=None,
        help=(
            "Override CLIP model identifier. For openai, this can be a checkpoint path. "
            "For openclip/siglip2, this can be an open_clip model name."
        ),
    )
    parser.add_argument(
        "--clip-pretrained",
        default=None,
        help=(
            "Override open_clip pretrained tag (e.g., openai, laion2b_s34b_b88k, webli). "
            "Only used by openclip/siglip2 backends."
        ),
    )
    parser.add_argument(
        "--dino-model",
        default=None,
        help="Override DINO model name in config (e.g., dinov2_vitb14_reg, xcit_medium_24_p16).",
    )
    parser.add_argument(
        "--dino-fts-dim",
        type=int,
        default=None,
        help="Override dino_fts_fuse_dim in config.",
    )
    parser.add_argument(
        "--dino-decoder-layers",
        type=int,
        default=None,
        help="Override decoder_layer in config.",
    )
    parser.set_defaults(setup_data=False)
    args = parser.parse_args()

    main(
        repo_root=args.repo_root,
        split_images_dir=args.split_images_dir,
        class_name=args.class_name,
        setup_data=args.setup_data,
        results_dir=args.results_dir,
        clip_backend=args.clip_backend,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        dino_model=args.dino_model,
        dino_fts_dim=args.dino_fts_dim,
        dino_decoder_layers=args.dino_decoder_layers,
    )
