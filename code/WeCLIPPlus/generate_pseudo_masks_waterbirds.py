import argparse
import os
import re
import shutil


def _resolve_paths(repo_root, voc_workspace_name):
    repo_root = os.path.abspath(repo_root)
    weclip_root = os.path.join(repo_root, "code", "WeCLIPPlus")
    workspace = voc_workspace_name or "VOC2012"
    # Keep a VOC-style layout for compatibility with WeCLIP internals.
    mask_data_root = os.path.join(weclip_root, "VOCdevkit", workspace)
    return {
        "weclip_root": weclip_root,
        "config": os.path.join(weclip_root, "configs", "voc_attn_reg.yaml"),
        "config_dir": os.path.join(weclip_root, "configs"),
        "mask_data_root": mask_data_root,
        "imageset_dir": os.path.join(mask_data_root, "ImageSets", "Main"),
        "jpegimages_dir": os.path.join(mask_data_root, "JPEGImages"),
        "clip_pretrain_path": os.path.join(weclip_root, "pretrained", "ViT-B-16.pt"),
    }


def _write_runtime_config(
    base_config,
    output_dir,
    mask_data_root,
    clip_pretrain_path,
    dino_model=None,
    dino_fts_dim=None,
    dino_decoder_layers=None,
):
    os.makedirs(output_dir, exist_ok=True)
    name_list_dir = os.path.join(mask_data_root, "ImageSets", "Main")

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(base_config)
        cfg.dataset.root_dir = mask_data_root
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

        content = re.sub(
            r"(root_dir:\s*')([^']*)(')",
            rf"\1{mask_data_root}\3",
            content,
        )
        content = re.sub(
            r"(name_list_dir:\s*')([^']*)(')",
            rf"\1{name_list_dir}\3",
            content,
        )
        content = re.sub(
            r"(clip_pretrain_path:\s*')([^']*)(')",
            rf"\1{clip_pretrain_path}\3",
            content,
        )
        if dino_model:
            content = re.sub(
                r"(dino_model:\s*')([^']*)(')",
                rf"\1{dino_model}\3",
                content,
            )
        if dino_fts_dim is not None:
            content = re.sub(
                r"(dino_fts_fuse_dim:\s*)([0-9]+)",
                rf"\1{int(dino_fts_dim)}",
                content,
            )
        if dino_decoder_layers is not None:
            content = re.sub(
                r"(decoder_layer:\s*)([0-9]+)",
                rf"\1{int(dino_decoder_layers)}",
                content,
            )

        output_path = os.path.join(output_dir, "voc_attn_reg_runtime.yaml")
        with open(output_path, "w") as f:
            f.write(content)
        return output_path


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _iter_image_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in _IMAGE_EXTS:
                yield os.path.join(dirpath, fname)


def _make_image_id(src_img_dir, image_path):
    rel_path = os.path.relpath(image_path, src_img_dir)
    rel_no_ext = os.path.splitext(rel_path)[0]
    flat = rel_no_ext.replace(os.sep, "_").replace("/", "_")
    flat = re.sub(r"[^A-Za-z0-9_-]+", "_", flat).strip("_")
    return flat


def _write_imagesets(set_dir, class_name, basenames):
    os.makedirs(set_dir, exist_ok=True)
    train_path = os.path.join(set_dir, "train.txt")
    val_path = os.path.join(set_dir, "val.txt")
    cls_train_path = os.path.join(set_dir, f"{class_name}_train.txt")
    cls_val_path = os.path.join(set_dir, f"{class_name}_val.txt")

    with open(train_path, "w") as f_train:
        f_train.write("\n".join(basenames) + "\n")
    with open(val_path, "w") as f_val:
        f_val.write("\n".join(basenames) + "\n")

    with open(cls_train_path, "w") as f_cls_train:
        f_cls_train.writelines(f"{b} 1\n" for b in basenames)
    with open(cls_val_path, "w") as f_cls_val:
        f_cls_val.writelines(f"{b} 1\n" for b in basenames)


def _prepare_single_class_dataset(src_img_dir, class_name, set_dir, dest_dir, copy_images=True):
    os.makedirs(dest_dir, exist_ok=True)
    basenames = []
    seen = set()

    for image_path in _iter_image_files(src_img_dir):
        base_id = _make_image_id(src_img_dir, image_path)
        unique_id = base_id
        suffix = 1
        while unique_id in seen:
            unique_id = f"{base_id}_{suffix}"
            suffix += 1
        seen.add(unique_id)

        ext = os.path.splitext(image_path)[1].lower() or ".jpg"
        dst_path = os.path.join(dest_dir, unique_id + ext)
        if not os.path.exists(dst_path):
            if copy_images:
                shutil.copyfile(image_path, dst_path)
            else:
                shutil.move(image_path, dst_path)

        basenames.append(unique_id)

    basenames = sorted(basenames)
    if not basenames:
        print(f"No images found under {src_img_dir}")
        return

    _write_imagesets(set_dir, class_name, basenames)


def main(
    repo_root,
    src_img_dir,
    setup_data,
    class_name,
    voc_workspace_name,
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
    os.environ["CLIP_TEXT_DATASET"] = "waterbirds"
    if clip_backend:
        os.environ["CLIP_BACKEND"] = clip_backend
    if clip_model:
        os.environ["CLIP_MODEL_NAME"] = clip_model
    if clip_pretrained:
        os.environ["CLIP_PRETRAINED"] = clip_pretrained

    from move_data import moveImageSets, convert_to_jpg
    from scripts import dist_clip_voc
    import test_msc_flip_voc

    paths = _resolve_paths(repo_root, voc_workspace_name)
    clip_pretrain_path = clip_model or paths["clip_pretrain_path"]
    config = _write_runtime_config(
        paths["config"],
        paths["mask_data_root"],
        paths["mask_data_root"],
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

    if setup_data:
        print("Setting up data")
        os.makedirs(paths["imageset_dir"], exist_ok=True)
        moveImageSets.main(paths["imageset_dir"])
        _prepare_single_class_dataset(
            src_img_dir,
            class_name,
            paths["imageset_dir"],
            paths["jpegimages_dir"],
            copy_images=True,
        )
    else:
        print("Skipping Setup")

    convert_to_jpg.convert_to_jpg(paths["jpegimages_dir"], True)
    final_path = dist_clip_voc.main(config)
    if results_dir:
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(paths["weclip_root"], results_dir)
        test_msc_flip_voc.args.work_dir = results_dir
    test_msc_flip_voc.outer_main(final_path, config, cfg_override=runtime_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/ryreu/guided_cnn/waterbirds/New_Teach/LearningToLook",
        help="Absolute path to the LearningToLook repo root.",
    )
    parser.add_argument(
        "--src-img-dir",
        default="/home/ryreu/guided_cnn/waterbirds/waterbird_complete95_forest2water2",
        help="Dataset root (expects class subfolders with images).",
    )
    parser.add_argument(
        "--class-name",
        default="bird",
        help="Single foreground class name for Waterbirds (default: bird).",
    )
    parser.add_argument(
        "--voc-workspace-name",
        default="VOC2012_waterbirds",
        help=(
            "Name of VOC workspace under code/WeCLIPPlus/VOCdevkit/. "
            "Use a distinct value per runner to avoid dataset overwrite."
        ),
    )
    parser.add_argument(
        "--setup-data",
        dest="setup_data",
        action="store_true",
        help="Run data setup steps (ImageSets + image moves).",
    )
    parser.add_argument(
        "--no-setup-data",
        dest="setup_data",
        action="store_false",
        help="Skip data setup steps.",
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
        "--results-dir",
        default="results",
        help="Output directory for prediction_cmap.",
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
        args.repo_root,
        args.src_img_dir,
        args.setup_data,
        args.class_name,
        args.voc_workspace_name,
        args.results_dir,
        args.clip_backend,
        args.clip_model,
        args.clip_pretrained,
        args.dino_model,
        args.dino_fts_dim,
        args.dino_decoder_layers,
    )
