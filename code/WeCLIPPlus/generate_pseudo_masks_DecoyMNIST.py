import argparse
import os
import re
import shutil


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
        # Fallback to text replacement if OmegaConf is unavailable.
        with open(base_config, "r") as f:
            content = f.read()

        content = re.sub(
            r"(root_dir:\s*')([^']*)(')",
            rf"\1{voc_root}\3",
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


def _resolve_split_dir(src_img_dir, split):
    candidate = os.path.join(src_img_dir, split)
    if os.path.isdir(candidate):
        return candidate
    return src_img_dir


def _resolve_dataset_root(src_img_dir):
    if os.path.isdir(os.path.join(src_img_dir, "train")) or os.path.isdir(os.path.join(src_img_dir, "test")):
        return src_img_dir
    if os.path.basename(src_img_dir) in {"train", "test"}:
        parent = os.path.dirname(src_img_dir)
        if os.path.isdir(os.path.join(parent, "train")) or os.path.isdir(os.path.join(parent, "test")):
            return parent
    return None


def main(
    repo_root,
    src_img_dir,
    setup_data,
    class_name,
    split,
    sort_by_label,
    results_dir,
    dino_model,
    dino_fts_dim,
    dino_decoder_layers,
):
    if class_name:
        os.environ["CLIP_TEXT_VERSION"] = class_name

    from move_data import moveImageSets, convert_to_jpg, sort_by_label as sort_by_label_mod
    from scripts import dist_clip_voc
    import test_msc_flip_voc

    paths = _resolve_paths(repo_root)
    config = _write_runtime_config(
        paths["config"],
        paths["config_dir"],
        paths["voc_root"],
        paths["clip_pretrain_path"],
        dino_model=dino_model,
        dino_fts_dim=dino_fts_dim,
        dino_decoder_layers=dino_decoder_layers,
    )

    if src_img_dir is None:
        src_img_dir = os.path.join(repo_root, "data", "saved", "DecoyMNIST_images", "digit")

    split_dir = _resolve_split_dir(src_img_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Image directory not found: {split_dir}")

    if setup_data:
        print("Setting up data")
        os.makedirs(paths["set_dir"], exist_ok=True)
        moveImageSets.main(paths["set_dir"])
        _prepare_single_class_dataset(
            split_dir,
            class_name,
            paths["set_dir"],
            paths["dest_dir"],
            copy_images=True,
        )
    else:
        print("Skipping Setup")

    convert_to_jpg.convert_to_jpg(paths["dest_dir"], True)
    final_path = dist_clip_voc.main(config)

    if results_dir:
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(paths["weclip_root"], results_dir)
        test_msc_flip_voc.args.work_dir = results_dir

    test_msc_flip_voc.outer_main(final_path, config)

    if sort_by_label:
        dataset_root = _resolve_dataset_root(src_img_dir)
        if dataset_root is None:
            print("Could not resolve dataset root for label sorting; skipping.")
        else:
            for split_name in ("train", "test"):
                split_path = os.path.join(dataset_root, split_name)
                if os.path.isdir(split_path):
                    sort_by_label_mod.main(split_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=_default_repo_root(),
        help="Absolute path to the LearningToLook repo root.",
    )
    parser.add_argument(
        "--src-img-dir",
        default=None,
        help="DecoyMNIST digit folder or split folder (default: <repo-root>/data/saved/DecoyMNIST_images/digit).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Which split to use when src-img-dir points at the digit root (default: train).",
    )
    parser.add_argument(
        "--class-name",
        default="digit",
        help="Foreground class name for DecoyMNIST (default: digit).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Output directory for prediction_cmap (default: results).",
    )
    parser.add_argument(
        "--dino-model",
        default=None,
        help="Override DINO model name in config (e.g., xcit_medium_24_p16).",
    )
    parser.add_argument(
        "--dino-fts-dim",
        type=int,
        default=None,
        help="Override dino_fts_fuse_dim in config (e.g., 512 for XCiT-Medium).",
    )
    parser.add_argument(
        "--dino-decoder-layers",
        type=int,
        default=None,
        help="Override decoder_layer in config.",
    )
    parser.add_argument(
        "--sort-by-label",
        action="store_true",
        help="Sort DecoyMNIST train/test images into label subfolders after masks are generated.",
    )
    parser.add_argument(
        "--setup-data",
        dest="setup_data",
        action="store_true",
        help="Run data setup steps (ImageSets + image copies).",
    )
    parser.add_argument(
        "--no-setup-data",
        dest="setup_data",
        action="store_false",
        help="Skip data setup steps.",
    )
    parser.set_defaults(setup_data=False)
    args = parser.parse_args()

    main(
        args.repo_root,
        args.src_img_dir,
        args.setup_data,
        args.class_name,
        args.split,
        args.sort_by_label,
        args.results_dir,
        args.dino_model,
        args.dino_fts_dim,
        args.dino_decoder_layers,
    )
