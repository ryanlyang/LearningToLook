import argparse
import os
import re
import shutil


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


def _write_runtime_config(base_config, output_dir, voc_root, clip_pretrain_path, clip_pretrained=None):
    os.makedirs(output_dir, exist_ok=True)
    with open(base_config, "r") as f:
        content = f.read()

    name_list_dir = os.path.join(voc_root, "ImageSets", "Main")

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
    if clip_pretrained is not None:
        if re.search(r"clip_pretrained:\s*'[^']*'", content):
            content = re.sub(
                r"(clip_pretrained:\s*')([^']*)(')",
                rf"\1{clip_pretrained}\3",
                content,
            )
        else:
            content = re.sub(
                r"(clip_pretrain_path:\s*'[^']*'\n)",
                rf"\1  clip_pretrained: '{clip_pretrained}'\n",
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
    clip_backend,
    clip_model,
    clip_pretrained,
    results_dir,
):
    if class_name:
        os.environ["CLIP_TEXT_VERSION"] = class_name
    if clip_backend:
        os.environ["CLIP_BACKEND"] = clip_backend
    if clip_model:
        os.environ["CLIP_MODEL_NAME"] = clip_model
    else:
        os.environ.pop("CLIP_MODEL_NAME", None)

    clip_pretrained_effective = clip_pretrained
    if clip_backend == "siglip2" and clip_pretrained in (None, "", "metaclip_fullcc"):
        # Clear config/env override so adapter can auto-pick SigLIP2 defaults.
        clip_pretrained_effective = ""

    if clip_pretrained_effective:
        os.environ["CLIP_PRETRAINED"] = clip_pretrained_effective
    else:
        os.environ.pop("CLIP_PRETRAINED", None)

    from move_data import moveImageSets, convert_to_jpg
    from scripts import dist_clip_voc
    import test_msc_flip_voc

    paths = _resolve_paths(repo_root)

    config = _write_runtime_config(
        paths["config"],
        paths["config_dir"],
        paths["voc_root"],
        clip_model or paths["clip_pretrain_path"],
        clip_pretrained_effective,
    )

    if setup_data:
        print("Setting up data")
        os.makedirs(paths["set_dir"], exist_ok=True)
        moveImageSets.main(paths["set_dir"])
        _prepare_single_class_dataset(
            src_img_dir,
            class_name,
            paths["set_dir"],
            paths["dest_dir"],
            copy_images=True,
        )
    else:
        print("Skipping Setup")

    print(
        f"Runtime CLIP: backend={clip_backend}, "
        f"model={clip_model or paths['clip_pretrain_path']}, "
        f"pretrained={clip_pretrained_effective}"
    )

    convert_to_jpg.convert_to_jpg(paths["dest_dir"], True)
    final_path = dist_clip_voc.main(config)
    if results_dir:
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(paths["weclip_root"], results_dir)
        test_msc_flip_voc.args.work_dir = results_dir
    test_msc_flip_voc.outer_main(final_path, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default="/home/ryreu/guided_cnn/waterbirds/LearningToLook",
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
        "--clip-pretrained",
        default="metaclip_fullcc",
        help=(
            "OpenCLIP/SigLIP2 pretrained tag (e.g., metaclip_fullcc, openai, webli). "
            "If backend is siglip2 and this is left as metaclip_fullcc, it auto-switches to SigLIP2 defaults."
        ),
    )
    parser.add_argument(
        "--clip-backend",
        default="openclip",
        choices=["openclip", "siglip2"],
        help="CLIP backend family. Use siglip2 to load SigLIP2 models via open_clip.",
    )
    parser.add_argument(
        "--clip-model",
        default=None,
        help="Optional model override (e.g., ViT-B-16-SigLIP2).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Output directory for predictions (prediction_cmap will be under <results-dir>/val).",
    )
    parser.set_defaults(setup_data=False)
    args = parser.parse_args()

    main(
        args.repo_root,
        args.src_img_dir,
        args.setup_data,
        args.class_name,
        args.clip_backend,
        args.clip_model,
        args.clip_pretrained,
        args.results_dir,
    )
