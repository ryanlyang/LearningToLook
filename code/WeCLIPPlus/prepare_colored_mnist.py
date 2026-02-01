#!/usr/bin/env python3
"""
prepare_colored_mnist.py

Copies ColorMNIST PNG images from their source directory into the
VOCdevkit/VOC2012/JPEGImages/ layout that WeCLIP+ expects, converting
them to JPEG along the way.  Also writes the ImageSets/Main/ text files.

The source directory is NEVER modified.

Usage:
    python prepare_colored_mnist.py \
        --src /home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/ColorMNIST_png/train \
        --class-name digit

    # Or let it use defaults:
    python prepare_colored_mnist.py
"""
import argparse
import os
import re

from PIL import Image

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}


def _default_repo_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def _voc_paths(repo_root):
    weclip_root = os.path.join(repo_root, "code", "WeCLIPPlus")
    voc_root = os.path.join(weclip_root, "VOCdevkit", "VOC2012")
    return {
        "voc_root": voc_root,
        "jpeg_dir": os.path.join(voc_root, "JPEGImages"),
        "set_dir": os.path.join(voc_root, "ImageSets", "Main"),
    }


def _make_image_id(src_root, image_path):
    """Flatten the relative path into a single ID string."""
    rel = os.path.relpath(image_path, src_root)
    stem = os.path.splitext(rel)[0]
    flat = stem.replace(os.sep, "_").replace("/", "_")
    flat = re.sub(r"[^A-Za-z0-9_-]+", "_", flat).strip("_")
    return flat


def _iter_images(root):
    for dirpath, _, filenames in os.walk(root):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in _IMAGE_EXTS:
                yield os.path.join(dirpath, fname)


def prepare(src_dir, repo_root, class_name):
    paths = _voc_paths(repo_root)
    jpeg_dir = paths["jpeg_dir"]
    set_dir = paths["set_dir"]
    os.makedirs(jpeg_dir, exist_ok=True)
    os.makedirs(set_dir, exist_ok=True)

    ids = []
    seen = set()
    copied = 0
    skipped = 0

    for img_path in _iter_images(src_dir):
        base_id = _make_image_id(src_dir, img_path)
        uid = base_id
        suffix = 1
        while uid in seen:
            uid = f"{base_id}_{suffix}"
            suffix += 1
        seen.add(uid)

        dst = os.path.join(jpeg_dir, uid + ".jpg")
        if not os.path.exists(dst):
            Image.open(img_path).convert("RGB").save(dst, "JPEG", quality=95)
            copied += 1
        else:
            skipped += 1

        ids.append(uid)

    ids.sort()

    if not ids:
        print(f"No images found under {src_dir}")
        return

    # Write ImageSets
    train_txt = os.path.join(set_dir, "train.txt")
    val_txt = os.path.join(set_dir, "val.txt")
    cls_train = os.path.join(set_dir, f"{class_name}_train.txt")
    cls_val = os.path.join(set_dir, f"{class_name}_val.txt")

    with open(train_txt, "w") as f:
        f.write("\n".join(ids) + "\n")
    with open(val_txt, "w") as f:
        f.write("\n".join(ids) + "\n")
    with open(cls_train, "w") as f:
        f.writelines(f"{i} 1\n" for i in ids)
    with open(cls_val, "w") as f:
        f.writelines(f"{i} 1\n" for i in ids)

    print(f"Done: {copied} images copied, {skipped} already existed, {len(ids)} total IDs written.")
    print(f"  JPEGImages -> {jpeg_dir}")
    print(f"  ImageSets  -> {set_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ColorMNIST data for WeCLIP+")
    parser.add_argument(
        "--src",
        default="/home/ryreu/guided_cnn/MNIST_AGAIN/MakeMNIST/data/ColorMNIST_png/train",
        help="Source directory containing class subdirs (0/, 1/, ..., 9/) of PNG images.",
    )
    parser.add_argument(
        "--repo-root",
        default=_default_repo_root(),
        help="Repository root (default: auto-detected).",
    )
    parser.add_argument(
        "--class-name",
        default="digit",
        help="Foreground class name (default: digit).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.src):
        raise FileNotFoundError(f"Source directory not found: {args.src}")

    prepare(args.src, args.repo_root, args.class_name)
