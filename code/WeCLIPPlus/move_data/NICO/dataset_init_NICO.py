import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def _safe(tok: str) -> str:
    """
    Sanitize a token so it contains no spaces or path separators.
    - Replace any run of non [A-Za-z0-9-_] with a single underscore.
    - Strip leading/trailing underscores.
    """
    tok = tok.replace("\\", "/")           # normalize first
    tok = tok.split("/")[-1]               # drop any accidental path parts
    tok = re.sub(r"[^A-Za-z0-9\-]+", "_", tok)
    tok = tok.strip("_")
    return tok

def iter_images(env_dir: str) -> Iterable[Tuple[str, str, str]]:
    """Yield (domain, label, image_path) for each image under env_dir/label/*."""
    for label in sorted(d for d in os.listdir(env_dir)
                        if os.path.isdir(os.path.join(env_dir, d))):
        lbl_dir = os.path.join(env_dir, label)
        for fname in os.listdir(lbl_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in IMG_EXTS:
                yield os.path.basename(env_dir), label, os.path.join(lbl_dir, fname)

def main(src_root: str,
         out_root: str,
         do_copy_images: bool = True,
         split_for_val: float = 0.0):
    """
    Build per-class VOC-style directories from NICO++ domains.
    Creates separate VOC2012{class_name} folders for each class.

    Args:
        src_root: path to NICO_DG (contains domain folders like autumn/dim/...)
        out_root: parent directory where VOC2012{class} folders will be created
        do_copy_images: if True, copy images into JPEGImages/ with sanitized names
        split_for_val: fraction in (0,1) for per-class val split; 0.0 = duplicate train/val
    """
    # Collect basenames per class and all items
    images_by_class = {}           # class -> set of sanitized IDs
    items_by_class = {}            # class -> list of (sanitized_id, src_path, ext)
    domains = sorted(d for d in os.listdir(src_root)
                     if os.path.isdir(os.path.join(src_root, d)))
    print(f"Found environments: {domains}")

    for domain in domains:
        env_dir = os.path.join(src_root, domain)
        for dom, label, src_path in iter_images(env_dir):
            base = Path(src_path).stem
            ext = Path(src_path).suffix.lower()

            # *** SANITIZED ID: include BOTH domain and label, no spaces ***
            sid = f"{_safe(dom)}_{_safe(label)}_{_safe(base)}"
            safe_label = _safe(label)

            images_by_class.setdefault(safe_label, set()).add(sid)
            items_by_class.setdefault(safe_label, []).append((sid, src_path, ext))

    # Build per-class VOC2012{class_name} folders
    import random
    random.seed(1337)

    class_count = len(images_by_class)
    for cls in sorted(images_by_class.keys()):
        # Create per-class VOC folder
        voc_cls_dir = os.path.join(out_root, "VOC2012", cls)
        jpeg_dir = os.path.join(voc_cls_dir, "JPEGImages")
        imagesets_dir = os.path.join(voc_cls_dir, "ImageSets", "Main")
        os.makedirs(imagesets_dir, exist_ok=True)
        if do_copy_images:
            os.makedirs(jpeg_dir, exist_ok=True)

        # Copy/move images for this class
        placed = 0
        for sid, src_path, ext in items_by_class[cls]:
            dst_path = os.path.join(jpeg_dir, sid + ext)
            if not os.path.exists(dst_path):
                os.makedirs(jpeg_dir, exist_ok=True)
                if do_copy_images:
                    shutil.copyfile(src_path, dst_path)
                else:
                    shutil.move(src_path, dst_path)
                placed += 1
        print(f"[{cls}] Placed {placed} images → {jpeg_dir}")

        # Build train/val split for this class
        members_list = sorted(images_by_class[cls])
        if split_for_val and 0.0 < split_for_val < 1.0:
            train_list, val_list = [], []
            shuffled = members_list[:]
            random.shuffle(shuffled)
            k = max(1, int(round(split_for_val * len(shuffled))))
            val_list = sorted(shuffled[:k])
            train_list = sorted(shuffled[k:])
        else:
            # identical lists (OK for wiring/tests; not for real HPO)
            train_list = members_list[:]
            val_list   = members_list[:]

        # Write class-specific train.txt and val.txt (only images of this class)
        with open(os.path.join(imagesets_dir, "train.txt"), "w") as f:
            f.write("\n".join(train_list) + "\n")
        with open(os.path.join(imagesets_dir, "val.txt"), "w") as f:
            f.write("\n".join(val_list) + "\n")

        # Write per-class label files (VOC Main format: "<id> 1|-1")
        members_set = set(members_list)
        for split_name, split_list in [("train", train_list), ("val", val_list)]:
            out_path = os.path.join(imagesets_dir, f"{cls}_{split_name}.txt")
            with open(out_path, "w") as f:
                _in = members_set.__contains__  # local bind for speed
                f.writelines(f"{b} {'1' if _in(b) else '-1'}\n" for b in split_list)

    print(f"Wrote {class_count} per-class VOC2012 folders in {out_root}")

if __name__ == "__main__":
    # Example usage:
    # main(
    #   src_root="/workspace/code/NICO-plus/datasets/NICO/DG_Benchmark/NICO_DG",
    #   out_root="WeCLIPPlus/VOCdevkit/VOC2012",
    #   do_copy_images=True,
    #   split_for_val=0.1
    # )
    pass
