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
    Build VOC-style files from NICO++ common domains (sanitized IDs, no spaces).

    Args:
        src_root: path to NICO_DG (contains domain folders like autumn/dim/...)
        out_root: path to VOC root (will create JPEGImages/ and ImageSets/Main/)
        do_copy_images: if True, copy images into JPEGImages/ with sanitized names
        split_for_val: fraction in (0,1) for per-class val split; 0.0 = duplicate train/val
    """
    jpeg_dir = os.path.join(out_root, "JPEGImages")
    imagesets_dir = os.path.join(out_root, "ImageSets", "Main")
    os.makedirs(imagesets_dir, exist_ok=True)
    if do_copy_images:
        os.makedirs(jpeg_dir, exist_ok=True)

    # Collect basenames per class and all items
    images_by_class = {}           # class -> set of sanitized IDs
    all_items = []                 # list of (sanitized_id, src_path, ext)
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

            images_by_class.setdefault(_safe(label), set()).add(sid)
            all_items.append((sid, src_path, ext))

    # Optional: copy/rename images to JPEGImages with the sanitized ID
    if do_copy_images:
        placed, total = 0, len(all_items)
        for i, (sid, src_path, ext) in enumerate(all_items, 1):
            dst_path = os.path.join(jpeg_dir, sid + ext)
            if not os.path.exists(dst_path):
                shutil.copyfile(src_path, dst_path)  # faster than copy2
                placed += 1
            if i % 1000 == 0:
                print(f"[stage:copy] {i}/{total} ({placed} new)")
        print(f"Placed {placed} images → {jpeg_dir}")

    # Build splits
    ids = sorted(sid for sid, _, _ in all_items)
    if split_for_val and 0.0 < split_for_val < 1.0:
        import random, collections
        random.seed(1337)
        by_class = collections.defaultdict(list)
        for cls, members in images_by_class.items():
            by_class[cls].extend(sorted(members))

        train_set, val_set = set(), set()
        for cls, members in by_class.items():
            random.shuffle(members)
            k = max(1, int(round(split_for_val * len(members))))
            val_set.update(members[:k])
            train_set.update(members[k:])
        # ensure any ids not seen above end up in train
        train_set.update([b for b in ids if b not in val_set])
        train_list = sorted(train_set)
        val_list   = sorted(val_set)
    else:
        # identical lists (OK for wiring/tests; not for real HPO)
        train_list = ids[:]
        val_list   = ids[:]

    # Write global split files (VOC-style IDs)
    with open(os.path.join(imagesets_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_list) + "\n")
    with open(os.path.join(imagesets_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_list) + "\n")
    print(f"Wrote train.txt ({len(train_list)}) and val.txt ({len(val_list)})")

    # Write per-class label files (VOC Main: "<id> 1|-1")
    class_count = len(images_by_class)
    for split_name, split_list in [("train", train_list), ("val", val_list)]:
        print(f"[stage:labels] writing {split_name} for {class_count} classes…")
        for j, (cls, members) in enumerate(sorted(images_by_class.items()), 1):
            out_path = os.path.join(imagesets_dir, f"{cls}_{split_name}.txt")
            with open(out_path, "w") as f:
                _in = members.__contains__  # local bind for speed
                f.writelines(f"{b} {'1' if _in(b) else '-1'}\n" for b in split_list)
            if j % 10 == 0:
                print(f"  {j}/{class_count}")

    print(f"Wrote {class_count} × per-class label files in {imagesets_dir}")

if __name__ == "__main__":
    # Example usage:
    # main(
    #   src_root="/workspace/code/NICO-plus/datasets/NICO/DG_Benchmark/NICO_DG",
    #   out_root="WeCLIPPlus/VOCdevkit/VOC2012",
    #   do_copy_images=True,
    #   split_for_val=0.1
    # )
    pass
