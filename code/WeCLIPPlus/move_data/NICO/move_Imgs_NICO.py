import os
import shutil
import re

# ——— EDIT THESE PATHS ———
base_dir = r"/workspace/code/NICO-plus/datasets/NICO/DG_Benchmark/NICO_DG"
# expects: base_dir/autumn/airplane/*.jpg, base_dir/dim/bear/*.jpg, etc.
output_parent_dir = r"WeCLIPPlus/VOCdevkit"
# ————————————————————————

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

def main(src_root, output_parent, do_copy=False):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    moved_files = 0

    # each top-level folder = environment (autumn, dim, etc.)
    for env in sorted(d for d in os.listdir(src_root)
                      if os.path.isdir(os.path.join(src_root, d))):
        env_dir = os.path.join(src_root, env)
        for cls in sorted(d for d in os.listdir(env_dir)
                          if os.path.isdir(os.path.join(env_dir, d))):
            cls_src = os.path.join(env_dir, cls)
            safe_cls = _safe(cls)
            # Images go to VOC2012/class/JPEGImages/
            cls_dst = os.path.join(output_parent, "VOC2012", safe_cls, "JPEGImages")
            os.makedirs(cls_dst, exist_ok=True)

            for fname in os.listdir(cls_src):
                if not fname.lower().endswith(image_extensions):
                    continue

                src_path = os.path.join(cls_src, fname)
                dst_path = os.path.join(cls_dst, fname)  # SAME NAME

                try:
                    if do_copy:
                        shutil.copy2(src_path, dst_path)
                    else:
                        shutil.move(src_path, dst_path)
                    moved_files += 1
                except Exception as e:
                    print(f"Error moving {src_path} → {dst_path}: {e}")

    print(f"{'Copied' if do_copy else 'Moved'} {moved_files} images "
          f"from '{src_root}' → '{output_parent}' into per-class VOC2012 folders.")

if __name__ == "__main__":
    main(base_dir, output_parent_dir, do_copy=False)  # set do_copy=True if you prefer copying
