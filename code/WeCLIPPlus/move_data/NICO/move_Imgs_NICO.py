import os
import shutil

# ——— EDIT THESE PATHS ———
base_dir = r"/workspace/code/NICO-plus/datasets/NICO/DG_Benchmark/NICO_DG"
# expects: base_dir/autumn/airplane/*.jpg, base_dir/dim/bear/*.jpg, etc.
output_dir = r"WeCLIPPlus/VOCdevkit/VOC2012/JPEGImages"
# ————————————————————————

def main(src_root, dst_root, do_copy=False):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    moved_files = 0
    os.makedirs(dst_root, exist_ok=True)

    # each top-level folder = environment (autumn, dim, etc.)
    for env in sorted(d for d in os.listdir(src_root)
                      if os.path.isdir(os.path.join(src_root, d))):
        env_dir = os.path.join(src_root, env)
        for cls in sorted(d for d in os.listdir(env_dir)
                          if os.path.isdir(os.path.join(env_dir, d))):
            cls_src = os.path.join(env_dir, cls)
            cls_dst = os.path.join(dst_root, env, cls)  # preserve structure
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
          f"from '{src_root}' → '{dst_root}' preserving filenames and hierarchy.")

if __name__ == "__main__":
    main(base_dir, output_dir, do_copy=False)  # set do_copy=True if you prefer copying
