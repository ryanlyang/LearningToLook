import os
import shutil

# ——— EDIT THESE PATHS ———
base_dir = r'WeCLIPPlus/image_sets/ColorMNIST'
# Expecting directory structure:
#   base_dir/digit/*.jpg, base_dir/fish/*.png, etc.

output_dir = r'WeCLIPPlus/VOCdevkit/VOC2012/JPEGImages'
class_names = ['digit']  # update with your class subfolders
# —————————————————————

# Create destination folder if it doesn’t exist
# os.makedirs(output_dir, exist_ok=True)


def main(src_root, dst_root, classes):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    moved_files = 0
    os.makedirs(dst_root, exist_ok=True)


    for cls in classes:
        cls_dir = src_root
        if not os.path.isdir(cls_dir):
            print(f"Warning: class folder not found: {cls_dir}")
            continue

        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(image_extensions):
                continue
            src_path = os.path.join(cls_dir, fname)
            dst_path = os.path.join(dst_root, fname)
            try:
                shutil.move(src_path, dst_path)
                moved_files += 1
            except Exception as e:
                print(f"Error moving {src_path} → {dst_path}: {e}")

    print(f"Moved {moved_files} image files from '{src_root}' to '{dst_root}'")


if __name__ == '__main__':
    main(base_dir, output_dir, class_names)
