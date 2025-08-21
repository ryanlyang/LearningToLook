import os

# ——— EDIT THIS PATH ———
base_img_dir = r"WeCLIPPlus/image_sets/color2.1"
# Expecting subfolders:
#   base_img_dir/fish/*.jpg
#   base_img_dir/chainsaw/*.jpg
#   etc.

min_output_dir = r'WeCLIPPlus/VOCdevkit/VOC2012/ImageSets/Main'
class_names = ['digit']  # update with your class names
# —————————————————————

# os.makedirs(output_dir, exist_ok=True)


def main(input_dir, classes, output_dir):
    # Collect all basenames from each class folder

    # img_dir = os.path.join(input_dir, 'train')
    # print(img_dir)
    all_basenames = set()
    images_by_class = {cls: set() for cls in classes}

    for cls in classes:
        cls_folder_old = os.path.join(input_dir, cls)
        cls_folder = os.path.join(cls_folder_old, 'train')
        if not os.path.isdir(cls_folder):
            print(f"Warning: class folder not found: {cls_folder}")
            continue
        for fname in os.listdir(cls_folder):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            base = os.path.splitext(fname)[0]
            all_basenames.add(base)
            images_by_class[cls].add(base)

    # Sort for consistency
    sorted_basenames = sorted(all_basenames)

    # For both train and val, generate identical lists
    for split in ['train', 'val']:
        # 1) Write the bare split file
        split_path = os.path.join(output_dir, f'{split}.txt')
        with open(split_path, 'w') as f_split:
            for base in sorted_basenames:
                f_split.write(f"{base}\n")

        # 2) Write per-class presence files
        for cls in classes:
            cls_split_path = os.path.join(output_dir, f'{cls}_{split}.txt')
            with open(cls_split_path, 'w') as f_cls:
                for base in sorted_basenames:
                    label = '1' if base in images_by_class[cls] else '-1'
                    f_cls.write(f"{base} {label}\n")

        print(f"Generated: {split}.txt ({len(sorted_basenames)} entries)")
        for cls in classes:
            print(f"           {cls}_{split}.txt")


if __name__ == '__main__':
    main(base_img_dir, class_names, min_output_dir)
