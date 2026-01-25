from scripts import dist_clip_voc
from move_data import dataset_initializer, moveImageSets, moveImgs, convert_to_jpg, sort_by_label
from clip import clip_text
import test_msc_flip_voc
import argparse
import shutil
import os
import re





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


def main(repo_root, src_img_dir, setup_data, clip_pretrained):
    paths = _resolve_paths(repo_root)
    config = _write_runtime_config(
        paths["config"],
        paths["config_dir"],
        paths["voc_root"],
        paths["clip_pretrain_path"],
        clip_pretrained,
    )
    class_names = clip_text.class_names
    
    if setup_data:
        print("Setting up data")
        moveImageSets.main(paths["set_dir"])

        dataset_initializer.main(src_img_dir, class_names, paths["set_dir"])

        moveImgs.main(src_img_dir + '/digit/train', paths["dest_dir"], class_names)
    else:
        print("Skipping Setup")

    convert_to_jpg.convert_to_jpg(paths["dest_dir"], True)
    final_path = dist_clip_voc.main(config)
    test_msc_flip_voc.outer_main(final_path, config)

    sort_by_label.main(paths["dest_dir"])
    sort_by_label.main(src_img_dir + '/digit/test')


    
    
    shutil.move(paths["dest_dir"], src_img_dir + '/digit/')
    os.rename(src_img_dir + '/digit/JPEGImages', src_img_dir + '/digit/train')

    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Absolute path to the LearningToLook repo root.",
    )
    parser.add_argument(
        "--src-img-dir",
        required=True,
        help="Path to dataset root (expects /digit/train and /digit/test).",
    )
    # Use either `--setup-data` or `--no-setup-data`
    parser.add_argument('--setup-data', dest='setup_data', action='store_true',
                        help='Run data setup steps (move ImageSets, init dataset, move images).')
    parser.add_argument('--no-setup-data', dest='setup_data', action='store_false',
                        help='Skip data setup steps.')
    parser.add_argument(
        "--clip-pretrained",
        default=None,
        help="OpenCLIP pretrained tag or local checkpoint (e.g., openai, laion2b_s34b_b88k).",
    )
    parser.set_defaults(setup_data=False)  # default = skip
    args = parser.parse_args()

    main(args.repo_root, args.src_img_dir, args.setup_data, args.clip_pretrained)


# After this is done you can run python run_guided_CNN.py to train the model.
# make sure you hand in the data path and the Gt (Psuedo Masks) path.
# Get the Gt_path from results/predictions/ and take specifically the prediction_cmap path
# Run it like this:
# python run_guided_CNN.py path/to/data ../code/WeCLIPPlus/results/prediction_cmap
