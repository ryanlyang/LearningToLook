from scripts import dist_clip_voc
from move_data import moveImageSets, convert_to_jpg, sort_by_label
from move_data.NICO import dataset_init_NICO, config_dupe
import test_msc_flip_voc
import argparse
import shutil
import os
import re


def _resolve_paths(repo_root, class_name):
    repo_root = os.path.abspath(repo_root)
    weclip_root = os.path.join(repo_root, "code", "WeCLIPPlus")
    base_config = os.path.join(weclip_root, "configs", "voc_attn_reg.yaml")
    config_dir = os.path.join(weclip_root, "configs", "NICO_configs")
    voc_root = os.path.join(weclip_root, "VOCdevkit", "VOC2012")

    return {
        "repo_root": repo_root,
        "weclip_root": weclip_root,
        "base_config": base_config,
        "config_dir": config_dir,
        "base_voc_root": voc_root,
        "set_dir": os.path.join(voc_root, class_name, "ImageSets", "Main"),
        "dest_dir": os.path.join(voc_root, class_name, "JPEGImages"),
        "dev_kit_dir": os.path.join(weclip_root, "VOCdevkit"),
        "clip_pretrain_path": os.path.join(weclip_root, "pretrained", "ViT-B-16.pt"),
    }


def _write_base_config(base_config, output_dir, base_voc_root, clip_pretrain_path):
    os.makedirs(output_dir, exist_ok=True)
    with open(base_config, "r") as f:
        content = f.read()

    name_list_dir = os.path.join(base_voc_root, "ImageSets", "Main")

    content = re.sub(
        r"(root_dir:\s*')([^']*)(')",
        rf"\1{base_voc_root}\3",
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

    output_path = os.path.join(output_dir, "voc_attn_reg_base.yaml")
    with open(output_path, "w") as f:
        f.write(content)
    return output_path

def main(repo_root, src_img_dir, setup_data, class_name=None):
    # Get class from CLIP_TEXT_VERSION environment variable, default to 'bear'
    # Keep underscores as-is because directory structure uses underscores.
    this_class = class_name or os.environ.get("CLIP_TEXT_VERSION", "bear")

    paths = _resolve_paths(repo_root, this_class)
    base_config = _write_base_config(
        paths["base_config"],
        paths["config_dir"],
        paths["base_voc_root"],
        paths["clip_pretrain_path"],
    )
    if setup_data:
        print("Setting up data")
        # moveImageSets.main(set_dir)

        dataset_init_NICO.main(src_img_dir, paths["dev_kit_dir"],
                               do_copy_images=True, split_for_val=0.0)
    else:
        print("Skipping Setup")

    new_config = config_dupe.main(base_config, paths["config_dir"], this_class)

    #convert_to_jpg.convert_to_jpg(dest_dir, True)

    final_path = dist_clip_voc.main(new_config)
    # final_path = r"/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/work_dir_voc/checkpoints/2025-11-13-12-07/wetr_iter_30000.pth"
    # final_path = r"/workspace/LearningToLook/code/WeCLIPPlus/work_dir_voc/checkpoints/2025-10-19-05-29/wetr_iter_5000.pth"
    # test_voc2.outer_main(final_path)
    test_msc_flip_voc.outer_main(final_path, config_path=new_config)

    # sort_by_label.main(paths["dest_dir"])
    # sort_by_label.main(src_img_dir + '/digit/test')


    
    
    # shutil.move(paths["dest_dir"], src_img_dir + '/digit/')
    # os.rename(src_img_dir + '/digit/JPEGImages', src_img_dir + '/digit/train')

    





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
        help="Path to NICO_DG (domain folders with class subfolders).",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="Override CLIP_TEXT_VERSION (default: env or 'bear').",
    )
    # Use either `--setup-data` or `--no-setup-data`
    parser.add_argument('--setup-data', dest='setup_data', action='store_true',
                        help='Run data setup steps (move ImageSets, init dataset, move images).')
    parser.add_argument('--no-setup-data', dest='setup_data', action='store_false',
                        help='Skip data setup steps.')
    parser.set_defaults(setup_data=False)  # default = skip
    args = parser.parse_args()

    main(args.repo_root, args.src_img_dir, args.setup_data, class_name=args.class_name)


# After this is done you can run python run_guided_CNN.py to train the model.
# make sure you hand in the data path and the Gt (Psuedo Masks) path.
# Get the Gt_path from results/predictions/ and take specifically the prediction_cmap path
# Run it like this:
# python run_guided_CNN.py path/to/data ../code/WeCLIPPlus/results/prediction_cmap
