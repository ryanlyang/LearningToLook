from scripts import dist_clip_voc
from move_data import moveImageSets, convert_to_jpg, sort_by_label
from move_data.NICO import dataset_init_NICO, move_Imgs_NICO
from clip import clip_text
import test_voc2
import argparse
import shutil
import os

#fortnite



# Before running, make sure that the paths to the config file and the src_img_dir are correct. 
# Make sure that the structure of the src_img_dir is: 
# src_img_dir/{train, val}/{folder for each class name}

# Make sure that you have edited BACKGROUND_CATEGORY and class names and new_class_names in 
# C:\Users\ryreu\Documents\CLIP Segmentation Work\WeCLIP\WeCLIP+\clip\clip_text.py

# Make sure you have edited the correct file paths in configs/voc_attn_reg.yaml


config = r"/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/configs/voc_attn_reg.yaml"

src_img_dir = r'/home/ryreu/guided_cnn/code/NICO-plus/data/Unzip_DG_Bench/DG_Benchmark/NICO_DG'
 
set_dir = r'/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/VOCdevkit/VOC2012/ImageSets/Main'

dest_dir = r'/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/VOCdevkit/VOC2012/JPEGImages'

class_names = clip_text.class_names

def main(setup_data):
    
    if(setup_data):
        print("Setting up data")
        moveImageSets.main(set_dir)

        dataset_init_NICO.main(src_img_dir, r'/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/VOCdevkit/VOC2012')

        move_Imgs_NICO.main(src_img_dir, dest_dir)
    else:
        print("Skipping Setup")

    #convert_to_jpg.convert_to_jpg(dest_dir, True)
    # final_path = dist_clip_voc.main(config)
    # final_path = r"/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/work_dir_voc/checkpoints/2025-11-13-12-07/wetr_iter_30000.pth"
    # final_path = r"/workspace/LearningToLook/code/WeCLIPPlus/work_dir_voc/checkpoints/2025-10-19-05-29/wetr_iter_5000.pth"
    # test_voc2.outer_main(final_path)

    # sort_by_label.main(dest_dir)
    # sort_by_label.main(src_img_dir + '/digit/test')


    
    
    # shutil.move(dest_dir, src_img_dir + '/digit/')
    # os.rename(src_img_dir + '/digit/JPEGImages', src_img_dir + '/digit/train')

    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Use either `--setup-data` or `--no-setup-data`
    parser.add_argument('--setup-data', dest='setup_data', action='store_true',
                        help='Run data setup steps (move ImageSets, init dataset, move images).')
    parser.add_argument('--no-setup-data', dest='setup_data', action='store_false',
                        help='Skip data setup steps.')
    parser.set_defaults(setup_data=True)  # default = skip
    args = parser.parse_args()

    main(args.setup_data)


# After this is done you can run python run_guided_CNN.py to train the model.
# make sure you hand in the data path and the Gt (Psuedo Masks) path.
# Get the Gt_path from results/predictions/ and take specifically the prediction_cmap path
# Run it like this:
# python run_guided_CNN.py path/to/data ../code/WeCLIPPlus/results/prediction_cmap