from scripts import dist_clip_voc
import dataset_initializer
from clip import clip_text
import moveImageSets
import moveImgs
import argparse
import convert_to_jpg



# Before running, make sure that the paths to the config file and the src_img_dir are correct. Make sure that the structure of the src_img_dir is: 
# src_img_dir/{train, val}/{folder for each class name}

# Make sure that you have edited BACKGROUND_CATEGORY and class names and new_class_names in C:\Users\ryreu\Documents\CLIP Segmentation Work\WeCLIP\WeCLIP+\clip\clip_text.py

# Make sure you have edited the correct file paths in config


config = r"/workspace/LearnToLook/WeCLIPPlus_Work/configs/voc_attn_reg.yaml"

src_img_dir = r'/workspace/LearnToLook/WeCLIPPlus_Work/VOCdevkit/VOC2012/funny_fold'

set_dir = r'/workspace/LearnToLook/WeCLIPPlus_Work/VOCdevkit/VOC2012/ImageSets/Main'

dest_dir = r'/workspace/LearnToLook/WeCLIPPlus_Work/VOCdevkit/VOC2012/JPEGImages'

class_names = clip_text.class_names

def main(setup_data):
    
    if(setup_data):
        print("Setting up data")
        moveImageSets.main(set_dir)

        dataset_initializer.main(src_img_dir,class_names, set_dir)

        moveImgs.main(src_img_dir, dest_dir, class_names)
    else:
        print("Skipping Setup")

    convert_to_jpg.convert_to_jpg(dest_dir, True)
    dist_clip_voc.main(config)



if __name__ == '__main__':

    setup_data = False
    main(setup_data)


# After this is done you can run 
# test_msc_flip_voc.py --model_path {best model path}