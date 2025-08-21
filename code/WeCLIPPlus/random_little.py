from move_data import dataset_initializer, moveImageSets, moveImgs, convert_to_jpg, moveBack, sort_by_label
import shutil
import os





src_img_dir = r'/workspace/LearningToLook/data/saved/ColorMNIST_images'


# sort_by_label.main(src_img_dir + '/digit/test')
old_path = os.path.join(src_img_dir, "digit", "train", "JPEGImages")
new_path = os.path.join(src_img_dir, "digit", "train", "train")

# os.rename(old_path, new_path)
# os.rename(src_img_dir + '/digit/train/', src_img_dir + '/digit/old/')


shutil.move(src_img_dir + '/digit/old/train', src_img_dir + '/digit/')
