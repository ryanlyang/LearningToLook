import os
import shutil
from datetime import datetime

# Define paths
set_dir = r'/home/ryan/ComputerScience/LearnToLook/Learning2/LearningToLook/code/WeCLIPPlus/VOCdevkit/VOC2012/ImageSets/Main'
# parent_dir = os.path.dirname(main_dir)


def main(img_set_dir):

    parent_dir = os.path.dirname(img_set_dir)
    # List all files in 'Main' (ignoring subdirectories)
    files = [f for f in os.listdir(img_set_dir) if os.path.isfile(os.path.join(img_set_dir, f))]

    # Proceed only if files exist
    if files:
        # Generate timestamped folder name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_folder = os.path.join(parent_dir, timestamp)

        # Create the new folder
        os.makedirs(new_folder, exist_ok=True)

        # Move each file
        for file in files:
            shutil.move(os.path.join(img_set_dir, file), os.path.join(new_folder, file))

        print(f"Moved {len(files)} files to {new_folder}")
    else:
        print("No files found in the 'Main' directory.")

if __name__ == '__main__':
    main(set_dir)
