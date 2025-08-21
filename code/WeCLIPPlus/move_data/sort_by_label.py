import os
import re
import shutil

# if you're running this script from the folder with the images, leave this as "."
SOURCE_DIR = r"data\saved\colored2\test"
# regex to capture the label digit after "lbl"
pattern = re.compile(r'^.+_lbl([0-9])\.(?:jpg|jpeg|png)$', re.IGNORECASE)

def main(src_dir):

    for filename in os.listdir(src_dir):
        match = pattern.match(filename)
        if not match:
            continue  # skip files that don't match the pattern

        label = match.group(1)            # the digit after "lbl"
        src_path = os.path.join(src_dir, filename)
        dest_dir = os.path.join(src_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, filename)

        shutil.move(src_path, dest_path)
        print(f"Moved {filename} → {label}/")

    print("Done!")

if __name__ == '__main__':
    main(SOURCE_DIR)