#!/usr/bin/env python3
"""
convert_to_jpg.py

Recursively converts all non-JPEG images in the specified directory to JPEG format,
then reports a summary of conversions.
Usage:
    python convert_to_jpg.py /path/to/folder [remove]
"""
import os
import sys
from PIL import Image

# Supported source image extensions (excluding JPEG)
INPUT_EXTENSIONS = {'.png', '.bmp', '.gif', '.tiff', '.jpeg'}


def convert_to_jpg(root_dir, remove_original=False):
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.")
        return

    converted = 0
    failed = 0

    # Walk and convert
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            base, ext = os.path.splitext(fname)
            ext_lower = ext.lower()

            # Skip if already a JPEG
            if ext_lower in {'.jpg', '.jpeg'}:
                continue

            # Only attempt conversion for supported extensions
            if ext_lower not in INPUT_EXTENSIONS:
                continue

            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(dirpath, base + '.jpg')
            try:
                Image.open(src_path).convert('RGB').save(dst_path, 'JPEG', quality=95)
                converted += 1
                if remove_original:
                    os.remove(src_path)
            except Exception:
                failed += 1

    # Count resulting JPEG files
    total_jpg = 0
    for _, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in {'.jpg', '.jpeg'}:
                total_jpg += 1

    # Summary
    print(f"Conversion complete! Converted {converted} files, failed {failed} conversions, and found {total_jpg} JPG files in '{root_dir}' .")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    directory = sys.argv[1]
    remove_flag = (len(sys.argv) > 2 and sys.argv[2].lower() == 'remove')
    convert_to_jpg(directory, remove_original=remove_flag)
