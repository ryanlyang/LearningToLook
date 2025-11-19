import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap
import imageio.v2 as imageio
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--logit_dir", type=str, required=True, help="Directory containing .npy logit files")
parser.add_argument("--output_dir", type=str, required=True, help="Base directory for output predictions and cmaps")
parser.add_argument("--use_crf", action="store_true", default=False, help="Apply DenseCRF post-processing")
parser.add_argument("--image_dir", type=str, default=None, help="Directory with original images (required if --use_crf is set)")
args = parser.parse_args()

def process_logits(logit_dir, output_dir, use_crf=False, image_dir=None):
    """
    Convert logit files to predictions and prediction cmaps.

    Args:
        logit_dir: Directory containing .npy logit files
        output_dir: Base output directory
        use_crf: Whether to apply DenseCRF post-processing
        image_dir: Directory with original images (required if use_crf=True)
    """

    if use_crf and image_dir is None:
        raise ValueError("image_dir is required when use_crf=True")

    # Create output directories
    pred_dir = os.path.join(output_dir, "prediction")
    cmap_dir = os.path.join(output_dir, "prediction_cmap")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(cmap_dir, exist_ok=True)

    # Initialize CRF post-processor if needed
    if use_crf:
        post_processor = DenseCRF(
            iter_max=10,
            pos_xy_std=3,
            pos_w=3,
            bi_xy_std=64,
            bi_rgb_std=5,
            bi_w=4,
        )

    # Get list of .npy files
    logit_dir = Path(logit_dir)
    npy_files = sorted(logit_dir.glob("*.npy"))

    if not npy_files:
        print(f"No .npy files found in {logit_dir}")
        return

    print(f"Found {len(npy_files)} logit files to process")

    for npy_file in tqdm(npy_files, desc="Converting logits to predictions"):
        name = npy_file.stem  # filename without extension

        # Load logit file
        logit_data = np.load(npy_file, allow_pickle=True).item()
        logit = logit_data['msc_segs']  # Use multi-scale predictions

        # Convert to torch and apply softmax
        logit = torch.FloatTensor(logit)

        if use_crf:
            # Load original image for CRF
            image_name = os.path.join(image_dir, name + ".jpg")
            if not os.path.exists(image_name):
                print(f"Warning: Image not found for {name}, skipping CRF")
                prob = F.softmax(logit, dim=1)[0].numpy()
                pred = np.argmax(prob, axis=0)
            else:
                image = imageio.imread(image_name).astype(np.float32)

                if image.ndim == 2:
                    image = np.stack([image, image, image], axis=-1)

                H, W, _ = image.shape
                logit_resized = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
                prob = F.softmax(logit_resized, dim=1)[0].numpy()

                image = image.astype(np.uint8)
                prob = post_processor(image, prob)
                pred = np.argmax(prob, axis=0)
        else:
            # No CRF, just softmax + argmax
            prob = F.softmax(logit, dim=1)[0].numpy()
            pred = np.argmax(prob, axis=0)

        # Save prediction
        pred_path = os.path.join(pred_dir, name + ".png")
        imageio.imsave(pred_path, np.squeeze(pred).astype(np.uint8))

        # Save prediction colormap
        cmap_path = os.path.join(cmap_dir, name + ".png")
        imageio.imsave(cmap_path, encode_cmap(np.squeeze(pred)).astype(np.uint8))

    print(f"\nProcessing complete!")
    print(f"Predictions saved to: {pred_dir}")
    print(f"Prediction cmaps saved to: {cmap_dir}")

if __name__ == "__main__":
    process_logits(
        logit_dir=args.logit_dir,
        output_dir=args.output_dir,
        use_crf=args.use_crf,
        image_dir=args.image_dir
    )
