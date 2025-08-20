import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

# -------------------------
# Paths
# -------------------------
OUT_ROOT = "saved/DecoyMNIST_images/digit"
TRAIN_DIR = os.path.join(OUT_ROOT, "train")
TEST_DIR  = os.path.join(OUT_ROOT, "test")
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(0)

# -------------------------
# Load MNIST
# -------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # uint8, shapes: (60000,28,28), (10000,28,28)

# -------------------------
# Helpers
# -------------------------
def save_png_gray(arr_hw_uint8: np.ndarray, path: str):
    """Save (H, W) uint8 array as grayscale PNG."""
    Image.fromarray(arr_hw_uint8, mode="L").save(path)

def add_decoy_square(img_hw_uint8: np.ndarray, label: int, train_split: bool, square_size: int = 5) -> np.ndarray:
    """
    Return a copy of img with a 5x5 decoy square in a random corner.
    Train: intensity = 255 - 25*label
    Test:  intensity = 0 + 25*label
    """
    assert img_hw_uint8.shape == (28, 28)
    out = img_hw_uint8.copy()

    # Choose top/left in {0, 23} -> corners (0,0), (0,23), (23,0), (23,23)
    top  = np.random.choice([0, 28 - square_size])
    left = np.random.choice([0, 28 - square_size])

    if train_split:
        val = 255 - 25 * int(label)
    else:
        val = 0 + 25 * int(label)
    val = int(np.clip(val, 0, 255))

    out[top:top+square_size, left:left+square_size] = val
    return out

def write_split(images: np.ndarray, labels: np.ndarray, out_dir: str, train_split: bool):
    """
    Write all images with decoy squares to out_dir.
    Filenames: 000000_lblY.png
    """
    n = images.shape[0]
    for i in tqdm(range(n), desc=f"Saving {'train' if train_split else 'test'}"):
        y = int(labels[i])
        img = images[i].astype(np.uint8)  # (28,28), uint8
        img_decoy = add_decoy_square(img, y, train_split=train_split)
        fname = f"{i:06d}_lbl{y}.png"
        save_png_gray(img_decoy, os.path.join(out_dir, fname))

# -------------------------
# Save images
# -------------------------
write_split(x_train, y_train, TRAIN_DIR, train_split=True)   # 60,000
write_split(x_test,  y_test,  TEST_DIR,  train_split=False)  # 10,000

print(f"Done! Images saved under: {os.path.abspath(OUT_ROOT)}")
