import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
from tqdm import tqdm
from colour import Color
from pathlib import Path
import random
from PIL import Image


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)


red = Color("red")
colors = list(red.range_to(Color("purple"), 10))  # 10 evenly-spaced colors
# Convert to np arrays in [0,1], shape (10, 3)
colors = np.stack([np.array(c.get_rgb(), dtype=np.float32) for c in colors], axis=0)  # (10, 3)


root = Path("saved")
out_root = Path("saved/ColorMNIST_images/digit")
(out_root / "train").mkdir(parents=True, exist_ok=True)
(out_root / "test").mkdir(parents=True, exist_ok=True)


def to_rgb_uint8(img_2d_uint8, rgb_in_01):
    """
    img_2d_uint8: (H, W) uint8 grayscale in [0, 255]
    rgb_in_01: (3,) float32 in [0, 1]
    returns: (H, W, 3) uint8
    """
    colored = (rgb_in_01[:, None, None] * img_2d_uint8.astype(np.float32))  # (3, H, W) float
    colored = np.clip(colored, 0, 255).astype(np.uint8)  # (3, H, W) uint8
    return np.moveaxis(colored, 0, 2)  # -> (H, W, 3)

def gray_to_rgb_uint8(img_2d_uint8):
    """Replicate a grayscale uint8 (H, W) to RGB (H, W, 3)."""
    return np.repeat(img_2d_uint8[:, :, None], 3, axis=2)

def save_png(arr_hw3_uint8, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_hw3_uint8, mode="RGB").save(path)

def get_label_from_dataset(ds, idx):
    # Torchvision MNIST uses .targets; older had .train_labels. Handle both safely.
    if hasattr(ds, "targets"):
        return int(ds.targets[idx].item())
    if hasattr(ds, "train_labels"):
        return int(ds.train_labels[idx].item())
    raise AttributeError("Dataset does not have 'targets' or 'train_labels'")

def get_image_from_dataset(ds, idx):
    # Raw tensor data: uint8, shape (H, W)
    return ds.data[idx].numpy()

def save_mnist_split(ds, split_name: str, p_color: float = 1.0, reverse_map: bool = False):
    """
    ds: MNIST dataset
    split_name: 'train' or 'test'
    p_color: probability to colorize; else grayscale replicated to RGB
    reverse_map: if True, use colors[9 - y] instead of colors[y]
    """
    base_dir = out_root / split_name
    n = len(ds)
    for i in tqdm(range(n), desc=f"Saving {split_name}"):
        y = get_label_from_dataset(ds, i)
        img2d = get_image_from_dataset(ds, i)  # (28, 28) uint8

        if random.random() <= p_color:
            rgb = colors[9 - y] if reverse_map else colors[y]
            rgb_img = to_rgb_uint8(img2d, rgb)
        else:
            rgb_img = gray_to_rgb_uint8(img2d)

        # Flat folder, no per-label subdirs; keep label in filename
        fname = f"{i:06d}_lbl{y}.png"
        save_png(rgb_img, base_dir / fname)


mnist_train = datasets.MNIST(root=root, train=True, download=True, transform=None)   # 60,000
mnist_test  = datasets.MNIST(root=root, train=False, download=True, transform=None)  # 10,000


# Train: colored with colors[y]
save_mnist_split(mnist_train, "train", p_color=1.0, reverse_map=False)

# Test: colored with reversed mapping colors[9 - y]
save_mnist_split(mnist_test, "test", p_color=1.0, reverse_map=True)

print(f"Done! Images saved under: {out_root.resolve()}")
