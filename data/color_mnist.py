import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.datasets import ColoredDataset
from utils.measure import *
# from utils.models import *
from multiprocessing import freeze_support

import os
from tqdm import tqdm
import torchvision.utils as vutils
from pathlib import Path

import argparse


try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
  
    SCRIPT_DIR = Path.cwd()


root = SCRIPT_DIR / "saved"
OUT_ROOT = os.path.join(SCRIPT_DIR, "saved", "ColorMNIST_images", "digit")




def save_dataset(dataset, out_dir):
    """
    Saves every image in `dataset` to disk under `out_dir`, naming them
    with a zero‑padded index and their label.
    """
    os.makedirs(out_dir, exist_ok=True)
    for idx in tqdm(range(len(dataset)), desc=f"Saving {out_dir}"):
        img, label = dataset[idx]               # img is a C×H×W tensor
        fname = f"{idx:05d}_lbl{label}.png"
        vutils.save_image(img, os.path.join(out_dir, fname),
                          normalize=True)       # rescales to [0,1] for PNG




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--model', default='lenet', choices=['lenet', 'mlp'], type=str)
    parser.add_argument('--color-std', type=float, default=0.1)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    # load data
    train_set = datasets.MNIST('data/github/', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST('data4/github/', train=False, download=True, transform=transforms.ToTensor())

    # colored_train_set = ColoredDataset(train_set,
    #                                classes=10,
    #                                colors=[0,1],
    #                                std=args.color_std)

    # 2) extract the actual colors that got assigned per‐class
    # train_colors = colored_train_set.colors    # e.g. a list of 10 RGB tuples

    # 3) reverse that list
    # train_colors = list(colored_train_set.colors)      # force it to a list
    # reversed_colors = train_colors[::-1]    

    # # 4) use the reversed mapping for your test split
    # colored_test_set = ColoredDataset(
    #                         test_set,
    #                         classes=10,
    #                         colors=reversed_colors,
    #                         std=args.color_std
    #                     )

    # # biased datasets, i.e. colored mnist
    # print('Coloring MNIST dataset with standard deviation = {:.2f}'.format(args.color_std))
    # # colored_train_set = ColoredDataset(train_set, classes=10, colors=[0, 1], std=args.color_std)
    # train_loader = DataLoader(colored_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # # colored_test_set = ColoredDataset(test_set, classes=10, colors=colored_train_set.colors, std=args.color_std)
    # test_loader = DataLoader(colored_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    # 1) train‐set
    colored_train_set = ColoredDataset(train_set, classes=10, colors=[0,1], std=args.color_std)
    train_colors  = colored_train_set.colors.clone()

    # 2) test‐set, then override
    colored_test_set  = ColoredDataset(test_set,  classes=10, colors=[0,1], std=args.color_std)
    colored_test_set.colors = train_colors.flip(0)

    # 3) loaders
    train_loader = DataLoader(colored_train_set, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(colored_test_set,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    # grount-truth datasets, i.e. grayscale mnist
    gt_train_set = ColoredDataset(train_set, classes=10, colors=[1, 1], std=0)
    gt_train_loader = DataLoader(gt_train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    gt_test_set = ColoredDataset(test_set, classes=10, colors=[1, 1], std=0)
    gt_test_loader = DataLoader(gt_test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)


    save_dataset(colored_train_set, os.path.join(OUT_ROOT, 'train'))
    save_dataset(colored_test_set, os.path.join(OUT_ROOT, 'test'))

    # (2) Grayscale “ground‑truth” MNIST
    # save_dataset(gt_train_set,      "saved/grayscale2/train")
    # save_dataset(gt_test_set,       "saved/grayscale2/test")


if __name__ == '__main__' :
    freeze_support()
    main()