#!/usr/bin/env python3
import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Root with train/ and test/ subdirs (DecoyMNIST_png).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-dir", default="LeNet_Checkpoints")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Match paper-style decoy setup: 1-channel, 28x28, and scale to [-1, 1].
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0),
    ])

    full_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.data_path, "test"), transform=transform)

    n_total = len(full_train)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_train, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    model = LeNet().to(device)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_wts = None
    since = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            opt.step()

        if epoch % args.log_interval == 0:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}/{args.epochs} Val set: Average loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_wts is not None:
        model.load_state_dict(best_wts)

    test_loss, test_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - since
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    print(f"Training complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.checkpoint_dir, f"lenet_decoy_paperstyle_{ts}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
