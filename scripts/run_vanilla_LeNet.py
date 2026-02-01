import os
import time
import copy
import argparse
from datetime import datetime
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms




batch_size = 32
num_epochs = 30
learning_rate = 0.001
momentum = 0.98
step_size = 7     # LR decay every 7 epochs
gamma = 0.1       # decay factor
weight_decay = 1e-4

checkpoint_dir = "LeNet_Checkpoints"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 37



def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        feats = x
        gap = self.gap(feats).view(feats.size(0), -1)
        out = self.classifier(gap)
        return out


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


class GuidedImageFolder(Dataset):
    def __init__(self, image_root: str, image_transform=None):
        self.images = datasets.ImageFolder(image_root, transform=image_transform)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img, label = self.images[idx]
        path, _ = self.images.samples[idx]
        return img, label, path

def train_model(model, dataloaders, dataset_sizes, num_epochs, test_loader=None):
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = -1.0
    since = time.time()

    # single optimizer + scheduler
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    sch = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for phase in ['train', 'val_in']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            for batch in dataloaders[phase]:
                inputs, labels, paths = batch

                inputs, labels = inputs.to(device), labels.to(device)
                if is_train:
                    opt.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = nn.functional.cross_entropy(outputs, labels)

                    if is_train:
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if is_train:
                sch.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val_in':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())

                if test_loader is not None:
                    test_loss, test_acc = evaluate_test(model, test_loader)
                    print(f"[TEST @ epoch {epoch + 1}] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")

    print()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # load best val_in weights before returning
    model.load_state_dict(best_wts)
    return model, best_acc




@torch.no_grad()
def evaluate_test(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    for images, labels, paths in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    avg_loss = total_loss / max(total, 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc




def run_single(args, attn_epoch, kl_value):
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'eval': transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    seed_everything(SEED)
    g = torch.Generator(); g.manual_seed(SEED)

    # Train + internal val split (from train/)
    full_train = GuidedImageFolder(
        image_root=os.path.join(args.data_path, 'train'),
        image_transform=data_transforms['train'],
    )
    n_total = len(full_train)
    n_val_in = max(1, int(0.16 * n_total))
    n_train = n_total - n_val_in
    train_subset, val_in_subset = random_split(full_train, [n_train, n_val_in], generator=g)

    # Test set (evaluated once at the end)
    test_dataset = ImageFolderWithPaths(
        root=os.path.join(args.data_path, 'test'),
        transform=data_transforms['eval']
    )

    dataloaders = {
        'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                            num_workers=4, worker_init_fn=seed_worker, generator=g),
        'val_in': DataLoader(val_in_subset, batch_size=batch_size, shuffle=False,
                             num_workers=4, worker_init_fn=seed_worker, generator=g),
    }
    dataset_sizes = {
        'train': len(train_subset),
        'val_in': len(val_in_subset),
    }
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, worker_init_fn=seed_worker, generator=g)

    model = LeNet(len(full_train.images.classes)).to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("\n=== RUN: vanilla LeNet ===", flush=True)
    best_model, best_score = train_model(
        model, dataloaders, dataset_sizes, num_epochs, test_loader=test_loader
    )

    # Evaluate once on TEST with the best val_in weights
    test_loss, test_acc = evaluate_test(best_model, test_loader)
    print(f"\n[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")

    # Save best model (named with hyperparams)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"lenet_final_{ts}.pth"
    save_path = os.path.join(checkpoint_dir, save_name)
    torch.save(best_model.state_dict(), save_path)

    print(f"[RUN DONE] best_valin_acc={best_score:.4f} "
          f"| test_acc={test_acc:.2f}% | saved: {save_path}", flush=True)
    return best_score, test_acc, save_path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Root with train/ and test/ subdirs')
    parser.add_argument('gt_path', nargs='?', default=None,
                        help='(ignored) Optional mask path for compatibility.')
    parser.add_argument('--sweep', action='store_true',
                        help='Run a repeat sweep for compatibility (same settings each run).')
    args = parser.parse_args()

    if not args.sweep:
        run_single(args, None, None)
        return

    best_overall = (-1.0, None)  # (best_acc, test_acc)
    for _ in range(1):
        try:
            score, test_acc, _ = run_single(args, None, None)
            if score > best_overall[0]:
                best_overall = (score, test_acc)
        except Exception as e:
            print(f"[SWEEP ERROR] -> {e}", flush=True)

    print("\n=== SWEEP COMPLETE ===")
    if best_overall[1] is not None:
        print(f"Best by internal val Acc: acc={best_overall[0]:.4f}, "
              f"corresponding test_acc={best_overall[1]:.2f}%", flush=True)
    else:
        print("No successful runs.", flush=True)


if __name__ == '__main__':
    main()
