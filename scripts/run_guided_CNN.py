import os
import time
import copy
import argparse
from datetime import datetime
import random

import cv2
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



class ExpandWhite(object):
    def __init__(self, thr: int = 10, radius: int = 3):
        self.thr = thr
        self.radius = radius
    def __call__(self, mask: Image.Image) -> Image.Image:
        arr = np.array(mask)
        white = (arr > self.thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * self.radius + 1, 2 * self.radius + 1))
        dil = cv2.dilate(white, k, iterations=1)
        return Image.fromarray((dil * 255).astype(np.uint8))


class EdgeExtract(object):
    def __init__(self, thr: int = 10, edge_width: int = 1):
        self.thr = thr
        self.edge_width = edge_width
    def __call__(self, mask: Image.Image) -> Image.Image:
        arr = np.array(mask)
        white = (arr > self.thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * self.edge_width + 1, 2 * self.edge_width + 1))
        edge = cv2.morphologyEx(white, cv2.MORPH_GRADIENT, k)
        return Image.fromarray((edge * 255).astype(np.uint8))


class Brighten(object):
    def __init__(self, factor: float):
        self.factor = factor
    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        return torch.clamp(mask * self.factor, 0.0, 1.0)



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


def make_cam_model(num_classes):
    base = LeNet(num_classes)
    class CAMWrap(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base = base_model
            self.features = None
            self.base.conv2.register_forward_hook(self._hook_fn)
        def _hook_fn(self, module, inp, out):
            self.features = out
        def forward(self, x):
            out = self.base(x)
            return out, self.features
    return CAMWrap(base)



class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


class GuidedImageFolder(Dataset):
    def __init__(self, image_root: str, mask_root: str, image_transform=None, mask_transform=None):
        self.images = datasets.ImageFolder(image_root, transform=image_transform)
        self.mask_root = mask_root
        self.mask_transform = mask_transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img, label = self.images[idx]
        path, _ = self.images.samples[idx]
        base = os.path.splitext(os.path.basename(path))[0]
        mask_path = os.path.join(self.mask_root, base + ".png")
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, label, mask, path



def compute_loss(outputs, labels, cams, gt_masks, kl_lambda, only_attn):
    ce_loss = nn.functional.cross_entropy(outputs, labels)
    B, Hf, Wf = cams.shape
    cam_flat = cams.view(B, -1)
    gt_flat = gt_masks.view(B, -1)
    log_p = nn.functional.log_softmax(cam_flat, dim=1)
    gt_prob = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-8)
    kl_div = nn.KLDivLoss(reduction='batchmean')
    attn_loss = kl_div(log_p, gt_prob)
    if only_attn:
        return attn_loss, attn_loss
    else:
        return ce_loss + kl_lambda * attn_loss, attn_loss



def train_model(model, weight_decay_on, dataloaders, dataset_sizes,
                attention_epoch, kl_lambda_start, num_epochs,
                lr2, kl_incr):
    best_wts = copy.deepcopy(model.state_dict())
    best_optim = -100.0
    since = time.time()

    # single optimizer + scheduler
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    sch = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    kl_lambda_real = kl_lambda_start

    for epoch in range(num_epochs):
        # restart at attention_epoch
        if epoch == attention_epoch:
            print(f"*** Attention epoch {epoch} reached: restarting optimizer & scheduler ***")
            opt = optim.SGD(model.parameters(), lr=lr2, momentum=momentum, weight_decay=weight_decay)
            sch = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
            best_wts = copy.deepcopy(model.state_dict())
            best_optim = -100.0

        # increase KL after attention_epoch
        if epoch > attention_epoch:
            kl_lambda_real += kl_incr

        print(f"Epoch {epoch + 1}/{num_epochs}")

        for phase in ['train', 'val_in']:
            is_train = (phase == 'train')
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_attn_loss = 0.0

            for batch in dataloaders[phase]:
                if phase in ['train', 'val_in']:
                    inputs, labels, gt_masks, paths = batch
                    gt_masks = gt_masks.to(device)
                    has_masks = True
                else:
                    raise RuntimeError("Unexpected phase.")

                # attention used on both train and val_in
                use_attention_this_batch = True

                inputs, labels = inputs.to(device), labels.to(device)
                if is_train:
                    opt.zero_grad()

                with torch.set_grad_enabled(is_train):
                    outputs, feats = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    if use_attention_this_batch and has_masks:
                        weights = model.base.classifier.weight[labels]
                        cams = torch.einsum('bc,bchw->bhw', weights, feats)
                        cams = torch.relu(cams)

                        flat = cams.view(cams.size(0), -1)
                        mn, _ = flat.min(dim=1, keepdim=True)
                        mx, _ = flat.max(dim=1, keepdim=True)
                        sal_norm = ((flat - mn) / (mx - mn + 1e-8)).view_as(cams)

                        gt_small = nn.functional.interpolate(
                            gt_masks, size=sal_norm.shape[1:], mode='nearest'
                        ).squeeze(1)

                        if epoch < attention_epoch:
                            loss_tuple = compute_loss(outputs, labels, sal_norm, gt_small, 333, True)
                        else:
                            loss_tuple = compute_loss(outputs, labels, sal_norm, gt_small, kl_lambda_real, False)

                        loss = loss_tuple[0]
                        attn_loss = loss_tuple[1]
                    else:
                        loss = nn.functional.cross_entropy(outputs, labels)
                        attn_loss = torch.tensor(0.0, device=outputs.device)

                    if is_train:
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_attn_loss += attn_loss.item() * inputs.size(0)

            if is_train:
                sch.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_attn_loss = running_attn_loss / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Attn_Loss: {epoch_attn_loss:.4f}")

            if phase == 'val_in':
                optim_num = epoch_acc * (1 - epoch_attn_loss)
                print(f"{phase} Optim Num: {optim_num:.4f}")
                if (epoch >= attention_epoch) and (optim_num > best_optim):
                    best_optim = optim_num
                    best_wts = copy.deepcopy(model.state_dict())

    print()
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # load best val_in weights before returning
    model.load_state_dict(best_wts)
    return model, best_optim




@torch.no_grad()
def evaluate_test(model, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0
    for images, labels, paths in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs, _ = model(images)
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
    mask_transforms = {
        'train': transforms.Compose([
            ExpandWhite(thr=10, radius=3),
            EdgeExtract(thr=10, edge_width=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            Brighten(8.0),
        ])
    }

    seed_everything(SEED)
    g = torch.Generator(); g.manual_seed(SEED)

    # Train + internal val split (from train/)
    full_train = GuidedImageFolder(
        image_root=os.path.join(args.data_path, 'train'),
        mask_root=args.gt_path,
        image_transform=data_transforms['train'],
        mask_transform=mask_transforms['train'],
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

    model = make_cam_model(len(full_train.images.classes)).to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n=== RUN: kl_lambda={kl_value}, attention_epoch={attn_epoch} ===", flush=True)
    best_model, best_score = train_model(
        model, True, dataloaders, dataset_sizes,
        attn_epoch, kl_value, num_epochs,
        lr2=learning_rate, kl_incr=(kl_value / 10)
    )

    # Evaluate once on TEST with the best val_in weights
    test_loss, test_acc = evaluate_test(best_model, test_loader)
    print(f"\n[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.2f}%")

    # Save best model (named with hyperparams)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"lenet_final_kl{int(kl_value)}_attn{attn_epoch}_{ts}.pth"
    save_path = os.path.join(checkpoint_dir, save_name)
    torch.save(best_model.state_dict(), save_path)

    print(f"[RUN DONE] kl={kl_value} attn={attn_epoch} | best_valin_optim={best_score:.4f} "
          f"| test_acc={test_acc:.2f}% | saved: {save_path}", flush=True)
    return best_score, test_acc, save_path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Root with train/ and test/ subdirs')
    parser.add_argument('gt_path', help='Folder with ground-truth mask PNGs (for train only)')
    parser.add_argument('--attention_epoch', type=int, default=11, help='Epoch at which to restart training')
    parser.add_argument('--kl_lambda', type=float, default=160.0, help='Weight for attention KL loss')
    parser.add_argument('--sweep', action='store_true',
                        help='Run the full hyperparameter sweep (kl 100..300 step 20; attn 5..25 step 2)')
    args = parser.parse_args()

    if not args.sweep:
        run_single(args, args.attention_epoch, args.kl_lambda)
        return

    kl_values = list(range(100, 301, 20))
    attn_values = list(range(5, 26, 2))

    best_overall = (-1.0, None, None, None)  # (best_optim, kl, attn, test_acc)
    for kl in kl_values:
        for attn in attn_values:
            try:
                score, test_acc, _ = run_single(args, attn, kl)
                if score > best_overall[0]:
                    best_overall = (score, kl, attn, test_acc)
            except Exception as e:
                print(f"[SWEEP ERROR] kl={kl} attn={attn} -> {e}", flush=True)

    print("\n=== SWEEP COMPLETE ===")
    if best_overall[1] is not None:
        print(f"Best by internal val Optim Num: optim={best_overall[0]:.4f}, "
              f"kl={best_overall[1]}, attn={best_overall[2]}, "
              f"corresponding test_acc={best_overall[3]:.2f}%", flush=True)
    else:
        print("No successful runs.", flush=True)


if __name__ == '__main__':
    main()
