"""
data_loader.py
==============
Handles dataset loading, preprocessing, and augmentation for SOD.
Supports DUTS, ECSSD, MSRA10K dataset formats (image + mask pairs).
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
IMAGE_SIZE = 224          # resize all images to this
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
BATCH_SIZE  = 16
NUM_WORKERS = 2
SEED        = 42


# ─────────────────────────────────────────────
# Augmentation Pipelines
# ─────────────────────────────────────────────
def get_train_transforms(image_size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomCrop(height=int(image_size * 0.9), width=int(image_size * 0.9), p=0.3),
        A.Resize(image_size, image_size),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_test_transforms(image_size=IMAGE_SIZE):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────
class SODDataset(Dataset):
    """
    Loads image-mask pairs for Salient Object Detection.
    
    Expected folder structure:
        root/
          images/   ← .jpg or .png files
          masks/    ← .png binary masks (same filename as images)
    """

    def __init__(self, image_paths, mask_paths, transform=None):
        assert len(image_paths) == len(mask_paths), \
            f"Mismatch: {len(image_paths)} images but {len(mask_paths)} masks"
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (BGR → RGB)
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale, binary 0/1 float)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)   # binarize

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]            # shape: (3, H, W) tensor
            mask  = augmented["mask"].unsqueeze(0)  # shape: (1, H, W) tensor

        return image, mask


# ─────────────────────────────────────────────
# Helper: find paired image/mask paths
# ─────────────────────────────────────────────
def get_paired_paths(root_dir):
    """
    Scans root_dir/images and root_dir/masks and returns
    matched (image_path, mask_path) pairs.
    """
    root_dir   = Path(root_dir)
    images_dir = root_dir / "images"
    masks_dir  = root_dir / "masks"

    if not images_dir.exists():
        raise FileNotFoundError(f"images/ not found in {root_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"masks/ not found in {root_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in image_exts])

    image_paths, mask_paths = [], []
    for img_path in image_files:
        # Try matching mask by same stem
        for ext in [".png", ".jpg", ".jpeg"]:
            mask_path = masks_dir / (img_path.stem + ext)
            if mask_path.exists():
                image_paths.append(img_path)
                mask_paths.append(mask_path)
                break

    print(f"[DataLoader] Found {len(image_paths)} matched image-mask pairs in {root_dir}")
    return image_paths, mask_paths


# ─────────────────────────────────────────────
# Main factory: returns train/val/test loaders
# ─────────────────────────────────────────────
def get_dataloaders(root_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS, seed=SEED):
    """
    Returns (train_loader, val_loader, test_loader) given a root dataset directory.
    """
    image_paths, mask_paths = get_paired_paths(root_dir)
    n = len(image_paths)

    # Reproducible shuffle
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    image_paths = [image_paths[i] for i in indices]
    mask_paths  = [mask_paths[i]  for i in indices]

    # Split
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    train_imgs, train_masks = image_paths[:n_train],           mask_paths[:n_train]
    val_imgs,   val_masks   = image_paths[n_train:n_train+n_val], mask_paths[n_train:n_train+n_val]
    test_imgs,  test_masks  = image_paths[n_train+n_val:],     mask_paths[n_train+n_val:]

    print(f"[DataLoader] Split → Train: {n_train} | Val: {n_val} | Test: {n_test}")

    train_ds = SODDataset(train_imgs, train_masks, transform=get_train_transforms(image_size))
    val_ds   = SODDataset(val_imgs,   val_masks,   transform=get_val_test_transforms(image_size))
    test_ds  = SODDataset(test_imgs,  test_masks,  transform=get_val_test_transforms(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# Quick visual sanity check
# ─────────────────────────────────────────────
def visualize_batch(loader, n=4):
    """Show n image-mask pairs from a DataLoader batch."""
    images, masks = next(iter(loader))
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))
    for i in range(n):
        img = (images[i] * std + mean).permute(1,2,0).clamp(0,1).numpy()
        msk = masks[i].squeeze().numpy()
        axes[0, i].imshow(img);          axes[0, i].set_title("Image");    axes[0, i].axis("off")
        axes[1, i].imshow(msk, cmap="gray"); axes[1, i].set_title("Mask"); axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig("batch_preview.png", dpi=100)
    plt.show()
    print("[DataLoader] Batch preview saved → batch_preview.png")


if __name__ == "__main__":
    # Quick test — replace with your actual dataset path
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    train_loader, val_loader, test_loader = get_dataloaders(root)
    visualize_batch(train_loader)
