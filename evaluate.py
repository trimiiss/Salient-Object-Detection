"""
evaluate.py
===========
Evaluation metrics + visualizations for SOD models.
Computes: IoU, Precision, Recall, F1, MAE
Generates: input / ground-truth / predicted / overlay grid visualizations
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pathlib import Path
from tqdm import tqdm

from data_loader import get_dataloaders
from sod_model   import SODNet, SODNetPlus


# ─────────────────────────────────────────────
# Metric utilities
# ─────────────────────────────────────────────
def compute_metrics(pred, target, threshold=0.5, smooth=1e-6):
    """
    pred, target : (B, 1, H, W) tensors on CPU, values in [0,1]
    Returns dict with iou, precision, recall, f1, mae
    """
    pred_bin = (pred > threshold).float()
    target   = target.float()

    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()

    iou       = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    f1        = 2 * precision * recall / (precision + recall + smooth)
    mae       = (pred - target).abs().mean().item()

    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1, "mae": mae}


def aggregate_metrics(metric_list):
    keys = metric_list[0].keys()
    return {k: float(np.mean([m[k] for m in metric_list])) for k in keys}


# ─────────────────────────────────────────────
# Denormalization helper
# ─────────────────────────────────────────────
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denorm(tensor):
    return (tensor.cpu() * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def visualize_predictions(model, loader, device, n_samples=8,
                           save_path="predictions.png", threshold=0.5):
    """
    Creates a 4-row grid:  Input | GT Mask | Predicted Mask | Overlay
    """
    model.eval()
    images_all, masks_all, preds_all = [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            preds  = model(images).cpu()
            images_all.append(images.cpu())
            masks_all.append(masks.cpu())
            preds_all.append(preds)
            if sum(x.shape[0] for x in images_all) >= n_samples:
                break

    images_all = torch.cat(images_all)[:n_samples]
    masks_all  = torch.cat(masks_all)[:n_samples]
    preds_all  = torch.cat(preds_all)[:n_samples]

    fig, axes = plt.subplots(4, n_samples, figsize=(3 * n_samples, 13))
    titles = ["Input Image", "Ground Truth", "Predicted Mask", "Overlay"]

    for i in range(n_samples):
        img  = denorm(images_all[i])
        gt   = masks_all[i].squeeze().numpy()
        pred = preds_all[i].squeeze().numpy()
        pred_bin = (pred > threshold).astype(np.float32)

        # Overlay: blend prediction as red channel on image
        overlay = img.copy()
        overlay[pred_bin == 1] = overlay[pred_bin == 1] * 0.5 + np.array([1, 0.2, 0.2]) * 0.5

        row_data = [img, gt, pred, overlay]
        cmaps    = [None, "gray", "hot", None]

        for row, (data, cmap, title) in enumerate(zip(row_data, cmaps, titles)):
            ax = axes[row, i]
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            if i == 0:
                ax.set_ylabel(title, fontsize=11, fontweight="bold")
            ax.axis("off")

    plt.suptitle("SOD Model Predictions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"[Evaluate] Visualization saved → {save_path}")


# ─────────────────────────────────────────────
# Training curve plot
# ─────────────────────────────────────────────
def plot_training_history(log_file="training_log.json", save_path="training_curves.png"):
    with open(log_file) as f:
        history = json.load(f)

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    train_iou  = [h["train_iou"]  for h in history]
    val_iou    = [h["val_iou"]    for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, val_loss,   label="Val Loss",   marker="o", markersize=3)
    ax1.set_title("Loss per Epoch"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_iou, label="Train IoU", marker="o", markersize=3)
    ax2.plot(epochs, val_iou,   label="Val IoU",   marker="o", markersize=3)
    ax2.set_title("IoU per Epoch"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.show()
    print(f"[Evaluate] Training curves saved → {save_path}")


# ─────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────
def evaluate(data_dir="dataset", checkpoint_path="checkpoints/best.pt",
             model_type="baseline", image_size=224, batch_size=16,
             visualize=True, output_dir="outputs"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────
    model = (SODNet() if model_type == "baseline" else SODNetPlus()).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[Evaluate] Loaded checkpoint: {checkpoint_path}")

    # ── Load test data ───────────────────────
    _, _, test_loader = get_dataloaders(data_dir, image_size=image_size,
                                        batch_size=batch_size, num_workers=2)

    # ── Compute metrics ───────────────────────
    all_metrics = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            preds  = model(images).cpu()
            for pred, mask in zip(preds, masks):
                all_metrics.append(compute_metrics(pred.unsqueeze(0), mask.unsqueeze(0)))

    final = aggregate_metrics(all_metrics)

    print("\n" + "="*45)
    print(f"  Test Results  ({len(all_metrics)} samples)")
    print("="*45)
    for k, v in final.items():
        print(f"  {k.upper():12s}: {v:.4f}")
    print("="*45 + "\n")

    # Save metrics JSON
    metrics_path = Path(output_dir) / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"[Evaluate] Metrics saved → {metrics_path}")

    # ── Visualizations ───────────────────────
    if visualize:
        visualize_predictions(
            model, test_loader, device, n_samples=8,
            save_path=str(Path(output_dir) / "predictions.png")
        )
        if Path("training_log.json").exists():
            plot_training_history(
                save_path=str(Path(output_dir) / "training_curves.png")
            )

    return final


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        default="dataset")
    parser.add_argument("--checkpoint_path", default="checkpoints/best.pt")
    parser.add_argument("--model_type",      default="baseline", choices=["baseline","improved"])
    parser.add_argument("--image_size",      type=int, default=224)
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--output_dir",      default="outputs")
    args = parser.parse_args()
    evaluate(**vars(args))
