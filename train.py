"""
train.py
========
Full training + validation loop for SOD models.
Features:
  • Per-epoch loss & IoU logging
  • Best model checkpoint saving
  • BONUS: full resume-from-checkpoint support
  • Early stopping
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data_loader import get_dataloaders
from sod_model   import SODNet, SODNetPlus, combined_loss


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_dir"        : "dataset",
    "model_type"      : "baseline",    # "baseline" | "improved"
    "image_size"      : 224,
    "batch_size"      : 16,
    "lr"              : 1e-3,
    "epochs"          : 25,
    "early_stop_patience": 5,
    "checkpoint_dir"  : "checkpoints",
    "log_file"        : "training_log.json",
    "num_workers"     : 2,
}


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────
def compute_iou(pred, target, threshold=0.5, smooth=1e-6):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_iou, n_batches = 0.0, 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, masks in tqdm(loader, desc="Train" if train else "Val  ", leave=False):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            preds = model(images)
            loss  = combined_loss(preds, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_iou  += compute_iou(preds.detach(), masks)
            n_batches  += 1

    return total_loss / n_batches, total_iou / n_batches


# ─────────────────────────────────────────────
# Checkpoint helpers  (BONUS)
# ─────────────────────────────────────────────
def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"  [Checkpoint] Saved → {path}")


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch  = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    history      = ckpt.get("history", [])
    print(f"  [Checkpoint] Resumed from epoch {ckpt['epoch']} | best_val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss, history


# ─────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────
def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  SOD Training  |  model={cfg['model_type']}  |  device={device}")
    print(f"{'='*55}\n")

    # ── Data ─────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        cfg["data_dir"],
        image_size  = cfg["image_size"],
        batch_size  = cfg["batch_size"],
        num_workers = cfg["num_workers"],
    )

    # ── Model ────────────────────────────────
    model = (SODNet() if cfg["model_type"] == "baseline" else SODNetPlus()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5, )

    # ── Checkpoint dir ───────────────────────
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.pt"
    best_ckpt   = ckpt_dir / "best.pt"

    # ── Resume if checkpoint exists (BONUS) ──
    start_epoch   = 0
    best_val_loss = float("inf")
    history       = []
    no_improve    = 0

    if latest_ckpt.exists():
        start_epoch, best_val_loss, history = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler
        )
    else:
        print("  [Checkpoint] No existing checkpoint found — starting fresh.\n")

    # ── Training loop ────────────────────────
    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_start = time.time()

        train_loss, train_iou = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss,   val_iou   = run_epoch(model, val_loader,   optimizer, device, train=False)

        scheduler.step(val_loss)
        elapsed = time.time() - epoch_start

        # Log
        log_entry = {
            "epoch"      : epoch + 1,
            "train_loss" : round(train_loss, 5),
            "val_loss"   : round(val_loss,   5),
            "train_iou"  : round(train_iou,  5),
            "val_iou"    : round(val_iou,    5),
            "lr"         : optimizer.param_groups[0]["lr"],
            "elapsed_s"  : round(elapsed, 1),
        }
        history.append(log_entry)

        print(f"Epoch [{epoch+1:03d}/{cfg['epochs']}] "
              f"| Train loss={train_loss:.4f} iou={train_iou:.4f} "
              f"| Val loss={val_loss:.4f} iou={val_iou:.4f} "
              f"| {elapsed:.1f}s")

        # Save latest checkpoint every epoch (BONUS)
        save_checkpoint({
            "epoch"          : epoch,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss"  : best_val_loss,
            "history"        : history,
            "config"         : cfg,
        }, latest_ckpt)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            save_checkpoint({"model_state": model.state_dict(), "epoch": epoch+1,
                              "val_loss": val_loss, "val_iou": val_iou}, best_ckpt)
            print(f"  ✔ New best model saved (val_loss={val_loss:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{cfg['early_stop_patience']} epoch(s)")

        # Early stopping
        if no_improve >= cfg["early_stop_patience"]:
            print(f"\n[EarlyStopping] Stopping at epoch {epoch+1}.")
            break

    # Save training history as JSON
    log_path = Path(cfg["log_file"])
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] History saved → {log_path}")
    print(f"[Train] Best val_loss = {best_val_loss:.4f}")
    print("[Train] Done ✓\n")

    return model, history


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SOD model")
    parser.add_argument("--data_dir",    default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--model_type",  default=DEFAULT_CONFIG["model_type"],
                        choices=["baseline", "improved"])
    parser.add_argument("--epochs",      type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size",  type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--image_size",  type=int, default=DEFAULT_CONFIG["image_size"])
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CONFIG["checkpoint_dir"])
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(vars(args))
    train(cfg)
