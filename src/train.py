import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import time
import json

# Import our modules
import sys

sys.path.append(str(Path(__file__).parent))
from dataset import get_dataloaders
from models import get_model, count_parameters

# Paths
BASE_DIR = Path(r"C:\Users\MSI\Downloads\dr_detection")
CHECKPOINTS = BASE_DIR / "checkpoints"
OUTPUTS = BASE_DIR / "outputs"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)


# QUADRATIC WEIGHTED KAPPA - primary metric


def compute_qwk(preds, labels):
    """
    Quadratic Weighted Kappa - official APTOS metric.
    Predictions are rounded to nearest integer grade.
    """
    preds_rounded = np.clip(np.round(preds), 0, 4).astype(int)
    labels_int = labels.astype(int)
    return cohen_kappa_score(labels_int, preds_rounded, weights="quadratic")


# TRAIN ONE EPOCH


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        # Progress every 40 batches
        if (batch_idx + 1) % 40 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"    Batch [{batch_idx + 1}/{len(loader)}]  Loss: {avg_loss:.4f}")

    avg_loss = total_loss / len(loader)
    qwk = compute_qwk(np.array(all_preds), np.array(all_labels))
    return avg_loss, qwk


# VALIDATE


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    qwk = compute_qwk(np.array(all_preds), np.array(all_labels))
    return avg_loss, qwk


# FULL TRAINING LOOP


def train_model(
    model_name="efficientnet_b5",
    img_size=456,
    batch_size=16,
    num_epochs=25,
    lr=1e-4,
    patience=7,  # early stopping patience
):
    print(f"\n  Training: {model_name.upper()}")
    print(f"  Image size : {img_size}px")
    print(f"  Batch size : {batch_size}")
    print(f"  Epochs     : {num_epochs}")
    print(f"  LR         : {lr}\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n"
    )

    # Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(img_size, batch_size)
    print(f"   Train batches : {len(train_loader)}")
    print(f"   Val batches   : {len(val_loader)}\n")

    # Model
    print("Loading model...")
    model, device = get_model(model_name, pretrained=True, device=device)
    total, trainable = count_parameters(model)
    print(f"   Parameters    : {total:,}")
    print(f"   Trainable     : {trainable:,}\n")

    # Loss, Optimizer, Scheduler
    criterion = nn.SmoothL1Loss()  # Huber loss - good for regression
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # restart every 10 epochs
        T_mult=1,
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda")  # mixed precision scaler

    # Training history
    history = {
        "train_loss": [],
        "train_qwk": [],
        "val_loss": [],
        "val_qwk": [],
        "lr": [],
    }

    best_val_qwk = -1.0
    patience_counter = 0
    best_epoch = 0

    print("Starting training...\n")
    print(
        f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train QWK':>10}  {'Val Loss':>10}  {'Val QWK':>10}  {'LR':>10}"
    )

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_qwk = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch
        )

        # Validate
        val_loss, val_qwk = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log history
        history["train_loss"].append(train_loss)
        history["train_qwk"].append(train_qwk)
        history["val_loss"].append(val_loss)
        history["val_qwk"].append(val_qwk)
        history["lr"].append(current_lr)

        elapsed = time.time() - start_time

        # Print epoch summary
        improved = "UP " if val_qwk > best_val_qwk else "   "
        print(
            f"  {epoch:>5}  {train_loss:>10.4f}  {train_qwk:>10.4f}  "
            f"{val_loss:>10.4f}  {val_qwk:>10.4f}  "
            f"{current_lr:>10.2e}  {improved} ({elapsed:.0f}s)"
        )

        # Save best model
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            best_epoch = epoch
            patience_counter = 0
            save_path = CHECKPOINTS / f"{model_name}_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "val_qwk": val_qwk,
                    "val_loss": val_loss,
                },
                save_path,
            )
            print(f"         Saved best model - Val QWK: {best_val_qwk:.4f}")
        else:
            patience_counter += 1
            print(f"         No improvement ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"   Best epoch was {best_epoch} with Val QWK: {best_val_qwk:.4f}")
            break

    # Save training history
    history_path = OUTPUTS / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete!")
    print(f"   Best Val QWK : {best_val_qwk:.4f}  (Epoch {best_epoch})")
    print(f"   Checkpoint   : {CHECKPOINTS / f'{model_name}_best.pth'}")
    print(f"   History      : {history_path}")

    return best_val_qwk, history


# MAIN

if __name__ == "__main__":

    # Train EfficientNet-B5
    print("\nTraining EfficientNet-B5")
    best_qwk, history = train_model(
        model_name = "efficientnet_b5",
        img_size   = 456,
        batch_size = 16,
        num_epochs = 25,
        lr         = 1e-4,
        patience   = 7,
    )