import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json

import sys
sys.path.append(str(Path(__file__).parent))
from dataset import get_dataloaders, get_val_transforms
from models  import get_model

# Paths
BASE_DIR    = Path(r"C:\Users\MSI\Downloads\dr_detection")
CHECKPOINTS = BASE_DIR / "checkpoints"
OUTPUTS     = BASE_DIR / "outputs"

GRADE_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}


# LOAD TRAINED MODEL FROM CHECKPOINT

def load_checkpoint(model_name, device):
    model, device = get_model(model_name, pretrained=False, device=device)
    checkpoint    = torch.load(
        CHECKPOINTS / f"{model_name}_best.pth",
        map_location=device,
        weights_only = True
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"  Loaded {model_name} - Val QWK: {checkpoint['val_qwk']:.4f}")
    return model


# GET PREDICTIONS FROM ONE MODEL

def get_predictions(model, loader, device):
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


# COMPUTE QWK

def compute_qwk(preds, labels):
    preds_rounded = np.clip(np.round(preds), 0, 4).astype(int)
    return cohen_kappa_score(labels.astype(int), preds_rounded, weights="quadratic")


# PLOT CONFUSION MATRIX

def plot_confusion_matrix(labels, preds, title, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = "d",
        cmap       = "Blues",
        xticklabels= list(GRADE_LABELS.values()),
        yticklabels= list(GRADE_LABELS.values()),
        linewidths = 0.5
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("True Grade",      fontsize=12)
    plt.xlabel("Predicted Grade", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Saved: {save_path}")


# PLOT TRAINING HISTORY

def plot_training_history():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    model_name = "efficientnet_b5"
    color = "#2E6DA4"
    history_path = OUTPUTS / f"{model_name}_history.json"

    if not history_path.exists():
        print(f"   No training history found at {history_path}")
        plt.close()
        return

    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["val_qwk"]) + 1)
    label  = "EfficientNet-B5"

    axes[0].plot(epochs, h["train_loss"], linestyle="--", color=color, alpha=0.6, label=f"{label} (train)")
    axes[0].plot(epochs, h["val_loss"],   linestyle="-",  color=color, label=f"{label} (val)")
    axes[1].plot(epochs, h["train_qwk"],  linestyle="--", color=color, alpha=0.6, label=f"{label} (train)")
    axes[1].plot(epochs, h["val_qwk"],    linestyle="-",  color=color, label=f"{label} (val)")

    axes[0].set_title("Loss per Epoch",    fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Val QWK per Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("QWK")
    axes[1].axhline(y=0.90, color="green", linestyle=":", label="Target 0.90")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training History - EfficientNet-B5",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = OUTPUTS / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {save_path}")


# MAIN - MODEL EVALUATION

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # Load model
    print("Loading trained model...")
    eff_model = load_checkpoint("efficientnet_b5", device)

    # Get test DataLoader
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders(img_size=456, batch_size=16)
    print("   Test loader ready\n")

    # Get predictions
    print(" Getting predictions...")
    print("   Running EfficientNet-B5...")
    eff_preds, labels = get_predictions(eff_model, test_loader, device)

    # Model score
    eff_qwk = compute_qwk(eff_preds, labels)
    eff_rounded = np.clip(np.round(eff_preds), 0, 4).astype(int)

    print(f"\n  Model Results (Test Set)")
    print(f"  EfficientNet-B5 QWK : {eff_qwk:.4f}")

    # Classification report
    print(f"\nClassification Report (EfficientNet-B5):")
    print(classification_report(
        labels.astype(int),
        eff_rounded,
        target_names=list(GRADE_LABELS.values())
    ))

    # Save results
    results_info = {
        "model": "efficientnet_b5",
        "test_qwk": eff_qwk,
    }
    with open(OUTPUTS / "evaluation_results.json", "w") as f:
        json.dump(results_info, f, indent=2)
    print(f"\n  Saved evaluation results to outputs/evaluation_results.json")

    # Plot confusion matrix
    print(f"\n Generating plots...")
    plot_confusion_matrix(
        labels.astype(int), eff_rounded,
        "EfficientNet-B5 - Confusion Matrix (Test Set)",
        OUTPUTS / "confusion_matrix.png"
    )
    plot_training_history()

    print(f"\nEvaluation complete!")
    print(f"   All plots saved to: {OUTPUTS}")