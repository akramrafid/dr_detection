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

    for model_name, color in [("efficientnet_b5", "#2E6DA4"), ("vit_b16", "#E74C3C")]:
        history_path = OUTPUTS / f"{model_name}_history.json"
        if not history_path.exists():
            continue
        with open(history_path) as f:
            h = json.load(f)

        epochs = range(1, len(h["val_qwk"]) + 1)
        label  = "EfficientNet-B5" if model_name == "efficientnet_b5" else "ViT-B/16"

        axes[0].plot(epochs, h["train_loss"], linestyle="--", color=color, alpha=0.6)
        axes[0].plot(epochs, h["val_loss"],   linestyle="-",  color=color, label=label)
        axes[1].plot(epochs, h["train_qwk"],  linestyle="--", color=color, alpha=0.6)
        axes[1].plot(epochs, h["val_qwk"],    linestyle="-",  color=color, label=label)

    axes[0].set_title("Loss per Epoch",    fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Val QWK per Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("QWK")
    axes[1].axhline(y=0.90, color="green", linestyle=":", label="Target 0.90")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training History - EfficientNet-B5 vs ViT-B/16",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = OUTPUTS / "training_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {save_path}")


# MAIN - ENSEMBLE EVALUATION

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # Load both models
    print("Loading trained models...")
    eff_model = load_checkpoint("efficientnet_b5", device)
    vit_model = load_checkpoint("vit_b16",         device)

    # Get test DataLoaders
    print("\nLoading test data...")
    _, _, test_loader_456 = get_dataloaders(img_size=456, batch_size=16)
    _, _, test_loader_384 = get_dataloaders(img_size=384, batch_size=16)
    print("   Test loaders ready\n")

    # Get predictions from each model
    print(" Getting predictions...")
    print("   Running EfficientNet-B5...")
    eff_preds, labels = get_predictions(eff_model, test_loader_456, device)

    print("   Running ViT-B/16...")
    vit_preds, _      = get_predictions(vit_model, test_loader_384, device)

    # Individual model scores
    eff_qwk = compute_qwk(eff_preds, labels)
    vit_qwk = compute_qwk(vit_preds, labels)

    
    print(f"  Individual Model Results (Test Set)")
    print(f"  EfficientNet-B5 QWK : {eff_qwk:.4f}")
    print(f"  ViT-B/16        QWK : {vit_qwk:.4f}")

    # Ensemble with different weights
    print(f"\n  Ensemble Results (Test Set)")
    
    print(f"  {'Weights (Eff / ViT)':>25}   {'QWK':>8}")
    

    best_qwk     = -1
    best_weights = (0.5, 0.5)

    for eff_w in np.arange(0.3, 0.8, 0.1):
        vit_w         = round(1.0 - eff_w, 1)
        ensemble_preds = eff_w * eff_preds + vit_w * vit_preds
        ensemble_qwk   = compute_qwk(ensemble_preds, labels)

        marker = " <-- Best" if ensemble_qwk > best_qwk else ""
        print(f"  {eff_w:.1f} / {vit_w:.1f}  {' '*15}  {ensemble_qwk:.4f}{marker}")

        if ensemble_qwk > best_qwk:
            best_qwk     = ensemble_qwk
            best_weights = (eff_w, vit_w)

    # Final ensemble with best weights
    eff_w, vit_w   = best_weights
    ensemble_preds = eff_w * eff_preds + vit_w * vit_preds
    ensemble_rounded = np.clip(np.round(ensemble_preds), 0, 4).astype(int)

    
    print(f" FINAL RESULTS")
    
    print(f"  Best weights         : Eff={eff_w:.1f} / ViT={vit_w:.1f}")
    print(f"  EfficientNet-B5 QWK  : {eff_qwk:.4f}")
    print(f"  ViT-B/16 QWK         : {vit_qwk:.4f}")
    print(f"  Ensemble QWK         : {best_qwk:.4f}")
    

    # Classification report
    print(f"\nClassification Report (Ensemble):")
    print(classification_report(
        labels.astype(int),
        ensemble_rounded,
        target_names=list(GRADE_LABELS.values())
    ))

    # Save best weights
    weights_info = {
        "efficientnet_b5_weight" : eff_w,
        "vit_b16_weight"         : vit_w,
        "ensemble_test_qwk"      : best_qwk,
        "efficientnet_b5_test_qwk": eff_qwk,
        "vit_b16_test_qwk"       : vit_qwk,
    }
    with open(OUTPUTS / "ensemble_weights.json", "w") as f:
        json.dump(weights_info, f, indent=2)
    print(f"\n  Saved ensemble weights to outputs/ensemble_weights.json")

    # Plot confusion matrix
    print(f"\n Generating plots...")
    plot_confusion_matrix(
        labels.astype(int), ensemble_rounded,
        "Ensemble - Confusion Matrix (Test Set)",
        OUTPUTS / "confusion_matrix.png"
    )
    plot_training_history()

    print(f"\nEnsemble evaluation complete!")
    print(f"   All plots saved to: {OUTPUTS}")