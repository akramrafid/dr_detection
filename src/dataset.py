import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR      = Path(r"C:\Users\MSI\Downloads\dr_detection")
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "train_images"
SPLITS_CSV    = BASE_DIR / "data" / "splits" / "splits.csv"


# ─────────────────────────────────────────────────────────────────
# AUGMENTATION PIPELINES
# ─────────────────────────────────────────────────────────────────

def get_train_transforms(img_size=456):
    """Heavy augmentation for training set only."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
        A.GridDistortion(p=0.3),
        A.CoarseDropout(
            
            
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean
            std=[0.229, 0.224, 0.225],    # ImageNet std
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=456):
    """Minimal transforms for validation and test — no augmentation."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────
# CUSTOM DATASET CLASS
# ─────────────────────────────────────────────────────────────────

class DRDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy Detection.
    Loads preprocessed fundus images and applies transforms.
    """

    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df        : DataFrame with columns [id_code, diagnosis]
            img_dir   : Path to processed images folder
            transform : Albumentations transform pipeline
        """
        self.df        = df.reset_index(drop=True)
        self.img_dir   = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.img_dir / f"{row['id_code']}.png"

        # ── Load image ────────────────────────────────────────────
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ── Apply transforms ──────────────────────────────────────
        if self.transform:
            img = self.transform(image=img)["image"]

        # ── Label as float for regression head ───────────────────
        # We use regression (not classification) for ordinal grading
        label = torch.tensor(row["diagnosis"], dtype=torch.float32)

        return img, label


# ─────────────────────────────────────────────────────────────────
# WEIGHTED SAMPLER — fixes class imbalance
# ─────────────────────────────────────────────────────────────────

def get_weighted_sampler(df):
    """
    Creates a WeightedRandomSampler so minority grades
    (Grade 1, 3) are sampled more frequently during training.
    """
    class_counts = df["diagnosis"].value_counts().sort_index()
    class_weights = 1.0 / class_counts

    # Assign weight to each sample based on its class
    sample_weights = df["diagnosis"].map(class_weights).values
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )
    return sampler


# ─────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────

def get_dataloaders(img_size=456, batch_size=16):
    """
    Returns train, val, and test DataLoaders ready for training.
    """
    # ── Load splits ───────────────────────────────────────────────
    df       = pd.read_csv(SPLITS_CSV)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    # ── Create datasets ───────────────────────────────────────────
    train_dataset = DRDataset(train_df, PROCESSED_DIR, get_train_transforms(img_size))
    val_dataset   = DRDataset(val_df,   PROCESSED_DIR, get_val_transforms(img_size))
    test_dataset  = DRDataset(test_df,  PROCESSED_DIR, get_val_transforms(img_size))

    # ── Weighted sampler for training only ────────────────────────
    sampler = get_weighted_sampler(train_df)

    # ── DataLoaders ───────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        sampler     = sampler,       # weighted sampling
        num_workers = 4,
        pin_memory  = True,          # faster CPU→GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 4,
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────
# TEST — verify everything works
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    print("🔄 Testing Dataset and DataLoader...\n")

    train_loader, val_loader, test_loader = get_dataloaders(
        img_size   = 456,
        batch_size = 16
    )

    print(f"📦 Train batches : {len(train_loader)}")
    print(f"📦 Val batches   : {len(val_loader)}")
    print(f"📦 Test batches  : {len(test_loader)}")

    # ── Load one batch and inspect ────────────────────────────────
    print("\n🔍 Inspecting one training batch...")
    images, labels = next(iter(train_loader))

    print(f"\n  Image batch shape : {images.shape}")
    print(f"  Label batch shape : {labels.shape}")
    print(f"  Image dtype       : {images.dtype}")
    print(f"  Label dtype       : {labels.dtype}")
    print(f"  Image min/max     : {images.min():.3f} / {images.max():.3f}")
    print(f"  Labels in batch   : {labels.tolist()}")

    # ── Check GPU transfer ────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    labels = labels.to(device)
    print(f"\n  Device            : {device}")
    print(f"  Images on GPU     : {images.is_cuda}")
    print(f"\n{'='*50}")
    print("✅ Dataset class working perfectly!")
    print("   Ready to build models.")