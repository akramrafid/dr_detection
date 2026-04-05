import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import os

# Paths
BASE_DIR      = Path(r"C:\Users\MSI\Downloads\dr_detection")
RAW_DIR       = BASE_DIR / "data" / "raw" / "aptos2019-blindness-detection"
TRAIN_IMG_DIR = RAW_DIR / "train_images"
TRAIN_CSV     = RAW_DIR / "train.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "train_images"
SPLITS_DIR    = BASE_DIR / "data" / "splits"

# Create output folders if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

print("Paths configured")
print(f"   Input  : {TRAIN_IMG_DIR}")
print(f"   Output : {PROCESSED_DIR}")

# BEN GRAHAM PREPROCESSING
# Used by top APTOS solutions - enhances retinal vessel visibility
def crop_image_from_gray(img, tol=7):
    """Remove dark borders around the fundus image."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def ben_graham_preprocess(img_path, img_size=456):
    """
    Full Ben Graham preprocessing pipeline:
    1. Load image
    2. Crop dark borders
    3. Resize to target size
    4. Apply CLAHE on green channel
    5. Subtract local average (Gaussian blur trick)
    6. Circular crop mask
    """
    # Load
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not load: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop dark borders
    img = crop_image_from_gray(img)

    # Resize
    img = cv2.resize(img, (img_size, img_size))

    # CLAHE on green channel
    # Green channel shows retinal vessels most clearly
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_g = clahe.apply(img[:, :, 1])  # apply only to green channel

    # Ben Graham Gaussian subtract trick
    # Removes uneven illumination and enhances local contrast
    img = cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), img_size / 30), -4,
        128
    )

    # Circular mask
    # Remove everything outside the circular fundus area
    mask = np.zeros(img.shape)
    cv2.circle(
        mask,
        (img_size // 2, img_size // 2),
        int(img_size * 0.475),
        (1, 1, 1),
        -1,
        8,
        0
    )
    img = (img * mask + 128 * (1 - mask)).astype(np.uint8)

    return img


# PROCESS ALL IMAGES
def process_all_images(img_size=456):
    df = pd.read_csv(TRAIN_CSV)
    total = len(df)
    success = 0
    failed  = 0

    print(f"\nProcessing {total} images at {img_size}x{img_size}px...")
    print("   This will take 5-10 minutes. Please wait.\n")

    for idx, row in df.iterrows():
        img_path  = TRAIN_IMG_DIR / f"{row['id_code']}.png"
        save_path = PROCESSED_DIR / f"{row['id_code']}.png"

        # Skip if already processed
        if save_path.exists():
            success += 1
            continue

        img = ben_graham_preprocess(img_path, img_size=img_size)

        if img is not None:
            cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            success += 1
        else:
            failed += 1

        # Progress update every 200 images
        if (idx + 1) % 200 == 0:
            print(f"   [{idx+1}/{total}] {success} processed  {failed} failed")

    print(f"\nProcessing complete!")
    print(f"   Success : {success}")
    print(f"   Failed  : {failed}")
    print(f"   Saved to: {PROCESSED_DIR}")


# CREATE TRAIN / VAL / TEST SPLITS
def create_splits():
    df = pd.read_csv(TRAIN_CSV)

    print("\nCreating stratified train/val/test splits...")
    print("   Strategy: 70% train | 15% val | 15% test\n")

    # Step 1: Split off 15% test
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        stratify=df["diagnosis"],
        random_state=42
    )

    # Step 2: Split remaining into train and val
    train, val = train_test_split(
        train_val,
        test_size=0.1765,   # 0.1765 of 85% ~ 15% of total
        stratify=train_val["diagnosis"],
        random_state=42
    )

    # Add split column and save
    train["split"] = "train"
    val["split"]   = "val"
    test["split"]  = "test"

    full_df = pd.concat([train, val, test]).reset_index(drop=True)
    full_df.to_csv(SPLITS_DIR / "splits.csv", index=False)

    # Print summary
    print(f"  Split Summary")
    print(f"  Train : {len(train):>4} images ({len(train)/len(df)*100:.1f}%)")
    print(f"  Val   : {len(val):>4} images ({len(val)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(test):>4} images ({len(test)/len(df)*100:.1f}%)")
    print(f"  Total : {len(full_df):>4} images")


    # Verify stratification
    grade_labels = {0:"No DR", 1:"Mild", 2:"Moderate", 3:"Severe", 4:"Proliferative"}
    print(f"\n  Grade distribution per split:")
    print(f"  {'Grade':<20} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*46}")
    for grade in range(5):
        t = len(train[train["diagnosis"] == grade])
        v = len(val[val["diagnosis"] == grade])
        te = len(test[test["diagnosis"] == grade])
        print(f"  Grade {grade} ({grade_labels[grade]:<13}) {t:>8} {v:>8} {te:>8}")

    print(f"\nSplits saved to: {SPLITS_DIR / 'splits.csv'}")
    return train, val, test


# MAIN - Run everything
if __name__ == "__main__":
    print("Starting preprocessing pipeline...\n")

    # Step 1: Process all images
    process_all_images(img_size=456)

    # Step 2: Create splits
    create_splits()

    print("\nPreprocessing pipeline complete!")
    print("   Next step: Build PyTorch Dataset class")