import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

import sys
sys.path.append(str(Path(__file__).parent))
from dataset import get_val_transforms
from models  import get_model

# Paths
BASE_DIR    = Path(r"C:\Users\MSI\Downloads\dr_detection")
CHECKPOINTS = BASE_DIR / "checkpoints"
PROCESSED   = BASE_DIR / "data" / "processed" / "train_images"
SPLITS_CSV  = BASE_DIR / "data" / "splits" / "splits.csv"
GRADCAM_DIR = BASE_DIR / "outputs" / "gradcam"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

GRADE_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}


# LOAD MODEL

def load_model(model_name, device):
    model, device = get_model(model_name, pretrained=False, device=device)
    checkpoint    = torch.load(
        CHECKPOINTS / f"{model_name}_best.pth",
        map_location  = device,
        weights_only  = True
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"  Loaded {model_name}")
    return model


# GET TARGET LAYER FOR GRAD-CAM++

def get_target_layer(model, model_name):
    if model_name == "efficientnet_b5":
        return [model.backbone.conv_head]
    else:
        raise ValueError(f"Unknown model: {model_name}. Only 'efficientnet_b5' is supported.")


# GENERATE HEATMAP FOR ONE IMAGE

def generate_gradcam(model, model_name, img_path, img_size, device):
    """
    Generates Grad-CAM++ heatmap for a single image.
    Returns original image and heatmap overlay.
    """
    # Load and preprocess image
    img_orig = cv2.imread(str(img_path))
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_orig = cv2.resize(img_orig, (img_size, img_size))

    # Transform for model
    transform  = get_val_transforms(img_size)
    img_tensor = transform(image=img_orig)["image"]
    img_tensor = img_tensor.unsqueeze(0).to(device)  # add batch dim

    # Normalize image for display
    img_display = img_orig.astype(np.float32) / 255.0

    # Get prediction
    with torch.no_grad():
        pred = model(img_tensor).item()
    pred_grade = int(np.clip(round(pred), 0, 4))

    # Generate Grad-CAM++
    target_layers = get_target_layer(model, model_name)
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    grayscale_cam = cam(
        input_tensor = img_tensor,
        targets      = [RawScoresOutputTarget()]
)
    grayscale_cam = grayscale_cam[0]  # remove batch dim

    # Overlay heatmap on image
    visualization = show_cam_on_image(
        img_display,
        grayscale_cam,
        use_rgb    = True,
        colormap   = cv2.COLORMAP_JET,
        image_weight = 0.5
    )

    return img_orig, visualization, pred_grade, pred


# GENERATE HEATMAPS FOR SAMPLE IMAGES PER GRADE

def generate_all_heatmaps(model, model_name, img_size, samples_per_grade=3):
    device = next(model.parameters()).device
    df     = pd.read_csv(SPLITS_CSV)

    # Use test set only
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"\n  Generating heatmaps for {model_name}...")
    print(f"  {samples_per_grade} samples per grade = {samples_per_grade * 5} total\n")

    fig, axes = plt.subplots(
        5, samples_per_grade * 2,
        figsize=(samples_per_grade * 8, 22)
    )

    for grade in range(5):
        grade_df = test_df[test_df["diagnosis"] == grade]
        samples  = grade_df.sample(
            min(samples_per_grade, len(grade_df)),
            random_state=42
        )

        for col_idx, (_, row) in enumerate(samples.iterrows()):
            img_path = PROCESSED / f"{row['id_code']}.png"

            try:
                orig, heatmap, pred_grade, pred_val = generate_gradcam(
                    model, model_name, img_path, img_size, device
                )

                # Original image
                ax_orig = axes[grade][col_idx * 2]
                ax_orig.imshow(orig)
                ax_orig.axis("off")
                ax_orig.set_title(
                    f"True: {GRADE_LABELS[grade]}",
                    fontsize=9, fontweight="bold", color="green"
                )

                # Heatmap overlay
                ax_heat = axes[grade][col_idx * 2 + 1]
                ax_heat.imshow(heatmap)
                ax_heat.axis("off")
                color = "green" if pred_grade == grade else "red"
                ax_heat.set_title(
                    f"Pred: {GRADE_LABELS[pred_grade]} ({pred_val:.2f})",
                    fontsize=9, fontweight="bold", color=color
                )

            except Exception as e:
                print(f"    Error on {row['id_code']}: {e}")

        # Grade label on left
        axes[grade][0].set_ylabel(
            f"Grade {grade}\n{GRADE_LABELS[grade]}",
            fontsize=11, fontweight="bold",
            rotation=0, labelpad=80, va="center"
        )

    plt.suptitle(
        f"Grad-CAM++ Heatmaps - {model_name.upper()}\n"
        f"Green title = correct prediction | Red title = wrong prediction",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    save_path = GRADCAM_DIR / f"{model_name}_gradcam.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# MAIN

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # EfficientNet-B5 heatmaps only
    print("Generating Grad-CAM++ for EfficientNet-B5")
    eff_model = load_model("efficientnet_b5", device)
    generate_all_heatmaps(
        eff_model, "efficientnet_b5",
        img_size=456,
        samples_per_grade=4    # 4 samples per grade = 20 total
    )

    del eff_model
    torch.cuda.empty_cache()

    print(f"\nGrad-CAM++ complete!")
    print(f"Heatmaps saved to: {GRADCAM_DIR}")