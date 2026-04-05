import torch
import numpy as np
import cv2
import base64
import json
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))
from dataset import get_val_transforms
from models import get_model

# Paths
# Use relative paths for cloud deployment
BASE_DIR = Path(__file__).parent.parent  # Project root
CHECKPOINTS = BASE_DIR / "checkpoints"
OUTPUTS = BASE_DIR / "outputs"

GRADE_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

GRADE_COLORS = {0: "#2ECC71", 1: "#F39C12", 2: "#E67E22", 3: "#E74C3C", 4: "#8E44AD"}

GRADE_ADVICE = {
    0: "No signs of diabetic retinopathy detected. Continue regular annual screening.",
    1: "Mild NPDR detected. Schedule follow-up in 12 months with your ophthalmologist.",
    2: "Moderate NPDR detected. Schedule follow-up in 6 months. Monitor blood sugar carefully.",
    3: "Severe NPDR detected. Urgent referral to ophthalmologist recommended within 1 month.",
    4: "Proliferative DR detected. Immediate referral to ophthalmologist required.",
}

# FastAPI app
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="AI-powered DR grading from retinal fundus images",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eff_model = None


@app.on_event("startup")
async def load_models():
    global eff_model
    print(f"Loading models on {device}...")

    eff_model, _ = get_model("efficientnet_b5", pretrained=False, device=device)
    checkpoint = torch.load(
        CHECKPOINTS / "efficientnet_b5_best.pth", map_location=device, weights_only=True
    )
    eff_model.load_state_dict(checkpoint["state_dict"])
    eff_model.eval()
    print("EfficientNet-B5 loaded!")


# PREPROCESSING


def preprocess_image(img_bytes, img_size=456):
    """Load image from bytes and apply Ben Graham preprocessing."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))

    # Ben Graham
    img_processed = cv2.addWeighted(
        img, 4, cv2.GaussianBlur(img, (0, 0), img_size / 30), -4, 128
    )

    # Circular mask
    mask = np.zeros(img_processed.shape)
    cv2.circle(
        mask, (img_size // 2, img_size // 2), int(img_size * 0.475), (1, 1, 1), -1, 8, 0
    )
    img_processed = (img_processed * mask + 128 * (1 - mask)).astype(np.uint8)

    return img, img_processed


# GRAD-CAM++ HEATMAP


def generate_heatmap(img_tensor, img_display):
    target_layers = [eff_model.backbone.conv_head]
    cam = GradCAMPlusPlus(model=eff_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor, targets=[RawScoresOutputTarget()])[0]

    img_float = img_display.astype(np.float32) / 255.0
    visualization = show_cam_on_image(
        img_float,
        grayscale_cam,
        use_rgb=True,
        colormap=cv2.COLORMAP_JET,
        image_weight=0.5,
    )
    return visualization


# PREDICT ENDPOINT


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        img_bytes = await file.read()

        # Preprocess
        img_orig, img_processed = preprocess_image(img_bytes, img_size=456)

        # Transform for model
        transform = get_val_transforms(456)
        img_tensor = transform(image=img_processed)["image"]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred_raw = eff_model(img_tensor).item()

        pred_grade = int(np.clip(round(pred_raw), 0, 4))
        confidence = round(float(1 - abs(pred_raw - round(pred_raw))), 3)

        # Heatmap
        heatmap = generate_heatmap(img_tensor, img_processed)
        heatmap_b64 = base64.b64encode(
            cv2.imencode(".png", cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))[1]
        ).decode("utf-8")

        # Original image b64
        orig_b64 = base64.b64encode(
            cv2.imencode(".png", cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR))[1]
        ).decode("utf-8")

        return {
            "grade": pred_grade,
            "grade_label": GRADE_LABELS[pred_grade],
            "grade_color": GRADE_COLORS[pred_grade],
            "raw_score": round(pred_raw, 3),
            "confidence": confidence,
            "advice": GRADE_ADVICE[pred_grade],
            "heatmap_b64": heatmap_b64,
            "original_b64": orig_b64,
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {"status": "running", "device": str(device), "model": "EfficientNet-B5"}


@app.get("/")
async def root():
    return {"message": "Diabetic Retinopathy Detection API is running!"}
