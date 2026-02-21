import torch
import torch.nn as nn
import timm
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR     = Path(r"C:\Users\MSI\Downloads\dr_detection")
CHECKPOINTS  = BASE_DIR / "checkpoints"
CHECKPOINTS.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# EFFICIENTNET-B5 MODEL
# ─────────────────────────────────────────────────────────────────

class EfficientNetB5(nn.Module):
    """
    EfficientNet-B5 with regression head for ordinal DR grading.
    Output: single value in range [0, 4] representing DR grade.
    """

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()

        # ── Load pretrained backbone ──────────────────────────────
        self.backbone = timm.create_model(
            "efficientnet_b5",
            pretrained   = pretrained,
            num_classes  = 0,          # remove default classifier
            global_pool  = "avg"       # global average pooling
        )

        # ── Get feature dimension ─────────────────────────────────
        num_features = self.backbone.num_features  # 2048 for B5

        # ── Custom regression head ────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
            nn.Sigmoid()               # output in [0, 1]
        )

    def forward(self, x):
        features = self.backbone(x)
        output   = self.head(features)
        # Scale sigmoid [0,1] → [0,4] for DR grades
        return output.squeeze(1) * 4.0


# ─────────────────────────────────────────────────────────────────
# VISION TRANSFORMER — ViT-B/16
# ─────────────────────────────────────────────────────────────────

class ViTB16(nn.Module):
    """
    Vision Transformer ViT-B/16 with regression head.
    Global attention captures peripheral retinal lesions.
    Output: single value in range [0, 4] representing DR grade.
    """

    def __init__(self, pretrained=True, dropout=0.3, img_size=384):
        super().__init__()

        # ── Load pretrained backbone ──────────────────────────────
        self.backbone = timm.create_model(
            "vit_base_patch16_384",
            pretrained  = pretrained,
            num_classes = 0,           # remove default classifier
            img_size    = img_size
        )

        # ── Get feature dimension ─────────────────────────────────
        num_features = self.backbone.num_features  # 768 for ViT-B

        # ── Custom regression head ────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        output   = self.head(features)
        return output.squeeze(1) * 4.0


# ─────────────────────────────────────────────────────────────────
# MODEL FACTORY — easy model creation
# ─────────────────────────────────────────────────────────────────

def get_model(model_name, pretrained=True, device=None):
    """
    Returns the requested model moved to the correct device.

    Args:
        model_name : 'efficientnet_b5' or 'vit_b16'
        pretrained : use ImageNet pretrained weights
        device     : torch device (auto-detected if None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "efficientnet_b5":
        model = EfficientNetB5(pretrained=pretrained)
    elif model_name == "vit_b16":
        model = ViTB16(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'efficientnet_b5' or 'vit_b16'")

    model = model.to(device)
    return model, device


# ─────────────────────────────────────────────────────────────────
# HELPER — count trainable parameters
# ─────────────────────────────────────────────────────────────────

def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────────────────────────
# TEST — verify models load and forward pass works
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}\n")

    # ── Test EfficientNet-B5 ──────────────────────────────────────
    print("=" * 50)
    print("  Testing EfficientNet-B5")
    print("=" * 50)
    model_eff, device = get_model("efficientnet_b5", pretrained=True)

    # Dummy batch — 2 images at 456x456
    dummy = torch.randn(2, 3, 456, 456).to(device)
    with torch.no_grad():
        out = model_eff(dummy)

    total, trainable = count_parameters(model_eff)
    print(f"  Input shape      : {dummy.shape}")
    print(f"  Output shape     : {out.shape}")
    print(f"  Output values    : {out.tolist()}")
    print(f"  Output range     : [{out.min():.3f}, {out.max():.3f}]  ✅ should be in [0,4]")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  VRAM used        : {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Clear GPU memory ──────────────────────────────────────────
    del model_eff, dummy, out
    torch.cuda.empty_cache()

    # ── Test ViT-B/16 ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  Testing ViT-B/16")
    print("=" * 50)
    model_vit, device = get_model("vit_b16", pretrained=True)

    # Dummy batch — 2 images at 384x384
    dummy = torch.randn(2, 3, 384, 384).to(device)
    with torch.no_grad():
        out = model_vit(dummy)

    total, trainable = count_parameters(model_vit)
    print(f"  Input shape      : {dummy.shape}")
    print(f"  Output shape     : {out.shape}")
    print(f"  Output values    : {out.tolist()}")
    print(f"  Output range     : [{out.min():.3f}, {out.max():.3f}]  ✅ should be in [0,4]")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  VRAM used        : {torch.cuda.memory_allocated()/1e9:.2f} GB")

    del model_vit, dummy, out
    torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print("✅ Both models working perfectly!")
    print("   Ready for training.")
    print(f"{'='*50}")