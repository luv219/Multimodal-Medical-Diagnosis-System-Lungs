"""
models/hybrid.py — CNN-Transformer hybrid classifier for NIH ChestX-ray14.

Architecture:
    ResNet-50 (layer4 output)  →  Conv2d projection  →  positional encoding
    →  TransformerEncoder  →  global average pool  →  classification head

ResNet-50 supplies spatially-aware local features (2048-channel, 7×7 map).
The Transformer encoder attends across all 49 spatial positions to capture
long-range relationships that pure CNNs miss.  Outputs raw logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HybridClassifier(nn.Module):
    """ResNet-50 CNN backbone + lightweight Transformer encoder.

    Forward pipeline
    ----------------
    1. CNN backbone (ResNet-50 layer4):  (B, 3, 224, 224) → (B, 2048, 7, 7)
    2. Conv2d projection:                (B, 2048, 7, 7)  → (B, transformer_dim, 7, 7)
    3. Flatten + add positional enc:     (B, transformer_dim, 7, 7) → (B, 49, transformer_dim)
    4. TransformerEncoder:               (B, 49, transformer_dim) → (B, 49, transformer_dim)
    5. Global average pool (seq dim):    → (B, transformer_dim)
    6. Classifier head:                  → (B, num_classes)

    Outputs raw logits — no sigmoid applied.
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_rate: float = 0.0,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        transformer_dim: int = 256,
    ) -> None:
        super().__init__()

        self.num_classes            = num_classes
        self.dropout_rate           = dropout_rate
        self.num_transformer_layers = num_transformer_layers
        self.num_heads              = num_heads
        self.transformer_dim        = transformer_dim

        # ---- CNN backbone --------------------------------------------------
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        _resnet = resnet50(weights=weights)
        # Remove avgpool and fc — we use layer4's spatial feature map directly.
        _resnet.avgpool = nn.Identity()
        _resnet.fc      = nn.Identity()
        self.cnn_backbone = _resnet

        # ---- Projection: 2048 → transformer_dim ----------------------------
        self.projection = nn.Conv2d(2048, transformer_dim, kernel_size=1)

        # ---- Learnable positional encoding: 49 tokens (7×7 spatial grid) ---
        self.pos_encoding = nn.Parameter(torch.zeros(1, 49, transformer_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        # ---- Transformer encoder -------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # ---- Classifier head -----------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, num_classes),
        )

    # ------------------------------------------------------------------
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze CNN backbone parameters only.

        Projection, positional encoding, transformer, and classifier head
        remain trainable regardless of this setting.
        """
        for param in self.cnn_backbone.parameters():
            param.requires_grad = not freeze

    # ------------------------------------------------------------------
    def get_cam_target_layer(self) -> nn.Module:
        """Return ResNet layer4 — the correct GradCAM++ hook point."""
        return self.cnn_backbone.layer4

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, num_classes)."""
        # 1. CNN feature map: (B, 2048, 7, 7)
        features = self.cnn_backbone(x)

        # ResNet with Identity avgpool/fc returns a flattened vector when the
        # spatial map collapses; reshape back to (B, 2048, 7, 7).
        B = features.size(0)
        features = features.view(B, 2048, 7, 7)

        # 2. Project channels: (B, 2048, 7, 7) → (B, transformer_dim, 7, 7)
        features = self.projection(features)

        # 3. Flatten spatial dims → sequence: (B, transformer_dim, 49) → (B, 49, transformer_dim)
        features = features.flatten(2).transpose(1, 2)

        # 4. Add learnable positional encoding (broadcast over batch)
        features = features + self.pos_encoding

        # 5. Transformer encoder: (B, 49, transformer_dim)
        features = self.transformer(features)

        # 6. Global average pool over sequence dimension: (B, transformer_dim)
        features = features.mean(dim=1)

        # 7. Classification head: (B, num_classes)
        return self.classifier(features)

    # ------------------------------------------------------------------
    def model_info(self) -> None:
        """Print architecture hyper-parameters and parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("=" * 60)
        print(f"  Model                  : HybridClassifier")
        print(f"  transformer_dim        : {self.transformer_dim}")
        print(f"  num_heads              : {self.num_heads}")
        print(f"  num_transformer_layers : {self.num_transformer_layers}")
        print(f"  spatial tokens (7×7)   : 49")
        print(f"  num_classes            : {self.num_classes}")
        print(f"  dropout_rate           : {self.dropout_rate}")
        print(f"  Total params           : {total:,}")
        print(f"  Trainable params       : {trainable:,}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_hybrid(config) -> HybridClassifier:
    """Instantiate HybridClassifier and move it to the correct device.

    Reads from config:
        DATASET["class_names"] or DATASET["classes"]  — list of class labels.
        MODEL.get("dropout_rate", 0.0)
        MODEL.get("num_transformer_layers", 2)
        MODEL.get("num_heads", 8)
        MODEL.get("transformer_dim", 256)

    Args:
        config: Project config module (or any namespace with DATASET / MODEL).

    Returns:
        HybridClassifier on cuda if available, otherwise cpu.
    """
    # Support both "class_names" (current config key) and "classes" (alias).
    classes     = config.DATASET.get("class_names", config.DATASET.get("classes", []))
    num_classes = len(classes)

    dropout_rate           = config.MODEL.get("dropout_rate", 0.0)
    num_transformer_layers = config.MODEL.get("num_transformer_layers", 2)
    num_heads              = config.MODEL.get("num_heads", 8)
    transformer_dim        = config.MODEL.get("transformer_dim", 256)

    model  = HybridClassifier(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        transformer_dim=transformer_dim,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    print(
        f"HybridClassifier  →  {device}  "
        f"({num_classes} classes, {num_transformer_layers}× Transformer, "
        f"dim={transformer_dim}, heads={num_heads})"
    )
    return model


# ---------------------------------------------------------------------------
# __main__ — smoke test (pretrained=False avoids network download)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("models/hybrid.py — smoke test")
    print("=" * 60)

    # ---- Instantiate (no download) ----------------------------------------
    model = HybridClassifier(
        num_classes=14,
        pretrained=False,
        dropout_rate=0.0,
        num_transformer_layers=2,
        num_heads=8,
        transformer_dim=256,
    )

    # ---- Forward pass -------------------------------------------------------
    x      = torch.randn(2, 3, 224, 224)
    logits = model(x)

    print(f"\nInput  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(logits.shape)}")
    assert tuple(logits.shape) == (2, 14), (
        f"Shape mismatch — expected (2, 14), got {tuple(logits.shape)}"
    )
    print("Output shape confirmed: (2, 14)  ✓")

    # ---- model_info ---------------------------------------------------------
    print()
    model.model_info()

    # ---- GradCAM++ target layer --------------------------------------------
    cam_layer = model.get_cam_target_layer()
    assert isinstance(cam_layer, nn.Module), "get_cam_target_layer() must return nn.Module"
    print(f"\nGradCAM++ target layer : {type(cam_layer).__name__}  ✓")

    # ---- freeze_backbone(True) ---------------------------------------------
    model.freeze_backbone(True)

    backbone_params  = list(model.cnn_backbone.parameters())
    frozen_count     = sum(1 for p in backbone_params if not p.requires_grad)
    backbone_total   = len(backbone_params)

    # Collect all non-backbone trainable params (projection, pos_enc, transformer, head)
    non_backbone_train = sum(
        p.requires_grad
        for name, p in model.named_parameters()
        if not name.startswith("cnn_backbone")
    )

    print(f"\nAfter freeze_backbone(True):")
    print(f"  CNN backbone params frozen       : {frozen_count} / {backbone_total}")
    print(f"  Non-backbone params trainable    : {non_backbone_train}")
    assert frozen_count == backbone_total, "All CNN backbone params should be frozen"
    assert non_backbone_train > 0,        "Projection / transformer / head must be trainable"
    print("  freeze_backbone(True)  ✓")

    # ---- freeze_backbone(False) --------------------------------------------
    model.freeze_backbone(False)
    unfrozen_count = sum(1 for p in backbone_params if p.requires_grad)

    print(f"\nAfter freeze_backbone(False):")
    print(f"  CNN backbone params trainable    : {unfrozen_count} / {backbone_total}")
    assert unfrozen_count == backbone_total, "All CNN backbone params should be trainable"
    print("  freeze_backbone(False)  ✓")

    print("\nDone.")
