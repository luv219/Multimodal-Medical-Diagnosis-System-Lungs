"""
models/swin.py — Swin Transformer classifier for NIH ChestX-ray14.

Backbone loaded via timm with num_classes=0 (head removed, returns pooled
features). Custom head uses LayerNorm + GELU to match Transformer conventions.
Outputs raw logits — no sigmoid — compatible with BCEWithLogitsLoss and
libauc AUCM_MultiLabel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SwinClassifier(nn.Module):
    """Swin Transformer backbone with a custom multi-label classification head.

    timm is called with ``num_classes=0`` so the pretrained head is removed and
    ``forward()`` returns a pooled feature vector of shape ``(B, feature_dim)``.
    The custom head is:

        LayerNorm(feature_dim) → Linear(feature_dim→512) → GELU
            → Dropout → Linear(512→num_classes)

    LayerNorm is used instead of BatchNorm because Swin outputs are derived
    from a layer-normalised sequence; GELU matches the Transformer convention.

    Outputs are raw logits — no sigmoid applied.
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_rate: float = 0.0,
        model_name: str = "swin_tiny_patch4_window7_224",
    ) -> None:
        super().__init__()

        self.model_name   = model_name
        self.num_classes  = num_classes
        self.dropout_rate = dropout_rate

        # num_classes=0 removes the timm head; backbone returns pooled features.
        self.backbone  = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim: int = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    # ------------------------------------------------------------------
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze all parameters in the backbone only."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    # ------------------------------------------------------------------
    def get_cam_target_layer(self) -> nn.Module:
        """Return the last Swin stage — the correct GradCAM++ hook point."""
        return self.backbone.layers[-1]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, num_classes).

        timm's Swin with num_classes=0 returns pooled features (B, feature_dim)
        directly, so no manual pooling is needed here.
        """
        features = self.backbone(x)      # (B, feature_dim)
        return self.classifier(features) # (B, num_classes)

    # ------------------------------------------------------------------
    def model_info(self) -> None:
        """Print model name, feature dimension, parameter counts, and head."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("=" * 60)
        print(f"  Model            : {self.model_name}")
        print(f"  feature_dim      : {self.feature_dim}")
        print(f"  num_classes      : {self.num_classes}")
        print(f"  dropout_rate     : {self.dropout_rate}")
        print(f"  Total params     : {total:,}")
        print(f"  Trainable params : {trainable:,}")
        print("  Classifier head  :")
        for i, layer in enumerate(self.classifier):
            print(f"    [{i}] {layer}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_swin(config) -> SwinClassifier:
    """Instantiate SwinClassifier and move it to the correct device.

    Reads from config:
        MODEL.get("swin_model_name", "swin_tiny_patch4_window7_224")
        MODEL.get("dropout_rate", 0.0)
        DATASET["class_names"] or DATASET["classes"]  — list of class labels.

    Args:
        config: Project config module (or any namespace with DATASET / MODEL).

    Returns:
        SwinClassifier on cuda if available, otherwise cpu.
    """
    model_name   = config.MODEL.get("swin_model_name", "swin_tiny_patch4_window7_224")
    dropout_rate = config.MODEL.get("dropout_rate", 0.0)

    # Support both "class_names" (current config key) and "classes" (alias).
    classes     = config.DATASET.get("class_names", config.DATASET.get("classes", []))
    num_classes = len(classes)

    model  = SwinClassifier(num_classes=num_classes, model_name=model_name,
                            dropout_rate=dropout_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    print(f"SwinClassifier ({model_name})  →  {device}  ({num_classes} classes)")
    return model


# ---------------------------------------------------------------------------
# __main__ — smoke test (pretrained=False avoids network download)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("models/swin.py — smoke test")
    print("=" * 60)

    # ---- Instantiate (no download) ----------------------------------------
    model = SwinClassifier(num_classes=14, pretrained=False, dropout_rate=0.0)

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

    backbone_params  = list(model.backbone.parameters())
    frozen_count     = sum(1 for p in backbone_params if not p.requires_grad)
    backbone_total   = len(backbone_params)
    classifier_train = sum(1 for p in model.classifier.parameters() if p.requires_grad)

    print(f"\nAfter freeze_backbone(True):")
    print(f"  Backbone params frozen     : {frozen_count} / {backbone_total}")
    print(f"  Classifier params trainable: {classifier_train}")
    assert frozen_count == backbone_total, "All backbone params should be frozen"
    assert classifier_train > 0,          "Classifier params must remain trainable"
    print("  freeze_backbone(True)  ✓")

    # ---- freeze_backbone(False) --------------------------------------------
    model.freeze_backbone(False)
    unfrozen_count = sum(1 for p in backbone_params if p.requires_grad)

    print(f"\nAfter freeze_backbone(False):")
    print(f"  Backbone params trainable  : {unfrozen_count} / {backbone_total}")
    assert unfrozen_count == backbone_total, "All backbone params should be trainable"
    print("  freeze_backbone(False)  ✓")

    print("\nDone.")
