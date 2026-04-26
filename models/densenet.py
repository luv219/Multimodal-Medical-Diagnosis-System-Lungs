"""
models/densenet.py — DenseNet-121 classifier for NIH ChestX-ray14.

Outputs raw logits (no sigmoid) for compatibility with BCEWithLogitsLoss
and libauc AUCM_MultiLabel.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DenseNet121Classifier(nn.Module):
    """DenseNet-121 backbone with a custom two-layer multi-label head.

    The original ImageNet classifier is replaced by:
        Linear(1024→512) → ReLU → Dropout → Linear(512→num_classes)

    No sigmoid is applied — outputs are raw logits compatible with
    BCEWithLogitsLoss and libauc AUCM_MultiLabel.
    """

    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = densenet121(weights=weights)

        # DenseNet-121 feature dimension is 1024.
        self.backbone.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

        self.num_classes  = num_classes
        self.dropout_rate = dropout_rate

    # ------------------------------------------------------------------
    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze all backbone parameters except the classifier head.

        Use freeze=True for a warmup phase; False to fine-tune end-to-end.
        """
        for name, param in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = not freeze

    # ------------------------------------------------------------------
    def get_cam_target_layer(self) -> nn.Module:
        """Return denseblock4 — the correct GradCAM++ target for DenseNet-121."""
        return self.backbone.features.denseblock4

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch, num_classes)."""
        return self.backbone(x)

    # ------------------------------------------------------------------
    def model_info(self) -> None:
        """Print parameter counts and classifier head architecture."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("=" * 60)
        print(f"  Model            : DenseNet121Classifier")
        print(f"  num_classes      : {self.num_classes}")
        print(f"  dropout_rate     : {self.dropout_rate}")
        print(f"  Total params     : {total:,}")
        print(f"  Trainable params : {trainable:,}")
        print("  Classifier head  :")
        for i, layer in enumerate(self.backbone.classifier):
            print(f"    [{i}] {layer}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_densenet(config) -> DenseNet121Classifier:
    """Instantiate DenseNet121Classifier and move it to the correct device.

    Reads from config:
        DATASET["num_classes"]          — number of output classes (default 14).
        MODEL.get("dropout_rate", 0.0)  — dropout before final linear layer.

    Args:
        config: Project config module (or any namespace with DATASET / MODEL).

    Returns:
        DenseNet121Classifier on cuda if available, otherwise cpu.
    """
    num_classes  = config.DATASET["num_classes"]
    dropout_rate = config.MODEL.get("dropout_rate", 0.0)

    model  = DenseNet121Classifier(num_classes=num_classes, dropout_rate=dropout_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)

    print(f"DenseNet121Classifier  →  {device}  ({num_classes} classes)")
    return model


# ---------------------------------------------------------------------------
# __main__ — smoke test (no GPU or dataset files required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("models/densenet.py — smoke test")
    print("=" * 60)

    # ---- Instantiate -------------------------------------------------------
    model = DenseNet121Classifier(num_classes=14, pretrained=True, dropout_rate=0.0)

    # ---- Forward pass ------------------------------------------------------
    x      = torch.randn(2, 3, 224, 224)
    logits = model(x)

    print(f"\nInput  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(logits.shape)}")
    assert tuple(logits.shape) == (2, 14), (
        f"Shape mismatch — expected (2, 14), got {tuple(logits.shape)}"
    )
    print("Output shape confirmed: (2, 14)  ✓")

    # ---- model_info --------------------------------------------------------
    print()
    model.model_info()

    # ---- GradCAM++ target layer --------------------------------------------
    cam_layer = model.get_cam_target_layer()
    assert isinstance(cam_layer, nn.Module), "get_cam_target_layer() must return nn.Module"
    print(f"\nGradCAM++ target layer : {type(cam_layer).__name__}  ✓")

    # ---- freeze_backbone ---------------------------------------------------
    model.freeze_backbone(True)

    backbone_params   = [(n, p) for n, p in model.backbone.named_parameters()
                         if not n.startswith("classifier")]
    frozen_count      = sum(1 for _, p in backbone_params if not p.requires_grad)
    backbone_total    = len(backbone_params)
    classifier_train  = sum(p.requires_grad
                            for p in model.backbone.classifier.parameters())

    print(f"\nAfter freeze_backbone(True):")
    print(f"  Backbone params frozen     : {frozen_count} / {backbone_total}")
    print(f"  Classifier params trainable: {classifier_train}")
    assert frozen_count == backbone_total, "All backbone params should be frozen"
    assert classifier_train > 0,          "Classifier params must remain trainable"
    print("  freeze_backbone(True)  ✓")

    model.freeze_backbone(False)
    unfrozen_count = sum(1 for _, p in backbone_params if p.requires_grad)
    print(f"\nAfter freeze_backbone(False):")
    print(f"  Backbone params trainable  : {unfrozen_count} / {backbone_total}")
    assert unfrozen_count == backbone_total, "All backbone params should be trainable"
    print("  freeze_backbone(False)  ✓")

    print("\nDone.")
