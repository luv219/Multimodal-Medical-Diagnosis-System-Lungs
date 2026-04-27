"""
losses.py — Loss functions and optimizer/scheduler factories for NIH ChestX-ray14.

Primary training loss : AUCMLoss  (wraps libauc MultiLabelAUCMLoss + PESG).
Ablation loss         : FocalLoss (standard sigmoid focal loss; uses AdamW in train.py).
"""
from __future__ import annotations

import functools
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from libauc.losses import MultiLabelAUCMLoss
from libauc.optimizers import PESG

import config as _default_config


# ---------------------------------------------------------------------------
# AUCMLoss
# ---------------------------------------------------------------------------

class AUCMLoss(nn.Module):
    """Wraps libauc MultiLabelAUCMLoss for multi-label AUC-margin optimisation.

    Workflow
    --------
    1. Instantiate (no imratio needed at construction time).
    2. Call ``update_imratio(dataset.class_weights)`` once the dataset is
       loaded so the internal loss uses real per-class positive rates.
    3. Build a PESG optimizer via ``get_optimizer`` — it reads ``self.aucm_loss``
       directly to obtain the margin-variable tensors (a, b, alpha).

    Attributes
    ----------
    aucm_loss : MultiLabelAUCMLoss
        Exposed so that the PESG optimizer can access ``.a``, ``.b``,
        ``.alpha`` directly.
    """

    def __init__(
        self,
        num_classes: int = 14,
        margin: float = 1.0,
        epoch_decay: float = 2e-3,
        gamma: float = 500,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.margin      = margin
        self.epoch_decay = epoch_decay
        self.gamma       = gamma

        self.aucm_loss = MultiLabelAUCMLoss(
            num_labels=num_classes,
            margin=margin,
            epoch_decay=epoch_decay,
            gamma=gamma,
        )

    # ------------------------------------------------------------------
    def update_imratio(self, class_weights_tensor: torch.Tensor) -> None:
        """Reinitialise the internal AUCM loss with real per-class positive rates.

        Converts NIHChestDataset.class_weights (neg/pos ratios) to imratio:

            imratio_i = 1 / (1 + class_weight_i)  =  pos_i / (pos_i + neg_i)

        The internal AUCM_MultiLabel is rebuilt so that the margin-variable
        tensors (a, b, alpha) are initialised correctly before PESG is created.

        Args:
            class_weights_tensor: 1-D float tensor, shape (num_classes,),
                where entry i is neg_count_i / pos_count_i.
        """
        imratio_list = (1.0 / (1.0 + class_weights_tensor.float())).tolist()
        self.aucm_loss = MultiLabelAUCMLoss(
            imratio=imratio_list,
            num_labels=self.num_classes,
            margin=self.margin,
            epoch_decay=self.epoch_decay,
            gamma=self.gamma,
        )

    # ------------------------------------------------------------------
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.aucm_loss(logits, labels)


# ---------------------------------------------------------------------------
# FocalLoss — ablation only
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Sigmoid focal loss for multi-label classification.

    **Ablation use only** — this is NOT the primary training loss.
    Use AUCMLoss + PESG for the main NIH ChestX-ray14 experiments.
    FocalLoss is paired with a standard AdamW optimizer in train.py.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean focal loss over all (batch, class) positions.

        Args:
            logits: raw model outputs, shape (B, C).
            labels: binary float targets,  shape (B, C).

        Returns:
            Scalar mean focal loss.
        """
        p       = torch.sigmoid(logits)
        p_t     = labels * p + (1.0 - labels) * (1.0 - p)
        alpha_t = labels * self.alpha + (1.0 - labels) * (1.0 - self.alpha)
        loss    = -alpha_t * (1.0 - p_t) ** self.gamma * torch.log(p_t.clamp(min=1e-7))
        return loss.mean()


# ---------------------------------------------------------------------------
# Optimizer / scheduler factories
# ---------------------------------------------------------------------------

def get_optimizer(model: nn.Module, loss: AUCMLoss, config=_default_config) -> PESG:
    """Return a PESG optimizer configured for AUCM_MultiLabel training.

    Call *after* ``loss.update_imratio()`` so that ``loss.aucm_loss.a``,
    ``.b``, and ``.alpha`` reflect the real class imbalance.

    Args:
        model:  The network whose parameters PESG will update.
        loss:   An AUCMLoss instance (exposes ``.aucm_loss`` for PESG).
        config: Config module; reads ``TRAINING["learning_rate"]`` and
                ``TRAINING["weight_decay"]``.

    Returns:
        Configured PESG optimizer.
    """
    lr           = config.TRAINING["learning_rate"]
    weight_decay = config.TRAINING["weight_decay"]
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    return PESG(
        model,
        a=loss.aucm_loss.a,
        b=loss.aucm_loss.b,
        alpha=loss.aucm_loss.alpha,
        lr=lr,
        gamma=loss.gamma,
        margin=loss.margin,
        epoch_decay=loss.epoch_decay,
        weight_decay=weight_decay,
        momentum=0.9,
        device=device,
    )


def get_scheduler(optimizer, config=_default_config) -> CosineAnnealingLR:
    """Return a CosineAnnealingLR scheduler.

    Args:
        optimizer: Any PyTorch optimizer (typically PESG or AdamW).
        config:    Config module; reads ``TRAINING["num_epochs"]``.

    Returns:
        CosineAnnealingLR with eta_min=1e-6.
    """
    return CosineAnnealingLR(
        optimizer,
        T_max=config.TRAINING["num_epochs"],
        eta_min=1e-6,
    )


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def get_loss(loss_name: str, dataset, config=_default_config):
    """Instantiate and configure a loss function for training.

    Args:
        loss_name: ``"aucm"``  — primary AUC-margin loss; returns a PESG
                                 optimizer factory alongside the loss.
                   ``"focal"`` — ablation focal loss; AdamW is wired in
                                 train.py, so optimizer_fn is None.
        dataset:   Object with a ``.class_weights`` tensor of shape
                   (num_classes,) — as produced by NIHChestDataset.
        config:    Config module (default: project config).

    Returns:
        ``(loss_fn, optimizer_fn)``

        * For ``"aucm"``:  ``optimizer_fn`` is a partial of ``get_optimizer``
          that accepts ``(model, loss)`` and returns a ready PESG optimizer.
        * For ``"focal"``: ``optimizer_fn`` is ``None``; use AdamW in train.py.

    Raises:
        ValueError: If ``loss_name`` is not ``"aucm"`` or ``"focal"``.
    """
    if loss_name == "aucm":
        num_classes = config.DATASET["num_classes"]
        loss_fn     = AUCMLoss(num_classes=num_classes)
        loss_fn.update_imratio(dataset.class_weights)
        optimizer_fn = functools.partial(get_optimizer, config=config)
        return loss_fn, optimizer_fn

    if loss_name == "focal":
        return FocalLoss(), None

    raise ValueError(
        f"Unknown loss name '{loss_name}'. Valid options: 'aucm', 'focal'."
    )


# ---------------------------------------------------------------------------
# __main__ — smoke test (no GPU or dataset files required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import types

    print("=" * 60)
    print("losses.py — smoke test")
    print("=" * 60)

    B, C = 4, 14
    torch.manual_seed(0)
    logits = torch.randn(B, C)
    labels = torch.randint(0, 2, (B, C)).float()

    # ---- AUCMLoss ---------------------------------------------------------
    print("\n[AUCMLoss]")
    aucm_loss = AUCMLoss(num_classes=C)
    print(f"  loss (before update_imratio): {aucm_loss(logits, labels).item():.6f}")

    fake_weights = torch.rand(C) * 9.0 + 1.0       # simulated neg/pos ratios in [1, 10]
    aucm_loss.update_imratio(fake_weights)
    print(f"  loss (after update_imratio): {aucm_loss(logits, labels).item():.6f}")

    # ---- FocalLoss --------------------------------------------------------
    print("\n[FocalLoss]")
    focal_loss = FocalLoss()
    print(f"  loss value : {focal_loss(logits, labels).item():.6f}")

    # ---- get_loss + optimizer_fn -----------------------------------------
    print("\n[get_loss — aucm]")

    mock_dataset = types.SimpleNamespace(
        class_weights=torch.rand(C) * 9.0 + 1.0
    )
    mock_config = types.SimpleNamespace(
        DATASET={"num_classes": C},
        TRAINING={
            "learning_rate": 1e-4,
            "weight_decay":  1e-5,
            "num_epochs":    50,
        },
    )

    loss_fn, optimizer_fn = get_loss("aucm", mock_dataset, config=mock_config)
    print(f"  loss type    : {type(loss_fn).__name__}")

    dummy_model = nn.Linear(10, C)
    optimizer   = optimizer_fn(dummy_model, loss_fn)
    print(f"  optimizer type : {type(optimizer).__name__}")

    print("\n[get_loss — focal]")
    focal_fn, focal_opt_fn = get_loss("focal", mock_dataset, config=mock_config)
    print(f"  loss type      : {type(focal_fn).__name__}")
    print(f"  optimizer_fn   : {focal_opt_fn}  (None — AdamW handled in train.py)")

    print("\nDone.")
