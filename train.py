"""
train.py — Training pipeline for NIH ChestX-ray14 multi-label classification.

Optimised for RTX 3050 6 GB:
  - Mixed-precision (AMP) with float16 forward/loss, manual PESG-safe scaler step
  - Gradient accumulation (effective batch = batch_size × grad_accum_steps)
  - cudnn.benchmark=True via config.set_seed()

Usage:
    python train.py --model densenet --loss aucm
    python train.py --model swin    --loss focal --epochs 30 --freeze-epochs 5
    python train.py --model hybrid  --resume checkpoints/latest.pth
    python train.py --model densenet --fast-dev-run   # smoke-test 2 epochs
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config as cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NIH ChestX-ray14 training script")
    p.add_argument("--model",           choices=["densenet", "swin", "hybrid"], required=True)
    p.add_argument("--loss",            choices=["aucm", "focal"], default="aucm")
    p.add_argument("--epochs",          type=int,  default=None,
                   help="Override config num_epochs")
    p.add_argument("--batch-size",      type=int,  default=None,
                   help="Override config batch_size")
    p.add_argument("--freeze-epochs",   type=int,  default=3,
                   help="Epochs to train with frozen backbone before full fine-tuning")
    p.add_argument("--resume",          type=str,  default=None,
                   help="Path to checkpoint (.pth) to resume from")
    p.add_argument("--experiment-name", type=str,  default=None,
                   help="Override config experiment_name")
    p.add_argument("--fast-dev-run",    action="store_true",
                   help="Smoke-test mode: 2 epochs, 100 train batches, 50 val batches")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    Default patience is 10 — rare-disease AUC can plateau for several epochs
    before improving, so a short patience causes premature termination.

    Args:
        patience:  Epochs with no improvement before stopping (default 10).
        min_delta: Minimum change to qualify as an improvement.
        mode:      "max" (higher is better, e.g. AUC) or "min".
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience    = patience
        self.min_delta   = min_delta
        self.mode        = mode
        self.best        = float("-inf") if mode == "max" else float("inf")
        self.counter     = 0
        self.epochs_seen = 0

    def __call__(self, metric: float) -> bool:
        """Return True if training should stop."""
        self.epochs_seen += 1

        improved = (
            metric > self.best + self.min_delta if self.mode == "max"
            else metric < self.best - self.min_delta
        )

        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience and self.epochs_seen >= self.patience


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> tuple[int, float]:
    """Load a checkpoint and restore model / optimizer / scheduler state.

    Returns:
        (start_epoch, best_auc)
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as exc:
        warnings.warn(f"Could not restore optimizer state: {exc}")

    if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as exc:
            warnings.warn(f"Could not restore scheduler state: {exc}")

    epoch    = ckpt.get("epoch", 0) + 1
    best_auc = ckpt.get("best_auc", 0.0)
    print(f"Resumed from '{path}'  (epoch {ckpt.get('epoch', '?')}, best_auc={best_auc:.4f})")
    return epoch, best_auc


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer,
    scaler,                     # torch.amp.GradScaler or None
    grad_accum_steps: int = 1,
    max_batches: int | None = None,   # for --fast-dev-run
) -> float:
    """Run one training epoch with optional AMP and gradient accumulation.

    AMP + PESG notes
    ----------------
    scaler.step(optimizer) cannot be used with PESG because PESG's step method
    has a non-standard signature.  Instead we:
      1. scaler.unscale_(optimizer) — convert scaled grads to true grads
      2. optimizer.step()           — PESG's custom update
      3. scaler.update()            — adjust scale factor for next iteration
    For standard optimizers (AdamW) the same path is used for consistency.

    Returns:
        Mean loss per batch (unscaled, for logging).
    """
    model.train()
    total_loss  = 0.0
    num_batches = 0
    is_cuda     = (device.type == "cuda")

    bar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False, dynamic_ncols=True)

    for batch_idx, (images, labels, _) in enumerate(bar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Zero gradients at the START of each accumulation cycle
        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad()

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        is_last_batch  = (batch_idx == len(loader) - 1)
        if max_batches is not None:
            is_last_batch = is_last_batch or (batch_idx == max_batches - 1)
        should_step = (batch_idx + 1) % grad_accum_steps == 0 or is_last_batch

        # ---- Forward + loss (inside autocast when AMP enabled) -----------
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss   = loss_fn(logits, labels) / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(images)
            loss   = loss_fn(logits, labels) / grad_accum_steps
            loss.backward()

        # ---- Optimizer step at end of accumulation cycle -----------------
        if should_step:
            if scaler is not None:
                # Unscale before optimizer.step() so PESG sees true gradients.
                scaler.unscale_(optimizer)
                optimizer.step()
                scaler.update()
            else:
                optimizer.step()

        batch_loss   = loss.item() * grad_accum_steps   # restore for display
        total_loss  += batch_loss
        num_batches += 1

        bar.set_postfix(loss=f"{batch_loss:.4f}")

        if writer is not None:
            global_step = (epoch - 1) * len(loader) + batch_idx
            writer.add_scalar("Loss/train_batch", batch_loss, global_step)

    # PESG requires one epoch-level regularizer decay call
    if hasattr(optimizer, "update_regularizer"):
        optimizer.update_regularizer(decay_factor=2)

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    class_names: list[str],
    max_batches: int | None = None,   # for --fast-dev-run
) -> dict:
    """Run inference and compute per-class AUC metrics.

    Returns:
        dict with keys: "mean_auc", "per_class_auc" (class→float), "loss".
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss  = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(
            tqdm(loader, desc="[eval]", leave=False, dynamic_ncols=True)
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = loss_fn(logits, labels)

            total_loss  += loss.item()
            num_batches += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_probs  = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    per_class_auc: dict[str, float] = {}
    valid_aucs: list[float] = []

    for i, name in enumerate(class_names):
        y_true  = all_labels[:, i]
        y_score = all_probs[:, i]

        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            warnings.warn(
                f"Class '{name}' has no minority-class samples in this split; AUC skipped."
            )
            continue

        auc = roc_auc_score(y_true, y_score)
        per_class_auc[name] = float(auc)
        valid_aucs.append(auc)

    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return {
        "mean_auc":      mean_auc,
        "per_class_auc": per_class_auc,
        "loss":          total_loss / max(num_batches, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Fast-dev-run warning --------------------------------------------
    if args.fast_dev_run:
        print("=" * 60)
        print("  WARNING: --fast-dev-run active")
        print("  Training: 100 batches × 2 epochs")
        print("  Validation: 50 batches per epoch")
        print("  This mode is for pipeline smoke-testing only.")
        print("=" * 60)

    # ---- Apply CLI overrides to config -----------------------------------
    if args.epochs is not None:
        cfg.TRAINING["num_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.TRAINING["batch_size"] = args.batch_size
    if args.experiment_name is not None:
        cfg.LOGGING["experiment_name"] = args.experiment_name
    if args.fast_dev_run:
        cfg.TRAINING["num_epochs"] = 2

    cfg.set_seed(cfg.TRAINING["seed"])

    grad_accum_steps = cfg.TRAINING.get("grad_accum_steps", 1)
    mixed_precision  = cfg.TRAINING.get("mixed_precision", False)

    # ---- Data ------------------------------------------------------------
    from dataset import get_dataloaders
    loaders      = get_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders["test"]

    train_dataset = train_loader.dataset
    class_names   = cfg.DATASET["class_names"]
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- AMP scaler ------------------------------------------------------
    use_amp = mixed_precision and device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print(f"Mixed precision (AMP float16) enabled — grad_accum_steps={grad_accum_steps}")
    else:
        print(f"Mixed precision disabled (device={device})")

    # ---- Model -----------------------------------------------------------
    from models import get_model
    model = get_model(args.model, cfg)

    # ---- Loss + optimizer ------------------------------------------------
    from losses import get_loss, get_scheduler
    loss_fn, optimizer_fn = get_loss(args.loss, train_dataset, cfg)

    if optimizer_fn is not None:       # AUCM → PESG
        optimizer = optimizer_fn(model, loss_fn)
    else:                              # focal → AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.TRAINING["learning_rate"],
            weight_decay=cfg.TRAINING["weight_decay"],
        )

    scheduler = get_scheduler(optimizer, cfg)

    # ---- Checkpoint dir --------------------------------------------------
    ckpt_dir = Path(cfg.PATHS["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume ----------------------------------------------------------
    start_epoch = 1
    best_auc    = 0.0
    if args.resume:
        start_epoch, best_auc = load_checkpoint(args.resume, model, optimizer, scheduler)

    # ---- TensorBoard -----------------------------------------------------
    writer = None
    if cfg.LOGGING.get("use_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = Path(cfg.PATHS["logs"]) / cfg.LOGGING["experiment_name"] / args.model
            writer  = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard logs → {log_dir}")
        except ImportError:
            warnings.warn("tensorboard not installed; skipping SummaryWriter.")

    # ---- Backbone warmup -------------------------------------------------
    freeze_epochs   = args.freeze_epochs
    backbone_frozen = False
    if freeze_epochs > 0 and hasattr(model, "freeze_backbone"):
        model.freeze_backbone(True)
        backbone_frozen = True
        print(f"Backbone frozen for first {freeze_epochs} epoch(s).")

    # ---- Early stopping --------------------------------------------------
    early_stopper = EarlyStopping(
        patience=cfg.TRAINING.get("early_stopping_patience", 10),
        mode="max",
    )

    num_epochs = cfg.TRAINING["num_epochs"]
    best_epoch = start_epoch

    eff_batch = cfg.TRAINING["batch_size"] * grad_accum_steps
    print(
        f"\nStarting training — model={args.model}, loss={args.loss}, "
        f"epochs={num_epochs}, device={device}, "
        f"batch={cfg.TRAINING['batch_size']}×{grad_accum_steps}={eff_batch} (effective)\n"
    )

    # fast-dev-run batch limits
    train_max = 100 if args.fast_dev_run else None
    val_max   = 50  if args.fast_dev_run else None

    # ---- Training loop ---------------------------------------------------
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start = time.time()

        # Unfreeze backbone after warmup
        if backbone_frozen and epoch == freeze_epochs + 1:
            model.freeze_backbone(False)
            backbone_frozen = False
            print(f"Backbone unfrozen at epoch {epoch}.")

        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, writer,
            scaler=scaler,
            grad_accum_steps=grad_accum_steps,
            max_batches=train_max,
        )
        val_metrics = evaluate(
            model, val_loader, loss_fn, device, class_names,
            max_batches=val_max,
        )
        val_loss = val_metrics["loss"]
        val_auc  = val_metrics["mean_auc"]

        scheduler.step()

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("Loss/train",   train_loss, epoch)
            writer.add_scalar("Loss/val",     val_loss,   epoch)
            writer.add_scalar("AUC/val_mean", val_auc,    epoch)
            for name, auc in val_metrics["per_class_auc"].items():
                writer.add_scalar(f"AUC_per_class/{name}", auc, epoch)

        current_lr = optimizer.param_groups[0]["lr"]

        # GPU memory stats
        if device.type == "cuda":
            mem_alloc    = torch.cuda.memory_allocated(0) / 1e9
            mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            gpu_str      = f" | GPU mem: {mem_alloc:.1f}/{mem_reserved:.1f}GB"
        else:
            gpu_str = ""

        # Epoch timing
        elapsed  = time.time() - epoch_start
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins}m {secs}s"

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.1e}"
            f"{gpu_str} | "
            f"Time: {time_str}"
        )

        # Save latest checkpoint every epoch
        ckpt_state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_auc":             best_auc,
            "args":                 vars(args),
        }
        save_checkpoint(ckpt_state, ckpt_dir / "latest.pth")

        # Save best checkpoint
        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch
            save_checkpoint(ckpt_state | {"best_auc": best_auc}, ckpt_dir / "best.pth")
            print(f"  → New best AUC: {best_auc:.4f}  (epoch {epoch})")

        if early_stopper(val_auc):
            print(
                f"\nEarly stopping triggered after epoch {epoch}. "
                f"Best epoch: {best_epoch} | best val AUC: {best_auc:.4f}"
            )
            break

    print(f"\nTraining complete. Best val AUC: {best_auc:.4f} at epoch {best_epoch}.")

    # ---- Test evaluation using best checkpoint ---------------------------
    if device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        best_ckpt = ckpt_dir / "best.pth"
        if best_ckpt.exists():
            print(f"\nLoading best checkpoint for test evaluation: {best_ckpt}")
            load_checkpoint(str(best_ckpt), model, optimizer, scheduler)

        test_metrics = evaluate(model, test_loader, loss_fn, device, class_names,
                                max_batches=val_max)

        print("\n" + "=" * 60)
        print("Final Test Results")
        print("=" * 60)
        print(f"  Mean AUC : {test_metrics['mean_auc']:.4f}")
        print(f"  Loss     : {test_metrics['loss']:.4f}")
        print()

        for name, auc in sorted(test_metrics["per_class_auc"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {name:<22} {auc:>6.4f}")

        if writer is not None:
            writer.add_scalar("AUC/test_mean", test_metrics["mean_auc"], num_epochs)
            for name, auc in test_metrics["per_class_auc"].items():
                writer.add_scalar(f"AUC_test_per_class/{name}", auc, num_epochs)
            writer.close()

    except Exception as e:
        print(f"\n[Warning] Test evaluation failed: {e}")
        print("This is non-critical in --fast-dev-run mode. Run evaluate.py for full test eval.")
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
