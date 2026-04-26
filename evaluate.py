"""
evaluate.py — Standalone evaluation script for NIH ChestX-ray14.

Loads a trained checkpoint, runs inference on val or test split, and produces
a per-class metrics report (AUC-ROC, F1, precision, recall) with optional
JSON export.

Usage:
    python evaluate.py --model densenet --checkpoint checkpoints/best.pth
    python evaluate.py --model swin --checkpoint checkpoints/best.pth --split val --save-report
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

import config as cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIH ChestX-ray14 — checkpoint evaluation"
    )
    p.add_argument("--model",       choices=["densenet", "swin", "hybrid"], required=True)
    p.add_argument("--checkpoint",  type=str,  required=True,
                   help="Path to .pth checkpoint file")
    p.add_argument("--split",       choices=["val", "test"], default="test")
    p.add_argument("--loss",        choices=["aucm", "focal"], default="aucm",
                   help="Loss used during training (needed for eval loss computation)")
    p.add_argument("--output-dir",  type=str,  default="results/",
                   help="Directory for saved report files")
    p.add_argument("--save-report", action="store_true",
                   help="Save per-class metrics as a JSON report")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run model in eval mode and collect predictions.

    Returns:
        all_labels    : (N, C) numpy float32 array of ground-truth labels.
        all_probs     : (N, C) numpy float32 array of sigmoid probabilities.
        all_filenames : list of N filename strings.
    """
    model.eval()
    label_list: list[torch.Tensor] = []
    prob_list:  list[torch.Tensor] = []
    filename_list: list[str] = []

    with torch.no_grad():
        for images, labels, filenames in tqdm(loader, desc="Inference", leave=True):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs  = torch.sigmoid(logits).cpu()

            prob_list.append(probs)
            label_list.append(labels.cpu())
            filename_list.extend(filenames)

    all_labels = torch.cat(label_list, dim=0).numpy().astype(np.float32)
    all_probs  = torch.cat(prob_list,  dim=0).numpy().astype(np.float32)
    return all_labels, all_probs, filename_list


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    all_labels: np.ndarray,
    all_probs:  np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute per-class AUC-ROC, F1, precision, recall, and optimal threshold.

    Optimal threshold per class is found by exhaustive search over
    [0.05, 0.10, …, 0.95], maximising F1 on the evaluation data.

    Args:
        all_labels:  (N, C) binary ground-truth array.
        all_probs:   (N, C) predicted probability array.
        class_names: list of C class name strings.

    Returns:
        dict mapping each class name to
            {"auc", "f1", "precision", "recall", "threshold"}
        plus the key "macro_auc" (mean of non-None per-class AUCs).
    """
    thresholds = np.arange(0.05, 1.0, 0.05)
    results: dict[str, dict | float | None] = {}
    valid_aucs: list[float] = []

    for i, name in enumerate(class_names):
        y_true  = all_labels[:, i]
        y_score = all_probs[:, i]

        # AUC-ROC requires at least one positive and one negative sample
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            warnings.warn(
                f"Class '{name}' has no minority-class samples in this split; "
                "AUC and threshold-based metrics skipped."
            )
            results[name] = {
                "auc":       None,
                "f1":        None,
                "precision": None,
                "recall":    None,
                "threshold": None,
            }
            continue

        auc = float(roc_auc_score(y_true, y_score))
        valid_aucs.append(auc)

        # Find threshold that maximises F1
        best_f1, best_thresh = -1.0, 0.5
        best_prec, best_rec  = 0.0, 0.0

        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(int)
            # zero_division=0 avoids warnings when predictions are all-negative
            f1  = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1     = f1
                best_thresh = float(thresh)
                best_prec   = float(precision_score(y_true, y_pred, zero_division=0))
                best_rec    = float(recall_score(y_true, y_pred, zero_division=0))

        results[name] = {
            "auc":       auc,
            "f1":        best_f1,
            "precision": best_prec,
            "recall":    best_rec,
            "threshold": best_thresh,
        }

    results["macro_auc"] = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(metrics: dict, class_names: list[str]) -> None:
    """Print a fixed-width per-class metrics table sorted by AUC descending."""

    def _fmt(val, fmt=".4f") -> str:
        return f"{val:{fmt}}" if val is not None else "N/A"

    # Sort by AUC descending; classes with None AUC go to the bottom
    sorted_names = sorted(
        class_names,
        key=lambda n: metrics[n]["auc"] if metrics[n]["auc"] is not None else -1.0,
        reverse=True,
    )

    col_w = 24
    header = (
        f"  {'Disease':<{col_w}} {'AUC':>6}  {'F1':>6}  "
        f"{'Prec':>6}  {'Recall':>6}  {'Thresh':>6}"
    )
    sep = "  " + "-" * (col_w + 44)

    print()
    print("=" * 70)
    print("  Per-Class Metrics Report")
    print("=" * 70)
    print(header)
    print(sep)

    for name in sorted_names:
        m = metrics[name]
        print(
            f"  {name:<{col_w}} "
            f"{_fmt(m['auc']):>6}  "
            f"{_fmt(m['f1']):>6}  "
            f"{_fmt(m['precision']):>6}  "
            f"{_fmt(m['recall']):>6}  "
            f"{_fmt(m['threshold']):>6}"
        )

    print(sep)
    print(f"  {'Macro AUC':<{col_w}} {_fmt(metrics['macro_auc']):>6}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Report saving
# ---------------------------------------------------------------------------

def save_report(metrics: dict, args: argparse.Namespace, output_dir: str) -> Path:
    """Save metrics dict as a JSON file.

    Filename: {output_dir}/{model}_{split}_{timestamp}.json

    Returns:
        Path to the saved file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"{args.model}_{args.split}_{timestamp}.json"
    filepath  = out_dir / filename

    report = {
        "meta": {
            "model":      args.model,
            "checkpoint": args.checkpoint,
            "split":      args.split,
            "loss":       args.loss,
            "timestamp":  timestamp,
        },
        "macro_auc":  metrics["macro_auc"],
        "per_class":  {
            name: metrics[name]
            for name in metrics
            if name != "macro_auc"
        },
    }

    with open(filepath, "w") as fh:
        json.dump(report, fh, indent=2)

    return filepath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.set_seed(cfg.TRAINING["seed"])

    # ---- Build dataloader for the requested split only -------------------
    from dataset import NIHChestDataset, get_transforms
    from torch.utils.data import DataLoader

    ds = NIHChestDataset(
        split=args.split,
        transform=get_transforms(args.split, cfg),
        cfg=cfg,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.TRAINING["batch_size"],
        shuffle=False,
        num_workers=cfg.DATASET["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    class_names = cfg.DATASET["class_names"]
    print(f"Split '{args.split}': {len(ds):,} images, {len(loader)} batches")

    # ---- Load model ------------------------------------------------------
    from models import get_model
    model = get_model(args.model, cfg)
    model = model.to(device)

    # ---- Load checkpoint (weights only — no optimizer state needed) ------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    ckpt_epoch   = ckpt.get("epoch",    "?")
    ckpt_best    = ckpt.get("best_auc", None)
    ckpt_info    = f"epoch={ckpt_epoch}"
    if ckpt_best is not None:
        ckpt_info += f", best_val_auc={ckpt_best:.4f}"
    print(f"Loaded checkpoint: {ckpt_path}  ({ckpt_info})")

    # ---- Initialise loss (for eval loss computation) ---------------------
    from losses import get_loss
    loss_fn, _ = get_loss(args.loss, ds, cfg)

    # ---- Inference -------------------------------------------------------
    t_start = time.perf_counter()
    all_labels, all_probs, all_filenames = run_inference(model, loader, device)
    elapsed = time.perf_counter() - t_start

    n_images    = len(all_filenames)
    throughput  = n_images / elapsed if elapsed > 0 else float("inf")
    print(f"\nInference: {n_images:,} images in {elapsed:.1f}s "
          f"({throughput:.1f} img/s)")

    # ---- Metrics ---------------------------------------------------------
    metrics = compute_metrics(all_labels, all_probs, class_names)

    print_report(metrics, class_names)

    # ---- Save report -----------------------------------------------------
    if args.save_report:
        report_path = save_report(metrics, args, args.output_dir)
        print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
