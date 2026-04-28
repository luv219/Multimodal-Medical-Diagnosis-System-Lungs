import os
import random
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Data root ────────────────────────────────────────────────────────────────
# Override by setting NIH_DATA_ROOT environment variable
DATA_ROOT = Path(os.environ.get("NIH_DATA_ROOT", "data/NIH Dataset"))

# ── Paths ─────────────────────────────────────────────────────────────────────
PATHS = {
    "data_root":   DATA_ROOT,
    "images_dirs": [DATA_ROOT / f"images_{i:03d}" / "images" for i in range(1, 13)],
    "labels_csv":  DATA_ROOT / "Data_Entry_2017.csv",
    "train_list":  DATA_ROOT / "train_val_list.txt",
    "test_list":   DATA_ROOT / "test_list.txt",
    "checkpoints": Path("checkpoints"),
    "logs":        Path("logs"),
    "results":     Path("results"),
}

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET = {
    "image_size": 224,
    "num_workers": 0,
    "num_classes": 14,
    "class_names": [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia",
    ],
}

# ── Training ──────────────────────────────────────────────────────────────────
TRAINING = {
    "batch_size":              32,
    "learning_rate":           1e-4,
    "weight_decay":            1e-5,
    "num_epochs":              5,
    "early_stopping_patience": 2,
    "seed":                    42,
    "grad_accum_steps":        4,
    "mixed_precision":         True,
    "num_workers":             0,
}

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL = {
    "image_mean":                [0.485, 0.456, 0.406],
    "image_std":                 [0.229, 0.224, 0.225],
    "dropout_rate":              0.0,
    "swin_model_name":           "swin_tiny_patch4_window7_224",
    "hybrid_transformer_layers": 2,
    "hybrid_transformer_dim":    256,
    "hybrid_num_heads":          8,
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOGGING = {
    "use_tensorboard":  True,
    "use_wandb":        False,
    "experiment_name":  "nih_chestxray_comparison",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_paths() -> bool:
    """Check all required input paths exist. Output dirs are created if missing."""
    required = {
        "labels_csv": PATHS["labels_csv"],
        "train_list": PATHS["train_list"],
        "test_list":  PATHS["test_list"],
    }

    missing = []
    for name, path in required.items():
        if not path.exists():
            missing.append(f"  MISSING  {name}: {path}")

    # Check at least one images_dir exists
    found_images = [p for p in PATHS["images_dirs"] if p.exists()]
    if not found_images:
        missing.append(
            f"  MISSING  images dirs: none of images_001..012/images/ "
            f"found under {PATHS['data_root']}"
        )
    else:
        print(f"  OK       images dirs: {len(found_images)}/12 found")

    if missing:
        print("\n[config] Path validation FAILED:")
        for m in missing:
            print(m)
        print(f"\nHint: set NIH_DATA_ROOT to your dataset folder, e.g.:")
        print(f'  export NIH_DATA_ROOT="/path/to/NIH Dataset"   # Linux/Mac')
        print(f'  set NIH_DATA_ROOT=C:\\path\\to\\NIH Dataset     # Windows cmd')
        return False

    for name, path in required.items():
        print(f"  OK       {name}: {path}")

    # Create output dirs on demand
    for key in ("checkpoints", "logs", "results"):
        PATHS[key].mkdir(parents=True, exist_ok=True)
        print(f"  READY    {key}: {PATHS[key]}")

    print("\n[config] All paths OK.")
    return True


def set_seed(seed: int | None = None) -> None:
    """Seed random, numpy, and torch for reproducibility."""
    if seed is None:
        seed = TRAINING["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # benchmark=True lets cuDNN auto-tune kernels for the input size (faster).
    # deterministic=False is required — the two flags are mutually exclusive.
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG SUMMARY")
    print("=" * 60)
    print(f"DATA_ROOT  : {DATA_ROOT}")
    print(f"DATASET    : {DATASET}")
    print(f"TRAINING   : {TRAINING}")
    print(f"MODEL      : {MODEL}")
    print(f"LOGGING    : {LOGGING}")
    print("=" * 60)
    validate_paths()
