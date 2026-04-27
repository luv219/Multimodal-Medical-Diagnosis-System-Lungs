"""
dataset.py — NIH ChestX-ray14 dataset for the multi-label classification project.

Images are split across 12 subdirectories (images_001/images/ … images_012/images/).
At startup the dataset scans all directories once and builds a filename→Path lookup
so __getitem__ can locate any image in O(1) regardless of which subfolder it lives in.

Patient-level 80/20 train/val split ensures no patient leaks between sets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

CLASSES: list[str] = config.DATASET["class_names"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Transform factory
# ---------------------------------------------------------------------------

def get_transforms(split: str, cfg=config) -> transforms.Compose:
    """Return the augmentation pipeline for the given split."""
    mean = cfg.MODEL["image_mean"]
    std  = cfg.MODEL["image_std"]

    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # val / test — deterministic centre crop
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _parse_labels(finding_labels: str) -> torch.Tensor:
    """Convert a pipe-separated NIH label string to a 14-d binary float tensor."""
    vec = torch.zeros(len(CLASSES), dtype=torch.float32)
    for label in finding_labels.split("|"):
        label = label.strip()
        if label in CLASS_TO_IDX:
            vec[CLASS_TO_IDX[label]] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Patient-level split helpers
# ---------------------------------------------------------------------------

def _patient_id(filename: str) -> str:
    """'00000001_000.png' → '00000001'  (prefix before first underscore)."""
    return filename.split("_")[0]


def _train_val_split(
    filenames: list[str],
    val_fraction: float = 0.2,
    seed: int = config.TRAINING["seed"],
) -> tuple[list[str], list[str]]:
    """Patient-level 80/20 split — no patient appears in both train and val.

    Patients are sorted then shuffled with a fixed seed for reproducibility.
    """
    patients = sorted(set(_patient_id(f) for f in filenames))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n_val       = max(1, int(len(patients) * val_fraction))
    val_set     = set(patients[:n_val])
    train_set   = set(patients[n_val:])

    return (
        [f for f in filenames if _patient_id(f) in train_set],
        [f for f in filenames if _patient_id(f) in val_set],
    )


# ---------------------------------------------------------------------------
# Image lookup builder
# ---------------------------------------------------------------------------

def _build_image_lookup(images_dirs: list[Path]) -> dict[str, Path]:
    """Scan all images_NNN/images/ subdirectories and return {filename: full_path}.

    Runs once at dataset construction time so __getitem__ is O(1).
    """
    lookup: dict[str, Path] = {}
    for directory in images_dirs:
        if not directory.exists():
            continue
        for img_path in directory.iterdir():
            if img_path.suffix.lower() == ".png":
                lookup[img_path.name] = img_path
    return lookup


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NIHChestDataset(Dataset):
    """NIH ChestX-ray14 multi-label classification dataset.

    Images are spread across up to 12 subdirectories; the constructor scans
    them all once and stores a filename→Path dict (``self.image_lookup``) used
    by ``__getitem__``.

    Each item is a tuple ``(image_tensor, label_tensor, filename)`` where:
      - image_tensor : float32 (3, 224, 224) normalised tensor
      - label_tensor : float32 (14,) multi-hot label vector
      - filename     : original PNG filename string
    """

    def __init__(
        self,
        split: str,
        transform: Optional[transforms.Compose] = None,
        cfg=config,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test'; got '{split}'")

        self.split     = split
        self.transform = transform or transforms.ToTensor()

        # ---- Build filename→path lookup across all 12 image subdirs --------
        self.image_lookup: dict[str, Path] = _build_image_lookup(
            cfg.PATHS["images_dirs"]
        )

        # ---- Load labels ---------------------------------------------------
        df = pd.read_csv(
            cfg.PATHS["labels_csv"],
            usecols=["Image Index", "Finding Labels"],
        )
        label_map: dict[str, torch.Tensor] = {
            row["Image Index"]: _parse_labels(row["Finding Labels"])
            for _, row in df.iterrows()
        }

        # ---- Resolve filenames for the requested split --------------------
        train_val_files = self._read_list(cfg.PATHS["train_list"])
        test_files      = self._read_list(cfg.PATHS["test_list"])

        if split == "test":
            split_files = test_files
        else:
            train_files, val_files = _train_val_split(
                train_val_files,
                val_fraction=0.2,
                seed=cfg.TRAINING["seed"],
            )
            split_files = train_files if split == "train" else val_files

        # Keep only files that have both a label entry and an image on disk
        self.filenames: list[str] = [
            f for f in split_files
            if f in label_map and f in self.image_lookup
        ]
        self.labels: list[torch.Tensor] = [label_map[f] for f in self.filenames]

        # ---- Class weights: neg/pos ratio capped at 50 --------------------
        label_matrix = torch.stack(self.labels)           # (N, 14)
        pos_counts   = label_matrix.sum(dim=0).clamp(min=1.0)
        neg_counts   = len(self.filenames) - pos_counts
        self.class_weights: torch.Tensor = (neg_counts / pos_counts).clamp(max=50.0)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.filenames[idx]
        img_path = self.image_lookup[filename]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, self.labels[idx], filename

    # ------------------------------------------------------------------
    @staticmethod
    def _read_list(path: Path) -> list[str]:
        with open(path, "r") as fh:
            return [line.strip() for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(cfg=config) -> dict[str, DataLoader]:
    """Build and return DataLoaders for train, val, and test splits.

    Returns:
        dict with keys "train", "val", "test".
    """
    pin      = torch.cuda.is_available()
    nworkers = cfg.DATASET.get("num_workers", 0)
    loaders: dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        ds = NIHChestDataset(
            split=split,
            transform=get_transforms(split, cfg),
            cfg=cfg,
        )
        # pin_memory is disabled for test: single sequential pass, compute-bound not
        # transfer-bound, and pin_memory can conflict with CUDA state post-training.
        loader_kwargs: dict = dict(
            batch_size=cfg.TRAINING["batch_size"],
            shuffle=(split == "train"),
            pin_memory=(pin and split != "test"),
        )
        if nworkers > 0:
            # persistent_workers and prefetch_factor are invalid with num_workers=0.
            loader_kwargs.update(
                num_workers=nworkers,
                persistent_workers=True,
                prefetch_factor=2,
            )
        loaders[split] = DataLoader(ds, **loader_kwargs)

    return loaders


# ---------------------------------------------------------------------------
# __main__ — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("NIH ChestX-ray14 — dataset.py smoke test")
    print("=" * 60)

    try:
        # ---- Image lookup --------------------------------------------------
        print("\n[1] Building image lookup...")
        lookup = _build_image_lookup(config.PATHS["images_dirs"])
        print(f"    Found {len(lookup):,} PNG images across "
              f"{sum(1 for d in config.PATHS['images_dirs'] if d.exists())} "
              f"image directories.")

        if not lookup:
            print("\nERROR: No images found. Check NIH_DATA_ROOT and folder structure.")
            print("Expected: data/NIH Dataset/images_001/images/*.png  ..  images_012/images/")
            raise SystemExit(1)

        # ---- Load splits --------------------------------------------------
        print("\n[2] Loading splits...")
        loaders = get_dataloaders()
        for split, loader in loaders.items():
            ds: NIHChestDataset = loader.dataset  # type: ignore[assignment]
            print(f"    {split:5s}: {len(ds):>6,} images")

        # ---- Sample item from train ---------------------------------------
        print("\n[3] Inspecting a training sample...")
        train_ds: NIHChestDataset = loaders["train"].dataset  # type: ignore[assignment]
        img_tensor, label_tensor, fname = train_ds[0]

        print(f"    filename     : {fname}")
        print(f"    image shape  : {tuple(img_tensor.shape)}  dtype={img_tensor.dtype}")
        print(f"    label shape  : {tuple(label_tensor.shape)}  dtype={label_tensor.dtype}")

        active = [CLASSES[i] for i, v in enumerate(label_tensor) if v == 1.0]
        print(f"    active labels: {active if active else ['No Finding']}")
        print(f"    label vector : {label_tensor.tolist()}")

        # ---- Class weight stats ------------------------------------------
        print("\n[4] Class weight statistics (train split):")
        w = train_ds.class_weights
        print(f"    min={w.min():.3f}  max={w.max():.3f}  mean={w.mean():.3f}")
        for name, weight in zip(CLASSES, w.tolist()):
            print(f"    {name:<22} {weight:>6.2f}")

        print("\nDone.")

    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        print(
            "\nMake sure the NIH ChestX-ray14 dataset is in place:\n"
            "  data/NIH Dataset/Data_Entry_2017.csv\n"
            "  data/NIH Dataset/train_val_list.txt\n"
            "  data/NIH Dataset/test_list.txt\n"
            "  data/NIH Dataset/images_001/images/*.png\n"
            "  ...\n"
            "  data/NIH Dataset/images_012/images/*.png\n\n"
            "Or set NIH_DATA_ROOT to point at the correct location."
        )
        raise SystemExit(1)
