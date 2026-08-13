# NIH ChestX-ray14 — Multimodal Lung Disease Classification

Three model architectures — DenseNet-121, Swin Transformer, and a CNN+Transformer Hybrid — trained and compared on the [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) dataset for multi-label classification of 14 thoracic diseases.  All models are optimised with the AUC-margin loss (libauc `AUCM_MultiLabel` + `PESG`) as the primary objective, with sigmoid focal loss available as an ablation baseline.

---

## Project structure

```
.
├── config.py          # Single source of truth: all paths, hyperparameters, class names
├── dataset.py         # NIHChestDataset, patient-level split, DataLoader factory
├── losses.py          # AUCMLoss (AUCM_MultiLabel wrapper), FocalLoss, loss factory
├── train.py           # Full training pipeline: warmup freeze, early stopping, checkpointing
├── evaluate.py        # Standalone eval: per-class AUC-ROC, F1, precision, recall + JSON report
├── explain.py         # GradCAM++ saliency maps for all three architectures
├── dashboard.py       # Multi-model comparison dashboard (heatmaps, bar chart, summary table)
├── requirements.txt   # Python dependencies
└── models/
    ├── __init__.py    # Model registry: get_model() factory
    ├── densenet.py    # DenseNet-121 classifier (Model 1)
    ├── swin.py        # Swin Transformer classifier (Model 2)
    └── hybrid.py      # CNN-Transformer hybrid: ResNet-50 + Transformer encoder (Model 3)
```

---

## Dataset setup

1. **Download** NIH ChestX-ray14 from https://nihcc.app.box.com/v/ChestXray-NIHCC

2. **Required files** — place them all under one directory:
   ```
   <data_root>/
   ├── images/                 # all 112,120 PNG chest X-ray images
   ├── Data_Entry_2017.csv     # image-level labels
   ├── train_val_list.txt      # official train+val image list
   └── test_list.txt           # official test image list
   ```

3. **Point the project at your data directory:**
   ```bash
   export NIH_DATA_ROOT=/path/to/your/nih_data
   ```
   If the variable is not set, the scripts default to `./data/`.

4. **Verify the setup:**
   ```bash
   python config.py
   ```
   This prints all resolved paths and confirms the required files exist.

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `timm`, `libauc`, `pytorch-grad-cam`, `scikit-learn`, `pandas`, `seaborn`, `matplotlib`, `tqdm`.

---

## Training

All commands use AUC-margin loss with PESG optimiser by default.  The backbone is frozen for the first `--freeze-epochs` (default 3) epochs to warm up the new classification head, then unfrozen for end-to-end fine-tuning.

**DenseNet-121**
```bash
python train.py --model densenet --loss aucm
```

**Swin Transformer**
```bash
python train.py --model swin --loss aucm
```

**CNN-Transformer Hybrid**
```bash
python train.py --model hybrid --loss aucm
```

**With CLI overrides**
```bash
python train.py --model densenet --epochs 30 --batch-size 64 --experiment-name my_run
```

**Resume from a checkpoint**
```bash
python train.py --model densenet --resume checkpoints/latest.pth
```

Checkpoints are written to `checkpoints/latest.pth` every epoch and `checkpoints/best.pth` whenever validation macro AUC improves.

---

## Evaluation

```bash
python evaluate.py \
  --model densenet \
  --checkpoint checkpoints/best.pth \
  --split test \
  --save-report
```

Outputs a per-class table (AUC-ROC, F1, precision, recall, optimal threshold) to stdout and, with `--save-report`, saves a JSON file to `results/`.

---

## Explainability

```bash
python explain.py \
  --model densenet \
  --checkpoint checkpoints/best.pth \
  --image path/to/image.png \
  --top-k 3
```

Generates GradCAM++ saliency maps for the top-3 predicted disease classes.  Use `--save-dir results/` to save the three-panel figures (original | heatmap | overlay) as PNGs instead of displaying them interactively.  Pass `--class-idx 0` to explain a specific class regardless of predicted probability.

---

## Comparison dashboard

> **Prerequisite:** run `evaluate.py --save-report` for all three models first so that JSON reports exist in `results/`.

```bash
python dashboard.py --results-dir results/ --output results/dashboard.png
```

Produces a 2×2 panel PNG:
- **Top-left** — AUC-ROC heatmap (14 diseases × 3 models)
- **Top-right** — F1 heatmap (swap metric with `--metric precision` or `--metric recall`)
- **Bottom-left** — Macro AUC bar chart + per-disease AUC line plot
- **Bottom-right** — Summary table with per-disease Best-Model column

---

## Architecture summary

| Model | Backbone | Classification head | GradCAM++ target | Params (approx) |
|---|---|---|---|---|
| **DenseNet-121** | DenseNet-121 (ImageNet) | Linear(1024 → 512) → ReLU → Linear(512 → 14) | `denseblock4` | ~7 M |
| **Swin-T** | Swin-Tiny (ImageNet) | LayerNorm(768) → Linear(768 → 512) → GELU → Linear(512 → 14) | `layers[-1]` | ~28 M |
| **Hybrid** | ResNet-50 (ImageNet) + 2-layer Transformer encoder | LayerNorm(256) → Linear(256 → 14) | `layer4` | ~28 M |

The Hybrid model projects ResNet-50's `(B, 2048, 7, 7)` feature map down to `(B, 256, 7, 7)` via a 1×1 convolution, then flattens to 49 tokens with learnable positional encoding before the Transformer encoder.

---

## Key design decisions

- **Uniform AUC-margin loss** — `AUCM_MultiLabel` from libauc directly optimises the multi-label AUC rather than a surrogate cross-entropy proxy.  Per-class positive rates (`imratio`) are computed from the training split and injected into the loss before training begins.
- **Patient-level train / val split** — images from the same patient never appear in both sets, preventing label leakage.
- **GradCAM++ unified across architectures** — a single `reshape_transform` function converts Transformer sequence outputs `(B, 49, C)` → `(B, C, 7, 7)` for GradCAM++, while leaving 4D CNN outputs unchanged, so all three models share the same saliency pipeline.
- **Threshold tuning per disease** — `evaluate.py` searches thresholds `[0.05, 0.10, …, 0.95]` per class to maximise F1, rather than using a fixed 0.5 cutoff, which is unreliable under class imbalance.
- **`config.py` as single source of truth** — all file paths, hyperparameters, and class names live in one place; every script imports from it so changing a path or learning rate propagates everywhere.

---

## Reproducing result

All scripts call `config.set_seed(config.TRAINING["seed"])` at startup (default seed `42`), which seeds Python `random`, NumPy, and PyTorch (CPU + CUDA with deterministic cuDNN).  The best checkpoint is saved by validation macro AUC to `checkpoints/best.pth` and is used automatically for test evaluation at the end of `train.py`.

To fully reproduce a run from a saved checkpoint:

```bash
python evaluate.py \
  --model densenet \
  --checkpoint checkpoints/best.pth \
  --split test \
  --save-report
```
