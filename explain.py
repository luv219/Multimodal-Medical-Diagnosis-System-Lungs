"""
explain.py — GradCAM++ saliency maps for NIH ChestX-ray14 models.

Generates per-class GradCAM++ visualisations for a single chest X-ray image
using any of the three model architectures (DenseNet-121, Swin Transformer,
CNN-Transformer Hybrid).

Usage:
    python explain.py --model densenet --checkpoint checkpoints/best.pth --image img.png
    python explain.py --model swin --checkpoint best.pth --image img.png --top-k 5 --save-dir out/
    python explain.py --model hybrid --checkpoint best.pth --image img.png --class-idx 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config as cfg


# ---------------------------------------------------------------------------
# Reshape transform for Transformer intermediate outputs
# ---------------------------------------------------------------------------

def reshape_transform_swin(tensor: torch.Tensor, height: int = 7, width: int = 7) -> torch.Tensor:
    """Reshape a Transformer sequence tensor to spatial (B, C, H, W) format.

    GradCAM++ expects 4-D activations.  Swin Transformer stage outputs have
    shape (B, seq_len, C).  This function converts (B, 49, C) → (B, C, 7, 7).

    If the tensor is already 4-D (e.g. ResNet / DenseNet conv output), it is
    returned unchanged so the same transform can be safely registered for the
    Hybrid model's ResNet layer4 hook.

    Args:
        tensor: Activation tensor from the hook target layer.
        height: Spatial height after reshape (default 7 for 224-px input).
        width:  Spatial width  after reshape (default 7 for 224-px input).

    Returns:
        Tensor of shape (B, C, height, width) or the original 4-D tensor.
    """
    if tensor.ndim == 4:
        return tensor
    # (B, seq_len, C) → (B, C, H, W)
    return tensor.permute(0, 2, 1).reshape(
        tensor.size(0), tensor.size(2), height, width
    )


# ---------------------------------------------------------------------------
# CAM extractor factory
# ---------------------------------------------------------------------------

def get_cam_extractor(model: torch.nn.Module, model_name: str) -> GradCAMPlusPlus:
    """Build a GradCAMPlusPlus extractor for the given model.

    The target layer is obtained via ``model.get_cam_target_layer()``.
    A reshape_transform is supplied for Swin and Hybrid to convert 3-D
    Transformer sequence activations into spatial 4-D tensors.

    Args:
        model:      The model instance (already on the correct device).
        model_name: One of "densenet", "swin", "hybrid" (case-insensitive).

    Returns:
        GradCAMPlusPlus instance to be used as a context manager.
    """
    target_layer = model.get_cam_target_layer()
    key = model_name.lower()

    if key == "swin":
        return GradCAMPlusPlus(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform_swin,
        )

    if key == "hybrid":
        # layer4 is a 4-D conv output; reshape_transform passes it through
        # unchanged, but registering it keeps the code path uniform.
        return GradCAMPlusPlus(
            model=model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform_swin,
        )

    # densenet — denseblock4 output is (B, C, 7, 7), no reshape needed
    return GradCAMPlusPlus(
        model=model,
        target_layers=[target_layer],
    )


# ---------------------------------------------------------------------------
# CAM generation
# ---------------------------------------------------------------------------

def generate_cam(
    model: torch.nn.Module,
    cam_extractor: GradCAMPlusPlus,
    image_tensor: torch.Tensor,
    target_class_idx: int,
) -> np.ndarray:
    """Generate a GradCAM++ saliency map for one class.

    Args:
        model:            The model (eval mode, on device).
        cam_extractor:    Active GradCAMPlusPlus context manager.
        image_tensor:     Shape (1, 3, 224, 224), on the model's device.
        target_class_idx: Class index (0–13) to explain.

    Returns:
        Grayscale CAM array of shape (224, 224), values in [0, 1].
    """
    targets = [ClassifierOutputTarget(target_class_idx)]
    grayscale_cam = cam_extractor(input_tensor=image_tensor, targets=targets)
    return grayscale_cam[0]  # (H, W)


# ---------------------------------------------------------------------------
# Image denormalisation
# ---------------------------------------------------------------------------

def denormalize(
    tensor: torch.Tensor,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """Reverse ImageNet normalisation and return a (H, W, 3) float32 array.

    Args:
        tensor: Normalised image tensor of shape (3, H, W).
        mean:   Per-channel means used during normalisation.
        std:    Per-channel standard deviations used during normalisation.

    Returns:
        Float32 numpy array of shape (H, W, 3) with values clipped to [0, 1].
    """
    mean_arr = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std_arr  = np.array(std,  dtype=np.float32).reshape(3, 1, 1)

    img = tensor.cpu().float().numpy()
    img = img * std_arr + mean_arr
    img = np.clip(img, 0.0, 1.0)
    return img.transpose(1, 2, 0)   # (H, W, 3)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_cam(
    original_image_np: np.ndarray,
    cam_map: np.ndarray,
    class_name: str,
    predicted_prob: float,
    save_path: str | None = None,
) -> None:
    """Create and display / save a three-panel GradCAM++ figure.

    Panels:
        1. Original image
        2. CAM heatmap (jet colormap)
        3. CAM overlay on original image

    Args:
        original_image_np: Float32 (H, W, 3) array in [0, 1].
        cam_map:           Float32 (H, W) array in [0, 1].
        class_name:        Disease class label.
        predicted_prob:    Sigmoid probability for this class.
        save_path:         If provided, save figure to this path and close it.
    """
    overlay = show_cam_on_image(original_image_np, cam_map, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"{class_name} — predicted prob: {predicted_prob:.3f}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].imshow(original_image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(cam_map, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("GradCAM++ heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# High-level explainer
# ---------------------------------------------------------------------------

def explain_sample(
    model: torch.nn.Module,
    model_name: str,
    image_path: str,
    class_names: list[str],
    config,
    top_k: int = 3,
    save_dir: str | None = None,
    target_class_idx: int | None = None,
) -> None:
    """Generate GradCAM++ maps for a single image.

    Loads the image, runs a forward pass to obtain per-class probabilities,
    then produces GradCAM++ visualisations for either the top_k predicted
    classes or a single specified class index.

    Args:
        model:            Loaded model in eval mode on the correct device.
        model_name:       "densenet", "swin", or "hybrid".
        image_path:       Path to the chest X-ray PNG/JPEG.
        class_names:      List of 14 disease class names.
        config:           Project config module (provides MODEL image stats).
        top_k:            Number of highest-probability classes to explain.
        save_dir:         If provided, save figures here instead of displaying.
        target_class_idx: If provided, explain only this class (ignores top_k).
    """
    device = next(model.parameters()).device

    # ---- Load and transform image ----------------------------------------
    from dataset import get_transforms
    transform = get_transforms("test", config)

    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # ---- Forward pass — probabilities ------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs  = torch.sigmoid(logits).squeeze(0).cpu()  # (14,)

    probs_np = probs.numpy()

    # ---- Select classes to explain ---------------------------------------
    if target_class_idx is not None:
        explain_indices = [target_class_idx]
    else:
        # argsort descending, take top_k
        explain_indices = list(np.argsort(probs_np)[::-1][:top_k])

    # ---- Summary ---------------------------------------------------------
    image_stem = Path(image_path).stem
    print(f"\nImage : {Path(image_path).name}")
    print(f"Top predictions:")
    for idx in np.argsort(probs_np)[::-1][:max(top_k, len(explain_indices))]:
        marker = " ←" if idx in explain_indices else ""
        print(f"  [{idx:2d}] {class_names[idx]:<26} {probs_np[idx]:.4f}{marker}")

    # ---- Denormalise image once for all subplots -------------------------
    original_np = denormalize(
        image_tensor.squeeze(0),
        mean=config.MODEL["image_mean"],
        std=config.MODEL["image_std"],
    )

    # ---- GradCAM++ -------------------------------------------------------
    with get_cam_extractor(model, model_name) as cam_extractor:
        for class_idx in explain_indices:
            cam_map  = generate_cam(model, cam_extractor, image_tensor, class_idx)
            class_name = class_names[class_idx]
            prob       = float(probs_np[class_idx])

            save_path = None
            if save_dir is not None:
                safe_name = class_name.replace(" ", "_").replace("/", "-")
                save_path = str(Path(save_dir) / f"{image_stem}_{safe_name}.png")

            visualize_cam(original_np, cam_map, class_name, prob, save_path=save_path)

            if save_path:
                print(f"  Saved: {save_path}")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIH ChestX-ray14 — GradCAM++ explainability"
    )
    p.add_argument("--model",      choices=["densenet", "swin", "hybrid"], required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pth checkpoint file")
    p.add_argument("--image",      type=str, required=True,
                   help="Path to a single chest X-ray image")
    p.add_argument("--top-k",     type=int, default=3,
                   help="Number of top predicted classes to explain (default: 3)")
    p.add_argument("--save-dir",  type=str, default=None,
                   help="Directory to save figures; if omitted, figures are shown interactively")
    p.add_argument("--class-idx", type=int, default=None,
                   help="Explain only this class index (0–13) instead of top-k")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.set_seed(cfg.TRAINING["seed"])

    # ---- Load model ------------------------------------------------------
    from models import get_model
    model = get_model(args.model, cfg)
    model = model.to(device)

    # ---- Load checkpoint (weights only) ----------------------------------
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    ckpt_epoch = ckpt.get("epoch",    "?")
    ckpt_auc   = ckpt.get("best_auc", None)
    info       = f"epoch={ckpt_epoch}"
    if ckpt_auc is not None:
        info += f", best_val_auc={ckpt_auc:.4f}"
    print(f"Loaded checkpoint: {ckpt_path}  ({info})")

    model.eval()

    # ---- Explain ---------------------------------------------------------
    class_names = cfg.DATASET["class_names"]

    explain_sample(
        model=model,
        model_name=args.model,
        image_path=args.image,
        class_names=class_names,
        config=cfg,
        top_k=args.top_k,
        save_dir=args.save_dir,
        target_class_idx=args.class_idx,
    )

    if args.save_dir:
        print(f"\nAll figures saved to: {args.save_dir}")
    else:
        print("\nExplanation complete.")


if __name__ == "__main__":
    main()
