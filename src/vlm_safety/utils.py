"""Image helpers, logging, config loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AttackConfig(BaseModel):
    clip_model: str = "ViT-L-14-336"
    clip_pretrained: str = "openai"
    clip_checkpoint: str | None = None  # local path to .safetensors / .bin checkpoint
    trigger_type: str = "visual"  # visual | ocr | textual | combined
    target_image: str | None = None
    target_text: str | None = None
    init_image: str | None = None  # path or "noise"
    image_size: int = 336
    epochs: int = 5000
    lr: float = 0.1
    output_dir: str = "outputs/"
    device: str = "auto"  # auto | cuda | mps | cpu


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> AttackConfig:
    """Load YAML config and apply CLI overrides."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return AttackConfig(**data)


def load_image(path: str | Path, size: int = 336) -> Image.Image:
    """Load and resize an image."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.LANCZOS)
    return img


def save_image(tensor: np.ndarray | Any, path: str | Path) -> Path:
    """Save a numpy array or torch tensor as an image."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.dim() == 4:
            tensor = tensor[0]
        # CHW -> HWC
        arr = tensor.permute(1, 2, 0).numpy()
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    else:
        arr = np.clip(np.array(tensor) * 255, 0, 255).astype(np.uint8)

    Image.fromarray(arr).save(path)
    logger.info(f"Saved image to {path}")
    return path


def plot_loss_curve(losses: list[float], path: str | Path) -> Path:
    """Plot and save loss curve."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (cosine distance)")
    plt.title("CLIP Optimization Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved loss curve to {path}")
    return path


def create_white_image(size: int = 336) -> Image.Image:
    """Create a white image."""
    return Image.fromarray(np.full((size, size, 3), 255, dtype=np.uint8))
