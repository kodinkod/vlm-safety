"""CLIP embedding optimization for adversarial image generation.

Based on "Jailbreak in Pieces" (arxiv 2307.14539).
Optimizes an image to match a target CLIP embedding (from text, image, or both).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from vlm_safety.utils import (
    AttackConfig,
    create_white_image,
    load_image,
    plot_loss_curve,
    save_image,
)

logger = logging.getLogger(__name__)


class ClipAttack:
    """CLIP-based adversarial image generator."""

    def __init__(self, config: AttackConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.losses: list[float] = []

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def load_model(self) -> None:
        """Load CLIP model and preprocessing."""
        cfg = self.config

        if cfg.clip_checkpoint:
            # Load from local checkpoint file
            logger.info(f"Loading CLIP model: {cfg.clip_model} from {cfg.clip_checkpoint}")
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg.clip_model,
                pretrained=cfg.clip_checkpoint,
                device=self.device,
            )
        else:
            # Download from HuggingFace
            logger.info(f"Loading CLIP model: {cfg.clip_model} ({cfg.clip_pretrained})")
            model, _, preprocess = open_clip.create_model_and_transforms(
                cfg.clip_model,
                pretrained=cfg.clip_pretrained,
                device=self.device,
            )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model)
        logger.info(f"Model loaded on {self.device}")

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode a PIL image to a normalized CLIP embedding."""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
        return F.normalize(emb, dim=-1)

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to a normalized CLIP embedding."""
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        return F.normalize(emb, dim=-1)

    def compute_target_embedding(self) -> torch.Tensor:
        """Compute the target embedding based on trigger_type."""
        cfg = self.config
        trigger = cfg.trigger_type.lower()

        if trigger == "visual":
            if not cfg.target_image:
                raise ValueError("target_image required for visual trigger")
            img = load_image(cfg.target_image, cfg.image_size)
            return self._encode_image(img)

        elif trigger == "textual":
            if not cfg.target_text:
                raise ValueError("target_text required for textual trigger")
            return self._encode_text(cfg.target_text)

        elif trigger == "ocr":
            if not cfg.target_text:
                raise ValueError("target_text required for OCR trigger")
            return self._encode_text(cfg.target_text)

        elif trigger == "combined":
            embeddings = []
            if cfg.target_image:
                img = load_image(cfg.target_image, cfg.image_size)
                embeddings.append(self._encode_image(img))
            if cfg.target_text:
                embeddings.append(self._encode_text(cfg.target_text))
            if not embeddings:
                raise ValueError("target_image and/or target_text required for combined trigger")
            combined = torch.mean(torch.stack(embeddings), dim=0)
            return F.normalize(combined, dim=-1)

        else:
            raise ValueError(f"Unknown trigger_type: {trigger}")

    def _init_adversarial(self) -> torch.Tensor:
        """Create the initial adversarial image tensor (requires grad)."""
        size = self.config.image_size
        if self.config.init_image and self.config.init_image != "noise":
            img = load_image(self.config.init_image, size)
            tensor = torch.tensor(np.array(img) / 255.0, dtype=torch.float32)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> 1CHW
        else:
            tensor = torch.rand(1, 3, size, size)

        return tensor.to(self.device).requires_grad_(True)

    def optimize(self) -> torch.Tensor:
        """Run the optimization loop. Returns the adversarial image tensor."""
        if self.model is None:
            self.load_model()

        target_emb = self.compute_target_embedding()
        adv = self._init_adversarial()
        optimizer = torch.optim.Adam([adv], lr=self.config.lr)
        self.losses = []

        logger.info(
            f"Starting optimization: {self.config.epochs} epochs, "
            f"lr={self.config.lr}, trigger={self.config.trigger_type}"
        )
        t0 = time.time()

        for epoch in range(self.config.epochs):
            optimizer.zero_grad()

            # Clamp to valid image range
            adv_clamped = adv.clamp(0, 1)

            # Encode current adversarial image
            adv_emb = self.model.encode_image(adv_clamped)
            adv_emb = F.normalize(adv_emb, dim=-1)

            # Cosine distance loss (minimize = maximize similarity)
            loss = 1 - F.cosine_similarity(adv_emb, target_emb).mean()
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())

            if (epoch + 1) % 500 == 0 or epoch == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Time: {elapsed:.1f}s"
                )

        final_loss = self.losses[-1]
        total_time = time.time() - t0
        logger.info(f"Optimization complete. Final loss: {final_loss:.6f}, Time: {total_time:.1f}s")

        return adv.detach().clamp(0, 1)

    def save_result(self, adv_tensor: torch.Tensor, tag: str = "") -> dict[str, Path]:
        """Save adversarial image, loss curve, and config to output_dir."""
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_{tag}" if tag else ""
        paths = {}

        paths["image"] = save_image(adv_tensor, out_dir / f"adversarial{suffix}.png")
        paths["loss_curve"] = plot_loss_curve(self.losses, out_dir / f"loss_curve{suffix}.png")

        loss_path = out_dir / f"losses{suffix}.json"
        with open(loss_path, "w") as f:
            json.dump(self.losses, f)
        paths["losses"] = loss_path

        config_path = out_dir / f"config{suffix}.json"
        with open(config_path, "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)
        paths["config"] = config_path

        logger.info(f"Results saved to {out_dir}")
        return paths
