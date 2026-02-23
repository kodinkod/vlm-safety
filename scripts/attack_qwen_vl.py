"""Direct Qwen2.5-VL vision encoder attack.

Optimizes adversarial image to match target image features in Qwen2.5-VL's
own vision encoder space (not CLIP).

Key insight: Qwen2.5-VL pixel_values are pre-patchified [num_patches, 1176],
not standard [B,C,H,W]. We replicate the preprocessing as differentiable
tensor ops so gradients flow back to the optimized image.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Qwen2.5-VL uses CLIP normalization
IMAGE_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
IMAGE_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2


def patchify_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert [1, 3, H, W] image tensor to Qwen2.5-VL patch format.

    Replicates the image_processor logic as differentiable tensor ops.
    H, W must be divisible by PATCH_SIZE * MERGE_SIZE (=28).

    Returns:
        pixel_values: [num_patches, channel * temporal_patch_size * patch_size^2]
        image_grid_thw: [1, 3] tensor with (grid_t, grid_h, grid_w)
    """
    _, c, h, w = x.shape
    assert h % (PATCH_SIZE * MERGE_SIZE) == 0 and w % (PATCH_SIZE * MERGE_SIZE) == 0, (
        f"Image size {h}x{w} must be divisible by {PATCH_SIZE * MERGE_SIZE}"
    )

    grid_h = h // PATCH_SIZE
    grid_w = w // PATCH_SIZE
    grid_t = 1  # single image, not video

    # Normalize: (x - mean) / std
    mean = IMAGE_MEAN.to(x.device, x.dtype).view(1, 3, 1, 1)
    std = IMAGE_STD.to(x.device, x.dtype).view(1, 3, 1, 1)
    x_norm = (x - mean) / std

    # For single image: temporal_patch_size=2, so we duplicate the frame
    # patches shape: [temporal_patch_size, C, H, W]
    patches = x_norm.squeeze(0).unsqueeze(0).expand(TEMPORAL_PATCH_SIZE, -1, -1, -1)

    # Reshape to extract patches:
    # [t, c, grid_h, patch_size, grid_w, patch_size]
    patches = patches.reshape(
        grid_t, TEMPORAL_PATCH_SIZE,
        c,
        grid_h // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
        grid_w // MERGE_SIZE, MERGE_SIZE, PATCH_SIZE,
    )
    # Permute to match official: transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    # dims: [grid_t, temporal, c, grid_h//m, merge_h, patch_h, grid_w//m, merge_w, patch_w]
    #    →   [grid_t, grid_h//m, grid_w//m, merge_h, merge_w, c, temporal, patch_h, patch_w]
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    # Flatten to: [num_patches, feature_dim]
    num_patches = grid_t * (grid_h // MERGE_SIZE) * (grid_w // MERGE_SIZE) * MERGE_SIZE * MERGE_SIZE
    feature_dim = c * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE
    pixel_values = patches.reshape(num_patches, feature_dim)

    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long, device=x.device)
    return pixel_values, grid_thw


def smart_resize(h: int, w: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 28 * 28 * 1280) -> tuple[int, int]:
    """Qwen2.5-VL smart resize: find target size divisible by factor within pixel budget."""
    if h < factor or w < factor:
        raise ValueError(f"Image too small: {h}x{w}")
    ratio = w / h
    # Find h_bar such that h_bar * w_bar is within [min_pixels, max_pixels]
    h_bar = max(factor, round(((min_pixels * max_pixels) ** 0.5 / ratio) ** 0.5 / factor) * factor)
    w_bar = max(factor, round(h_bar * ratio / factor) * factor)
    # Clamp
    total = h_bar * w_bar
    if total > max_pixels:
        scale = (max_pixels / total) ** 0.5
        h_bar = max(factor, round(h_bar * scale / factor) * factor)
        w_bar = max(factor, round(w_bar * scale / factor) * factor)
    if h_bar * w_bar < min_pixels:
        scale = (min_pixels / (h_bar * w_bar)) ** 0.5
        h_bar = max(factor, round(h_bar * scale / factor) * factor)
        w_bar = max(factor, round(w_bar * scale / factor) * factor)
    return h_bar, w_bar


def load_model(model_name: str, device: str = "cuda"):
    """Load Qwen2.5-VL model, return (vision_encoder, processor)."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    logger.info(f"Loading processor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    logger.info(f"Loading model: {model_name}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Vision encoder: model.model.visual or model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        vision_encoder = model.model.visual
    elif hasattr(model, "visual"):
        vision_encoder = model.visual
    else:
        raise AttributeError(
            f"Cannot find vision encoder. Available attrs: "
            f"{[a for a in dir(model) if not a.startswith('_')]}"
        )

    vision_encoder = vision_encoder.to(device).eval()
    logger.info(f"Vision encoder: {vision_encoder.__class__.__name__} on {device}")

    # Free language model VRAM
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        model.model.language_model.cpu()
        torch.cuda.empty_cache()
        logger.info("Language model offloaded to CPU")

    return vision_encoder, processor, model


def encode_image_pil(image: Image.Image, vision_encoder, processor, device: str = "cuda") -> torch.Tensor:
    """Encode PIL image (no grad). Used for target image."""
    inputs = processor.image_processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device=device, dtype=vision_encoder.dtype)
    image_grid_thw = inputs["image_grid_thw"].to(device)

    with torch.no_grad():
        outputs = vision_encoder(pixel_values, grid_thw=image_grid_thw)
        # outputs.last_hidden_state: [num_patches, hidden_dim] or similar
        # Use pooler_output if available, else mean over patches
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output.mean(dim=0, keepdim=True)
        else:
            feats = outputs.last_hidden_state.mean(dim=0, keepdim=True)
    return feats


def encode_tensor(x: torch.Tensor, vision_encoder) -> torch.Tensor:
    """Encode [1, 3, H, W] tensor through vision encoder (with grad).

    x should be in [0, 1] range.
    """
    pixel_values, grid_thw = patchify_tensor(x)
    pixel_values = pixel_values.to(dtype=vision_encoder.dtype)

    outputs = vision_encoder(pixel_values, grid_thw=grid_thw)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        feats = outputs.pooler_output.mean(dim=0, keepdim=True)
    else:
        feats = outputs.last_hidden_state.mean(dim=0, keepdim=True)
    return feats


def optimize(
    target_feats: torch.Tensor,
    vision_encoder,
    epochs: int = 500,
    lr: float = 0.01,
    image_h: int = 448,
    image_w: int = 448,
    device: str = "cuda",
    init_image: str | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Optimize adversarial image to match target features."""
    # Initialize in logit space (sigmoid maps to [0,1])
    if init_image and init_image != "noise":
        img = Image.open(init_image).convert("RGB").resize((image_w, image_h))
        x = torch.tensor(np.array(img) / 255.0, dtype=torch.float32)
        x = x.permute(2, 0, 1).unsqueeze(0)
        x_adv = torch.logit(x.clamp(1e-4, 1 - 1e-4)).detach().to(device).requires_grad_(True)
    else:
        x_adv = (torch.randn(1, 3, image_h, image_w) * 0.1).to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([x_adv], lr=lr)
    losses = []
    target_feats = target_feats.detach()

    logger.info(f"Optimizing: {epochs} steps, lr={lr}, size={image_h}x{image_w}")
    t0 = time.time()

    for step in range(epochs):
        optimizer.zero_grad()

        # Sigmoid → [0, 1] image, then encode through differentiable pipeline
        x_img = torch.sigmoid(x_adv)
        adv_feats = encode_tensor(x_img, vision_encoder)

        loss = F.mse_loss(adv_feats.float(), target_feats.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"Step {step}/{epochs} | loss={loss.item():.6f} | time={elapsed:.1f}s")

    logger.info(f"Done. Final loss={losses[-1]:.6f}, time={time.time() - t0:.1f}s")
    return torch.sigmoid(x_adv).detach(), losses


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL vision encoder adversarial attack")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--target-image", type=str, required=True, help="Harm trigger image")
    parser.add_argument("--init-image", type=str, default="noise", help="Init image or 'noise'")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--image-size", type=int, default=448, help="Image size (must be divisible by 28)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs/qwen_attack")
    args = parser.parse_args()

    # Ensure size is divisible by 28
    size = args.image_size
    factor = PATCH_SIZE * MERGE_SIZE  # 28
    if size % factor != 0:
        size = (size // factor) * factor
        logger.warning(f"Image size adjusted to {size} (must be divisible by {factor})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    vision_encoder, processor, model = load_model(args.model, args.device)

    # Encode target
    target_image = Image.open(args.target_image).convert("RGB")
    logger.info(f"Encoding target: {args.target_image}")
    target_feats = encode_image_pil(target_image, vision_encoder, processor, args.device)
    logger.info(f"Target features: {target_feats.shape}")

    # Optimize
    adv_tensor, losses = optimize(
        target_feats, vision_encoder,
        epochs=args.epochs, lr=args.lr,
        image_h=size, image_w=size, device=args.device,
        init_image=args.init_image,
    )

    # Save results
    adv_np = adv_tensor[0].cpu().permute(1, 2, 0).numpy()
    adv_img = Image.fromarray((adv_np * 255).astype(np.uint8))
    adv_img.save(out_dir / "adversarial.png")

    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(target_image)
    axes[0].set_title("Target (harm trigger)")
    axes[0].axis("off")
    axes[1].imshow(adv_np)
    axes[1].set_title(f"Adversarial (loss={losses[-1]:.4f})")
    axes[1].axis("off")
    axes[2].plot(losses)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("MSE Loss")
    axes[2].set_title("Loss Curve")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "result.png", dpi=150)
    plt.close()

    logger.info(f"Saved: {out_dir}/adversarial.png, {out_dir}/result.png")


if __name__ == "__main__":
    main()
