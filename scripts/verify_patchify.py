"""Verify that patchify_tensor produces the same output as the official processor."""

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attack_qwen_vl import patchify_tensor, IMAGE_MEAN, IMAGE_STD, PATCH_SIZE, MERGE_SIZE

def main():
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Create test image 448x448
    img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))

    # Path 1: official processor
    official = processor.image_processor(images=[img], return_tensors="pt")
    pv_official = official["pixel_values"]
    thw_official = official["image_grid_thw"]
    print(f"Official pixel_values: {pv_official.shape}, dtype={pv_official.dtype}")
    print(f"Official grid_thw: {thw_official}")

    # Path 2: our patchify_tensor
    x = torch.tensor(np.array(img) / 255.0, dtype=torch.float32)
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 448, 448]
    pv_ours, thw_ours = patchify_tensor(x)
    print(f"\nOur pixel_values: {pv_ours.shape}, dtype={pv_ours.dtype}")
    print(f"Our grid_thw: {thw_ours}")

    # Compare
    print(f"\nShapes match: {pv_official.shape == pv_ours.shape}")
    print(f"Grid THW match: {torch.equal(thw_official, thw_ours)}")

    if pv_official.shape == pv_ours.shape:
        diff = (pv_official.float() - pv_ours.float()).abs()
        print(f"Max abs diff: {diff.max().item():.6f}")
        print(f"Mean abs diff: {diff.mean().item():.6f}")
        print(f"Values close (atol=1e-3): {torch.allclose(pv_official.float(), pv_ours.float(), atol=1e-3)}")

        # Show first few values
        print(f"\nOfficial[0,:5]: {pv_official[0,:5]}")
        print(f"Ours[0,:5]:     {pv_ours[0,:5]}")
    else:
        print("SHAPE MISMATCH — need to fix patchify_tensor")
        print(f"Official: {pv_official.shape}")
        print(f"Ours: {pv_ours.shape}")

if __name__ == "__main__":
    main()
