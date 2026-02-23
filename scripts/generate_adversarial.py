"""CLI: Generate adversarial images via CLIP embedding optimization."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vlm_safety.clip_attack import ClipAttack
from vlm_safety.utils import AttackConfig, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial images via CLIP optimization")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--target-image", type=str, default=None, help="Path to target image")
    parser.add_argument("--target-text", type=str, default=None, help="Target text for textual/OCR trigger")
    parser.add_argument("--init-image", type=str, default=None, help="Init image path or 'noise'")
    parser.add_argument("--trigger-type", type=str, default=None, choices=["visual", "textual", "ocr", "combined"])
    parser.add_argument("--clip-model", type=str, default=None, help="CLIP model name")
    parser.add_argument("--clip-pretrained", type=str, default=None, help="CLIP pretrained weights")
    parser.add_argument("--clip-checkpoint", type=str, default=None, help="Local path to CLIP checkpoint (.safetensors/.bin)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tag", type=str, default="", help="Tag for output filenames")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    overrides = {
        "target_image": args.target_image,
        "target_text": args.target_text,
        "init_image": args.init_image,
        "trigger_type": args.trigger_type,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "clip_checkpoint": args.clip_checkpoint,
        "epochs": args.epochs,
        "lr": args.lr,
        "output_dir": args.output_dir,
        "device": args.device,
    }

    if args.config:
        config = load_config(args.config, overrides)
    else:
        config = AttackConfig(**{k: v for k, v in overrides.items() if v is not None})

    attack = ClipAttack(config)
    adv_tensor = attack.optimize()
    paths = attack.save_result(adv_tensor, tag=args.tag)

    print(f"\nDone! Output files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
