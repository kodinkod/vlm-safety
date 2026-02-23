"""CLI: Evaluate VLM responses to adversarial images."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vlm_safety.evaluate import evaluate_batch
from vlm_safety.remote_vlm import RemoteVlm, RemoteVlmConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VLM responses to adversarial images")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to adversarial images")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send with images")
    parser.add_argument("--model", type=str, default="gpt-4o", help="VLM model name")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    vlm = RemoteVlm(RemoteVlmConfig(model=args.model))

    results_data = []
    for img_path in args.images:
        logging.info(f"Querying VLM with {img_path}")
        response = vlm.query(img_path, args.prompt)
        results_data.append({
            "image_path": img_path,
            "prompt": args.prompt,
            "response": response,
        })

    results, asr = evaluate_batch(results_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "asr": asr,
                "results": [
                    {
                        "image_path": r.image_path,
                        "prompt": r.prompt,
                        "response": r.response,
                        "is_refusal": r.is_refusal,
                        "matched_categories": r.matched_categories,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"ASR: {asr:.2%}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
