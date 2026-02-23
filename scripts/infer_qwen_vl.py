"""Inference [Secrets122] with image + text prompt to verify attack results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def query_vlm(
    model_name: str,
    image_path: str,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 512,
) -> str:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode only the generated part
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def main():
    parser = argparse.ArgumentParser(description="Query [Secrets123] with image + prompt")
    parser.add_argument("--image", type=str, required=True, help="Image path")
    parser.add_argument("--prompt", type=str, default="What do you see in this image? Describe in detail.", help="Text prompt")
    parser.add_argument("--model", type=str, default="Qwen/[Secrets124]-7B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    response = query_vlm(
        model_name=args.model,
        image_path=args.image,
        prompt=args.prompt,
        device=args.device,
        max_new_tokens=args.max_tokens,
    )

    print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
