"""Generate OCR trigger images: render text onto images for use as CLIP optimization targets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render_text_image(
    text: str,
    size: int = 336,
    font_size: int = 14,
    padding: int = 12,
    bg_color: str = "white",
    bg_image: str | Path | Image.Image | None = None,
    text_color: str = "black",
    font_path: str | None = None,
) -> Image.Image:
    """Render text onto an image, wrapping lines to fit.

    Args:
        bg_image: Background image (path or PIL Image). Resized to fit.
                  Overrides bg_color when set.
    """
    if bg_image is not None:
        if isinstance(bg_image, (str, Path)):
            bg_image = Image.open(bg_image).convert("RGB")
        img = bg_image.resize((size, size), Image.LANCZOS)
    else:
        img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font = ImageFont.truetype("Courier", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Courier.dfont", font_size)
            except OSError:
                font = ImageFont.load_default(size=font_size)

    # Wrap text to fit within image width
    max_width = size - 2 * padding
    lines = []
    for raw_line in text.split("\n"):
        if not raw_line:
            lines.append("")
            continue
        # Word-wrap long lines
        current = ""
        for char in raw_line:
            test = current + char
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] > max_width:
                lines.append(current)
                current = char
            else:
                current = test
        if current:
            lines.append(current)

    # Draw lines
    y = padding
    line_height = font_size + 4
    for line in lines:
        if y + line_height > size - padding:
            # Draw ellipsis if text overflows
            draw.text((padding, y), "...", fill=text_color, font=font)
            break
        draw.text((padding, y), line, fill=text_color, font=font)
        y += line_height

    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OCR trigger images with rendered text")
    parser.add_argument("--text", type=str, default=None, help="Text to render (inline)")
    parser.add_argument("--file", type=str, default=None, help="Read text from file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output image path")
    parser.add_argument("--size", type=int, default=336, help="Image size (default: 336)")
    parser.add_argument("--font-size", type=int, default=14, help="Font size (default: 14)")
    parser.add_argument("--font", type=str, default=None, help="Path to .ttf font file")
    parser.add_argument("--bg", type=str, default="white", help="Background color")
    parser.add_argument("--bg-image", type=str, default=None, help="Background image path (overrides --bg)")
    parser.add_argument("--fg", type=str, default="black", help="Text color")
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text()
    elif args.text:
        text = args.text.replace("\\n", "\n").replace("\\t", "\t")
    else:
        print("Reading from stdin (Ctrl+D to finish)...")
        text = sys.stdin.read()

    if not text.strip():
        print("Error: no text provided", file=sys.stderr)
        sys.exit(1)

    img = render_text_image(
        text=text,
        size=args.size,
        font_size=args.font_size,
        font_path=args.font,
        bg_color=args.bg,
        bg_image=args.bg_image,
        text_color=args.fg,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"Saved {out} ({args.size}x{args.size}, font={args.font_size}px)")


if __name__ == "__main__":
    main()
