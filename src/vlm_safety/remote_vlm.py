"""Remote VLM client for evaluating adversarial images."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RemoteVlmConfig(BaseModel):
    model: str = "gpt-4o"
    max_tokens: int = 1024
    temperature: float = 0.0


class RemoteVlm:
    """Send images to a remote VLM API for evaluation."""

    def __init__(self, config: RemoteVlmConfig | None = None):
        self.config = config or RemoteVlmConfig()

    def _encode_image(self, image: str | Path | Image.Image) -> str:
        """Encode image to base64 data URI."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def query(self, image: str | Path | Image.Image, prompt: str) -> str:
        """Send image + prompt to VLM and return response text."""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        data_uri = self._encode_image(image)
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
        response = llm.invoke([message])
        return response.content

    def batch_query(
        self, images: list[str | Path | Image.Image], prompt: str
    ) -> list[str]:
        """Query VLM for multiple images."""
        return [self.query(img, prompt) for img in images]
