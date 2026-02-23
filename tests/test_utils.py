"""Basic tests for vlm_safety."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vlm_safety.utils import AttackConfig, create_white_image, load_config, load_image, save_image


def test_attack_config_defaults():
    cfg = AttackConfig()
    assert cfg.clip_model == "ViT-L-14-336"
    assert cfg.epochs == 5000
    assert cfg.device == "auto"


def test_load_config(tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("epochs: 100\nlr: 0.01\n")
    cfg = load_config(config_file)
    assert cfg.epochs == 100
    assert cfg.lr == 0.01


def test_load_config_with_overrides(tmp_path):
    config_file = tmp_path / "test.yaml"
    config_file.write_text("epochs: 100\n")
    cfg = load_config(config_file, overrides={"epochs": 200})
    assert cfg.epochs == 200


def test_create_white_image():
    img = create_white_image(64)
    assert img.size == (64, 64)
    arr = np.array(img)
    assert arr.shape == (64, 64, 3)
    assert (arr == 255).all()


def test_save_image(tmp_path):
    arr = np.ones((3, 64, 64), dtype=np.float32) * 0.5
    import torch

    tensor = torch.tensor(arr).unsqueeze(0)
    path = save_image(tensor, tmp_path / "test.png")
    assert path.exists()
    img = Image.open(path)
    assert img.size == (64, 64)


def test_load_image(tmp_path):
    img = create_white_image(100)
    img.save(tmp_path / "test.png")
    loaded = load_image(tmp_path / "test.png", size=64)
    assert loaded.size == (64, 64)
