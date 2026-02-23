"""Tests for evaluation module."""

from vlm_safety.evaluate import compute_asr, evaluate_response, is_refusal


def test_is_refusal_positive():
    assert is_refusal("I'm sorry, I can't help with that.")
    assert is_refusal("As an AI, I cannot provide this information.")


def test_is_refusal_negative():
    assert not is_refusal("Here is the information you requested.")
    assert not is_refusal("The image shows a landscape.")


def test_evaluate_response():
    result = evaluate_response("img.png", "describe this", "I'm sorry, I cannot help.")
    assert result.is_refusal


def test_compute_asr():
    from vlm_safety.evaluate import EvalResult

    results = [
        EvalResult(image_path="a.png", prompt="p", response="ok", is_refusal=False),
        EvalResult(image_path="b.png", prompt="p", response="sorry", is_refusal=True),
    ]
    assert compute_asr(results) == 0.5
