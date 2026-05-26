# -*- coding: utf-8 -*-
"""Chinese BERT text-risk prediction entry point."""
from __future__ import annotations

try:
    from .ensemble_utils import ensemble_inference
except Exception:
    from ChineseBERTModel.ensemble_utils import ensemble_inference


def predict(text: str):
    """Return the current Chinese BERT phishing-risk result for a single text."""
    return ensemble_inference(text)


if __name__ == "__main__":
    sample_text = "您好，请马上把验证码发给我。"
    print(predict(sample_text))
