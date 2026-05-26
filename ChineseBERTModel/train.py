# -*- coding: utf-8 -*-
"""Lightweight entry point for the current Chinese BERT training script.

The project keeps the full Kaggle-exported training workflow in
train_chinese_bert_kaggle.py. This file intentionally stays small so the legacy
training implementation is not maintained in two places.
"""
from __future__ import annotations

import runpy
from pathlib import Path


KAGGLE_TRAINING_SCRIPT = Path(__file__).resolve().with_name("train_chinese_bert_kaggle.py")


def main() -> None:
    if not KAGGLE_TRAINING_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {KAGGLE_TRAINING_SCRIPT}")
    runpy.run_path(KAGGLE_TRAINING_SCRIPT.as_posix(), run_name="__main__")


if __name__ == "__main__":
    main()
