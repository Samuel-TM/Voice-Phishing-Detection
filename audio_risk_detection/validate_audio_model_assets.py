#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_DIR / "audio_risk_detection" / "model"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close_enough(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except Exception:
        return False


def validate_assets(model_dir: Path) -> List[str]:
    blockers: List[str] = []
    model_path = model_dir / "best_f1_model.pt"
    config_path = model_dir / "audio_risk_config.json"
    meta_path = model_dir / "training_meta.json"

    for path in (model_path, config_path, meta_path):
        if not path.exists():
            blockers.append(f"Missing required asset: {path}")

    if blockers:
        return blockers

    config = load_json(config_path)
    meta = load_json(meta_path)
    decision_params = config.get("decision_params") or {}
    threshold = decision_params.get("decision_threshold")

    if threshold is None:
        blockers.append("audio_risk_config.json has no decision_params.decision_threshold.")
    if meta.get("deployment_ready") is not True:
        blockers.append("training_meta.json does not mark this model as deployment_ready=true.")
    if meta.get("deployment_blockers"):
        blockers.extend(f"Kaggle blocker: {item}" for item in meta["deployment_blockers"])
    if threshold is not None and not close_enough(threshold, meta.get("decision_threshold")):
        blockers.append("decision_threshold differs between audio_risk_config.json and training_meta.json.")

    required_meta_keys = [
        "balanced_validation_metrics",
        "full_combined_validation_metrics",
        "in_the_wild_ood_metrics",
        "selected_threshold_candidates",
    ]
    for key in required_meta_keys:
        if key not in meta:
            blockers.append(f"training_meta.json is missing {key}.")

    return blockers


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Kaggle-exported audio model assets before local deployment.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    blockers = validate_assets(args.model_dir)
    if blockers:
        print("Audio model assets are NOT deployment-ready:")
        for blocker in blockers:
            print(f"- {blocker}")
        return 2

    print("Audio model assets are deployment-ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
