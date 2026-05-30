#!/usr/bin/env python3
"""Replay baseline dynamic predictions with shared gated_v1 scoring."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

DEFAULT_INPUT = PROJECT_ROOT / ".cache/evaluation_reports/dynamic_predictions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / ".cache/evaluation_reports/dynamic_predictions_gated_v1.json"

from streaming_analysis.risk_scoring import (  # noqa: E402
    RiskScoringState,
    final_label_from_score,
    safe_float,
    score_window,
)


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        records = data.get("records", [])
    else:
        records = data
    if not isinstance(records, list):
        raise ValueError("Input predictions must be a list or an object with records.")
    return records


def apply_gated_record(record: Dict[str, Any]) -> Dict[str, Any]:
    output = copy.deepcopy(record)
    state = RiskScoringState()
    gated_timeline = []

    for point in output.get("timeline", []):
        gated_point = copy.deepcopy(point)
        scoring = score_window(
            raw_text_score=point.get("raw_text_score", point.get("text_score")),
            voice_score=point.get("voice_score"),
            text=str(point.get("text", "")),
            state=state,
            scoring_mode="gated_v1",
            case_type=str(output.get("case_type", "")),
        )
        gated_point.update(scoring)
        gated_timeline.append(gated_point)

    highest = max(gated_timeline, key=lambda item: safe_float(item.get("smoothed_score")), default=None)
    final_score = safe_float(gated_timeline[-1].get("smoothed_score")) if gated_timeline else 0.0
    max_score = safe_float(highest.get("smoothed_score")) if highest else 0.0
    output["timeline"] = gated_timeline
    output["final_score"] = round(final_score, 2)
    output["max_score"] = round(max_score, 2)
    output["highest_risk_window"] = highest
    output["final_label"] = final_label_from_score(final_score)
    output["scoring_mode"] = "gated_v1"
    return output


def write_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    records = load_records(args.input.resolve())
    gated = [apply_gated_record(record) for record in records]
    write_records(args.output.resolve(), gated)
    print(f"input={args.input.resolve()}")
    print(f"output={args.output.resolve()}")
    print(f"records={len(gated)}")


if __name__ == "__main__":
    main()
