#!/usr/bin/env python3
"""Run the corrected-180 window/step sensitivity experiment.

The experiment keeps the runtime baseline fixed (text weight 0.8, voice
weight 0.2, previous-score smoothing weight 0.65, alert threshold 70) and
changes only the streaming window and step.  By default this script prints
the commands; pass ``--run`` to execute them.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "test_samples" / "metadata_all_corrected.csv"
GENERATE_SCRIPT = PROJECT_ROOT / "evaluation" / "generate_dynamic_predictions.py"
METRICS_SCRIPT = PROJECT_ROOT / "evaluation" / "dynamic_metrics.py"
DEFAULT_CONFIGS = ((5.0, 2.5), (10.0, 5.0), (20.0, 10.0))
EXPECTED_SAMPLES = 180


@dataclass(frozen=True)
class WindowStepConfig:
    window_seconds: float
    step_seconds: float

    @property
    def run_name(self) -> str:
        return (
            "defense_all180_baseline_"
            f"w{format_number(self.window_seconds)}_s{format_number(self.step_seconds)}"
        )


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_config(value: str) -> WindowStepConfig:
    try:
        window_text, step_text = value.split(",", maxsplit=1)
        config = WindowStepConfig(float(window_text), float(step_text))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid configuration {value!r}; expected WINDOW,STEP (for example 5,2.5)."
        ) from exc
    if config.window_seconds <= 0 or config.step_seconds <= 0:
        raise argparse.ArgumentTypeError("Window and step must both be positive.")
    if config.step_seconds > config.window_seconds:
        raise argparse.ArgumentTypeError("Step must not be greater than window.")
    return config


def validate_metadata(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_SAMPLES:
        raise SystemExit(f"Expected {EXPECTED_SAMPLES} metadata rows, found {len(rows)}: {path}")

    required = {"sample_id", "audio_path", "label", "case_type"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise SystemExit(f"Metadata is missing required columns: {sorted(missing)}")

    invalid_labels = sorted({str(row["label"]).strip() for row in rows} - {"normal", "fraud"})
    if invalid_labels:
        raise SystemExit(f"Unexpected labels in corrected metadata: {invalid_labels}")


def prediction_command(config: WindowStepConfig, metadata: Path, resume: bool) -> list[str]:
    command = [
        sys.executable,
        GENERATE_SCRIPT.as_posix(),
        "--metadata",
        metadata.as_posix(),
        "--run-name",
        config.run_name,
        "--scoring-mode",
        "baseline",
        "--window-seconds",
        f"{config.window_seconds:g}",
        "--step-seconds",
        f"{config.step_seconds:g}",
        "--text-weight",
        "0.8",
        "--smoothing-previous-weight",
        "0.65",
    ]
    if resume:
        command.append("--resume")
    return command


def metrics_command(config: WindowStepConfig) -> list[str]:
    predictions = (
        PROJECT_ROOT / "evaluation" / "predictions" / config.run_name / "dynamic_predictions.json"
    )
    return [
        sys.executable,
        METRICS_SCRIPT.as_posix(),
        "--predictions",
        predictions.as_posix(),
        "--run-name",
        config.run_name,
        "--alert-threshold",
        "70",
    ]


def display_command(command: Sequence[str]) -> str:
    return " ".join(command)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--config",
        action="append",
        type=parse_config,
        help="WINDOW,STEP pair; repeatable. Defaults to 5,2.5; 10,5; and 20,10.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume incomplete run directories.")
    parser.add_argument("--run", action="store_true", help="Execute instead of only printing commands.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "dissertation":
        raise SystemExit("Activate the project environment first: conda activate dissertation")

    metadata = args.metadata.resolve()
    validate_metadata(metadata)
    configs = args.config or [WindowStepConfig(*values) for values in DEFAULT_CONFIGS]

    for config in configs:
        commands = (prediction_command(config, metadata, args.resume), metrics_command(config))
        for command in commands:
            print(display_command(command), flush=True)
            if args.run:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
