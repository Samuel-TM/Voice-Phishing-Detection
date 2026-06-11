#!/usr/bin/env python3
"""Run controlled long-sample evaluation batches from test_samples/metadata_long.csv."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "test_samples/metadata_final.csv"
PREDICTION_SCRIPT = PROJECT_ROOT / "evaluation/generate_dynamic_predictions.py"
METRICS_SCRIPT = PROJECT_ROOT / "evaluation/dynamic_metrics.py"
DEFAULT_SCORING_MODE = "baseline"
DEFAULT_WINDOW_SECONDS = 10.0
DEFAULT_STEP_SECONDS = 5.0


@dataclass(frozen=True)
class Batch:
    name: str
    sample_ids: tuple[str, ...]
    description: str


def ids(prefix: str, start: int, end: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_long_{index:02d}" for index in range(start, end + 1))


BATCHES: dict[str, Batch] = {
    "smoke_4": Batch(
        name="smoke_4",
        sample_ids=("ND_long_01", "SF_long_01", "MR_long_01", "SV_long_01"),
        description="One sample per long case type; use this first.",
    ),
    "balanced_8": Batch(
        name="balanced_8",
        sample_ids=(
            "ND_long_01",
            "ND_long_11",
            "SF_long_01",
            "SF_long_11",
            "MR_long_01",
            "MR_long_11",
            "SV_long_01",
            "SV_long_11",
        ),
        description="Two samples per case type for a quick controlled comparison.",
    ),
    "nd_01_10": Batch("nd_01_10", ids("ND", 1, 10), "Normal daily long samples 01-10."),
    "nd_13": Batch("nd_13", ("ND_long_13",), "Re-run normal daily long sample 13."),
    "nd_11_20": Batch("nd_11_20", ids("ND", 11, 20), "Normal daily long samples 11-20."),
    "nd_01_20": Batch("nd_01_20", ids("ND", 1, 20), "All normal daily long samples."),
    "sf_01_10": Batch("sf_01_10", ids("SF", 1, 10), "Semantic fraud long samples 01-10."),
    "sf_11_20": Batch("sf_11_20", ids("SF", 11, 20), "Semantic fraud long samples 11-20."),
    "sf_01_20": Batch("sf_01_20", ids("SF", 1, 20), "All semantic fraud long samples."),
    "mr_01_10": Batch("mr_01_10", ids("MR", 1, 10), "Google TTS mixed-risk samples 01-10."),
    "mr_11_20": Batch("mr_11_20", ids("MR", 11, 20), "Google TTS mixed-risk samples 11-20."),
    "mr_01_20": Batch("mr_01_20", ids("MR", 1, 20), "All Google TTS mixed-risk samples."),
    "sv_01_10": Batch("sv_01_10", ids("SV", 1, 10), "Google TTS synthetic-voice samples 01-10."),
    "sv_11_20": Batch("sv_11_20", ids("SV", 11, 20), "Google TTS synthetic-voice samples 11-20."),
    "sv_01_20": Batch("sv_01_20", ids("SV", 1, 20), "All Google TTS synthetic-voice samples."),
}


def assert_project_environment() -> None:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "dissertation":
        raise SystemExit(
            "Please activate the project conda environment before running evaluation:\n"
            "  conda activate dissertation"
        )


def load_metadata_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row.get("sample_id", "") for row in csv.DictReader(handle)}


def validate_batch(metadata_path: Path, batch: Batch) -> None:
    available_ids = load_metadata_sample_ids(metadata_path)
    missing = [sample_id for sample_id in batch.sample_ids if sample_id not in available_ids]
    if missing:
        raise SystemExit(f"Batch {batch.name} has sample IDs missing from metadata: {missing}")


def format_number_for_name(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_name_for(batch: Batch, args: argparse.Namespace) -> str:
    window = format_number_for_name(args.window_seconds)
    step = format_number_for_name(args.step_seconds)
    return f"final_{args.scoring_mode}_w{window}_s{step}"


def predictions_filename(scoring_mode: str) -> str:
    return "progression_predictions.json" if scoring_mode == "progression_v1" else "dynamic_predictions.json"


def prediction_path_for(batch: Batch, args: argparse.Namespace) -> Path:
    run_name = run_name_for(batch, args)
    return PROJECT_ROOT / "evaluation/predictions" / run_name / predictions_filename(args.scoring_mode)


def build_prediction_command(batch: Batch, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        PREDICTION_SCRIPT.as_posix(),
        "--metadata",
        args.metadata.as_posix(),
        "--run-name",
        run_name_for(batch, args),
        "--scoring-mode",
        args.scoring_mode,
        "--window-seconds",
        str(args.window_seconds),
        "--step-seconds",
        str(args.step_seconds),
        "--text-weight",
        str(args.text_weight),
        "--smoothing-previous-weight",
        str(args.smoothing_previous_weight),
        "--resume",
    ]
    for sample_id in batch.sample_ids:
        command.extend(["--sample-id", sample_id])
    return command


def build_metrics_command(batch: Batch, args: argparse.Namespace) -> list[str]:
    run_name = run_name_for(batch, args)
    prediction_path = prediction_path_for(batch, args)
    command = [
        sys.executable,
        METRICS_SCRIPT.as_posix(),
        "--predictions",
        prediction_path.as_posix(),
        "--run-name",
        run_name,
        "--alert-threshold",
        str(args.alert_threshold),
    ]
    if args.alert_thresholds:
        command.extend(["--alert-thresholds", args.alert_thresholds])
    if args.fusion_text_weights:
        command.extend(["--fusion-text-weights", args.fusion_text_weights])
    if args.smoothing_previous_weights:
        command.extend(["--smoothing-previous-weights", args.smoothing_previous_weights])
    if args.sweep_scoring_mode:
        command.extend(["--sweep-scoring-mode", args.sweep_scoring_mode])
    return command


def shell_join(command: Sequence[str]) -> str:
    return shlex.join(command)


def print_batches(names: Iterable[str]) -> None:
    for name in names:
        batch = BATCHES[name]
        print(f"{batch.name}: {len(batch.sample_ids)} samples - {batch.description}")
        print("  " + ", ".join(batch.sample_ids))


def run_command(command: Sequence[str]) -> None:
    print(shell_join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def force_rerun_existing_samples(prediction_path: Path, sample_ids: Sequence[str]) -> None:
    """Make selected records non-resumable while preserving all other cached predictions."""
    if not prediction_path.exists():
        print(f"force-rerun: prediction file does not exist yet: {prediction_path}")
        return

    with prediction_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Prediction file has no records list: {prediction_path}")

    selected = set(sample_ids)
    touched = 0
    for record in records:
        if isinstance(record, dict) and str(record.get("sample_id") or "") in selected:
            record["error"] = "force_rerun_requested"
            touched += 1

    if isinstance(payload, dict):
        payload["records"] = records
    else:
        payload = records

    with prediction_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"force-rerun: marked {touched} existing record(s) in {prediction_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--list-batches", action="store_true", help="List available controlled batches and exit.")
    parser.add_argument("--print-commands", metavar="BATCH", choices=sorted(BATCHES), help="Print commands for one batch.")
    parser.add_argument("--run-batch", metavar="BATCH", choices=sorted(BATCHES), help="Run predictions for one batch.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Recompute selected sample IDs even when cached predictions already exist.",
    )
    parser.add_argument("--with-metrics", action="store_true", help="Run dynamic_metrics.py after the selected batch finishes.")
    parser.add_argument(
        "--scoring-mode",
        choices=["progression_v1", "baseline", "gated_v1", "gated_v2", "gated_v3"],
        default=DEFAULT_SCORING_MODE,
    )
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--step-seconds", type=float, default=DEFAULT_STEP_SECONDS)
    parser.add_argument("--text-weight", type=float, default=0.8)
    parser.add_argument("--smoothing-previous-weight", type=float, default=0.65)
    parser.add_argument("--alert-threshold", type=float, default=70.0)
    parser.add_argument("--alert-thresholds", help="Optional comma-separated metrics threshold sweep, e.g. 50,60,70,80.")
    parser.add_argument("--fusion-text-weights", help="Optional comma-separated offline fusion sweep, e.g. 0.7,0.8,0.9.")
    parser.add_argument("--smoothing-previous-weights", help="Optional comma-separated smoothing sweep, e.g. 0.5,0.65,0.8.")
    parser.add_argument(
        "--sweep-scoring-mode",
        choices=["baseline", "gated_v1", "gated_v2", "gated_v3"],
        help="Optional scoring mode for offline fusion/smoothing sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.metadata = args.metadata.resolve()

    if args.list_batches or (not args.print_commands and not args.run_batch):
        print_batches(sorted(BATCHES))
        return

    batch_name = args.print_commands or args.run_batch
    batch = BATCHES[batch_name]
    validate_batch(args.metadata, batch)

    prediction_command = build_prediction_command(batch, args)
    metrics_command = build_metrics_command(batch, args)

    if args.print_commands:
        if args.force_rerun:
            print(
                "# --force-rerun is handled by run_long_controlled_eval.py before executing "
                "generate_dynamic_predictions.py."
            )
        print(shell_join(prediction_command))
        if args.with_metrics:
            print(shell_join(metrics_command))
        return

    assert_project_environment()
    if args.force_rerun:
        force_rerun_existing_samples(prediction_path_for(batch, args), batch.sample_ids)
    run_command(prediction_command)
    if args.with_metrics:
        run_command(metrics_command)


if __name__ == "__main__":
    main()
