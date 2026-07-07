#!/usr/bin/env python3
"""Sync corrected fraud labels and rerun dynamic ablation metrics.

The fraud label is semantic/behavioral: synthetic speech reading benign content
is normal for fraud detection, while still positive for acoustic authenticity.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from evaluation import dynamic_metrics


GENERATE_PREDICTIONS = PROJECT_ROOT / "evaluation" / "generate_dynamic_predictions.py"
PREDICTION_ROOT = PROJECT_ROOT / "evaluation" / "predictions"
REPORT_ROOT = PROJECT_ROOT / "evaluation" / "reports"
FINAL_METADATA = PROJECT_ROOT / "test_samples" / "metadata_final.csv"
EXTERNAL_V2_METADATA = PROJECT_ROOT / "test_samples" / "metadata_external_frozen_v2.csv"
AUDIO_ALL_DIR = PROJECT_ROOT / "test_samples" / "audio_all"
ALL180_METADATA = PROJECT_ROOT / "test_samples" / "metadata_all_corrected.csv"
DEFAULT_FUSION_TEXT_WEIGHTS = "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00"

CASE_FRAUD_LABELS = {
    "normal_daily": "normal",
    "normal_finance": "normal",
    "synthetic_voice": "normal",
    "semantic_fraud": "fraud",
    "mixed_risk": "fraud",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    metadata: Path
    prediction_run_name: str
    report_name: str
    source_metadata: tuple[Path, ...] = ()
    audio_dir: Path | None = None
    core_report_name: str | None = None
    exclude_core_case_types: tuple[str, ...] = ()
    expected_records: int | None = None


DATASETS = {
    "defense_all180": DatasetConfig(
        name="defense_all180",
        metadata=ALL180_METADATA,
        source_metadata=(FINAL_METADATA, EXTERNAL_V2_METADATA),
        audio_dir=AUDIO_ALL_DIR,
        prediction_run_name="defense_all180_baseline_w10_s5",
        report_name="defense_all180_baseline_w10_s5",
        expected_records=180,
    ),
    "final": DatasetConfig(
        name="final",
        metadata=FINAL_METADATA,
        prediction_run_name="final_baseline_w10_s5",
        report_name="final_baseline_w10_s5",
        expected_records=80,
    ),
    "external_v2": DatasetConfig(
        name="external_v2",
        metadata=EXTERNAL_V2_METADATA,
        prediction_run_name="external_frozen_v2_baseline_w10_s5",
        report_name="external_frozen_v2_expanded100",
        core_report_name="external_frozen_v2_core80",
        exclude_core_case_types=("normal_finance",),
        expected_records=100,
    ),
}


def assert_project_environment() -> None:
    if os.environ.get("CONDA_DEFAULT_ENV") != "dissertation":
        raise SystemExit(
            "Please activate the project conda environment before running this script:\n"
            "  conda activate dissertation"
        )


def normalize_label(value: Any) -> int:
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "true", "fraud", "phishing", "positive", "risk"} else 0


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def corrected_label_for_case(case_type: str, fallback: str) -> str:
    return CASE_FRAUD_LABELS.get(str(case_type or "").strip(), fallback)


def sync_metadata_labels(metadata_path: Path, dry_run: bool = False) -> dict[str, Any]:
    fieldnames, rows = read_csv_rows(metadata_path)
    changed = 0
    counts: dict[str, dict[str, int]] = {}

    for row in rows:
        case_type = str(row.get("case_type") or "").strip()
        old_label = str(row.get("label") or "").strip()
        new_label = corrected_label_for_case(case_type, old_label)
        if old_label != new_label:
            row["label"] = new_label
            changed += 1
        if new_label == "normal" and row.get("event_time_sec"):
            row["event_time_sec"] = ""
            changed += 1
        counts.setdefault(case_type, {})
        counts[case_type][new_label] = counts[case_type].get(new_label, 0) + 1

    if changed and not dry_run:
        write_csv_rows(metadata_path, fieldnames, rows)

    return {"path": metadata_path.relative_to(PROJECT_ROOT).as_posix(), "changed_fields": changed, "counts": counts}


def audio_files_by_sample_id(audio_dir: Path) -> dict[str, Path]:
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    mapping: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for path in sorted(audio_dir.iterdir()):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        sample_id = path.stem
        if sample_id in mapping:
            duplicates.setdefault(sample_id, [mapping[sample_id].name]).append(path.name)
        mapping[sample_id] = path
    if duplicates:
        raise ValueError(f"Duplicate audio files for sample IDs: {duplicates}")
    return mapping


def build_combined_metadata(config: DatasetConfig, dry_run: bool = False) -> dict[str, Any]:
    if not config.source_metadata:
        return {"created": False, "path": config.metadata.relative_to(PROJECT_ROOT).as_posix()}
    if config.audio_dir is None:
        raise ValueError(f"{config.name} has source metadata but no audio_dir")

    audio_by_id = audio_files_by_sample_id(config.audio_dir)
    source_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    origin_by_sample_id: dict[str, str] = {}
    for metadata_path in config.source_metadata:
        current_fields, rows = read_csv_rows(metadata_path)
        for field in current_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        for row in rows:
            sample_id = str(row.get("sample_id") or "").strip()
            if sample_id:
                origin_by_sample_id[sample_id] = metadata_path.relative_to(PROJECT_ROOT).as_posix()
        source_rows.extend(rows)

    if "origin_metadata" not in fieldnames:
        fieldnames.append("origin_metadata")

    seen: set[str] = set()
    missing_audio: list[str] = []
    output_rows: list[dict[str, str]] = []
    for row in source_rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id across metadata sources: {sample_id}")
        seen.add(sample_id)
        audio_path = audio_by_id.get(sample_id)
        if audio_path is None:
            missing_audio.append(sample_id)
            continue
        output = {field: row.get(field, "") for field in fieldnames}
        output["origin_metadata"] = origin_by_sample_id.get(sample_id, "")
        output["audio_path"] = audio_path.relative_to(PROJECT_ROOT).as_posix()
        output["label"] = corrected_label_for_case(str(output.get("case_type") or ""), str(output.get("label") or ""))
        if output.get("source") == "mimo_tts_voicedesign":
            output["source"] = "real_recording"
        if output["label"] == "normal":
            output["event_time_sec"] = ""
        output_rows.append(output)

    extra_audio = sorted(set(audio_by_id) - seen)
    if missing_audio or extra_audio:
        raise ValueError({
            "missing_audio_for_metadata": missing_audio,
            "extra_audio_without_metadata": extra_audio,
            "metadata_rows": len(source_rows),
            "audio_files": len(audio_by_id),
        })
    if config.expected_records is not None and len(output_rows) != config.expected_records:
        raise ValueError(f"Expected {config.expected_records} rows for {config.name}, found {len(output_rows)}")

    if not dry_run:
        write_csv_rows(config.metadata, fieldnames, output_rows)
    counts: dict[str, dict[str, int]] = {}
    for row in output_rows:
        case_type = str(row.get("case_type") or "")
        label = str(row.get("label") or "")
        counts.setdefault(case_type, {})
        counts[case_type][label] = counts[case_type].get(label, 0) + 1
    return {
        "created": True,
        "path": config.metadata.relative_to(PROJECT_ROOT).as_posix(),
        "records": len(output_rows),
        "audio_dir": config.audio_dir.relative_to(PROJECT_ROOT).as_posix(),
        "counts": counts,
    }


def metadata_by_sample_id(metadata_path: Path) -> dict[str, dict[str, str]]:
    _, rows = read_csv_rows(metadata_path)
    return {str(row.get("sample_id") or ""): row for row in rows}


def load_prediction_records(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Prediction file has no records list: {path}")
    return payload, records


def write_prediction_payload(path: Path, payload: Any, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["records"] = list(records)
    else:
        payload = list(records)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def sync_prediction_labels(config: DatasetConfig, dry_run: bool = False) -> dict[str, Any]:
    prediction_path = prediction_path_for(config)
    if not prediction_path.exists():
        return {"path": prediction_path.relative_to(PROJECT_ROOT).as_posix(), "exists": False, "changed_fields": 0}

    metadata = metadata_by_sample_id(config.metadata)
    payload, records = load_prediction_records(prediction_path)
    changed = 0

    for record in records:
        sample_id = str(record.get("sample_id") or "")
        row = metadata.get(sample_id)
        if not row:
            continue
        new_label = normalize_label(row.get("label"))
        if record.get("label") != new_label:
            record["label"] = new_label
            changed += 1
        event_time = row.get("event_time_sec")
        new_event_time: float | None = None if event_time in ("", None) else float(event_time)
        if record.get("event_time_sec") != new_event_time:
            record["event_time_sec"] = new_event_time
            changed += 1
        for key in ("case_type", "audio_path", "source"):
            if row.get(key) and record.get(key) != row.get(key):
                record[key] = row.get(key)
                changed += 1

    if changed and not dry_run:
        write_prediction_payload(prediction_path, payload, records)

    return {"path": prediction_path.relative_to(PROJECT_ROOT).as_posix(), "exists": True, "changed_fields": changed}


def prediction_path_for(config: DatasetConfig) -> Path:
    return PREDICTION_ROOT / config.prediction_run_name / "dynamic_predictions.json"


def core_prediction_path_for(config: DatasetConfig) -> Path:
    return PREDICTION_ROOT / config.prediction_run_name / "core80_dynamic_predictions.json"


def build_core_prediction_view(config: DatasetConfig, dry_run: bool = False) -> dict[str, Any]:
    if not config.core_report_name:
        return {"created": False}
    prediction_path = prediction_path_for(config)
    payload, records = load_prediction_records(prediction_path)
    excluded = set(config.exclude_core_case_types)
    core_records = [record for record in records if str(record.get("case_type") or "") not in excluded]
    output_path = core_prediction_path_for(config)
    if not dry_run:
        write_prediction_payload(output_path, {"records": core_records}, core_records)
    return {
        "created": True,
        "path": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "records": len(core_records),
        "excluded_case_types": sorted(excluded),
    }


def prediction_record_count(path: Path) -> int:
    if not path.exists():
        return 0
    _, records = load_prediction_records(path)
    return len(records)


def prediction_ready(config: DatasetConfig, allow_partial: bool = False) -> dict[str, Any]:
    path = prediction_path_for(config)
    count = prediction_record_count(path)
    expected = config.expected_records
    ready = path.exists() and (allow_partial or expected is None or count == expected)
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "exists": path.exists(),
        "records": count,
        "expected_records": expected,
        "ready": ready,
    }


def case_type_sample_ids(metadata_path: Path, case_types: Iterable[str]) -> list[str]:
    selected_case_types = set(case_types)
    if not selected_case_types:
        return []
    _, rows = read_csv_rows(metadata_path)
    return [
        str(row.get("sample_id") or "")
        for row in rows
        if str(row.get("case_type") or "") in selected_case_types
    ]


def mark_records_for_rerun(prediction_path: Path, sample_ids: Sequence[str]) -> int:
    if not prediction_path.exists() or not sample_ids:
        return 0
    payload, records = load_prediction_records(prediction_path)
    selected = set(sample_ids)
    touched = 0
    for record in records:
        if str(record.get("sample_id") or "") in selected:
            record["error"] = "force_rerun_requested"
            touched += 1
    write_prediction_payload(prediction_path, payload, records)
    return touched


def run_command(command: Sequence[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_predictions(config: DatasetConfig, args: argparse.Namespace) -> dict[str, Any]:
    prediction_path = prediction_path_for(config)
    if args.force_rerun_all:
        sample_ids = list(metadata_by_sample_id(config.metadata))
    else:
        sample_ids = case_type_sample_ids(config.metadata, args.force_rerun_case_type)
    touched = mark_records_for_rerun(prediction_path, sample_ids) if sample_ids else 0

    command = [
        sys.executable,
        GENERATE_PREDICTIONS.as_posix(),
        "--metadata",
        config.metadata.as_posix(),
        "--run-name",
        config.prediction_run_name,
        "--scoring-mode",
        "baseline",
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
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    run_command(command)
    return {"force_rerun_marked_records": touched, "prediction_path": prediction_path.relative_to(PROJECT_ROOT).as_posix()}


def run_metrics_for_prediction(prediction_path: Path, report_name: str, args: argparse.Namespace) -> dict[str, Any]:
    fusion_text_weights = args.fusion_text_weights or DEFAULT_FUSION_TEXT_WEIGHTS
    smoothing_previous_weights = args.smoothing_previous_weights or str(args.smoothing_previous_weight)
    report = dynamic_metrics.run_evaluation(
        prediction_path=prediction_path,
        output_dir=REPORT_ROOT / report_name,
        alert_threshold=args.alert_threshold,
        thresholds=dynamic_metrics.MetricThresholds(
            final_f1=args.min_final_f1,
            mean_lead_time_sec=args.min_mean_lead_time_sec,
            mean_detection_delay_sec=args.max_mean_detection_delay_sec,
        ),
        alert_thresholds=dynamic_metrics.parse_float_list(args.alert_thresholds),
        fusion_text_weights=dynamic_metrics.parse_float_list(fusion_text_weights),
        smoothing_previous_weights=dynamic_metrics.parse_float_list(smoothing_previous_weights),
        sweep_scoring_mode=args.sweep_scoring_mode or "baseline",
    )
    return {
        "report_dir": (REPORT_ROOT / report_name).relative_to(PROJECT_ROOT).as_posix(),
        "ablation_summary": report["ablation_summary"],
        "case_type_summary": (REPORT_ROOT / report_name / "dynamic_eval_case_type_summary.csv").relative_to(PROJECT_ROOT).as_posix(),
        "fusion_sweep": (REPORT_ROOT / report_name / "dynamic_eval_fusion_smoothing_sweep.csv").relative_to(PROJECT_ROOT).as_posix(),
        "fusion_case_type_sweep": (REPORT_ROOT / report_name / "dynamic_eval_fusion_smoothing_case_type_sweep.csv").relative_to(PROJECT_ROOT).as_posix(),
    }


def selected_configs(dataset_names: Sequence[str]) -> list[DatasetConfig]:
    names = list(dataset_names or ["defense_all180"])
    if "all" in names:
        names = list(DATASETS)
    return [DATASETS[name] for name in names]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted([*DATASETS.keys(), "all"]),
        default=None,
        help="Dataset to process. Defaults to defense_all180. Repeatable. Use --dataset all for every configured dataset.",
    )
    parser.add_argument("--rerun-predictions", action="store_true", help="Call the live stream route before metrics.")
    parser.add_argument("--force-rerun-all", action="store_true", help="When rerunning predictions, recompute every metadata record.")
    parser.add_argument(
        "--force-rerun-case-type",
        action="append",
        default=[],
        choices=sorted(CASE_FRAUD_LABELS.keys()),
        help="When rerunning predictions, recompute cached records for this case_type. Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true", help="Only sync/build metadata and predictions; do not write reports.")
    parser.add_argument("--limit", type=int, help="Debug only: run at most this many prediction records.")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--text-weight", type=float, default=0.8)
    parser.add_argument("--smoothing-previous-weight", type=float, default=0.65)
    parser.add_argument("--alert-threshold", type=float, default=70.0)
    parser.add_argument("--min-final-f1", type=float, default=0.80)
    parser.add_argument("--min-mean-lead-time-sec", type=float, default=5.0)
    parser.add_argument("--max-mean-detection-delay-sec", type=float, default=15.0)
    parser.add_argument("--alert-thresholds", help="Optional comma-separated alert threshold sweep.")
    parser.add_argument(
        "--fusion-text-weights",
        default=DEFAULT_FUSION_TEXT_WEIGHTS,
        help="Comma-separated offline fusion sweep. Defaults to 0.50,0.55,...,1.00.",
    )
    parser.add_argument(
        "--smoothing-previous-weights",
        help="Comma-separated smoothing sweep. Defaults to --smoothing-previous-weight.",
    )
    parser.add_argument(
        "--sweep-scoring-mode",
        choices=["baseline", "gated_v1", "gated_v2", "gated_v3"],
        help="Optional scoring mode for offline fusion/smoothing sweep.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    assert_project_environment()

    results: dict[str, Any] = {}
    for config in selected_configs(args.dataset):
        dataset_result: dict[str, Any] = {}
        dataset_result["combined_metadata"] = build_combined_metadata(config, dry_run=args.dry_run)
        if args.dry_run and dataset_result["combined_metadata"].get("created") and not config.metadata.exists():
            dataset_result["metadata"] = {
                "path": config.metadata.relative_to(PROJECT_ROOT).as_posix(),
                "skipped": "dry_run_combined_metadata_not_written",
            }
        else:
            dataset_result["metadata"] = sync_metadata_labels(config.metadata, dry_run=args.dry_run)

        if args.rerun_predictions and not args.dry_run:
            dataset_result["prediction_run"] = run_predictions(config, args)

        dataset_result["prediction_labels"] = sync_prediction_labels(config, dry_run=args.dry_run)
        dataset_result["prediction_ready"] = prediction_ready(config, allow_partial=args.limit is not None)
        if not args.dry_run and not args.skip_metrics and dataset_result["prediction_ready"]["ready"]:
            dataset_result["metrics"] = run_metrics_for_prediction(prediction_path_for(config), config.report_name, args)
            core_view = build_core_prediction_view(config, dry_run=args.dry_run)
            dataset_result["core_view"] = core_view
            if core_view.get("created") and config.core_report_name:
                dataset_result["core_metrics"] = run_metrics_for_prediction(
                    core_prediction_path_for(config),
                    config.core_report_name,
                    args,
                )
        elif not args.dry_run and not args.skip_metrics:
            dataset_result["metrics"] = {
                "skipped": "prediction_file_missing_or_incomplete",
                "hint": "Run again with --rerun-predictions after confirming metadata/audio files.",
            }
        results[config.name] = dataset_result

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
