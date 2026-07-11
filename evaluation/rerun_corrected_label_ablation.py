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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from evaluation import dynamic_metrics
from evaluation import causal_late_fusion_v2


GENERATE_PREDICTIONS = PROJECT_ROOT / "evaluation" / "generate_dynamic_predictions.py"
PREDICTION_ROOT = PROJECT_ROOT / "evaluation" / "predictions"
REPORT_ROOT = PROJECT_ROOT / "evaluation" / "reports"
FINAL_METADATA = PROJECT_ROOT / "test_samples" / "metadata_final.csv"
EXTERNAL_V2_METADATA = PROJECT_ROOT / "test_samples" / "metadata_external_frozen_v2.csv"
AUDIO_ALL_DIR = PROJECT_ROOT / "test_samples" / "audio_all"
ALL180_METADATA = PROJECT_ROOT / "test_samples" / "metadata_all_corrected.csv"
DEFAULT_FUSION_TEXT_WEIGHTS = "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00"
LEARNED_OOF_RUN_NAME = "defense_all180_causal_learned_oof_w10_s5"

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


def learned_oof_prediction_path_for(config: DatasetConfig) -> Path:
    return PREDICTION_ROOT / LEARNED_OOF_RUN_NAME / "dynamic_predictions.json"


def learned_oof_report_path_for(config: DatasetConfig) -> Path:
    return PREDICTION_ROOT / LEARNED_OOF_RUN_NAME / "nested_oof_report.json"


def metadata_group_id(row: dict[str, str], sample_id: str) -> str:
    origin = str(row.get("origin_metadata") or "unknown_origin")
    script_id = str(row.get("script_id") or "").strip()
    if script_id:
        return f"{origin}:script:{script_id}"
    return f"{origin}:sample:{sample_id}"


def grouped_metadata_folds(
    records: Sequence[dict[str, Any]],
    metadata: dict[str, dict[str, str]],
    n_splits: int,
) -> list[list[int]]:
    """Build sample-level folds and keep paired script variants together."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        sample_id = str(record.get("sample_id") or "")
        row = metadata.get(sample_id, {})
        groups[metadata_group_id(row, sample_id)].append(index)

    buckets: dict[tuple[tuple[str, str], ...], list[tuple[str, list[int]]]] = defaultdict(list)
    for group_id, indices in groups.items():
        signature = tuple(sorted(
            (
                str(records[index].get("case_type") or "unknown"),
                str(records[index].get("label", records[index].get("is_fraud", 0))),
            )
            for index in indices
        ))
        buckets[signature].append((group_id, indices))

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for signature in sorted(buckets, key=str):
        for offset, (group_id, indices) in enumerate(sorted(buckets[signature], key=lambda item: item[0])):
            folds[offset % n_splits].extend(indices)
    return [sorted(fold) for fold in folds]


def subset_records(records: Sequence[dict[str, Any]], indices: Iterable[int]) -> list[dict[str, Any]]:
    return [records[index] for index in indices]


def select_learned_candidate(
    train_records: Sequence[dict[str, Any]],
    inner_splits: int,
    random_state: int,
) -> dict[str, Any]:
    selected: dict[str, Any] | None = None
    candidate_reports: list[dict[str, Any]] = []
    for candidate in causal_late_fusion_v2.candidates(random_state):
        series = causal_late_fusion_v2.oof_probability_series(candidate, train_records, inner_splits)
        threshold, summary = causal_late_fusion_v2.select_threshold(train_records, series)
        score = (causal_late_fusion_v2.threshold_score(summary), -candidate.preference_rank)
        candidate_report = {
            "model_name": candidate.name,
            "threshold_probability": round(float(threshold), 8),
            "inner_oof_summary": summary,
        }
        candidate_reports.append(candidate_report)
        payload = {
            "candidate": candidate,
            "threshold": threshold,
            "summary": summary,
            "score": score,
        }
        if selected is None or payload["score"] > selected["score"]:
            selected = payload
    if selected is None:
        raise RuntimeError("No learned fusion candidate selected.")
    selected["candidate_reports"] = candidate_reports
    return selected


def build_learned_oof_predictions(
    config: DatasetConfig,
    prediction_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload, records = load_prediction_records(prediction_path)
    metadata = metadata_by_sample_id(config.metadata)
    if len(records) != config.expected_records:
        raise ValueError(f"Expected {config.expected_records} records for learned OOF, found {len(records)}")
    for record in records:
        if not record.get("timeline"):
            raise ValueError(f"Missing timeline for learned OOF: {record.get('sample_id')}")

    outer_splits = args.learned_outer_splits
    inner_splits = args.learned_inner_splits
    folds = grouped_metadata_folds(records, metadata, outer_splits)
    all_indices = set(range(len(records)))
    oof_records: list[dict[str, Any] | None] = [None for _ in records]
    fold_reports: list[dict[str, Any]] = []

    causal_late_fusion_v2.assert_feature_contract()
    for outer_fold, test_indices in enumerate(folds, start=1):
        train_indices = sorted(all_indices - set(test_indices))
        train_records = subset_records(records, train_indices)
        test_records = subset_records(records, test_indices)
        selected = select_learned_candidate(
            train_records,
            inner_splits=inner_splits,
            random_state=args.learned_random_state + outer_fold * 100,
        )
        candidate = selected["candidate"]
        model = causal_late_fusion_v2.fit_model(
            candidate,
            train_records,
            calibration_splits=min(3, max(2, inner_splits - 1)),
        )
        test_series = causal_late_fusion_v2.predict_probability_series(model, test_records)
        annotated = causal_late_fusion_v2.annotate(
            test_records,
            test_series,
            selected["threshold"],
            f"nested_oof_{candidate.name}",
        )
        for record_index, annotated_record in zip(test_indices, annotated):
            annotated_record["causal_late_fusion_v2_cv_protocol"] = (
                f"{outer_splits}-fold outer / {inner_splits}-fold inner grouped OOF on corrected 180"
            )
            annotated_record["causal_late_fusion_v2_outer_fold"] = outer_fold
            oof_records[record_index] = annotated_record
        fold_reports.append({
            "outer_fold": outer_fold,
            "train_samples": len(train_records),
            "test_samples": len(test_records),
            "test_case_counts": {
                case_type: sum(1 for record in test_records if str(record.get("case_type") or "") == case_type)
                for case_type in sorted({str(record.get("case_type") or "") for record in test_records})
            },
            "selected_model": candidate.name,
            "decision_threshold_probability": round(float(selected["threshold"]), 8),
            "inner_selected_summary": selected["summary"],
            "candidate_reports": selected["candidate_reports"],
        })

    final_records = [record for record in oof_records if record is not None]
    if len(final_records) != len(records):
        raise RuntimeError("Incomplete learned OOF predictions.")

    output_path = learned_oof_prediction_path_for(config)
    report_path = learned_oof_report_path_for(config)
    write_prediction_payload(output_path, payload, final_records)

    learned_rows = [
        dynamic_metrics.evaluate_record(record, causal_late_fusion_v2.SCORE_KEY, args.alert_threshold)
        for record in final_records
    ]
    learned_summary = dynamic_metrics.summarize_dynamic_metrics(learned_rows)
    learned_case_summary = {
        case_type: dynamic_metrics.summarize_dynamic_metrics(
            [row for row in learned_rows if row["case_type"] == case_type]
        )
        for case_type in sorted({row["case_type"] for row in learned_rows})
    }
    report = {
        "mode": "corrected_180_nested_grouped_oof_causal_late_fusion_v2",
        "input_predictions": prediction_path.relative_to(PROJECT_ROOT).as_posix(),
        "output_predictions": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "random_state": args.learned_random_state,
        "feature_contract_version": causal_late_fusion_v2.FEATURE_CONTRACT_VERSION,
        "feature_names": causal_late_fusion_v2.FEATURE_NAMES,
        "threshold_selection": "inner grouped OOF within each outer training fold",
        "final_evaluation": "outer grouped OOF predictions only",
        "summary": learned_summary,
        "case_type_summary": learned_case_summary,
        "fold_reports": fold_reports,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "prediction_path": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "report_path": report_path.relative_to(PROJECT_ROOT).as_posix(),
        "summary": learned_summary,
    }


def metric_prediction_path_for(config: DatasetConfig, prediction_path: Path, args: argparse.Namespace) -> tuple[Path, dict[str, Any] | None]:
    if config.name != "defense_all180" or args.skip_learned_fusion_oof:
        return prediction_path, None
    learned_result = build_learned_oof_predictions(config, prediction_path, args)
    return PROJECT_ROOT / learned_result["prediction_path"], learned_result


def run_metrics_for_prediction(prediction_path: Path, report_name: str, args: argparse.Namespace, config: DatasetConfig | None = None) -> dict[str, Any]:
    metric_prediction_path = prediction_path
    learned_result = None
    if config is not None:
        metric_prediction_path, learned_result = metric_prediction_path_for(config, prediction_path, args)
    fusion_text_weights = args.fusion_text_weights or DEFAULT_FUSION_TEXT_WEIGHTS
    smoothing_previous_weights = args.smoothing_previous_weights or str(args.smoothing_previous_weight)
    output_dir = REPORT_ROOT / report_name
    report = dynamic_metrics.run_evaluation(
        prediction_path=metric_prediction_path,
        output_dir=output_dir,
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
    strategy_comparison_path = build_fusion_strategy_comparison(output_dir)
    return {
        "report_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
        "ablation_summary": report["ablation_summary"],
        "case_type_summary": (output_dir / "dynamic_eval_case_type_summary.csv").relative_to(PROJECT_ROOT).as_posix(),
        "fusion_sweep": (output_dir / "dynamic_eval_fusion_smoothing_sweep.csv").relative_to(PROJECT_ROOT).as_posix(),
        "fusion_case_type_sweep": (output_dir / "dynamic_eval_fusion_smoothing_case_type_sweep.csv").relative_to(PROJECT_ROOT).as_posix(),
        "fusion_strategy_comparison": strategy_comparison_path.relative_to(PROJECT_ROOT).as_posix(),
        "learned_oof": learned_result,
    }


def read_report_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_value(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    return str(value)


def comparison_row(
    row: dict[str, str],
    strategy: str,
    comparison_group: str,
    text_weight: str,
    voice_weight: str,
    smoothing: str,
    validation_protocol: str,
    defense_role: str,
    interpretation: str,
    selected_mainline: bool = False,
) -> dict[str, str]:
    fields = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "tn",
        "fp",
        "fn",
        "fraud_alert_recall",
        "normal_alert_false_positive_rate",
        "normal_final_false_positive_rate",
        "mean_time_to_alert_sec",
        "mean_early_warning_lead_time_sec",
        "mean_detection_delay_sec",
    ]
    output = {
        "comparison_group": comparison_group,
        "strategy": strategy,
        "text_weight": text_weight,
        "voice_weight": voice_weight,
        "smoothing_previous_weight": smoothing,
        "validation_protocol": validation_protocol,
        "selected_mainline": "yes" if selected_mainline else "no",
        "defense_role": defense_role,
        "interpretation": interpretation,
    }
    output.update({field: metric_value(row, field) for field in fields})
    return output


def build_fusion_strategy_comparison(report_dir: Path) -> Path:
    """Write a slide-ready comparison of the fusion strategies used in the defense."""
    ablation_rows = read_report_csv(report_dir / "dynamic_eval_ablation_summary.csv")
    sweep_rows = read_report_csv(report_dir / "dynamic_eval_fusion_smoothing_sweep.csv")
    by_variant = {row.get("variant", ""): row for row in ablation_rows}
    by_weight = {row.get("text_weight", ""): row for row in sweep_rows}

    rows: list[dict[str, str]] = []
    if "text_only" in by_variant:
        rows.append(comparison_row(
            by_variant["text_only"],
            strategy="text_only",
            comparison_group="single_modality",
            text_weight="1.00",
            voice_weight="0.00",
            smoothing="none",
            validation_protocol="same corrected 180 cached predictions",
            defense_role="semantic-risk reference",
            interpretation="Strong semantic branch, but more final false positives than smoothed fusion.",
        ))
    if "voice_only" in by_variant:
        rows.append(comparison_row(
            by_variant["voice_only"],
            strategy="voice_only",
            comparison_group="single_modality",
            text_weight="0.00",
            voice_weight="1.00",
            smoothing="none",
            validation_protocol="same corrected 180 cached predictions",
            defense_role="authenticity cue diagnostic",
            interpretation="Not a standalone fraud detector; mainly shows the acoustic branch is auxiliary.",
        ))
    if "fusion_without_smoothing" in by_variant:
        rows.append(comparison_row(
            by_variant["fusion_without_smoothing"],
            strategy="fixed_8_2_without_smoothing",
            comparison_group="fixed_fusion",
            text_weight="0.80",
            voice_weight="0.20",
            smoothing="none",
            validation_protocol="same corrected 180 cached predictions",
            defense_role="fusion ablation",
            interpretation="Shows late fusion benefit before temporal stabilization.",
        ))
    if "causal_learned_late_fusion_v2" in by_variant:
        rows.append(comparison_row(
            by_variant["causal_learned_late_fusion_v2"],
            strategy="learned_late_fusion_nested_oof",
            comparison_group="learned_fusion",
            text_weight="learned",
            voice_weight="learned",
            smoothing="learned",
            validation_protocol="5-fold outer / 4-fold inner grouped OOF on corrected 180",
            defense_role="diagnostic comparison, not selected mainline",
            interpretation="Tests whether a learned decision layer should replace the interpretable fixed rule.",
        ))

    sweep_specs = [
        ("0.7", "fixed_7_3_with_smoothing", "weight_sensitivity", "More voice-sensitive but loses fraud recall."),
        ("0.8", "fixed_8_2_with_smoothing", "mainline", "Selected mainline: interpretable text-dominant fusion while preserving an acoustic cue."),
        ("0.9", "fixed_9_1_with_smoothing", "weight_sensitivity", "Slightly higher recall, but reduces the practical role of the acoustic branch."),
        ("1.0", "text_only_with_smoothing", "weight_sensitivity", "Upper text-dominant reference; useful but effectively removes multimodal fusion."),
    ]
    for weight, strategy, group, interpretation in sweep_specs:
        row = by_weight.get(weight)
        if not row:
            continue
        rows.append(comparison_row(
            row,
            strategy=strategy,
            comparison_group=group,
            text_weight=metric_value(row, "text_weight"),
            voice_weight=metric_value(row, "voice_weight"),
            smoothing=metric_value(row, "smoothing_previous_weight"),
            validation_protocol="offline fixed-rule rescore on corrected 180",
            defense_role="selected main method" if weight == "0.8" else "comparison point",
            interpretation=interpretation,
            selected_mainline=(weight == "0.8"),
        ))

    output_path = report_dir / "dynamic_eval_fusion_strategy_comparison.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        write_csv_rows(output_path, fieldnames, rows)
    else:
        write_csv_rows(output_path, ["strategy", "error"], [{"strategy": "", "error": "no comparison rows"}])
    return output_path


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
    parser.add_argument(
        "--skip-learned-fusion-oof",
        action="store_true",
        help="Skip corrected-180 nested OOF learned late-fusion comparison.",
    )
    parser.add_argument("--learned-outer-splits", type=int, default=5)
    parser.add_argument("--learned-inner-splits", type=int, default=4)
    parser.add_argument("--learned-random-state", type=int, default=42)
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
            dataset_result["metrics"] = run_metrics_for_prediction(
                prediction_path_for(config),
                config.report_name,
                args,
                config=config,
            )
            core_view = build_core_prediction_view(config, dry_run=args.dry_run)
            dataset_result["core_view"] = core_view
            if core_view.get("created") and config.core_report_name:
                dataset_result["core_metrics"] = run_metrics_for_prediction(
                    core_prediction_path_for(config),
                    config.core_report_name,
                    args,
                    config=config,
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
