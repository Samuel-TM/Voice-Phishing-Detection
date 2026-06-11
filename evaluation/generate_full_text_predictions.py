#!/usr/bin/env python3
"""Generate post-hoc full-audio text-risk predictions.

This script tests whether the baseline ChineseBERT text model performs better
when it sees the whole transcript after the audio has ended, instead of short
in-event windows. It writes JSON records, a per-sample CSV summary, and compact
metrics for the full-audio ASR transcript and the optional metadata transcript.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

DEFAULT_METADATA = PROJECT_ROOT / "test_samples" / "metadata.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluation" / "predictions"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
SCORE_FIELDS = ("full_asr_text_score", "metadata_text_score")


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value else 0
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "true", "fraud", "phishing", "positive", "risk"} else 0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_rows(
    rows: Sequence[Dict[str, str]],
    sample_ids: set[str],
    case_types: set[str],
    limit: int | None,
) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        case_type = row.get("case_type", "").strip()
        audio_path = resolve_project_path(row.get("audio_path", ""))
        if sample_ids and sample_id not in sample_ids:
            continue
        if case_types and case_type not in case_types:
            continue
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        if not audio_path.exists():
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def load_existing_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = data.get("records", []) if isinstance(data, dict) else data
    if not isinstance(records, list):
        return {}
    return {str(item.get("sample_id")): item for item in records if isinstance(item, dict)}


def write_json_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "records": list(records),
        "description": "Post-hoc full-audio text-risk predictions from baseline ChineseBERT.",
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def predict_text(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {
            "llm_score": 0.0,
            "final_label": "empty",
            "phishing_detected": False,
            "error": "empty text",
        }

    from ChineseBERTModel.ensemble_utils import ensemble_inference

    result = ensemble_inference(text)
    return {
        "llm_score": round(safe_float(result.get("llm_score")), 2),
        "final_label": result.get("final_label", ""),
        "phishing_detected": bool(result.get("phishing_detected", False)),
        "error": result.get("error", ""),
    }


def transcribe_full_audio(audio_path: Path) -> Dict[str, Any]:
    from speaker_analysis.asr_backend import transcribe_segment_with_metadata

    result = transcribe_segment_with_metadata(audio_path.as_posix())
    text = result.text or ""
    error = result.error or ""
    if text.startswith("(STT"):
        error = error or text
        text = ""
    return {
        "text": text,
        "raw_text": result.raw_text or "",
        "backend": result.backend,
        "model_name": result.model_name,
        "error": error,
    }


def build_record(row: Dict[str, str]) -> Dict[str, Any]:
    from speaker_analysis.asr_backend import clean_stt_text

    sample_id = row.get("sample_id", "").strip()
    audio_path = resolve_project_path(row.get("audio_path", ""))
    label = normalize_label(row.get("label", 0))
    event_time = row.get("event_time_sec", "")
    start = time.time()
    error = ""

    try:
        asr = transcribe_full_audio(audio_path)
    except Exception as exc:
        asr = {"text": "", "raw_text": "", "backend": "", "model_name": "", "error": str(exc)}
    full_asr_text = asr.get("text", "")
    full_asr_prediction = predict_text(full_asr_text)

    metadata_text = clean_stt_text(row.get("transcript_text", ""))
    metadata_prediction = predict_text(metadata_text) if metadata_text else {
        "llm_score": "",
        "final_label": "",
        "phishing_detected": "",
        "error": "metadata transcript missing",
    }

    if asr.get("error"):
        error = f"ASR: {asr.get('error')}"
    if full_asr_prediction.get("error") and full_asr_text:
        error = f"{error}; model: {full_asr_prediction.get('error')}".strip("; ")

    return {
        "sample_id": sample_id,
        "label": label,
        "case_type": row.get("case_type", ""),
        "event_time_sec": None if event_time in ("", None) else safe_float(event_time),
        "audio_path": row.get("audio_path", ""),
        "source": row.get("source", ""),
        "full_asr_text": full_asr_text,
        "full_asr_raw_text": asr.get("raw_text", ""),
        "full_asr_backend": asr.get("backend", ""),
        "full_asr_model_name": asr.get("model_name", ""),
        "full_asr_text_score": full_asr_prediction.get("llm_score", 0.0),
        "full_asr_text_label": full_asr_prediction.get("final_label", ""),
        "full_asr_phishing_detected": full_asr_prediction.get("phishing_detected", False),
        "metadata_text": metadata_text,
        "metadata_text_score": metadata_prediction.get("llm_score", ""),
        "metadata_text_label": metadata_prediction.get("final_label", ""),
        "metadata_phishing_detected": metadata_prediction.get("phishing_detected", ""),
        "elapsed_sec": round(time.time() - start, 2),
        "error": error,
    }


def summarize_record(record: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    label = normalize_label(record.get("label", 0))
    summary = {
        "sample_id": record.get("sample_id", ""),
        "label": label,
        "case_type": record.get("case_type", ""),
        "audio_path": record.get("audio_path", ""),
        "source": record.get("source", ""),
        "elapsed_sec": record.get("elapsed_sec", ""),
        "error": record.get("error", ""),
        "full_asr_backend": record.get("full_asr_backend", ""),
        "full_asr_text_score": record.get("full_asr_text_score", ""),
        "metadata_text_score": record.get("metadata_text_score", ""),
        "full_asr_text_preview": str(record.get("full_asr_text", ""))[:160],
        "metadata_text_preview": str(record.get("metadata_text", ""))[:160],
    }
    for field in SCORE_FIELDS:
        score = record.get(field, "")
        if score == "":
            summary[f"{field}_prediction"] = ""
            summary[f"{field}_correct"] = ""
            summary[f"{field}_false_positive"] = ""
            summary[f"{field}_false_negative"] = ""
            continue
        prediction = 1 if safe_float(score) >= threshold else 0
        summary[f"{field}_prediction"] = prediction
        summary[f"{field}_correct"] = int(prediction == label)
        summary[f"{field}_false_positive"] = int(label == 0 and prediction == 1)
        summary[f"{field}_false_negative"] = int(label == 1 and prediction == 0)
    return summary


def metric_row(records: Sequence[Dict[str, Any]], score_field: str, threshold: float) -> Dict[str, Any]:
    rows = [record for record in records if record.get(score_field, "") != ""]
    y_true = [normalize_label(record.get("label", 0)) for record in rows]
    y_pred = [1 if safe_float(record.get(score_field)) >= threshold else 0 for record in rows]
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    normal_count = sum(1 for truth in y_true if truth == 0)
    fraud_count = sum(1 for truth in y_true if truth == 1)
    scores = [safe_float(record.get(score_field)) for record in rows]
    normal_scores = [safe_float(record.get(score_field)) for record in rows if normalize_label(record.get("label")) == 0]
    fraud_scores = [safe_float(record.get(score_field)) for record in rows if normalize_label(record.get("label")) == 1]
    return {
        "score_field": score_field,
        "threshold": threshold,
        "samples": len(rows),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "normal_samples": normal_count,
        "fraud_samples": fraud_count,
        "normal_false_positives": fp,
        "fraud_false_negatives": fn,
        "normal_fp_rate": round(fp / normal_count, 4) if normal_count else 0.0,
        "fraud_recall": round(tp / fraud_count, 4) if fraud_count else 0.0,
        "mean_score": round(statistics.mean(scores), 2) if scores else 0.0,
        "normal_mean_score": round(statistics.mean(normal_scores), 2) if normal_scores else 0.0,
        "fraud_mean_score": round(statistics.mean(fraud_scores), 2) if fraud_scores else 0.0,
    }


def metric_rows(records: Sequence[Dict[str, Any]], thresholds: Iterable[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        for score_field in SCORE_FIELDS:
            rows.append(metric_row(records, score_field, threshold))
    return rows


def case_metric_rows(records: Sequence[Dict[str, Any]], thresholds: Iterable[float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    case_types = sorted({str(record.get("case_type", "")) for record in records})
    for threshold in thresholds:
        for score_field in SCORE_FIELDS:
            for case_type in case_types:
                group = [record for record in records if str(record.get("case_type", "")) == case_type]
                row = metric_row(group, score_field, threshold)
                row["case_type"] = case_type
                rows.append(row)
    return rows


def run_predictions(args: argparse.Namespace) -> List[Dict[str, Any]]:
    logging.getLogger().setLevel(logging.ERROR)
    for name in ["speaker_analysis.asr_backend", "ChineseBERTModel.ensemble_utils"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    metadata_rows = load_metadata(args.metadata)
    selected = filter_rows(
        metadata_rows,
        sample_ids=set(args.sample_id or []),
        case_types=set(args.case_type or []),
        limit=args.limit,
    )
    existing = load_existing_predictions(args.predictions) if args.resume else {}
    records: Dict[str, Dict[str, Any]] = dict(existing)

    for index, row in enumerate(selected, start=1):
        sample_id = row.get("sample_id", "")
        if args.resume and sample_id in records and not records[sample_id].get("error"):
            print(f"[{index:02d}/{len(selected)}] {sample_id} skipped (resume)")
            continue

        record = build_record(row)
        records[sample_id] = record
        current_records = list(records.values())
        write_json_records(args.predictions, current_records)
        write_csv_rows(args.summary, [summarize_record(item, args.threshold) for item in current_records])
        write_csv_rows(args.metrics, metric_rows(current_records, args.metric_threshold))
        write_csv_rows(args.case_metrics, case_metric_rows(current_records, args.metric_threshold))
        print(
            f"[{index:02d}/{len(selected)}] {sample_id} {row.get('case_type', '')} "
            f"full_asr={record.get('full_asr_text_score')} "
            f"metadata={record.get('metadata_text_score')} "
            f"elapsed={record.get('elapsed_sec')}s error={bool(record.get('error'))}",
            flush=True,
        )

    final_records = list(records.values())
    write_json_records(args.predictions, final_records)
    write_csv_rows(args.summary, [summarize_record(item, args.threshold) for item in final_records])
    write_csv_rows(args.metrics, metric_rows(final_records, args.metric_threshold))
    write_csv_rows(args.case_metrics, case_metric_rows(final_records, args.metric_threshold))
    return final_records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, help="Run output directory. Defaults to evaluation/predictions/<run-name>.")
    parser.add_argument("--run-name", default="full_text_posthoc_baseline", help="Stable run folder name.")
    parser.add_argument("--predictions", type=Path, help="Prediction JSON output path.")
    parser.add_argument("--summary", type=Path, help="Prediction summary CSV output path.")
    parser.add_argument("--metrics", type=Path, help="Metrics CSV output path.")
    parser.add_argument("--case-metrics", type=Path, help="Case-type metrics CSV output path.")
    parser.add_argument("--threshold", type=float, default=70.0, help="Decision threshold used in the summary CSV.")
    parser.add_argument("--metric-threshold", type=float, action="append", default=[50.0, 70.0])
    parser.add_argument("--sample-id", action="append", help="Run only this sample_id. Repeatable.")
    parser.add_argument("--case-type", action="append", help="Run only this case_type. Repeatable.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.metadata = args.metadata.resolve()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / args.run_name
    args.output_dir = args.output_dir.resolve()
    if args.predictions is None:
        args.predictions = args.output_dir / "full_text_predictions.json"
    if args.summary is None:
        args.summary = args.output_dir / "full_text_predictions_summary.csv"
    if args.metrics is None:
        args.metrics = args.output_dir / "full_text_metrics.csv"
    if args.case_metrics is None:
        args.case_metrics = args.output_dir / "full_text_case_metrics.csv"
    args.predictions = args.predictions.resolve()
    args.summary = args.summary.resolve()
    args.metrics = args.metrics.resolve()
    args.case_metrics = args.case_metrics.resolve()
    records = run_predictions(args)
    print(f"wrote_predictions={args.predictions}")
    print(f"wrote_summary={args.summary}")
    print(f"wrote_metrics={args.metrics}")
    print(f"wrote_case_metrics={args.case_metrics}")
    print(f"records={len(records)}")


if __name__ == "__main__":
    main()
