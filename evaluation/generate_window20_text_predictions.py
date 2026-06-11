#!/usr/bin/env python3
"""Generate 20-second window text-risk predictions.

This script tests baseline ChineseBERT text scoring on simulated in-event
20-second ASR windows with the same rolling-tail context schema used by the
streaming baseline. It mirrors generate_full_text_predictions.py by writing
JSON records, a per-sample CSV summary, compact metrics, and case-type metrics.
The main decision field is max_window_text_score, because a streaming alert can
fire before the final window.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from pydub import AudioSegment


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

DEFAULT_METADATA = PROJECT_ROOT / "test_samples" / "metadata.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluation" / "predictions"
DEFAULT_SEGMENT_CACHE_ROOT = PROJECT_ROOT / ".cache" / "window20_text_segments"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
SCORE_FIELDS = (
    "max_window_text_score",
    "final_window_text_score",
    "mean_window_text_score",
)


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


def format_number_for_name(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_project_cache_path(path: Path) -> Path:
    resolved = path.resolve()
    project_cache = (PROJECT_ROOT / ".cache").resolve()
    try:
        resolved.relative_to(project_cache)
    except ValueError as exc:
        raise ValueError(f"--segment-cache-root must be under {project_cache}") from exc
    return resolved


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
        "description": "20-second simulated streaming text-risk predictions from baseline ChineseBERT.",
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


def transcribe_window(segment_path: Path) -> Dict[str, Any]:
    from speaker_analysis.asr_backend import transcribe_segment_with_metadata

    result = transcribe_segment_with_metadata(segment_path.as_posix())
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


def window_starts(duration_ms: int, window_ms: int, step_ms: int) -> List[int]:
    starts = list(range(0, max(duration_ms - 1, 0), step_ms))
    if starts and starts[-1] + window_ms < duration_ms:
        starts.append(max(0, duration_ms - window_ms))
    elif not starts:
        starts = [0]
    return starts


def build_window_records(
    audio_path: Path,
    sample_id: str,
    window_seconds: float,
    step_seconds: float,
    segment_cache_root: Path,
) -> tuple[List[Dict[str, Any]], str, str]:
    from streaming_analysis.risk_scoring import combine_baseline_text_scores
    from streaming_analysis.window_pipeline import build_rolling_context_fields

    audio = AudioSegment.from_file(audio_path.as_posix())
    duration_ms = len(audio)
    window_ms = int(window_seconds * 1000)
    step_ms = int(step_seconds * 1000)
    cache_dir = segment_cache_root / sample_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    windows: List[Dict[str, Any]] = []
    transcript_parts: List[str] = []
    backend = ""
    model_name = ""
    starts = window_starts(duration_ms, window_ms, step_ms)

    for index, start_ms in enumerate(starts):
        end_ms = min(start_ms + window_ms, duration_ms)
        if end_ms <= start_ms:
            continue

        segment_path = cache_dir / f"window_{index:04d}.wav"
        audio[start_ms:end_ms].export(segment_path.as_posix(), format="wav")

        start_sec = round(start_ms / 1000.0, 2)
        end_sec = round(end_ms / 1000.0, 2)
        try:
            asr = transcribe_window(segment_path)
        except Exception as exc:
            asr = {"text": "", "raw_text": "", "backend": "", "model_name": "", "error": str(exc)}

        text = asr.get("text", "")
        rolling_context = build_rolling_context_fields(
            previous_window_texts=transcript_parts[-3:],
            current_window_text=text,
        )
        prediction = predict_text(text)
        raw_text_score = safe_float(prediction.get("llm_score", 0.0))
        context_prediction: Dict[str, Any] = {}
        context_text_score = 0.0
        rolling_context_text = rolling_context["rolling_context_text"]
        if text.strip() and rolling_context_text:
            if rolling_context_text == text.strip():
                context_prediction = dict(prediction)
                context_text_score = raw_text_score
            else:
                context_prediction = predict_text(rolling_context_text)
                context_text_score = safe_float(context_prediction.get("llm_score", 0.0))
        text_score = combine_baseline_text_scores(raw_text_score, context_text_score) if text.strip() else 0.0
        if text.strip():
            transcript_parts.append(text.strip())

        backend = backend or asr.get("backend", "")
        model_name = model_name or asr.get("model_name", "")
        windows.append({
            "index": index,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "text": text,
            "current_window_text": rolling_context["current_window_text"],
            "recent_context_text": rolling_context["recent_context_text"],
            "rolling_context_text": rolling_context["rolling_context_text"],
            "raw_text": asr.get("raw_text", ""),
            "raw_text_score": raw_text_score,
            "raw_window_text_score": raw_text_score,
            "context_text_score": context_text_score,
            "text_score": text_score,
            "text_label": prediction.get("final_label", ""),
            "phishing_detected": prediction.get("phishing_detected", False),
            "asr_error": asr.get("error", ""),
            "model_error": prediction.get("error", "") if text.strip() else "",
            "context_model_error": context_prediction.get("error", "") if text.strip() else "",
            "segment_cache_path": segment_path.relative_to(PROJECT_ROOT).as_posix(),
        })

    return windows, " ".join(transcript_parts).strip(), f"{backend}|{model_name}".strip("|")


def build_record(row: Dict[str, str], args: argparse.Namespace) -> Dict[str, Any]:
    sample_id = row.get("sample_id", "").strip()
    audio_path = resolve_project_path(row.get("audio_path", ""))
    label = normalize_label(row.get("label", 0))
    event_time = row.get("event_time_sec", "")
    start = time.time()
    error = ""

    try:
        windows, full_transcript, asr_backend_model = build_window_records(
            audio_path=audio_path,
            sample_id=sample_id,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            segment_cache_root=args.segment_cache_root,
        )
    except Exception as exc:
        windows = []
        full_transcript = ""
        asr_backend_model = ""
        error = str(exc)

    scored_windows = [window for window in windows if window.get("text_score", "") != ""]
    max_window = max(scored_windows, key=lambda item: safe_float(item.get("text_score")), default={})
    final_window = scored_windows[-1] if scored_windows else {}
    high_windows = [
        window for window in scored_windows
        if safe_float(window.get("text_score")) >= args.threshold
    ]
    scores = [safe_float(window.get("text_score")) for window in scored_windows]
    raw_scores = [safe_float(window.get("raw_text_score")) for window in scored_windows]
    context_scores = [safe_float(window.get("context_text_score")) for window in scored_windows]
    asr_errors = [window.get("asr_error") for window in windows if window.get("asr_error")]
    model_errors = [window.get("model_error") for window in windows if window.get("model_error")]
    first_alert_end_sec = high_windows[0].get("end_sec", "") if high_windows else ""
    event_time_sec = None if event_time in ("", None) else safe_float(event_time)
    detection_delay_sec = ""
    early_warning_lead_time_sec = ""
    if first_alert_end_sec != "" and event_time_sec is not None:
        detection_delay_sec = round(safe_float(first_alert_end_sec) - event_time_sec, 2)
        early_warning_lead_time_sec = round(event_time_sec - safe_float(first_alert_end_sec), 2)

    if asr_errors:
        error = f"{error}; ASR windows with errors: {len(asr_errors)}".strip("; ")
    if model_errors:
        error = f"{error}; text windows with model errors: {len(model_errors)}".strip("; ")

    return {
        "sample_id": sample_id,
        "label": label,
        "case_type": row.get("case_type", ""),
        "event_time_sec": event_time_sec,
        "audio_path": row.get("audio_path", ""),
        "source": row.get("source", ""),
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "scoring_mode": "baseline",
        "decision_threshold": args.threshold,
        "windows": windows,
        "window_count": len(windows),
        "full_transcript_from_windows": full_transcript,
        "asr_backend_model": asr_backend_model,
        "max_window_raw_text_score": round(safe_float(max_window.get("raw_text_score")), 2),
        "max_window_context_text_score": round(safe_float(max_window.get("context_text_score")), 2),
        "max_window_text_score": round(safe_float(max_window.get("text_score")), 2),
        "max_window_index": max_window.get("index", ""),
        "max_window_start_sec": max_window.get("start_sec", ""),
        "max_window_end_sec": max_window.get("end_sec", ""),
        "max_window_text": max_window.get("text", ""),
        "final_window_raw_text_score": round(safe_float(final_window.get("raw_text_score")), 2),
        "final_window_context_text_score": round(safe_float(final_window.get("context_text_score")), 2),
        "final_window_text_score": round(safe_float(final_window.get("text_score")), 2),
        "final_window_index": final_window.get("index", ""),
        "final_window_end_sec": final_window.get("end_sec", ""),
        "mean_window_raw_text_score": round(statistics.mean(raw_scores), 2) if raw_scores else 0.0,
        "mean_window_context_text_score": round(statistics.mean(context_scores), 2) if context_scores else 0.0,
        "mean_window_text_score": round(statistics.mean(scores), 2) if scores else 0.0,
        "first_alert_end_sec": first_alert_end_sec,
        "detection_delay_sec": detection_delay_sec,
        "early_warning_lead_time_sec": early_warning_lead_time_sec,
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
        "window_seconds": record.get("window_seconds", ""),
        "step_seconds": record.get("step_seconds", ""),
        "scoring_mode": record.get("scoring_mode", ""),
        "window_count": record.get("window_count", ""),
        "elapsed_sec": record.get("elapsed_sec", ""),
        "error": record.get("error", ""),
        "asr_backend_model": record.get("asr_backend_model", ""),
        "max_window_raw_text_score": record.get("max_window_raw_text_score", ""),
        "max_window_context_text_score": record.get("max_window_context_text_score", ""),
        "max_window_text_score": record.get("max_window_text_score", ""),
        "max_window_index": record.get("max_window_index", ""),
        "max_window_start_sec": record.get("max_window_start_sec", ""),
        "max_window_end_sec": record.get("max_window_end_sec", ""),
        "final_window_raw_text_score": record.get("final_window_raw_text_score", ""),
        "final_window_context_text_score": record.get("final_window_context_text_score", ""),
        "final_window_text_score": record.get("final_window_text_score", ""),
        "final_window_end_sec": record.get("final_window_end_sec", ""),
        "mean_window_raw_text_score": record.get("mean_window_raw_text_score", ""),
        "mean_window_context_text_score": record.get("mean_window_context_text_score", ""),
        "mean_window_text_score": record.get("mean_window_text_score", ""),
        "first_alert_end_sec": record.get("first_alert_end_sec", ""),
        "detection_delay_sec": record.get("detection_delay_sec", ""),
        "early_warning_lead_time_sec": record.get("early_warning_lead_time_sec", ""),
        "max_window_text_preview": str(record.get("max_window_text", ""))[:160],
        "transcript_preview": str(record.get("full_transcript_from_windows", ""))[:160],
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

        record = build_record(row, args)
        records[sample_id] = record
        current_records = list(records.values())
        write_json_records(args.predictions, current_records)
        write_csv_rows(args.summary, [summarize_record(item, args.threshold) for item in current_records])
        write_csv_rows(args.metrics, metric_rows(current_records, args.metric_threshold))
        write_csv_rows(args.case_metrics, case_metric_rows(current_records, args.metric_threshold))
        print(
            f"[{index:02d}/{len(selected)}] {sample_id} {row.get('case_type', '')} "
            f"max_window={record.get('max_window_text_score')} "
            f"final_window={record.get('final_window_text_score')} "
            f"windows={record.get('window_count')} "
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
    parser.add_argument("--run-name", help="Stable run folder name used when --output-dir is omitted.")
    parser.add_argument("--predictions", type=Path, help="Prediction JSON output path.")
    parser.add_argument("--summary", type=Path, help="Prediction summary CSV output path.")
    parser.add_argument("--metrics", type=Path, help="Metrics CSV output path.")
    parser.add_argument("--case-metrics", type=Path, help="Case-type metrics CSV output path.")
    parser.add_argument("--segment-cache-root", type=Path, default=DEFAULT_SEGMENT_CACHE_ROOT)
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--step-seconds", type=float, default=5.0)
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
    args.segment_cache_root = ensure_project_cache_path(args.segment_cache_root)
    if args.output_dir is None:
        run_name = args.run_name or (
            f"window_text_w{format_number_for_name(args.window_seconds)}_"
            f"s{format_number_for_name(args.step_seconds)}_baseline"
        )
        args.output_dir = DEFAULT_OUTPUT_ROOT / run_name
    args.output_dir = args.output_dir.resolve()
    if args.predictions is None:
        args.predictions = args.output_dir / "window_text_predictions.json"
    if args.summary is None:
        args.summary = args.output_dir / "window_text_predictions_summary.csv"
    if args.metrics is None:
        args.metrics = args.output_dir / "window_text_metrics.csv"
    if args.case_metrics is None:
        args.case_metrics = args.output_dir / "window_text_case_metrics.csv"
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
