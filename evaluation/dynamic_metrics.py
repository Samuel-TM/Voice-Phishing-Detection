# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_DIR / ".cache" / "evaluation_reports"
DEFAULT_ALERT_THRESHOLD = 70.0


@dataclass(frozen=True)
class MetricThresholds:
    final_f1: float = 0.80
    mean_lead_time_sec: float = 5.0
    mean_detection_delay_sec: float = 15.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value else 0
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "true", "fraud", "phishing", "positive", "risk"} else 0


def load_prediction_records(path: Path) -> List[Dict[str, Any]]:
    """Load JSON/JSONL predictions produced by the streaming pipeline."""
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    raise ValueError("Prediction JSON must be a list or an object with a records list.")


def score_to_label(score: float, threshold: float) -> int:
    return 1 if score >= threshold else 0


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    total = max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def first_crossing_time(timeline: Sequence[Dict[str, Any]], score_key: str, threshold: float) -> Optional[float]:
    for point in timeline:
        if safe_float(point.get(score_key)) >= threshold:
            return safe_float(point.get("end_sec", point.get("start_sec")))
    return None


def peak_time(timeline: Sequence[Dict[str, Any]], score_key: str) -> Optional[float]:
    if not timeline:
        return None
    peak = max(timeline, key=lambda point: safe_float(point.get(score_key)))
    return safe_float(peak.get("end_sec", peak.get("start_sec")))


def final_score(timeline: Sequence[Dict[str, Any]], score_key: str) -> float:
    if not timeline:
        return 0.0
    return safe_float(timeline[-1].get(score_key))


def get_timeline(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    timeline = record.get("timeline", [])
    if not isinstance(timeline, list):
        raise ValueError(f"Record {record.get('sample_id', '<unknown>')} has invalid timeline.")
    return timeline


def evaluate_record(
    record: Dict[str, Any],
    score_key: str,
    alert_threshold: float,
) -> Dict[str, Any]:
    timeline = get_timeline(record)
    label = normalize_label(record.get("label", record.get("is_fraud", 0)))
    event_time = record.get("event_time_sec", record.get("fraud_instruction_sec"))
    event_time_float = None if event_time in (None, "") else safe_float(event_time)
    alert_time = first_crossing_time(timeline, score_key, alert_threshold)
    sample_final_score = final_score(timeline, score_key)
    prediction = score_to_label(sample_final_score, alert_threshold)
    max_score = max((safe_float(point.get(score_key)) for point in timeline), default=0.0)

    lead_time = None
    detection_delay = None
    if label == 1 and event_time_float is not None and alert_time is not None:
        lead_time = event_time_float - alert_time
        detection_delay = max(alert_time - event_time_float, 0.0)
    elif label == 1 and event_time_float is not None:
        detection_delay = None

    return {
        "sample_id": record.get("sample_id", record.get("id", "")),
        "label": label,
        "prediction": prediction,
        "score_key": score_key,
        "final_score": round(sample_final_score, 4),
        "max_score": round(max_score, 4),
        "alert_time_sec": alert_time,
        "event_time_sec": event_time_float,
        "early_warning_lead_time_sec": lead_time,
        "detection_delay_sec": detection_delay,
        "peak_risk_time_sec": peak_time(timeline, score_key),
        "window_seconds": record.get("window_seconds"),
        "step_seconds": record.get("step_seconds"),
    }


def summarize_dynamic_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [safe_int(row["label"]) for row in rows]
    y_pred = [safe_int(row["prediction"]) for row in rows]
    metrics = classification_metrics(y_true, y_pred)
    fraud_rows = [row for row in rows if safe_int(row["label"]) == 1]
    alert_rows = [row for row in fraud_rows if row["alert_time_sec"] is not None]
    lead_times = [
        safe_float(row["early_warning_lead_time_sec"])
        for row in fraud_rows
        if row["early_warning_lead_time_sec"] is not None
    ]
    delays = [
        safe_float(row["detection_delay_sec"])
        for row in fraud_rows
        if row["detection_delay_sec"] is not None
    ]
    return {
        **{key: round(value, 4) for key, value in metrics.items()},
        "samples": len(rows),
        "fraud_samples": len(fraud_rows),
        "fraud_alert_rate": round(len(alert_rows) / max(len(fraud_rows), 1), 4),
        "mean_time_to_alert_sec": round(mean([safe_float(row["alert_time_sec"]) for row in alert_rows]), 4)
        if alert_rows else None,
        "mean_early_warning_lead_time_sec": round(mean(lead_times), 4) if lead_times else None,
        "mean_detection_delay_sec": round(mean(delays), 4) if delays else None,
    }


def build_ablation_report(
    records: Sequence[Dict[str, Any]],
    alert_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    score_keys = [
        ("text_only", "text_score"),
        ("voice_only", "voice_score"),
        ("fusion_without_smoothing", "fused_score"),
        ("fusion_with_smoothing", "smoothed_score"),
    ]
    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for variant, score_key in score_keys:
        variant_rows = []
        for record in records:
            row = evaluate_record(record, score_key, alert_threshold)
            row["variant"] = variant
            variant_rows.append(row)
        detail_rows.extend(variant_rows)
        summary = summarize_dynamic_metrics(variant_rows)
        summary["variant"] = variant
        summary_rows.append(summary)

    return detail_rows, summary_rows


def build_window_report(records: Sequence[Dict[str, Any]], alert_threshold: float) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for record in records:
        key = (record.get("window_seconds"), record.get("step_seconds"))
        grouped.setdefault(key, []).append(record)

    rows = []
    for (window_seconds, step_seconds), group in sorted(grouped.items(), key=lambda item: str(item[0])):
        evaluated = [evaluate_record(record, "smoothed_score", alert_threshold) for record in group]
        summary = summarize_dynamic_metrics(evaluated)
        summary["window_seconds"] = window_seconds
        summary["step_seconds"] = step_seconds
        rows.append(summary)
    return rows


def retraining_recommendation(summary_rows: Sequence[Dict[str, Any]], thresholds: MetricThresholds) -> Dict[str, Any]:
    by_variant = {row["variant"]: row for row in summary_rows}
    text_f1 = safe_float(by_variant.get("text_only", {}).get("f1"))
    voice_f1 = safe_float(by_variant.get("voice_only", {}).get("f1"))
    fusion = by_variant.get("fusion_with_smoothing", {})
    fusion_f1 = safe_float(fusion.get("f1"))
    mean_lead = fusion.get("mean_early_warning_lead_time_sec")
    mean_delay = fusion.get("mean_detection_delay_sec")

    weak_modules = []
    if text_f1 < thresholds.final_f1:
        weak_modules.append("text")
    if voice_f1 < thresholds.final_f1:
        weak_modules.append("voice")
    if fusion_f1 < thresholds.final_f1:
        weak_modules.append("fusion")
    if mean_lead is not None and safe_float(mean_lead) < thresholds.mean_lead_time_sec:
        weak_modules.append("early_warning")
    if mean_delay is not None and safe_float(mean_delay) > thresholds.mean_detection_delay_sec:
        weak_modules.append("latency")

    if not weak_modules:
        decision = "No retraining required. Tune fusion thresholds and smoothing only if needed."
    elif "text" in weak_modules and "voice" in weak_modules:
        decision = "Retrain one module at a time. Start with the module that has lower F1, then re-evaluate fusion."
    elif "text" in weak_modules:
        decision = "Retrain or fine-tune the text model first, then re-run dynamic fusion evaluation."
    elif "voice" in weak_modules:
        decision = "Retrain the audio model first, then re-run dynamic fusion evaluation."
    else:
        decision = "Do not retrain models yet. Tune fusion weights, alert threshold, smoothing, and window parameters."

    return {
        "triggered": bool(weak_modules),
        "weak_modules": weak_modules,
        "decision": decision,
        "thresholds": {
            "final_f1": thresholds.final_f1,
            "mean_lead_time_sec": thresholds.mean_lead_time_sec,
            "mean_detection_delay_sec": thresholds.mean_detection_delay_sec,
        },
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_evaluation(
    prediction_path: Path,
    output_dir: Path,
    alert_threshold: float,
    thresholds: MetricThresholds,
) -> Dict[str, Any]:
    records = load_prediction_records(prediction_path)
    detail_rows, ablation_summary = build_ablation_report(records, alert_threshold)
    window_summary = build_window_report(records, alert_threshold)
    recommendation = retraining_recommendation(ablation_summary, thresholds)

    report = {
        "prediction_path": prediction_path.as_posix(),
        "alert_threshold": alert_threshold,
        "ablation_summary": ablation_summary,
        "window_parameter_summary": window_summary,
        "retraining_recommendation": recommendation,
    }

    write_csv(output_dir / "dynamic_eval_detail.csv", detail_rows)
    write_csv(output_dir / "dynamic_eval_ablation_summary.csv", ablation_summary)
    write_csv(output_dir / "dynamic_eval_window_summary.csv", window_summary)
    write_json(output_dir / "dynamic_eval_report.json", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute thesis-oriented dynamic risk tracking metrics from timeline predictions."
    )
    parser.add_argument("--predictions", required=True, type=Path, help="JSON/JSONL file containing sample timelines.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Report output directory.")
    parser.add_argument("--alert-threshold", type=float, default=DEFAULT_ALERT_THRESHOLD)
    parser.add_argument("--min-final-f1", type=float, default=0.80)
    parser.add_argument("--min-mean-lead-time-sec", type=float, default=5.0)
    parser.add_argument("--max-mean-detection-delay-sec", type=float, default=15.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    thresholds = MetricThresholds(
        final_f1=args.min_final_f1,
        mean_lead_time_sec=args.min_mean_lead_time_sec,
        mean_detection_delay_sec=args.max_mean_detection_delay_sec,
    )
    report = run_evaluation(
        prediction_path=args.predictions,
        output_dir=args.output_dir,
        alert_threshold=args.alert_threshold,
        thresholds=thresholds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
