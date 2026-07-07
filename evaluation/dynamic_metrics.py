# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[1]
if PROJECT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_DIR.as_posix())
DEFAULT_REPORT_ROOT = PROJECT_DIR / "evaluation" / "reports"
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


def parse_float_list(value: Optional[str]) -> List[float]:
    if not value:
        return []
    numbers: List[float] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            numbers.append(float(item))
    return numbers


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
        "case_type": record.get("case_type", ""),
        "source": record.get("source", ""),
        "scoring_mode": record.get("scoring_mode", ""),
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
    normal_rows = [row for row in rows if safe_int(row["label"]) == 0]
    alert_rows = [row for row in fraud_rows if row["alert_time_sec"] is not None]
    normal_alert_rows = [row for row in normal_rows if row["alert_time_sec"] is not None]
    normal_final_fp_rows = [row for row in normal_rows if safe_int(row["prediction"]) == 1]
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
        "normal_samples": len(normal_rows),
        "fraud_alert_rate": round(len(alert_rows) / max(len(fraud_rows), 1), 4),
        "fraud_alert_recall": round(len(alert_rows) / max(len(fraud_rows), 1), 4),
        "normal_alert_false_positive_rate": round(len(normal_alert_rows) / max(len(normal_rows), 1), 4),
        "normal_final_false_positive_rate": round(len(normal_final_fp_rows) / max(len(normal_rows), 1), 4),
        "normal_alert_false_positives": len(normal_alert_rows),
        "normal_final_false_positives": len(normal_final_fp_rows),
        "mean_time_to_alert_sec": round(mean([safe_float(row["alert_time_sec"]) for row in alert_rows]), 4)
        if alert_rows else None,
        "mean_early_warning_lead_time_sec": round(mean(lead_times), 4) if lead_times else None,
        "mean_detection_delay_sec": round(mean(delays), 4) if delays else None,
    }


def build_ablation_report(
    records: Sequence[Dict[str, Any]],
    alert_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    score_keys = score_key_variants(records)
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


def has_timeline_score(records: Sequence[Dict[str, Any]], score_key: str) -> bool:
    for record in records:
        for point in get_timeline(record):
            if score_key in point:
                return True
    return False


def score_key_variants(records: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    variants = [
        ("text_only", "text_score"),
        ("voice_only", "voice_score"),
        ("fusion_without_smoothing", "fused_score"),
        ("fusion_with_smoothing", "smoothed_score"),
    ]
    if has_timeline_score(records, "learned_late_fusion_score"):
        variants.append(("learned_late_fusion", "learned_late_fusion_score"))
    if has_timeline_score(records, "causal_late_fusion_v2_score"):
        variants.append(("causal_learned_late_fusion_v2", "causal_late_fusion_v2_score"))
    return variants


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


def build_case_type_report(records: Sequence[Dict[str, Any]], alert_threshold: float) -> List[Dict[str, Any]]:
    score_keys = score_key_variants(records)
    rows: List[Dict[str, Any]] = []
    case_types = sorted({str(record.get("case_type") or "unknown") for record in records})

    for case_type in case_types:
        group = [record for record in records if str(record.get("case_type") or "unknown") == case_type]
        for variant, score_key in score_keys:
            evaluated = [evaluate_record(record, score_key, alert_threshold) for record in group]
            summary = summarize_dynamic_metrics(evaluated)
            summary["case_type"] = case_type
            summary["variant"] = variant
            rows.append(summary)
    return rows


def infer_record_scoring_mode(record: Dict[str, Any], fallback: str = "gated_v1") -> str:
    mode = str(record.get("scoring_mode") or "").strip()
    if mode:
        return mode
    timeline = get_timeline(record)
    for point in timeline:
        mode = str(point.get("scoring_mode") or "").strip()
        if mode:
            return mode
    return fallback


def rescore_record(
    record: Dict[str, Any],
    text_weight: float,
    smoothing_previous_weight: float,
    scoring_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Recompute fusion and smoothing from cached per-window text/voice outputs."""
    from streaming_analysis.risk_scoring import RiskScoringState, score_window

    output = copy.deepcopy(record)
    case_type = str(output.get("case_type") or "")
    mode = scoring_mode or infer_record_scoring_mode(output)
    voice_weight = round(1.0 - text_weight, 4)
    state = RiskScoringState()
    rescored_timeline: List[Dict[str, Any]] = []

    for point in get_timeline(output):
        raw_text_score = point.get("raw_text_score", point.get("original_text_score", point.get("text_score", 0.0)))
        context_text_score = point.get("context_text_score", raw_text_score)
        scoring = score_window(
            raw_text_score=raw_text_score,
            context_text_score=context_text_score,
            voice_score=point.get("voice_score", 0.0),
            text=str(point.get("text") or ""),
            state=state,
            scoring_mode=mode,
            case_type=str(point.get("case_type") or case_type),
            text_weight=text_weight,
            voice_weight=voice_weight,
            smoothing_previous_weight=smoothing_previous_weight,
        )
        new_point = dict(point)
        new_point.update(scoring)
        new_point["weights"] = {
            "text": round(text_weight, 4),
            "voice": round(voice_weight, 4),
            "smoothing_previous": round(smoothing_previous_weight, 4),
            "smoothing_current": round(1.0 - smoothing_previous_weight, 4),
        }
        rescored_timeline.append(new_point)

    output["timeline"] = rescored_timeline
    output["scoring_mode"] = mode
    output["weights"] = {
        "text": round(text_weight, 4),
        "voice": round(voice_weight, 4),
        "smoothing_previous": round(smoothing_previous_weight, 4),
        "smoothing_current": round(1.0 - smoothing_previous_weight, 4),
    }
    output["final_score"] = round(final_score(rescored_timeline, "smoothed_score"), 4)
    highest = max(rescored_timeline, key=lambda point: safe_float(point.get("smoothed_score")), default=None)
    output["max_score"] = round(safe_float(highest.get("smoothed_score")), 4) if highest else 0.0
    output["highest_risk_window"] = highest
    output["rescore_source"] = "dynamic_metrics_offline"
    return output


def rescore_records(
    records: Sequence[Dict[str, Any]],
    text_weight: float,
    smoothing_previous_weight: float,
    scoring_mode: Optional[str],
) -> List[Dict[str, Any]]:
    return [
        rescore_record(
            record,
            text_weight=text_weight,
            smoothing_previous_weight=smoothing_previous_weight,
            scoring_mode=scoring_mode,
        )
        for record in records
    ]


def build_threshold_sweep_report(
    records: Sequence[Dict[str, Any]],
    alert_thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for threshold in alert_thresholds:
        _, summary_rows = build_ablation_report(records, threshold)
        for row in summary_rows:
            enriched = dict(row)
            enriched["alert_threshold"] = threshold
            rows.append(enriched)
    return rows


def build_fusion_smoothing_sweep_report(
    records: Sequence[Dict[str, Any]],
    text_weights: Sequence[float],
    smoothing_weights: Sequence[float],
    alert_threshold: float,
    scoring_mode: Optional[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []

    for text_weight in text_weights:
        for smoothing_weight in smoothing_weights:
            rescored = rescore_records(
                records,
                text_weight=text_weight,
                smoothing_previous_weight=smoothing_weight,
                scoring_mode=scoring_mode,
            )
            evaluated = [evaluate_record(record, "smoothed_score", alert_threshold) for record in rescored]
            summary = summarize_dynamic_metrics(evaluated)
            summary.update({
                "variant": "fusion_with_smoothing",
                "text_weight": round(text_weight, 4),
                "voice_weight": round(1.0 - text_weight, 4),
                "smoothing_previous_weight": round(smoothing_weight, 4),
                "smoothing_current_weight": round(1.0 - smoothing_weight, 4),
                "alert_threshold": alert_threshold,
                "scoring_mode": scoring_mode or "from_records",
            })
            summary_rows.append(summary)

            for case_summary in build_case_type_report(rescored, alert_threshold):
                if case_summary.get("variant") != "fusion_with_smoothing":
                    continue
                case_summary.update({
                    "text_weight": round(text_weight, 4),
                    "voice_weight": round(1.0 - text_weight, 4),
                    "smoothing_previous_weight": round(smoothing_weight, 4),
                    "smoothing_current_weight": round(1.0 - smoothing_weight, 4),
                    "alert_threshold": alert_threshold,
                    "scoring_mode": scoring_mode or "from_records",
                })
                case_rows.append(case_summary)

    return summary_rows, case_rows


def build_high_raw_unalerted_report(
    records: Sequence[Dict[str, Any]],
    raw_threshold: float = 70.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        if normalize_label(record.get("label", record.get("is_fraud", 0))) != 0:
            continue
        for point in get_timeline(record):
            raw_score = safe_float(point.get("raw_window_text_score", point.get("text_score", 0.0)))
            decision = str(point.get("alert_decision", ""))
            if raw_score < raw_threshold or decision in {"High Risk", "Critical"}:
                continue
            rows.append({
                "sample_id": record.get("sample_id", record.get("id", "")),
                "case_type": record.get("case_type", ""),
                "source": record.get("source", ""),
                "index": point.get("index", ""),
                "start_sec": point.get("start_sec", ""),
                "end_sec": point.get("end_sec", ""),
                "raw_window_text_score": round(raw_score, 2),
                "short_context_text_score": round(safe_float(point.get("short_context_text_score")), 2),
                "alert_decision": decision,
                "alert_reason": point.get("alert_reason", ""),
                "scam_stage": point.get("scam_stage", ""),
                "benign_finance_context": point.get("benign_finance_context", ""),
                "text": point.get("text", ""),
                "short_context_text": point.get("short_context_text", ""),
            })
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
    alert_thresholds: Sequence[float] = (),
    fusion_text_weights: Sequence[float] = (),
    smoothing_previous_weights: Sequence[float] = (),
    sweep_scoring_mode: Optional[str] = None,
) -> Dict[str, Any]:
    records = load_prediction_records(prediction_path)
    detail_rows, ablation_summary = build_ablation_report(records, alert_threshold)
    window_summary = build_window_report(records, alert_threshold)
    case_type_summary = build_case_type_report(records, alert_threshold)
    threshold_sweep = build_threshold_sweep_report(records, alert_thresholds) if alert_thresholds else []
    high_raw_unalerted = build_high_raw_unalerted_report(records, raw_threshold=alert_threshold)
    fusion_smoothing_sweep: List[Dict[str, Any]] = []
    fusion_smoothing_case_type_sweep: List[Dict[str, Any]] = []
    if fusion_text_weights and smoothing_previous_weights:
        fusion_smoothing_sweep, fusion_smoothing_case_type_sweep = build_fusion_smoothing_sweep_report(
            records=records,
            text_weights=fusion_text_weights,
            smoothing_weights=smoothing_previous_weights,
            alert_threshold=alert_threshold,
            scoring_mode=sweep_scoring_mode,
        )
    recommendation = retraining_recommendation(ablation_summary, thresholds)

    report = {
        "prediction_path": prediction_path.as_posix(),
        "alert_threshold": alert_threshold,
        "ablation_summary": ablation_summary,
        "case_type_summary": case_type_summary,
        "threshold_sweep": threshold_sweep,
        "fusion_smoothing_sweep": fusion_smoothing_sweep,
        "fusion_smoothing_case_type_sweep": fusion_smoothing_case_type_sweep,
        "window_parameter_summary": window_summary,
        "high_raw_unalerted_count": len(high_raw_unalerted),
        "retraining_recommendation": recommendation,
    }

    write_csv(output_dir / "dynamic_eval_detail.csv", detail_rows)
    write_csv(output_dir / "dynamic_eval_ablation_summary.csv", ablation_summary)
    write_csv(output_dir / "dynamic_eval_case_type_summary.csv", case_type_summary)
    write_csv(output_dir / "dynamic_eval_window_summary.csv", window_summary)
    write_csv(output_dir / "dynamic_eval_high_raw_unalerted.csv", high_raw_unalerted)
    if threshold_sweep:
        write_csv(output_dir / "dynamic_eval_threshold_sweep.csv", threshold_sweep)
    if fusion_smoothing_sweep:
        write_csv(output_dir / "dynamic_eval_fusion_smoothing_sweep.csv", fusion_smoothing_sweep)
    if fusion_smoothing_case_type_sweep:
        write_csv(output_dir / "dynamic_eval_fusion_smoothing_case_type_sweep.csv", fusion_smoothing_case_type_sweep)
    write_json(output_dir / "dynamic_eval_report.json", report)
    return report


def default_run_name(prediction_path: Path) -> str:
    parent = prediction_path.parent.name
    stem = prediction_path.stem
    if parent and parent not in {".", "predictions", "evaluation"}:
        return parent
    return f"{stem}_metrics"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute thesis-oriented dynamic risk tracking metrics from timeline predictions."
    )
    parser.add_argument("--predictions", required=True, type=Path, help="JSON/JSONL file containing sample timelines.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report output directory. Defaults to evaluation/reports/<run-name>.",
    )
    parser.add_argument(
        "--run-name",
        help="Stable run folder name used when --output-dir is omitted.",
    )
    parser.add_argument("--alert-threshold", type=float, default=DEFAULT_ALERT_THRESHOLD)
    parser.add_argument("--min-final-f1", type=float, default=0.80)
    parser.add_argument("--min-mean-lead-time-sec", type=float, default=5.0)
    parser.add_argument("--max-mean-detection-delay-sec", type=float, default=15.0)
    parser.add_argument("--alert-thresholds", help="Comma-separated alert threshold sweep, e.g. 50,60,70,80.")
    parser.add_argument("--fusion-text-weights", help="Comma-separated text weights, e.g. 0.7,0.8,0.9.")
    parser.add_argument("--smoothing-previous-weights", help="Comma-separated smoothing weights, e.g. 0.5,0.65,0.8.")
    parser.add_argument(
        "--sweep-scoring-mode",
        choices=["baseline", "gated_v1", "gated_v2", "gated_v3"],
        help="Force a scoring mode for offline fusion/smoothing sweep. Defaults to each record's mode.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.predictions = args.predictions.resolve()
    if args.output_dir is None:
        run_name = args.run_name or default_run_name(args.predictions)
        args.output_dir = DEFAULT_REPORT_ROOT / run_name
    args.output_dir = args.output_dir.resolve()
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
        alert_thresholds=parse_float_list(args.alert_thresholds),
        fusion_text_weights=parse_float_list(args.fusion_text_weights),
        smoothing_previous_weights=parse_float_list(args.smoothing_previous_weights),
        sweep_scoring_mode=args.sweep_scoring_mode,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
