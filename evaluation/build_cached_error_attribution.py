#!/usr/bin/env python3
"""Build sample-level error attribution from cached dynamic evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLD = 70.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_label(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"fraud", "risk", "phishing", "1", "true"}:
            return 1
        if value in {"normal", "benign", "0", "false"}:
            return 0
    return 1 if safe_float(value) >= 1 else 0


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return {str(record.get("sample_id")): record for record in records}


def max_window(record: dict[str, Any], score_key: str) -> dict[str, Any]:
    timeline = record.get("timeline") or []
    if not timeline:
        return {}
    return max(timeline, key=lambda point: safe_float(point.get(score_key)))


def score_stats(record: dict[str, Any], threshold: float) -> dict[str, Any]:
    timeline = record.get("timeline") or []
    text_peak = max_window(record, "text_score")
    voice_peak = max_window(record, "voice_score")
    fused_peak = max_window(record, "fused_score")
    smoothed_peak = max_window(record, "smoothed_score")

    def peak_score(point: dict[str, Any], key: str) -> float:
        return round(safe_float(point.get(key)), 2) if point else 0.0

    def peak_time(point: dict[str, Any]) -> float | None:
        if not point:
            return None
        return safe_float(point.get("start_sec"))

    text_alert = peak_score(text_peak, "text_score") >= threshold
    voice_alert = peak_score(voice_peak, "voice_score") >= threshold
    fused_alert = peak_score(fused_peak, "fused_score") >= threshold
    smoothed_alert = peak_score(smoothed_peak, "smoothed_score") >= threshold

    return {
        "window_count": len(timeline),
        "final_score": round(safe_float(record.get("final_score")), 2),
        "max_score": round(safe_float(record.get("max_score")), 2),
        "text_peak": peak_score(text_peak, "text_score"),
        "voice_peak": peak_score(voice_peak, "voice_score"),
        "fused_peak": peak_score(fused_peak, "fused_score"),
        "smoothed_peak": peak_score(smoothed_peak, "smoothed_score"),
        "text_peak_time_sec": peak_time(text_peak),
        "voice_peak_time_sec": peak_time(voice_peak),
        "fused_peak_time_sec": peak_time(fused_peak),
        "smoothed_peak_time_sec": peak_time(smoothed_peak),
        "text_alert": text_alert,
        "voice_alert": voice_alert,
        "fused_alert": fused_alert,
        "smoothed_alert": smoothed_alert,
    }


def bool_cell(value: bool) -> int:
    return 1 if value else 0


def attribution(label: int, baseline: dict[str, Any], gated: dict[str, Any], threshold: float) -> str:
    b = score_stats(baseline, threshold)
    g = score_stats(gated, threshold)

    if label == 0:
        if b["smoothed_alert"] and not g["smoothed_alert"]:
            return "fixed_by_gated_calibration"
        if g["smoothed_alert"]:
            if g["text_alert"] and g["voice_alert"]:
                return "residual_normal_fp_text_and_voice"
            if g["text_alert"]:
                return "residual_normal_fp_text"
            if g["voice_alert"]:
                return "residual_normal_fp_voice"
            if g["fused_alert"]:
                return "residual_normal_fp_fusion"
            return "residual_normal_fp_smoothing_or_latch"
        if b["smoothed_alert"]:
            return "baseline_only_normal_alert"
        return "normal_ok"

    if g["smoothed_alert"]:
        if b["smoothed_alert"]:
            return "fraud_detected_by_both"
        return "fraud_recovered_by_gated"
    if b["smoothed_alert"]:
        return "fraud_alert_lost_after_gating"
    if g["text_alert"] or b["text_alert"]:
        return "fraud_text_signal_not_converted_to_alert"
    if g["voice_alert"] or b["voice_alert"]:
        return "fraud_voice_signal_not_converted_to_alert"
    return "fraud_missed_by_all"


def build_rows(baseline_records: dict[str, dict[str, Any]], gated_records: dict[str, dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_ids = sorted(set(baseline_records) | set(gated_records))

    for sample_id in sample_ids:
        baseline = baseline_records.get(sample_id, {})
        gated = gated_records.get(sample_id, {})
        source = gated or baseline
        label = normalize_label(source.get("label"))
        b = score_stats(baseline, threshold) if baseline else {}
        g = score_stats(gated, threshold) if gated else {}
        reason = attribution(label, baseline, gated, threshold) if baseline and gated else "missing_cached_record"

        row = {
            "sample_id": sample_id,
            "label": label,
            "case_type": source.get("case_type", ""),
            "event_time_sec": source.get("event_time_sec", ""),
            "attribution": reason,
        }

        for prefix, stats in (("baseline", b), ("gated", g)):
            for key in [
                "final_score",
                "max_score",
                "text_peak",
                "voice_peak",
                "fused_peak",
                "smoothed_peak",
                "text_peak_time_sec",
                "voice_peak_time_sec",
                "fused_peak_time_sec",
                "smoothed_peak_time_sec",
                "window_count",
            ]:
                row[f"{prefix}_{key}"] = stats.get(key, "")
            for key in ["text_alert", "voice_alert", "fused_alert", "smoothed_alert"]:
                row[f"{prefix}_{key}"] = bool_cell(bool(stats.get(key)))

        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    counts = Counter(row["attribution"] for row in rows)
    by_case = Counter((row["case_type"], row["attribution"]) for row in rows)
    summary_rows = [
        {"group": "all", "case_type": "", "attribution": key, "count": count}
        for key, count in sorted(counts.items())
    ]
    summary_rows.extend(
        {"group": "case_type", "case_type": case_type, "attribution": key, "count": count}
        for (case_type, key), count in sorted(by_case.items())
    )
    write_csv(path, summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path(".cache/evaluation_reports"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", type=Path, default=Path(".cache/evaluation_reports/cached_error_attribution.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(".cache/evaluation_reports/cached_error_attribution_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = args.reports_dir / "dynamic_predictions.json"
    gated_path = args.reports_dir / "dynamic_predictions_gated_v1.json"
    baseline_records = load_records(baseline_path)
    gated_records = load_records(gated_path)
    rows = build_rows(baseline_records, gated_records, args.threshold)
    write_csv(args.output, rows)
    write_summary(args.summary_output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote summary to {args.summary_output}")


if __name__ == "__main__":
    main()
