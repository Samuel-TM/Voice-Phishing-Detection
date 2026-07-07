#!/usr/bin/env python3
"""Corrected-label ablation: synthetic_voice = normal (not fraud).

Reads EXISTING prediction timelines from final_baseline_w10_s5 and
external_frozen_v2_baseline_w10_s5.  Does NOT regenerate predictions.
Only relabels synthetic_voice → normal and recomputes metrics.

Output tables:
  1. Ablation summary (text_only / voice_only / fusion_without_smoothing /
     fusion_with_smoothing)
  2. Case-type breakdown (ND / NF / SV / SF / MR)
  3. Final / alert / timing metrics
  4. Fusion text-weight sweep (w ∈ [0.50, 1.00])

Usage:
  python evaluation/rerun_corrected_label_ablation.py [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FINAL_PREDICTIONS = PROJECT_ROOT / "evaluation/predictions/final_baseline_w10_s5/dynamic_predictions.json"
V2_PREDICTIONS = PROJECT_ROOT / "evaluation/predictions/external_frozen_v2_baseline_w10_s5/dynamic_predictions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/reports/corrected_label_ablation"

# ---------------------------------------------------------------------------
# Corrected labels
# ---------------------------------------------------------------------------
# Fraud = content-based: only SF (real voice + fraud text) and MR (synthetic + fraud text)
# Normal = everything else, INCLUDING SV (synthetic + benign text → not fraud)
FRAUD_CASE_TYPES = {"semantic_fraud", "mixed_risk"}
NORMAL_CASE_TYPES = {"normal_daily", "normal_finance", "synthetic_voice"}

# Ablation variants
VARIANTS = [
    ("text_only", "text_score"),
    ("voice_only", "voice_score"),
    ("fusion_without_smoothing", "fused_score"),
    ("fusion_with_smoothing", "smoothed_score"),
]

# Weight sweep range (inclusive)
WEIGHT_SWEEP = [round(w, 2) for w in np.arange(0.50, 1.01, 0.05)]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def correct_label(case_type: str) -> int:
    return 1 if case_type in FRAUD_CASE_TYPES else 0


def load_all_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen: set = set()
    for origin, path in [("audio_final", FINAL_PREDICTIONS), ("external_frozen_v2", V2_PREDICTIONS)]:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        raw = data if isinstance(data, list) else data.get("records", data)
        if not isinstance(raw, list):
            raise ValueError(f"No records list in {path}")
        for r in raw:
            sid = str(r.get("sample_id", ""))
            if not sid or sid in seen:
                continue
            seen.add(sid)
            r["_origin"] = origin
            records.append(r)
    if len(records) != 180:
        print(f"Warning: expected 180 records, got {len(records)}", file=sys.stderr)
    return records


def smooth(series: Sequence[float], alpha: float = 0.35) -> List[float]:
    if not series:
        return []
    out = [series[0]]
    for v in series[1:]:
        out.append(alpha * v + (1.0 - alpha) * out[-1])
    return out


def fused_timeline(
    timeline: List[Dict[str, Any]],
    text_weight: float,
    smoothing_alpha: float = 0.35,
) -> List[float]:
    """Compute smoothed fused score for a given text_weight."""
    text = [safe_float(p.get("text_score")) / 100.0 for p in timeline]
    voice = [safe_float(p.get("voice_score")) / 100.0 for p in timeline]
    fused_raw = [text_weight * t + (1.0 - text_weight) * v for t, v in zip(text, voice)]
    return smooth(fused_raw, smoothing_alpha)


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------
def evaluate_sample(
    record: Dict[str, Any],
    score_key: str,
    alert_threshold: float = 70.0,
) -> Dict[str, Any]:
    """Return classification row for a single sample (compatible with dynamic_metrics)."""
    timeline = record["timeline"]
    label = correct_label(str(record.get("case_type", "")))
    scores = [safe_float(p.get(score_key)) for p in timeline]

    final_score = scores[-1] if scores else 0.0
    max_score = max(scores) if scores else 0.0
    prediction = int(final_score >= alert_threshold)
    alert = int(max_score >= alert_threshold)

    # Timing
    event_sec = safe_float(record.get("event_time_sec"), default=None)
    alert_sec = None
    for i, s in enumerate(scores):
        if s >= alert_threshold:
            alert_sec = safe_float(timeline[i].get("end_sec"))
            break

    lead_sec = None
    delay_sec = None
    if alert_sec is not None and event_sec is not None and event_sec > 0:
        lead_sec = max(0.0, event_sec - alert_sec)
        delay_sec = max(0.0, alert_sec - event_sec)

    return {
        "sample_id": record.get("sample_id", ""),
        "case_type": record.get("case_type", ""),
        "label": label,
        "prediction": prediction,
        "final_score": final_score,
        "max_score": max_score,
        "alert": alert,
        "alert_sec": alert_sec,
        "lead_sec": lead_sec,
        "delay_sec": delay_sec,
    }


def evaluate_weighted_sample(
    record: Dict[str, Any],
    text_weight: float,
    alert_threshold: float = 70.0,
) -> Dict[str, Any]:
    """Like evaluate_sample but computes fused score on-the-fly for weight sweep."""
    timeline = record["timeline"]
    label = correct_label(str(record.get("case_type", "")))
    sm = fused_timeline(timeline, text_weight)
    scores_100 = [s * 100.0 for s in sm]

    final_score = scores_100[-1] if scores_100 else 0.0
    max_score = max(scores_100) if scores_100 else 0.0
    prediction = int(final_score >= alert_threshold)
    alert = int(max_score >= alert_threshold)

    event_sec = safe_float(record.get("event_time_sec"), default=None)
    alert_sec = None
    for i, s in enumerate(scores_100):
        if s >= alert_threshold:
            alert_sec = safe_float(timeline[i].get("end_sec"))
            break

    lead_sec = None
    delay_sec = None
    if alert_sec is not None and event_sec is not None and event_sec > 0:
        lead_sec = max(0.0, event_sec - alert_sec)
        delay_sec = max(0.0, alert_sec - event_sec)

    return {
        "sample_id": record.get("sample_id", ""),
        "case_type": record.get("case_type", ""),
        "label": label,
        "prediction": prediction,
        "final_score": final_score,
        "max_score": max_score,
        "alert": alert,
        "alert_sec": alert_sec,
        "lead_sec": lead_sec,
        "delay_sec": delay_sec,
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def classification_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [r["label"] for r in rows]
    y_pred = [r["prediction"] for r in rows]
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    fpr = fp / max(fp + tn, 1)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Alert metrics
    fraud_rows = [r for r in rows if r["label"] == 1]
    normal_rows = [r for r in rows if r["label"] == 0]
    fraud_alerted = sum(1 for r in fraud_rows if r["alert"])
    normal_alerted = sum(1 for r in normal_rows if r["alert"])
    fraud_alert_recall = fraud_alerted / max(len(fraud_rows), 1)
    normal_alert_fpr = normal_alerted / max(len(normal_rows), 1)

    # Timing (fraud samples that were detected)
    detected = [r for r in fraud_rows if r["prediction"] == 1]
    lead_times = [r["lead_sec"] for r in detected if r["lead_sec"] is not None]
    delays = [r["delay_sec"] for r in detected if r["delay_sec"] is not None]
    alert_times = [r["alert_sec"] for r in detected if r["alert_sec"] is not None]

    # Case-type breakdown
    case_final = {}
    for ct in ["normal_daily", "normal_finance", "synthetic_voice", "semantic_fraud", "mixed_risk"]:
        ct_rows = [r for r in rows if r["case_type"] == ct]
        if ct_rows:
            ct_pos = sum(1 for r in ct_rows if r["prediction"] == 1)
            case_final[ct] = round(ct_pos / len(ct_rows), 4)

    return {
        "accuracy": round((tp + tn) / max(tp + tn + fp + fn, 1), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "samples": len(rows),
        "fraud_samples": len(fraud_rows),
        "normal_samples": len(normal_rows),
        "fraud_alert_recall": round(fraud_alert_recall, 4),
        "normal_alert_fpr": round(normal_alert_fpr, 4),
        "normal_final_fpr": round(fpr, 4),
        "mean_time_to_alert_sec": round(mean(alert_times), 4) if alert_times else None,
        "mean_early_warning_lead_time_sec": round(mean(lead_times), 4) if lead_times else None,
        "mean_detection_delay_sec": round(mean(delays), 4) if delays else None,
        "case_final": case_final,
    }


# ---------------------------------------------------------------------------
# Table printers
# ---------------------------------------------------------------------------
def print_ablation_table(records: List[Dict[str, Any]], alert_threshold: float = 70.0):
    print()
    print("=" * 110)
    print("TABLE 1 — ABLATION SUMMARY")
    print("  Labels: Fraud = SF + MR (80 samples), Normal = ND + NF + SV (100 samples)")
    print("=" * 110)
    header = f"{'Variant':<30s} {'Recall':>7s} {'Prec':>7s} {'F1':>7s} {'FPR':>7s} {'TP':>5s} {'TN':>5s} {'FP':>5s} {'FN':>5s} {'AlertR':>7s} {'LeadT':>8s}"
    print(header)
    print("-" * 110)

    for variant_name, score_key in VARIANTS:
        rows = [evaluate_sample(r, score_key, alert_threshold) for r in records]
        s = classification_summary(rows)
        lead = f"{s['mean_early_warning_lead_time_sec']:.1f}s" if s['mean_early_warning_lead_time_sec'] else "N/A"
        print(f"{variant_name:<30s} {s['recall']:7.4f} {s['precision']:7.4f} {s['f1']:7.4f} {s['normal_final_fpr']:7.4f} {s['tp']:5d} {s['tn']:5d} {s['fp']:5d} {s['fn']:5d} {s['fraud_alert_recall']:7.4f} {lead:>8s}")


def print_case_type_table(records: List[Dict[str, Any]], alert_threshold: float = 70.0):
    print()
    print("=" * 100)
    print("TABLE 2 — CASE-TYPE BREAKDOWN (final score >= 70 → alert)")
    print("=" * 100)
    case_types = ["normal_daily", "normal_finance", "synthetic_voice", "semantic_fraud", "mixed_risk"]
    counts = {ct: len([r for r in records if r["case_type"] == ct]) for ct in case_types}

    header = f"{'Variant':<30s}"
    for ct in case_types:
        header += f" {ct[:4]:>6s}({counts[ct]})"
    print(header)
    print("-" * 100)

    for variant_name, score_key in VARIANTS:
        line = f"{variant_name:<30s}"
        for ct in case_types:
            ct_rows = [r for r in records if r["case_type"] == ct]
            alerts = sum(1 for r in ct_rows if safe_float(r["timeline"][-1].get(score_key)) >= alert_threshold)
            line += f" {alerts:>4}/{counts[ct]:<2}"
        print(line)


def print_timing_table(records: List[Dict[str, Any]], alert_threshold: float = 70.0):
    print()
    print("=" * 100)
    print("TABLE 3 — FINAL / ALERT / TIMING METRICS")
    print("=" * 100)

    # Text-only scores for semantic_fraud reference
    sf_rows = [r for r in records if r["case_type"] == "semantic_fraud"]
    sf_text_final = sum(1 for r in sf_rows if safe_float(r["timeline"][-1].get("text_score")) >= alert_threshold)

    print(f"  text-only semantic_fraud reference recall: {sf_text_final}/{len(sf_rows)} = {sf_text_final/len(sf_rows):.4f}" if sf_rows else "  (no SF samples)")
    print()

    header = f"{'Variant':<30s} {'Final F1':>9s} {'Final FP':>9s} {'Alert Rec':>9s} {'Alert FPR':>9s} {'Alert T(s)':>10s} {'Lead T(s)':>10s} {'Delay(s)':>9s}"
    print(header)
    print("-" * 100)

    for variant_name, score_key in VARIANTS:
        rows = [evaluate_sample(r, score_key, alert_threshold) for r in records]
        s = classification_summary(rows)
        alert_t = f"{s['mean_time_to_alert_sec']:.1f}" if s['mean_time_to_alert_sec'] else "N/A"
        lead_t = f"{s['mean_early_warning_lead_time_sec']:.1f}" if s['mean_early_warning_lead_time_sec'] else "N/A"
        delay_t = f"{s['mean_detection_delay_sec']:.1f}" if s['mean_detection_delay_sec'] else "N/A"
        print(f"{variant_name:<30s} {s['f1']:9.4f} {s['fp']:>4}/{s['normal_samples']:<4} {s['fraud_alert_recall']:9.4f} {s['normal_alert_fpr']:9.4f} {alert_t:>10s} {lead_t:>10s} {delay_t:>9s}")


def print_weight_sweep(records: List[Dict[str, Any]], alert_threshold: float = 70.0):
    print()
    print("=" * 115)
    print("TABLE 4 — FUSION TEXT-WEIGHT SWEEP (w × text + (1−w) × voice, α=0.35 smoothing)")
    print("=" * 115)

    constraints_passed = []
    header = f"{'w':>6s} {'Recall':>8s} {'Prec':>8s} {'F1':>8s} {'FPR':>8s} {'SV':>6s} {'ND':>6s} {'NF':>6s} {'SF':>6s} {'MR':>6s} {'AlertR':>8s} {'LeadT':>7s} {'Status':>8s}"
    print(header)
    print("-" * 115)

    best_f1, best_w = 0.0, 0.80
    for w in WEIGHT_SWEEP:
        rows = [evaluate_weighted_sample(r, w, alert_threshold) for r in records]
        s = classification_summary(rows)

        # Per-type
        by_type = {}
        for ct in ["normal_daily", "normal_finance", "synthetic_voice", "semantic_fraud", "mixed_risk"]:
            ct_r = [r for r in records if r["case_type"] == ct]
            ct_rows = [evaluate_weighted_sample(r, w, alert_threshold) for r in ct_r]
            by_type[ct] = sum(1 for r in ct_rows if r["prediction"] == 1)

        sv_fpr = by_type["synthetic_voice"] / 40
        lead_t = f"{s['mean_early_warning_lead_time_sec']:.1f}s" if s['mean_early_warning_lead_time_sec'] else "N/A"

        passes = s["recall"] >= 0.90 and s["normal_final_fpr"] <= 0.10 and sv_fpr <= 0.05
        status = "✓ PASS" if passes else ""
        if passes and s["f1"] > best_f1:
            best_f1, best_w = s["f1"], w
            status += " ★"

        marker = " ← chosen" if abs(w - 0.80) < 0.001 else ""
        print(f"{w:5.2f}  {s['recall']:8.4f} {s['precision']:8.4f} {s['f1']:8.4f} {s['normal_final_fpr']:8.4f} {by_type['synthetic_voice']:>3}/40 {by_type['normal_daily']:>3}/40 {by_type['normal_finance']:>3}/20 {by_type['semantic_fraud']:>3}/40 {by_type['mixed_risk']:>3}/40 {s['fraud_alert_recall']:8.4f} {lead_t:>7s} {status}{marker}")

        if passes:
            constraints_passed.append(w)

    print()
    if constraints_passed:
        print(f"  Passing weights: {[f'{w:.2f}' for w in constraints_passed]}")
        print(f"  Best F1 among passing: w={best_w:.2f} (F1={best_f1:.4f})")
    else:
        print("  No weight passes all constraints.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--alert-threshold", type=float, default=70.0)
    p.add_argument("--json", action="store_true", help="Also write JSON report")
    return p


def main():
    args = build_parser().parse_args()
    records = load_all_records()

    # Verify composition
    ct_counts = Counter(r["case_type"] for r in records)
    print(f"Loaded {len(records)} samples: {dict(ct_counts)}")
    print(f"  Fraud (SF+MR):     {sum(ct_counts.get(ct,0) for ct in FRAUD_CASE_TYPES)}")
    print(f"  Normal (ND+NF+SV): {sum(ct_counts.get(ct,0) for ct in NORMAL_CASE_TYPES)}")

    threshold = args.alert_threshold

    # Print all tables
    print_ablation_table(records, threshold)
    print_case_type_table(records, threshold)
    print_timing_table(records, threshold)
    print_weight_sweep(records, threshold)

    # -------------------------------------------------------------------
    # JSON output (optional)
    # -------------------------------------------------------------------
    if args.json:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "protocol": "corrected_label_ablation_on_existing_predictions",
            "fraud_case_types": sorted(FRAUD_CASE_TYPES),
            "normal_case_types": sorted(NORMAL_CASE_TYPES),
            "samples": len(records),
            "alert_threshold": threshold,
            "ablation": {},
            "weight_sweep": {},
        }
        for variant_name, score_key in VARIANTS:
            rows = [evaluate_sample(r, score_key, threshold) for r in records]
            report["ablation"][variant_name] = classification_summary(rows)

        for w in WEIGHT_SWEEP:
            rows = [evaluate_weighted_sample(r, w, threshold) for r in records]
            report["weight_sweep"][f"{w:.2f}"] = classification_summary(rows)

        json_path = output_dir / "corrected_label_ablation.json"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {json_path}")


if __name__ == "__main__":
    main()