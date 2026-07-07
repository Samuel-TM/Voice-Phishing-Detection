#!/usr/bin/env python3
"""Train or apply the causal prefix-trained late-fusion v2 decision layer.

This module is intentionally versioned separately from calibrated_late_fusion.py
so the already-observed external_frozen_v1 experiment remains reproducible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "evaluation/predictions/final_baseline_w10_s5/dynamic_predictions.json"
DEFAULT_MODEL = PROJECT_ROOT / "evaluation/models/causal_late_fusion_v2_w10_s5.joblib"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/predictions/final_causal_late_fusion_v2_w10_s5/dynamic_predictions.json"
DEFAULT_REPORT = PROJECT_ROOT / "evaluation/predictions/final_causal_late_fusion_v2_w10_s5/training_report.json"
SCORE_KEY = "causal_late_fusion_v2_score"
PROBABILITY_KEY = "causal_late_fusion_v2_probability"
FEATURE_CONTRACT_VERSION = "causal_prefix_v2"
ALERT_THRESHOLD_SCORE = 70.0

FEATURE_NAMES = [
    "text_current",
    "text_max",
    "text_top3_mean",
    "voice_current",
    "voice_max",
    "voice_top3_mean",
    "voice_ge70_ratio",
    "voice_ge80_ratio",
    "voice_ge90_ratio",
    "voice_run70_ratio",
    "voice_run80_ratio",
    "fused_current",
    "fused_max",
    "smoothed_current",
    "smoothed_max",
    "text_voice_product",
    "voice_minus_text",
    "observed_window_count",
    "observed_end_sec",
]

FORBIDDEN_FEATURE_TOKENS = {
    "case_type",
    "sample_id",
    "source",
    "audio_path",
    "keyword",
    "term",
    "window_progress",
    "total_window",
    "duration_ratio",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: Any
    preference_rank: int


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    return int(str(value or "").strip().lower() in {"1", "true", "fraud", "phishing", "positive", "risk"})


def get_timeline(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    timeline = record.get("timeline") or []
    return timeline if isinstance(timeline, list) else []


def top_mean(values: Sequence[float], limit: int = 3) -> float:
    selected = sorted(values, reverse=True)[:limit]
    return sum(selected) / len(selected) if selected else 0.0


def max_run_at_least(values: Sequence[float], threshold: float) -> int:
    current = 0
    best = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        best = max(best, current)
    return best


def assert_feature_contract() -> None:
    lowered = [name.lower() for name in FEATURE_NAMES]
    for token in FORBIDDEN_FEATURE_TOKENS:
        if any(token in name for name in lowered):
            raise AssertionError(f"Forbidden feature token present: {token}")


def extract_prefix_features(timeline: Sequence[Dict[str, Any]], window_index: int) -> List[float]:
    """Use only the current and already-observed windows; never read total length."""
    if not timeline or window_index < 0:
        return [0.0 for _ in FEATURE_NAMES]
    bounded = min(window_index, len(timeline) - 1)
    prefix = list(timeline[: bounded + 1])
    text = [safe_float(point.get("text_score")) for point in prefix]
    voice = [safe_float(point.get("voice_score")) for point in prefix]
    fused = [safe_float(point.get("fused_score")) for point in prefix]
    smoothed = [safe_float(point.get("smoothed_score")) for point in prefix]
    count = float(len(prefix))
    values = [
        text[-1],
        max(text, default=0.0),
        top_mean(text),
        voice[-1],
        max(voice, default=0.0),
        top_mean(voice),
        sum(value >= 70.0 for value in voice) / count,
        sum(value >= 80.0 for value in voice) / count,
        sum(value >= 90.0 for value in voice) / count,
        max_run_at_least(voice, 70.0) / count,
        max_run_at_least(voice, 80.0) / count,
        fused[-1],
        max(fused, default=0.0),
        smoothed[-1],
        max(smoothed, default=0.0),
        text[-1] * voice[-1] / 100.0,
        voice[-1] - text[-1],
        count,
        safe_float(prefix[-1].get("end_sec"), count),
    ]
    return [round(float(value), 6) for value in values]


def prefix_label(record: Dict[str, Any], point: Dict[str, Any]) -> int:
    """Label a prefix positive only after its known event onset is observable."""
    if normalize_label(record.get("label", record.get("is_fraud", 0))) == 0:
        return 0
    event_time_raw = record.get("event_time_sec")
    if event_time_raw in (None, ""):
        return 1
    event_time = safe_float(event_time_raw, 0.0)
    return int(safe_float(point.get("end_sec"), 0.0) >= event_time)


def build_prefix_training_matrix(records: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: List[List[float]] = []
    labels: List[int] = []
    groups: List[int] = []
    for record_index, record in enumerate(records):
        timeline = get_timeline(record)
        for window_index, point in enumerate(timeline):
            rows.append(extract_prefix_features(timeline, window_index))
            labels.append(prefix_label(record, point))
            groups.append(record_index)
    if not rows:
        raise ValueError("No timeline prefixes available for training.")
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), np.asarray(groups, dtype=int)


def balanced_fold_indices(records: Sequence[Dict[str, Any]], n_splits: int) -> List[List[int]]:
    strata: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        key = str(record.get("case_type") or normalize_label(record.get("label", 0)))
        strata.setdefault(key, []).append(index)
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for indices in strata.values():
        ordered = sorted(indices, key=lambda idx: str(records[idx].get("sample_id") or idx))
        for offset, record_index in enumerate(ordered):
            folds[offset % n_splits].append(record_index)
    return [sorted(fold) for fold in folds]


def grouped_calibration_splits(records: Sequence[Dict[str, Any]], row_groups: np.ndarray, n_splits: int):
    sample_folds = balanced_fold_indices(records, n_splits)
    all_rows = np.arange(len(row_groups))
    splits = []
    for sample_fold in sample_folds:
        validation_mask = np.isin(row_groups, np.asarray(sample_fold, dtype=int))
        validation_rows = all_rows[validation_mask]
        training_rows = all_rows[~validation_mask]
        if len(validation_rows) and len(training_rows):
            splits.append((training_rows, validation_rows))
    return splits


def candidates(random_state: int) -> List[Candidate]:
    return [
        Candidate(
            "logistic_regression_group_calibrated",
            make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0, solver="liblinear", random_state=random_state)),
            0,
        ),
        Candidate(
            "gradient_boosting_group_calibrated",
            GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, random_state=random_state),
            1,
        ),
    ]


def fit_model(candidate: Candidate, records: Sequence[Dict[str, Any]], calibration_splits: int):
    features, labels, groups = build_prefix_training_matrix(records)
    cv = grouped_calibration_splits(records, groups, calibration_splits)
    model = CalibratedClassifierCV(estimator=clone(candidate.estimator), method="sigmoid", cv=cv)
    model.fit(features, labels)
    return model


def predict_probability_series(model: Any, records: Sequence[Dict[str, Any]]) -> List[List[float]]:
    result: List[List[float]] = []
    for record in records:
        timeline = get_timeline(record)
        if not timeline:
            result.append([])
            continue
        features = np.asarray([extract_prefix_features(timeline, index) for index in range(len(timeline))])
        result.append([float(value) for value in model.predict_proba(features)[:, 1]])
    return result


def threshold_candidates(series: Sequence[Sequence[float]]) -> List[float]:
    values = sorted({round(float(value), 10) for sample in series for value in sample if 0.0 < float(value) < 1.0})
    output = {0.01, 0.99}
    output.update(values)
    output.update((left + right) / 2.0 for left, right in zip(values, values[1:]))
    return sorted(output)


def summarize(records: Sequence[Dict[str, Any]], series: Sequence[Sequence[float]], threshold: float) -> Dict[str, Any]:
    labels = [normalize_label(record.get("label", record.get("is_fraud", 0))) for record in records]
    final = [int(bool(values) and values[-1] >= threshold) for values in series]
    alert = [int(bool(values) and max(values) >= threshold) for values in series]

    def metrics(predictions: Sequence[int]) -> Dict[str, Any]:
        tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
        tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
        fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
        fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    return {
        "threshold_probability": round(float(threshold), 8),
        "samples": len(records),
        "final": metrics(final),
        "alert": metrics(alert),
    }


def threshold_score(summary: Dict[str, Any]) -> Tuple[Any, ...]:
    """Predeclared label-only objective; deliberately independent of case_type."""
    final = summary["final"]
    alert = summary["alert"]
    return (
        float(final["f1"]),
        float(final["precision"]),
        float(final["recall"]),
        float(alert["f1"]),
        -int(alert["fp"]),
        -float(summary["threshold_probability"]),
    )


def select_threshold(records: Sequence[Dict[str, Any]], series: Sequence[Sequence[float]]) -> Tuple[float, Dict[str, Any]]:
    best_threshold = 0.5
    best_summary = summarize(records, series, best_threshold)
    best_score = threshold_score(best_summary)
    for threshold in threshold_candidates(series):
        candidate_summary = summarize(records, series, threshold)
        score = threshold_score(candidate_summary)
        if score > best_score:
            best_threshold = threshold
            best_summary = candidate_summary
            best_score = score
    return float(best_threshold), best_summary


def oof_probability_series(candidate: Candidate, records: Sequence[Dict[str, Any]], n_splits: int) -> List[List[float]]:
    folds = balanced_fold_indices(records, n_splits)
    output: List[List[float]] = [[] for _ in records]
    all_indices = set(range(len(records)))
    for fold in folds:
        train_indices = sorted(all_indices - set(fold))
        train_records = [records[index] for index in train_indices]
        validation_records = [records[index] for index in fold]
        model = fit_model(candidate, train_records, calibration_splits=min(3, n_splits - 1))
        predictions = predict_probability_series(model, validation_records)
        for record_index, values in zip(fold, predictions):
            output[record_index] = values
    return output


def probability_to_score(probability: float, threshold: float) -> float:
    probability = max(0.0, min(1.0, float(probability)))
    threshold = max(1e-6, min(1.0 - 1e-6, float(threshold)))
    if probability < threshold:
        return round(min(69.99, ALERT_THRESHOLD_SCORE * probability / threshold), 4)
    return round(ALERT_THRESHOLD_SCORE + (100.0 - ALERT_THRESHOLD_SCORE) * (probability - threshold) / (1.0 - threshold), 4)


def annotate(records: Sequence[Dict[str, Any]], series: Sequence[Sequence[float]], threshold: float, model_name: str):
    annotated = []
    for record, values in zip(records, series):
        output = copy.deepcopy(record)
        for point, probability in zip(get_timeline(output), values):
            point[PROBABILITY_KEY] = round(float(probability), 8)
            point[SCORE_KEY] = round(probability_to_score(probability, threshold), 2)
        output["causal_late_fusion_v2_model"] = model_name
        output["causal_late_fusion_v2_threshold_probability"] = round(float(threshold), 8)
        output["causal_late_fusion_v2_final_score"] = get_timeline(output)[-1].get(SCORE_KEY, 0.0) if get_timeline(output) else 0.0
        output["causal_late_fusion_v2_max_score"] = max((safe_float(point.get(SCORE_KEY)) for point in get_timeline(output)), default=0.0)
        annotated.append(output)
    return annotated


def load_records(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError(f"No prediction records found in {path}")
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(args: argparse.Namespace) -> Dict[str, Any]:
    assert_feature_contract()
    records = load_records(args.predictions)
    candidate_reports = []
    selected = None
    for candidate in candidates(args.random_state):
        oof_series = oof_probability_series(candidate, records, args.cv_splits)
        threshold, summary = select_threshold(records, oof_series)
        payload = {"candidate": candidate, "threshold": threshold, "summary": summary, "score": (threshold_score(summary), -candidate.preference_rank)}
        candidate_reports.append({"model_name": candidate.name, "threshold_probability": round(threshold, 8), "oof_summary": summary})
        if selected is None or payload["score"] > selected["score"]:
            selected = payload
    if selected is None:
        raise RuntimeError("No model candidate selected.")
    candidate = selected["candidate"]
    model = fit_model(candidate, records, calibration_splits=args.cv_splits)
    artifact = {
        "model": model,
        "model_name": candidate.name,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "decision_threshold_probability": selected["threshold"],
        "alert_threshold_score": ALERT_THRESHOLD_SCORE,
        "threshold_selection_objective": "max final F1, precision, recall, alert F1; label-only OOF",
        "training_predictions_sha256": sha256_file(args.predictions),
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path)
    series = predict_probability_series(model, records)
    write_json(args.output_predictions, {"records": annotate(records, series, selected["threshold"], candidate.name)})
    report = {
        "mode": "causal_prefix_training_grouped_oof",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "feature_names": FEATURE_NAMES,
        "training_rows": int(len(build_prefix_training_matrix(records)[1])),
        "training_samples": len(records),
        "selected_model": candidate.name,
        "decision_threshold_probability": round(float(selected["threshold"]), 8),
        "threshold_selection_uses_case_type": False,
        "selected_oof_summary": selected["summary"],
        "candidate_reports": candidate_reports,
        "model_path": args.model_path.as_posix(),
        "model_sha256": sha256_file(args.model_path),
    }
    write_json(args.report_path, report)
    return report


def apply_frozen(args: argparse.Namespace) -> Dict[str, Any]:
    assert_feature_contract()
    records = load_records(args.predictions)
    before = sha256_file(args.model_path)
    artifact = joblib.load(args.model_path)
    if artifact.get("feature_contract_version") != FEATURE_CONTRACT_VERSION or artifact.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Frozen model does not use the causal prefix v2 feature contract.")
    threshold = float(artifact["decision_threshold_probability"])
    series = predict_probability_series(artifact["model"], records)
    annotated = annotate(records, series, threshold, str(artifact["model_name"]))
    write_json(args.output_predictions, {"records": annotated})
    after = sha256_file(args.model_path)
    if before != after:
        raise RuntimeError("Frozen model changed during inference.")
    report = {
        "mode": "frozen_causal_late_fusion_v2_no_fit_no_threshold_selection",
        "input_predictions_sha256": sha256_file(args.predictions),
        "model_sha256": before,
        "decision_threshold_probability": round(threshold, 8),
        "records": len(records),
        "summary": summarize(records, series, threshold),
    }
    write_json(args.report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "apply-frozen"))
    parser.add_argument("--predictions", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-predictions", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("predictions", "model_path", "output_predictions", "report_path"):
        setattr(args, name, getattr(args, name).resolve())
    report = train(args) if args.command == "train" else apply_frozen(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
