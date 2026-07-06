#!/usr/bin/env python3
"""Train and apply a calibrated learned late-fusion layer from cached timelines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

DEFAULT_INPUT = PROJECT_ROOT / "evaluation/predictions/final_baseline_w10_s5/dynamic_predictions.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation/predictions/final_learned_late_fusion_w10_s5/dynamic_predictions.json"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "evaluation/models/calibrated_late_fusion_w10_s5.joblib"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "evaluation/predictions/final_learned_late_fusion_w10_s5/calibrated_late_fusion_report.json"
DEFAULT_CV_OUTPUT = PROJECT_ROOT / "evaluation/predictions/final_learned_late_fusion_w10_s5/sample_level_cv_predictions.json"
DEFAULT_CV_REPORT_PATH = PROJECT_ROOT / "evaluation/predictions/final_learned_late_fusion_w10_s5/sample_level_cv_report.json"
DEFAULT_ALERT_THRESHOLD_SCORE = 70.0
LEARNED_SCORE_KEY = "learned_late_fusion_score"
LEARNED_PROBABILITY_KEY = "learned_late_fusion_probability"

FEATURE_NAMES = [
    "text_current",
    "text_max",
    "text_top3_mean",
    "voice_current",
    "voice_max",
    "voice_top3_mean",
    "voice_ge70_count",
    "voice_ge80_count",
    "voice_ge90_count",
    "voice_run70",
    "voice_run80",
    "fused_current",
    "fused_max",
    "smoothed_current",
    "smoothed_max",
    "text_voice_product",
    "voice_minus_text",
    "window_progress",
]

FORBIDDEN_FEATURE_TOKENS = {
    "case_type",
    "sample_id",
    "source",
    "audio_path",
    "text_keyword",
    "keyword",
    "term",
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    estimator: Any
    final_calibration_cv: int
    validation_calibration_cv: int
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
        return 1 if value else 0
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "true", "fraud", "phishing", "positive", "risk"} else 0


def risk_level(score: float) -> str:
    if score >= 90:
        return "Critical"
    if score >= 70:
        return "High Risk"
    if score >= 50:
        return "Suspicious"
    return "Normal"


def top_mean(values: Sequence[float], limit: int = 3) -> float:
    top_values = sorted(values, reverse=True)[:limit]
    return sum(top_values) / len(top_values) if top_values else 0.0


def max_run_at_least(values: Sequence[float], threshold: float) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        best = max(best, current)
    return best


def assert_feature_contract() -> None:
    lowered = [name.lower() for name in FEATURE_NAMES]
    for token in FORBIDDEN_FEATURE_TOKENS:
        if any(token in name for name in lowered):
            raise AssertionError(f"Forbidden feature token present: {token}")


def load_prediction_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return data["records"]
    raise ValueError("Prediction JSON must be a list or an object with a records list.")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_artifact(artifact: Any) -> Dict[str, Any]:
    if not isinstance(artifact, dict):
        raise ValueError("Frozen late-fusion artifact must be a dictionary.")
    required = {
        "model",
        "model_name",
        "feature_names",
        "decision_threshold_probability",
        "alert_threshold_score",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"Frozen late-fusion artifact is missing fields: {missing}")
    if list(artifact["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Frozen model feature contract does not match the current feature extractor.")
    threshold = float(artifact["decision_threshold_probability"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"Invalid frozen decision threshold: {threshold}")
    return artifact


def get_timeline(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    timeline = record.get("timeline") or []
    return timeline if isinstance(timeline, list) else []


def extract_prefix_features(timeline: Sequence[Dict[str, Any]], window_index: int) -> List[float]:
    """Extract numeric late-fusion features from current and previous windows only."""
    if not timeline or window_index < 0:
        return [0.0 for _ in FEATURE_NAMES]

    bounded_index = min(window_index, len(timeline) - 1)
    prefix = list(timeline[: bounded_index + 1])
    text_scores = [safe_float(point.get("text_score")) for point in prefix]
    voice_scores = [safe_float(point.get("voice_score")) for point in prefix]
    fused_scores = [safe_float(point.get("fused_score")) for point in prefix]
    smoothed_scores = [safe_float(point.get("smoothed_score")) for point in prefix]

    text_current = text_scores[-1] if text_scores else 0.0
    voice_current = voice_scores[-1] if voice_scores else 0.0
    fused_current = fused_scores[-1] if fused_scores else 0.0
    smoothed_current = smoothed_scores[-1] if smoothed_scores else 0.0

    values = [
        text_current,
        max(text_scores, default=0.0),
        top_mean(text_scores),
        voice_current,
        max(voice_scores, default=0.0),
        top_mean(voice_scores),
        float(sum(score >= 70.0 for score in voice_scores)),
        float(sum(score >= 80.0 for score in voice_scores)),
        float(sum(score >= 90.0 for score in voice_scores)),
        float(max_run_at_least(voice_scores, 70.0)),
        float(max_run_at_least(voice_scores, 80.0)),
        fused_current,
        max(fused_scores, default=0.0),
        smoothed_current,
        max(smoothed_scores, default=0.0),
        (text_current * voice_current) / 100.0,
        voice_current - text_current,
        (bounded_index + 1) / max(len(timeline), 1),
    ]
    return [round(float(value), 6) for value in values]


def build_training_matrix(records: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[List[float]] = []
    labels: List[int] = []
    for record in records:
        timeline = get_timeline(record)
        rows.append(extract_prefix_features(timeline, len(timeline) - 1))
        labels.append(normalize_label(record.get("label", record.get("is_fraud", 0))))
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def balanced_fold_indices(records: Sequence[Dict[str, Any]], n_splits: int = 5) -> List[List[int]]:
    groups: Dict[str, List[int]] = {}
    for index, record in enumerate(records):
        case_type = str(record.get("case_type") or "unknown")
        groups.setdefault(case_type, []).append(index)

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for _, indices in sorted(groups.items()):
        ordered = sorted(indices, key=lambda idx: str(records[idx].get("sample_id") or idx))
        for offset, index in enumerate(ordered):
            folds[offset % n_splits].append(index)
    return [sorted(fold) for fold in folds if fold]


def candidate_specs(random_state: int) -> List[CandidateSpec]:
    return [
        CandidateSpec(
            name="logistic_regression_calibrated",
            estimator=make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", C=1.0, solver="liblinear", random_state=random_state),
            ),
            final_calibration_cv=5,
            validation_calibration_cv=3,
            preference_rank=0,
        ),
        CandidateSpec(
            name="gradient_boosting_calibrated",
            estimator=GradientBoostingClassifier(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.05,
                random_state=random_state,
            ),
            final_calibration_cv=5,
            validation_calibration_cv=3,
            preference_rank=1,
        ),
    ]


def calibrated_estimator(spec: CandidateSpec, cv: int) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(estimator=clone(spec.estimator), method="sigmoid", cv=cv)


def oof_probabilities(
    spec: CandidateSpec,
    records: Sequence[Dict[str, Any]],
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
) -> np.ndarray:
    folds = balanced_fold_indices(records, n_splits=n_splits)
    probabilities = np.zeros(len(records), dtype=float)
    all_indices = set(range(len(records)))
    for fold in folds:
        validation_indices = np.asarray(fold, dtype=int)
        train_indices = np.asarray(sorted(all_indices - set(fold)), dtype=int)
        model = calibrated_estimator(spec, cv=spec.validation_calibration_cv)
        model.fit(features[train_indices], labels[train_indices])
        probabilities[validation_indices] = model.predict_proba(features[validation_indices])[:, 1]
    return probabilities


def final_fit_model(spec: CandidateSpec, features: np.ndarray, labels: np.ndarray) -> CalibratedClassifierCV:
    model = calibrated_estimator(spec, cv=spec.final_calibration_cv)
    model.fit(features, labels)
    return model


def predict_window_probabilities(model: Any, records: Sequence[Dict[str, Any]]) -> List[List[float]]:
    all_probabilities: List[List[float]] = []
    for record in records:
        timeline = get_timeline(record)
        if not timeline:
            all_probabilities.append([])
            continue
        features = np.asarray(
            [extract_prefix_features(timeline, index) for index in range(len(timeline))],
            dtype=float,
        )
        probabilities = model.predict_proba(features)[:, 1]
        all_probabilities.append([float(value) for value in probabilities])
    return all_probabilities


def threshold_candidates(probability_series: Sequence[Sequence[float]]) -> List[float]:
    values = sorted({
        round(float(value), 10)
        for series in probability_series
        for value in series
        if 0.0 <= float(value) <= 1.0
    })
    candidates = {0.01, 0.99}
    candidates.update(value for value in values if 0.0 < value < 1.0)
    for left, right in zip(values, values[1:]):
        midpoint = (left + right) / 2.0
        if 0.0 < midpoint < 1.0:
            candidates.add(midpoint)
    return sorted(candidates)


def probability_to_score(probability: float, decision_threshold: float, alert_threshold_score: float) -> float:
    probability = max(0.0, min(1.0, float(probability)))
    decision_threshold = max(1e-6, min(1.0 - 1e-6, float(decision_threshold)))
    if probability < decision_threshold:
        below_threshold_cap = max(alert_threshold_score - 0.01, 0.0)
        return round(min(below_threshold_cap, alert_threshold_score * probability / decision_threshold), 4)
    tail = (probability - decision_threshold) / max(1.0 - decision_threshold, 1e-6)
    return round(alert_threshold_score + (100.0 - alert_threshold_score) * tail, 4)


def summarize_probability_decisions(
    records: Sequence[Dict[str, Any]],
    probability_series: Sequence[Sequence[float]],
    threshold: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for record, probabilities in zip(records, probability_series):
        label = normalize_label(record.get("label", record.get("is_fraud", 0)))
        final_probability = probabilities[-1] if probabilities else 0.0
        max_probability = max(probabilities, default=0.0)
        rows.append({
            "sample_id": record.get("sample_id", ""),
            "case_type": str(record.get("case_type") or "unknown"),
            "label": label,
            "final_prediction": int(final_probability >= threshold),
            "alert_prediction": int(max_probability >= threshold),
            "final_probability": final_probability,
            "max_probability": max_probability,
        })

    tp = sum(1 for row in rows if row["label"] == 1 and row["final_prediction"] == 1)
    tn = sum(1 for row in rows if row["label"] == 0 and row["final_prediction"] == 0)
    fp = sum(1 for row in rows if row["label"] == 0 and row["final_prediction"] == 1)
    fn = sum(1 for row in rows if row["label"] == 1 and row["final_prediction"] == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    case_summary: Dict[str, Dict[str, Any]] = {}
    for case_type in sorted({row["case_type"] for row in rows}):
        group = [row for row in rows if row["case_type"] == case_type]
        case_tp = sum(1 for row in group if row["label"] == 1 and row["final_prediction"] == 1)
        case_fp = sum(1 for row in group if row["label"] == 0 and row["final_prediction"] == 1)
        case_alert_tp = sum(1 for row in group if row["label"] == 1 and row["alert_prediction"] == 1)
        case_alert_fp = sum(1 for row in group if row["label"] == 0 and row["alert_prediction"] == 1)
        positives = sum(1 for row in group if row["label"] == 1)
        normals = sum(1 for row in group if row["label"] == 0)
        case_summary[case_type] = {
            "samples": len(group),
            "positives": positives,
            "normals": normals,
            "final_tp": case_tp,
            "final_fp": case_fp,
            "final_recall": round(case_tp / max(positives, 1), 4) if positives else 0.0,
            "alert_tp": case_alert_tp,
            "alert_fp": case_alert_fp,
            "alert_recall": round(case_alert_tp / max(positives, 1), 4) if positives else 0.0,
        }

    return {
        "threshold_probability": round(float(threshold), 8),
        "samples": len(rows),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "case_type_summary": case_summary,
    }


def threshold_score_tuple(
    summary: Dict[str, Any],
    min_synthetic_voice_final_tp: int,
    max_normal_alert_fp: int,
    min_overall_f1: float,
) -> Tuple[Any, ...]:
    cases = summary["case_type_summary"]
    normal = cases.get("normal_daily", {})
    semantic = cases.get("semantic_fraud", {})
    mixed = cases.get("mixed_risk", {})
    synthetic = cases.get("synthetic_voice", {})
    accepted = acceptance_passed(summary, min_synthetic_voice_final_tp, max_normal_alert_fp, min_overall_f1)
    return (
        int(accepted),
        -int(normal.get("final_fp", 0)),
        -max(int(normal.get("alert_fp", 0)) - max_normal_alert_fp, 0),
        int(semantic.get("final_tp", 0)),
        int(mixed.get("final_tp", 0)),
        int(synthetic.get("final_tp", 0)),
        float(summary.get("precision", 0.0)),
        float(summary.get("f1", 0.0)),
        -int(normal.get("alert_fp", 0)),
        -float(summary.get("threshold_probability", 0.0)),
    )


def acceptance_passed(
    summary: Dict[str, Any],
    min_synthetic_voice_final_tp: int,
    max_normal_alert_fp: int,
    min_overall_f1: float,
) -> bool:
    cases = summary["case_type_summary"]
    normal = cases.get("normal_daily", {})
    semantic = cases.get("semantic_fraud", {})
    mixed = cases.get("mixed_risk", {})
    synthetic = cases.get("synthetic_voice", {})
    return (
        int(normal.get("final_fp", 0)) == 0
        and int(normal.get("alert_fp", 0)) <= max_normal_alert_fp
        and int(semantic.get("final_tp", 0)) == int(semantic.get("positives", 0))
        and int(mixed.get("final_tp", 0)) == int(mixed.get("positives", 0))
        and int(synthetic.get("final_tp", 0)) >= min_synthetic_voice_final_tp
        and float(summary.get("precision", 0.0)) >= 1.0
        and float(summary.get("f1", 0.0)) >= min_overall_f1
    )


def select_threshold(
    records: Sequence[Dict[str, Any]],
    probability_series: Sequence[Sequence[float]],
    min_synthetic_voice_final_tp: int,
    max_normal_alert_fp: int,
    min_overall_f1: float,
) -> Tuple[float, Dict[str, Any]]:
    best_threshold = 0.5
    best_summary = summarize_probability_decisions(records, probability_series, best_threshold)
    best_score = threshold_score_tuple(
        best_summary,
        min_synthetic_voice_final_tp=min_synthetic_voice_final_tp,
        max_normal_alert_fp=max_normal_alert_fp,
        min_overall_f1=min_overall_f1,
    )

    for threshold in threshold_candidates(probability_series):
        summary = summarize_probability_decisions(records, probability_series, threshold)
        score = threshold_score_tuple(
            summary,
            min_synthetic_voice_final_tp=min_synthetic_voice_final_tp,
            max_normal_alert_fp=max_normal_alert_fp,
            min_overall_f1=min_overall_f1,
        )
        if score > best_score:
            best_threshold = threshold
            best_summary = summary
            best_score = score
    return best_threshold, best_summary


def sample_probability_series(probabilities: Sequence[float]) -> List[List[float]]:
    return [[float(value)] for value in probabilities]


def derive_baseline_normal_alert_fp(records: Sequence[Dict[str, Any]], alert_threshold_score: float) -> int:
    count = 0
    for record in records:
        if normalize_label(record.get("label", record.get("is_fraud", 0))) != 0:
            continue
        timeline = get_timeline(record)
        max_score = max((safe_float(point.get("smoothed_score")) for point in timeline), default=0.0)
        if max_score >= alert_threshold_score:
            count += 1
    return count


def case_positive_count(records: Sequence[Dict[str, Any]], case_type: str) -> int:
    return sum(
        1
        for record in records
        if str(record.get("case_type") or "") == case_type
        and normalize_label(record.get("label", record.get("is_fraud", 0))) == 1
    )


def subset_by_indices(records: Sequence[Dict[str, Any]], indices: Sequence[int]) -> List[Dict[str, Any]]:
    return [records[index] for index in indices]


def annotate_records(
    records: Sequence[Dict[str, Any]],
    probability_series: Sequence[Sequence[float]],
    decision_threshold: float,
    alert_threshold_score: float,
    model_name: str,
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for record, probabilities in zip(records, probability_series):
        output = copy.deepcopy(record)
        timeline = get_timeline(output)
        for point, probability in zip(timeline, probabilities):
            score = probability_to_score(probability, decision_threshold, alert_threshold_score)
            point[LEARNED_PROBABILITY_KEY] = round(float(probability), 8)
            point[LEARNED_SCORE_KEY] = round(score, 2)
            point["learned_late_fusion_label"] = (
                "Fraud Risk Detected" if score >= alert_threshold_score else "No High Risk Detected"
            )
            point["learned_late_fusion_risk_level"] = risk_level(score)
        final_probability = probabilities[-1] if probabilities else 0.0
        max_probability = max(probabilities, default=0.0)
        final_score_value = probability_to_score(final_probability, decision_threshold, alert_threshold_score)
        max_score_value = max(
            (safe_float(point.get(LEARNED_SCORE_KEY)) for point in timeline),
            default=0.0,
        )
        output["learned_late_fusion_model"] = model_name
        output["learned_late_fusion_decision_threshold_probability"] = round(float(decision_threshold), 8)
        output["learned_late_fusion_final_probability"] = round(float(final_probability), 8)
        output["learned_late_fusion_max_probability"] = round(float(max_probability), 8)
        output["learned_late_fusion_final_score"] = round(final_score_value, 2)
        output["learned_late_fusion_max_score"] = round(max_score_value, 2)
        annotated.append(output)
    return annotated


def summarize_score_decisions(
    records: Sequence[Dict[str, Any]],
    score_key: str,
    alert_threshold_score: float,
) -> Dict[str, Any]:
    probability_like_series: List[List[float]] = []
    for record in records:
        timeline = get_timeline(record)
        scores = [safe_float(point.get(score_key)) / 100.0 for point in timeline]
        probability_like_series.append(scores)
    return summarize_probability_decisions(records, probability_like_series, alert_threshold_score / 100.0)


def run_sample_level_cv(args: argparse.Namespace, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    folds = balanced_fold_indices(records, n_splits=args.cv_splits)
    if len(folds) != args.cv_splits:
        raise ValueError(f"Expected {args.cv_splits} non-empty folds, got {len(folds)}.")

    all_indices = set(range(len(records)))
    annotated_by_sample_id: Dict[str, Dict[str, Any]] = {}
    fold_reports: List[Dict[str, Any]] = []

    for fold_index, test_indices in enumerate(folds, start=1):
        train_indices = sorted(all_indices - set(test_indices))
        train_records = subset_by_indices(records, train_indices)
        test_records = subset_by_indices(records, test_indices)
        train_features, train_labels = build_training_matrix(train_records)
        max_train_normal_alert_fp = derive_baseline_normal_alert_fp(train_records, args.alert_threshold_score)
        min_train_synthetic_voice_tp = case_positive_count(train_records, "synthetic_voice")

        candidate_reports: List[Dict[str, Any]] = []
        best_payload: Optional[Dict[str, Any]] = None
        best_score: Optional[Tuple[Any, ...]] = None

        for spec in candidate_specs(args.random_state):
            inner_oof_prob = oof_probabilities(
                spec,
                train_records,
                train_features,
                train_labels,
                n_splits=args.inner_cv_splits,
            )
            dev_threshold, dev_summary = select_threshold(
                records=train_records,
                probability_series=sample_probability_series(inner_oof_prob),
                min_synthetic_voice_final_tp=min_train_synthetic_voice_tp,
                max_normal_alert_fp=max_train_normal_alert_fp,
                min_overall_f1=args.min_overall_f1,
            )
            model = final_fit_model(spec, train_features, train_labels)
            test_probability_series = predict_window_probabilities(model, test_records)
            test_summary = summarize_probability_decisions(test_records, test_probability_series, dev_threshold)
            dev_summary["accepted"] = acceptance_passed(
                dev_summary,
                min_synthetic_voice_final_tp=min_train_synthetic_voice_tp,
                max_normal_alert_fp=max_train_normal_alert_fp,
                min_overall_f1=args.min_overall_f1,
            )

            candidate_report = {
                "model_name": spec.name,
                "preference_rank": spec.preference_rank,
                "dev_threshold_probability": round(float(dev_threshold), 8),
                "dev_summary": dev_summary,
                "test_summary": test_summary,
            }
            candidate_reports.append(candidate_report)
            score = (
                threshold_score_tuple(
                    dev_summary,
                    min_synthetic_voice_final_tp=min_train_synthetic_voice_tp,
                    max_normal_alert_fp=max_train_normal_alert_fp,
                    min_overall_f1=args.min_overall_f1,
                ),
                -spec.preference_rank,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_payload = {
                    "spec": spec,
                    "model": model,
                    "threshold": dev_threshold,
                    "test_probability_series": test_probability_series,
                    "dev_summary": dev_summary,
                    "test_summary": test_summary,
                }

        if best_payload is None:
            raise RuntimeError(f"No candidate selected for fold {fold_index}.")

        selected_spec = best_payload["spec"]
        selected_threshold = float(best_payload["threshold"])
        annotated_test_records = annotate_records(
            records=test_records,
            probability_series=best_payload["test_probability_series"],
            decision_threshold=selected_threshold,
            alert_threshold_score=args.alert_threshold_score,
            model_name=f"{selected_spec.name}_fold_{fold_index}",
        )
        for record in annotated_test_records:
            record["learned_late_fusion_cv_fold"] = fold_index
            record["learned_late_fusion_cv_train_samples"] = len(train_records)
            record["learned_late_fusion_cv_test_samples"] = len(test_records)
            annotated_by_sample_id[str(record.get("sample_id") or "")] = record

        fold_reports.append({
            "fold": fold_index,
            "train_sample_ids": [records[index].get("sample_id", "") for index in train_indices],
            "test_sample_ids": [records[index].get("sample_id", "") for index in test_indices],
            "selected_model": selected_spec.name,
            "selected_threshold_probability": round(selected_threshold, 8),
            "max_train_normal_alert_fp": max_train_normal_alert_fp,
            "min_train_synthetic_voice_final_tp": min_train_synthetic_voice_tp,
            "selected_dev_summary": best_payload["dev_summary"],
            "selected_test_summary": best_payload["test_summary"],
            "candidate_reports": candidate_reports,
        })

    cv_records = [
        annotated_by_sample_id[str(record.get("sample_id") or "")]
        for record in records
        if str(record.get("sample_id") or "") in annotated_by_sample_id
    ]
    aggregate_summary = summarize_score_decisions(cv_records, LEARNED_SCORE_KEY, args.alert_threshold_score)
    report = {
        "input_predictions": args.predictions.as_posix(),
        "output_predictions": args.cv_output_predictions.as_posix(),
        "cv_report_path": args.cv_report_path.as_posix(),
        "cv_splits": args.cv_splits,
        "inner_cv_splits": args.inner_cv_splits,
        "feature_names": FEATURE_NAMES,
        "forbidden_feature_tokens": sorted(FORBIDDEN_FEATURE_TOKENS),
        "protocol": (
            "Sample-level nested CV: each outer fold holds out one balanced test fold; "
            "model selection and threshold selection use only the corresponding training fold."
        ),
        "aggregate_summary": aggregate_summary,
        "fold_reports": fold_reports,
    }
    write_json(args.cv_output_predictions, {"records": cv_records})
    write_json(args.cv_report_path, report)
    return report


def train_and_apply(args: argparse.Namespace) -> Dict[str, Any]:
    assert_feature_contract()
    records = load_prediction_records(args.predictions)
    if not records:
        raise ValueError("No prediction records found.")

    features, labels = build_training_matrix(records)
    baseline_normal_alert_fp = derive_baseline_normal_alert_fp(records, args.alert_threshold_score)
    max_normal_alert_fp = (
        baseline_normal_alert_fp if args.max_normal_alert_fp is None else int(args.max_normal_alert_fp)
    )

    candidate_reports: List[Dict[str, Any]] = []
    best_payload: Optional[Dict[str, Any]] = None
    best_score: Optional[Tuple[Any, ...]] = None

    for spec in candidate_specs(args.random_state):
        oof_prob = oof_probabilities(spec, records, features, labels, n_splits=args.cv_splits)
        oof_threshold, oof_summary = select_threshold(
            records=records,
            probability_series=sample_probability_series(oof_prob),
            min_synthetic_voice_final_tp=args.min_synthetic_voice_final_tp,
            max_normal_alert_fp=max_normal_alert_fp,
            min_overall_f1=args.min_overall_f1,
        )

        model = final_fit_model(spec, features, labels)
        final_probability_series = predict_window_probabilities(model, records)
        decision_threshold, final_summary = select_threshold(
            records=records,
            probability_series=final_probability_series,
            min_synthetic_voice_final_tp=args.min_synthetic_voice_final_tp,
            max_normal_alert_fp=max_normal_alert_fp,
            min_overall_f1=args.min_overall_f1,
        )

        final_summary["accepted"] = acceptance_passed(
            final_summary,
            min_synthetic_voice_final_tp=args.min_synthetic_voice_final_tp,
            max_normal_alert_fp=max_normal_alert_fp,
            min_overall_f1=args.min_overall_f1,
        )
        oof_summary["accepted"] = acceptance_passed(
            oof_summary,
            min_synthetic_voice_final_tp=args.min_synthetic_voice_final_tp,
            max_normal_alert_fp=max_normal_alert_fp,
            min_overall_f1=args.min_overall_f1,
        )
        report = {
            "model_name": spec.name,
            "preference_rank": spec.preference_rank,
            "oof_threshold_probability": round(float(oof_threshold), 8),
            "oof_summary": oof_summary,
            "final_fit_threshold_probability": round(float(decision_threshold), 8),
            "final_fit_summary": final_summary,
        }
        candidate_reports.append(report)
        score = (
            threshold_score_tuple(
                final_summary,
                min_synthetic_voice_final_tp=args.min_synthetic_voice_final_tp,
                max_normal_alert_fp=max_normal_alert_fp,
                min_overall_f1=args.min_overall_f1,
            ),
            -spec.preference_rank,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_payload = {
                "spec": spec,
                "model": model,
                "probability_series": final_probability_series,
                "decision_threshold": decision_threshold,
                "summary": final_summary,
            }

    if best_payload is None:
        raise RuntimeError("No late-fusion model candidate could be trained.")

    selected_spec = best_payload["spec"]
    selected_model = best_payload["model"]
    decision_threshold = float(best_payload["decision_threshold"])
    selected_summary = best_payload["summary"]
    accepted = bool(selected_summary.get("accepted"))

    annotated_records = annotate_records(
        records=records,
        probability_series=best_payload["probability_series"],
        decision_threshold=decision_threshold,
        alert_threshold_score=args.alert_threshold_score,
        model_name=selected_spec.name,
    )

    artifact = {
        "model": selected_model,
        "model_name": selected_spec.name,
        "feature_names": FEATURE_NAMES,
        "decision_threshold_probability": decision_threshold,
        "alert_threshold_score": args.alert_threshold_score,
        "score_key": LEARNED_SCORE_KEY,
    }

    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_predictions, {"records": annotated_records})
    joblib.dump(artifact, args.model_path)

    report = {
        "input_predictions": args.predictions.as_posix(),
        "output_predictions": args.output_predictions.as_posix(),
        "model_path": args.model_path.as_posix(),
        "feature_names": FEATURE_NAMES,
        "forbidden_feature_tokens": sorted(FORBIDDEN_FEATURE_TOKENS),
        "selected_model": selected_spec.name,
        "decision_threshold_probability": round(decision_threshold, 8),
        "alert_threshold_score": args.alert_threshold_score,
        "baseline_normal_alert_fp": baseline_normal_alert_fp,
        "max_normal_alert_fp": max_normal_alert_fp,
        "min_synthetic_voice_final_tp": args.min_synthetic_voice_final_tp,
        "min_overall_f1": args.min_overall_f1,
        "accepted": accepted,
        "acceptance_summary": selected_summary,
        "candidate_reports": candidate_reports,
    }
    write_json(args.report_path, report)
    return report


def apply_frozen_model(args: argparse.Namespace) -> Dict[str, Any]:
    """Apply an existing model artifact without fitting or selecting a threshold."""
    assert_feature_contract()
    records = load_prediction_records(args.predictions)
    if not records:
        raise ValueError("No prediction records found.")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Frozen model artifact not found: {args.model_path}")

    model_sha256_before = sha256_file(args.model_path)
    artifact = validate_frozen_artifact(joblib.load(args.model_path))
    decision_threshold = float(artifact["decision_threshold_probability"])
    alert_threshold_score = float(artifact["alert_threshold_score"])
    probability_series = predict_window_probabilities(artifact["model"], records)
    annotated_records = annotate_records(
        records=records,
        probability_series=probability_series,
        decision_threshold=decision_threshold,
        alert_threshold_score=alert_threshold_score,
        model_name=str(artifact["model_name"]),
    )
    write_json(args.output_predictions, {"records": annotated_records})
    model_sha256_after = sha256_file(args.model_path)
    if model_sha256_after != model_sha256_before:
        raise RuntimeError("Frozen model artifact changed during inference.")

    report = {
        "mode": "frozen_inference_no_fit_no_threshold_selection",
        "input_predictions": args.predictions.as_posix(),
        "input_predictions_sha256": sha256_file(args.predictions),
        "output_predictions": args.output_predictions.as_posix(),
        "model_path": args.model_path.as_posix(),
        "model_sha256": model_sha256_before,
        "model_name": str(artifact["model_name"]),
        "feature_names": FEATURE_NAMES,
        "forbidden_feature_tokens": sorted(FORBIDDEN_FEATURE_TOKENS),
        "decision_threshold_probability": round(decision_threshold, 8),
        "alert_threshold_score": alert_threshold_score,
        "records": len(records),
        "summary": summarize_score_decisions(
            annotated_records,
            LEARNED_SCORE_KEY,
            alert_threshold_score,
        ),
    }
    write_json(args.report_path, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-predictions", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--cv-output-predictions", type=Path, default=DEFAULT_CV_OUTPUT)
    parser.add_argument("--cv-report-path", type=Path, default=DEFAULT_CV_REPORT_PATH)
    parser.add_argument("--alert-threshold-score", type=float, default=DEFAULT_ALERT_THRESHOLD_SCORE)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--inner-cv-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-normal-alert-fp", type=int)
    parser.add_argument("--min-synthetic-voice-final-tp", type=int, default=20)
    parser.add_argument("--min-overall-f1", type=float, default=0.80)
    parser.add_argument("--sample-level-cv", action="store_true")
    parser.add_argument("--sample-level-cv-only", action="store_true")
    parser.add_argument(
        "--apply-frozen-model",
        action="store_true",
        help="Load --model-path and apply its saved model and threshold without fitting or selection.",
    )
    parser.add_argument("--allow-failed-acceptance", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.predictions = args.predictions.resolve()
    args.output_predictions = args.output_predictions.resolve()
    args.model_path = args.model_path.resolve()
    args.report_path = args.report_path.resolve()
    args.cv_output_predictions = args.cv_output_predictions.resolve()
    args.cv_report_path = args.cv_report_path.resolve()

    if args.apply_frozen_model:
        if args.sample_level_cv or args.sample_level_cv_only:
            raise SystemExit("--apply-frozen-model cannot be combined with CV modes.")
        report = apply_frozen_model(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.sample_level_cv_only:
        records = load_prediction_records(args.predictions)
        report = run_sample_level_cv(args, records)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    report = train_and_apply(args)
    if args.sample_level_cv:
        records = load_prediction_records(args.predictions)
        report["sample_level_cv"] = run_sample_level_cv(args, records)
        write_json(args.report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report.get("accepted") and not args.allow_failed_acceptance:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
