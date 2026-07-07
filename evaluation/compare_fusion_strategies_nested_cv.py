#!/usr/bin/env python3
"""Compare preregistered fusion strategies with nested grouped CV over 180 samples."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from evaluation import dynamic_metrics


FINAL_PREDICTIONS = PROJECT_ROOT / "evaluation/predictions/final_baseline_w10_s5/dynamic_predictions.json"
V2_PREDICTIONS = PROJECT_ROOT / "evaluation/predictions/external_frozen_v2_baseline_w10_s5/dynamic_predictions.json"
V2_METADATA = PROJECT_ROOT / "test_samples/metadata_external_frozen_v2.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation/predictions/fusion_strategy_nested_cv_180"
REPORT_DIR = PROJECT_ROOT / "evaluation/reports/fusion_strategy_nested_cv_180"
ALERT_SCORE = 70.0

STRATEGIES = (
    "fixed_fusion_smoothing",
    "calibrated_max",
    "calibrated_noisy_or",
    "monotonic_evidence_preserving",
    "unconstrained_learned",
)

SCORE_KEYS = {
    "fixed_fusion_smoothing": "smoothed_score",
    "calibrated_max": "nested_cv_calibrated_max_score",
    "calibrated_noisy_or": "nested_cv_calibrated_noisy_or_score",
    "monotonic_evidence_preserving": "nested_cv_monotonic_evidence_preserving_score",
    "unconstrained_learned": "nested_cv_unconstrained_learned_score",
}
PROBABILITY_KEYS = {
    strategy: f"{score_key.removesuffix('_score')}_probability"
    for strategy, score_key in SCORE_KEYS.items()
}

FIXED_THRESHOLD_STRATEGIES = {
    "fixed_fusion_smoothing": 0.70,
    "monotonic_evidence_preserving": 0.70,
}

CONSTRAINTS = {
    "synthetic_voice_min_recall": 0.80,
    "semantic_fraud_max_recall_drop_vs_text": 0.05,
    "mixed_risk_min_recall": 0.90,
    "normal_daily_max_final_fpr": 0.10,
    "normal_finance_max_final_fpr": 0.10,
}


@dataclass
class FittedModels:
    text_calibrator: Any | None = None
    voice_calibrator: Any | None = None
    voice_reliability: Any | None = None
    unconstrained: Any | None = None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_label(value: Any) -> int:
    return dynamic_metrics.normalize_label(value)


def load_records(path: Path) -> List[Dict[str, Any]]:
    return dynamic_metrics.load_prediction_records(path)


def load_v2_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def prepare_records(
    final_path: Path = FINAL_PREDICTIONS,
    v2_path: Path = V2_PREDICTIONS,
    v2_metadata_path: Path = V2_METADATA,
) -> List[Dict[str, Any]]:
    v2_meta = load_v2_metadata(v2_metadata_path)
    combined: List[Dict[str, Any]] = []
    seen = set()
    for origin, path in (("audio_final", final_path), ("external_frozen_v2", v2_path)):
        for original in load_records(path):
            record = copy.deepcopy(original)
            sample_id = str(record.get("sample_id") or "")
            if not sample_id or sample_id in seen:
                raise ValueError(f"Duplicate or empty sample_id: {sample_id}")
            seen.add(sample_id)
            if record.get("error") or not record.get("timeline"):
                raise ValueError(f"Invalid cached timeline: {sample_id}")
            record["dataset_origin"] = origin
            if origin == "external_frozen_v2":
                metadata = v2_meta.get(sample_id)
                if not metadata:
                    raise ValueError(f"Missing v2 metadata: {sample_id}")
                record["fusion_cv_group_id"] = f"v2:{metadata['script_id']}"
                record["fusion_cv_script_id"] = metadata["script_id"]
            else:
                record["fusion_cv_group_id"] = f"final:{sample_id}"
                record["fusion_cv_script_id"] = sample_id
            combined.append(record)
    if len(combined) != 180:
        raise ValueError(f"Expected 180 records, found {len(combined)}")
    return combined


def grouped_stratified_folds(records: Sequence[Dict[str, Any]], n_splits: int) -> List[List[int]]:
    """Keep paired scripts together and round-robin each origin/case signature."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[str(record["fusion_cv_group_id"])].append(index)

    buckets: Dict[Tuple[Any, ...], List[Tuple[str, List[int]]]] = defaultdict(list)
    for group_id, indices in groups.items():
        signature = tuple(sorted(
            (str(records[index]["dataset_origin"]), str(records[index].get("case_type") or ""))
            for index in indices
        ))
        buckets[signature].append((group_id, indices))

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for signature in sorted(buckets, key=str):
        ordered = sorted(buckets[signature], key=lambda item: item[0])
        for offset, (_group_id, indices) in enumerate(ordered):
            folds[offset % n_splits].extend(indices)
    return [sorted(fold) for fold in folds]


def prefix_label(record: Dict[str, Any], point: Dict[str, Any]) -> int:
    label = normalize_label(record.get("label", record.get("is_fraud", 0)))
    if label == 0:
        return 0
    event = record.get("event_time_sec")
    if event in (None, ""):
        return 1
    return int(safe_float(point.get("end_sec")) >= safe_float(event))


def top_mean(values: Sequence[float], limit: int = 3) -> float:
    selected = sorted(values, reverse=True)[:limit]
    return mean(selected) if selected else 0.0


def max_run(values: Sequence[float], threshold: float) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value >= threshold else 0
        best = max(best, current)
    return best


def prefix_features(timeline: Sequence[Dict[str, Any]], index: int, feature_set: str) -> List[float]:
    prefix = timeline[: index + 1]
    text = [safe_float(point.get("text_score")) / 100.0 for point in prefix]
    voice = [safe_float(point.get("voice_score")) / 100.0 for point in prefix]
    fused = [safe_float(point.get("fused_score")) / 100.0 for point in prefix]
    smoothed = [safe_float(point.get("smoothed_score")) / 100.0 for point in prefix]
    count = max(len(prefix), 1)
    if feature_set == "text_current":
        return [text[-1]]
    if feature_set == "voice_current":
        return [voice[-1]]

    voice_features = [
        voice[-1],
        max(voice, default=0.0),
        top_mean(voice),
        sum(value >= 0.70 for value in voice) / count,
        sum(value >= 0.80 for value in voice) / count,
        sum(value >= 0.90 for value in voice) / count,
        max_run(voice, 0.70) / count,
        max_run(voice, 0.80) / count,
        float(np.std(voice)),
    ]
    if feature_set == "voice_reliability":
        return voice_features
    if feature_set != "unconstrained":
        raise ValueError(feature_set)
    return [
        text[-1], max(text, default=0.0), top_mean(text),
        *voice_features,
        fused[-1], max(fused, default=0.0),
        smoothed[-1], max(smoothed, default=0.0),
        text[-1] * voice[-1], voice[-1] - text[-1],
        float(count), safe_float(prefix[-1].get("end_sec")),
    ]


def training_matrix(records: Sequence[Dict[str, Any]], feature_set: str):
    rows: List[List[float]] = []
    labels: List[int] = []
    weights: List[float] = []
    for record in records:
        timeline = record["timeline"]
        sample_weight = 1.0 / max(len(timeline), 1)
        for index, point in enumerate(timeline):
            rows.append(prefix_features(timeline, index, feature_set))
            labels.append(prefix_label(record, point))
            weights.append(sample_weight)
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), np.asarray(weights, dtype=float)


def fit_logistic(records: Sequence[Dict[str, Any]], feature_set: str, random_state: int):
    features, labels, weights = training_matrix(records, feature_set)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", random_state=random_state),
    )
    model.fit(features, labels, logisticregression__sample_weight=weights)
    return model


def grouped_calibration_splits(records: Sequence[Dict[str, Any]], n_splits: int = 3):
    """Build row-level calibration splits while keeping every sample group intact."""
    record_folds = grouped_stratified_folds(records, n_splits)
    row_ranges = []
    offset = 0
    for record in records:
        rows = np.arange(offset, offset + len(record["timeline"]), dtype=int)
        row_ranges.append(rows)
        offset += len(rows)
    all_rows = np.arange(offset, dtype=int)
    splits = []
    for fold in record_folds:
        validation = np.concatenate([row_ranges[index] for index in fold])
        training = np.setdiff1d(all_rows, validation, assume_unique=True)
        splits.append((training, validation))
    return splits


def fit_models(strategy: str, records: Sequence[Dict[str, Any]], random_state: int) -> FittedModels:
    if strategy == "fixed_fusion_smoothing":
        return FittedModels()
    if strategy in {"calibrated_max", "calibrated_noisy_or"}:
        return FittedModels(
            text_calibrator=fit_logistic(records, "text_current", random_state),
            voice_calibrator=fit_logistic(records, "voice_current", random_state),
        )
    if strategy == "monotonic_evidence_preserving":
        return FittedModels(voice_reliability=fit_logistic(records, "voice_reliability", random_state))
    if strategy == "unconstrained_learned":
        features, labels, weights = training_matrix(records, "unconstrained")
        estimator = GradientBoostingClassifier(
            n_estimators=80, max_depth=2, learning_rate=0.05, random_state=random_state,
        )
        model = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=grouped_calibration_splits(records),
        )
        model.fit(features, labels, sample_weight=weights)
        return FittedModels(unconstrained=model)
    raise ValueError(strategy)


def predict_one(model: Any, features: List[float]) -> float:
    probabilities = np.asarray(model.predict_proba(np.asarray([features], dtype=float)))
    return float(probabilities[0, 1])


def predict_probability_series(strategy: str, models: FittedModels, records: Sequence[Dict[str, Any]]):
    output: List[List[float]] = []
    for record in records:
        timeline = record["timeline"]
        series = []
        for index, point in enumerate(timeline):
            if strategy == "fixed_fusion_smoothing":
                probability = safe_float(point.get("smoothed_score")) / 100.0
            elif strategy in {"calibrated_max", "calibrated_noisy_or"}:
                text_probability = predict_one(models.text_calibrator, prefix_features(timeline, index, "text_current"))
                voice_probability = predict_one(models.voice_calibrator, prefix_features(timeline, index, "voice_current"))
                probability = (
                    max(text_probability, voice_probability)
                    if strategy == "calibrated_max"
                    else 1.0 - (1.0 - text_probability) * (1.0 - voice_probability)
                )
            elif strategy == "monotonic_evidence_preserving":
                raw_text_probability = safe_float(point.get("text_score")) / 100.0
                voice_reliability = predict_one(
                    models.voice_reliability,
                    prefix_features(timeline, index, "voice_reliability"),
                )
                probability = max(raw_text_probability, voice_reliability)
            elif strategy == "unconstrained_learned":
                probability = predict_one(models.unconstrained, prefix_features(timeline, index, "unconstrained"))
            else:
                raise ValueError(strategy)
            series.append(max(0.0, min(1.0, probability)))
        output.append(series)
    return output


def probability_to_score(probability: float, threshold: float) -> float:
    probability = max(0.0, min(1.0, probability))
    threshold = max(1e-6, min(1.0 - 1e-6, threshold))
    if probability < threshold:
        return min(69.99, 70.0 * probability / threshold)
    return 70.0 + 30.0 * (probability - threshold) / (1.0 - threshold)


def decision_rows(records: Sequence[Dict[str, Any]], series: Sequence[Sequence[float]], threshold: float):
    rows = []
    for record, values in zip(records, series):
        label = normalize_label(record.get("label", 0))
        rows.append({
            "sample_id": record["sample_id"],
            "case_type": record["case_type"],
            "label": label,
            "final_prediction": int(bool(values) and values[-1] >= threshold),
            "alert_prediction": int(bool(values) and max(values) >= threshold),
        })
    return rows


def case_recall_or_fpr(rows: Sequence[Dict[str, Any]], case_type: str) -> float:
    selected = [row for row in rows if row["case_type"] == case_type]
    if not selected:
        return 0.0
    if selected[0]["label"] == 1:
        return sum(row["final_prediction"] for row in selected) / len(selected)
    return sum(row["final_prediction"] for row in selected) / len(selected)


def classification_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [row["label"] for row in rows]
    y_pred = [row["final_prediction"] for row in rows]
    base = dynamic_metrics.classification_metrics(y_true, y_pred)
    alert_tp = sum(row["label"] == 1 and row["alert_prediction"] == 1 for row in rows)
    alert_fp = sum(row["label"] == 0 and row["alert_prediction"] == 1 for row in rows)
    positives = sum(y_true)
    normals = len(y_true) - positives
    return {
        **{key: round(value, 4) for key, value in base.items()},
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "fraud_alert_recall": round(alert_tp / max(positives, 1), 4),
        "normal_alert_fpr": round(alert_fp / max(normals, 1), 4),
        "case_final": {
            case_type: round(case_recall_or_fpr(rows, case_type), 4)
            for case_type in ("normal_daily", "normal_finance", "synthetic_voice", "semantic_fraud", "mixed_risk")
        },
    }


def text_only_semantic_recall(records: Sequence[Dict[str, Any]]) -> float:
    semantic = [record for record in records if record["case_type"] == "semantic_fraud"]
    return sum(record["timeline"][-1]["text_score"] >= 70 for record in semantic) / max(len(semantic), 1)


def constraint_results(summary: Mapping[str, Any], semantic_text_recall: float) -> Dict[str, bool]:
    case = summary["case_final"]
    return {
        "synthetic_voice_recall": case["synthetic_voice"] >= CONSTRAINTS["synthetic_voice_min_recall"],
        "semantic_fraud_recall": case["semantic_fraud"] >= semantic_text_recall - CONSTRAINTS["semantic_fraud_max_recall_drop_vs_text"],
        "mixed_risk_recall": case["mixed_risk"] >= CONSTRAINTS["mixed_risk_min_recall"],
        "normal_daily_fpr": case["normal_daily"] <= CONSTRAINTS["normal_daily_max_final_fpr"],
        "normal_finance_fpr": case["normal_finance"] <= CONSTRAINTS["normal_finance_max_final_fpr"],
    }


def threshold_candidates(series: Sequence[Sequence[float]]) -> List[float]:
    values = sorted({round(value, 8) for sample in series for value in sample if 0.05 <= value <= 0.95})
    if len(values) > 250:
        indices = np.linspace(0, len(values) - 1, 250).astype(int)
        values = [values[index] for index in indices]
    return sorted(set([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, *values]))


def threshold_rank(summary: Mapping[str, Any], threshold: float) -> Tuple[Any, ...]:
    """Label-only inner objective; case-specific constraints are admission tests only."""
    return (
        summary["macro_f1"],
        summary["precision"],
        summary["recall"],
        summary["fraud_alert_recall"],
        -summary["normal_alert_fpr"],
        -threshold,
    )


def select_threshold(records: Sequence[Dict[str, Any]], series: Sequence[Sequence[float]]):
    semantic_text_recall = text_only_semantic_recall(records)
    best = None
    for threshold in threshold_candidates(series):
        rows = decision_rows(records, series, threshold)
        summary = classification_summary(rows)
        constraints = constraint_results(summary, semantic_text_recall)
        payload = {
            "threshold": threshold,
            "summary": summary,
            "constraints": constraints,
            "rank": threshold_rank(summary, threshold),
        }
        if best is None or payload["rank"] > best["rank"]:
            best = payload
    return best


def subset(records: Sequence[Dict[str, Any]], indices: Iterable[int]) -> List[Dict[str, Any]]:
    return [records[index] for index in indices]


def inner_oof_series(strategy: str, records: Sequence[Dict[str, Any]], n_splits: int, random_state: int):
    folds = grouped_stratified_folds(records, n_splits)
    all_indices = set(range(len(records)))
    output: List[List[float]] = [[] for _ in records]
    for fold_index, validation_indices in enumerate(folds):
        training_indices = sorted(all_indices - set(validation_indices))
        models = fit_models(strategy, subset(records, training_indices), random_state + fold_index)
        predictions = predict_probability_series(strategy, models, subset(records, validation_indices))
        for record_index, values in zip(validation_indices, predictions):
            output[record_index] = values
    return output


def annotate_scores(
    records: Sequence[Dict[str, Any]],
    strategy: str,
    series: Sequence[Sequence[float]],
    threshold: float,
    outer_fold: int,
):
    key = SCORE_KEYS[strategy]
    for record, values in zip(records, series):
        if strategy != "fixed_fusion_smoothing":
            for point, probability in zip(record["timeline"], values):
                point[PROBABILITY_KEYS[strategy]] = round(probability, 8)
                point[key] = round(probability_to_score(probability, threshold), 4)
        record.setdefault("fusion_nested_cv", {})[strategy] = {
            "outer_fold": outer_fold,
            "decision_threshold_probability": round(threshold, 8),
        }


def strategy_dynamic_summary(records: Sequence[Dict[str, Any]], strategy: str):
    key = SCORE_KEYS[strategy]
    rows = [dynamic_metrics.evaluate_record(record, key, ALERT_SCORE) for record in records]
    summary = dynamic_metrics.summarize_dynamic_metrics(rows)
    summary["macro_f1"] = round(float(f1_score(
        [row["label"] for row in rows],
        [row["prediction"] for row in rows],
        average="macro",
        zero_division=0,
    )), 4)
    by_case = {}
    for case_type in sorted({row["case_type"] for row in rows}):
        by_case[case_type] = dynamic_metrics.summarize_dynamic_metrics(
            [row for row in rows if row["case_type"] == case_type]
        )
    return summary, by_case


def posthoc_threshold_sensitivity(records: Sequence[Dict[str, Any]], strategy: str) -> Dict[str, Any]:
    """Check robustness only; this diagnostic never changes a selected threshold."""
    if strategy == "fixed_fusion_smoothing":
        return {"mode": "not_applicable_to_fixed_baseline"}
    probability_key = PROBABILITY_KEYS[strategy]
    series = [
        [safe_float(point.get(probability_key)) for point in record["timeline"]]
        for record in records
    ]
    semantic_text_recall = text_only_semantic_recall(records)
    candidates = threshold_candidates(series)
    passing = []
    for threshold in candidates:
        summary = classification_summary(decision_rows(records, series, threshold))
        if all(constraint_results(summary, semantic_text_recall).values()):
            passing.append(threshold)
    return {
        "mode": "posthoc_robustness_diagnostic_not_used_for_selection",
        "thresholds_checked": len(candidates),
        "passing_threshold_count": len(passing),
        "passing_threshold_range": (
            [round(min(passing), 8), round(max(passing), 8)] if passing else None
        ),
    }


def run_nested_cv(records: List[Dict[str, Any]], outer_splits: int, inner_splits: int, random_state: int):
    outer_folds = grouped_stratified_folds(records, outer_splits)
    all_indices = set(range(len(records)))
    oof_by_id = {record["sample_id"]: copy.deepcopy(record) for record in records}
    fold_reports = []

    for outer_fold, test_indices in enumerate(outer_folds, start=1):
        train_indices = sorted(all_indices - set(test_indices))
        train_records = subset(records, train_indices)
        test_records = subset(records, test_indices)
        fold_report = {
            "outer_fold": outer_fold,
            "train_samples": len(train_records),
            "test_samples": len(test_records),
            "test_case_counts": dict(Counter(record["case_type"] for record in test_records)),
            "strategies": {},
        }
        for strategy_index, strategy in enumerate(STRATEGIES):
            if strategy in FIXED_THRESHOLD_STRATEGIES:
                threshold = FIXED_THRESHOLD_STRATEGIES[strategy]
                inner_selection = {"mode": "predefined_fixed_threshold", "threshold": threshold}
            else:
                inner_series = inner_oof_series(
                    strategy,
                    train_records,
                    inner_splits,
                    random_state + outer_fold * 100 + strategy_index * 10,
                )
                selected = select_threshold(train_records, inner_series)
                threshold = float(selected["threshold"])
                inner_selection = {
                    "mode": "inner_grouped_oof_selection",
                    "threshold": round(threshold, 8),
                    "summary": selected["summary"],
                    "constraints": selected["constraints"],
                }
            models = fit_models(strategy, train_records, random_state + outer_fold * 1000 + strategy_index)
            test_series = predict_probability_series(strategy, models, test_records)
            test_rows = decision_rows(test_records, test_series, threshold)
            test_summary = classification_summary(test_rows)
            annotate_scores(test_records, strategy, test_series, threshold, outer_fold)
            for record in test_records:
                target = oof_by_id[record["sample_id"]]
                target["timeline"] = record["timeline"]
                target.setdefault("fusion_nested_cv", {}).update(record.get("fusion_nested_cv", {}))
            fold_report["strategies"][strategy] = {
                "inner_selection": inner_selection,
                "outer_test_summary": test_summary,
            }
        fold_reports.append(fold_report)

    oof_records = [oof_by_id[record["sample_id"]] for record in records]
    semantic_text_recall = text_only_semantic_recall(oof_records)
    aggregate = {}
    for strategy in STRATEGIES:
        summary, by_case = strategy_dynamic_summary(oof_records, strategy)
        decision_summary = classification_summary(decision_rows(
            oof_records,
            [[safe_float(point.get(SCORE_KEYS[strategy])) / 100.0 for point in record["timeline"]] for record in oof_records],
            0.70,
        ))
        constraints = constraint_results(decision_summary, semantic_text_recall)
        fold_macro_f1 = [fold["strategies"][strategy]["outer_test_summary"]["macro_f1"] for fold in fold_reports]
        aggregate[strategy] = {
            "summary": summary,
            "case_type_summary": by_case,
            "selection_metrics": decision_summary,
            "constraints": constraints,
            "all_constraints_passed": all(constraints.values()),
            "outer_fold_macro_f1_mean": round(mean(fold_macro_f1), 4),
            "outer_fold_macro_f1_std": round(pstdev(fold_macro_f1), 4),
            "outer_fold_macro_f1": fold_macro_f1,
            "posthoc_threshold_sensitivity": posthoc_threshold_sensitivity(oof_records, strategy),
        }
    eligible = [name for name in STRATEGIES if aggregate[name]["all_constraints_passed"]]
    if eligible:
        def mainline_rank(name: str):
            item = aggregate[name]
            summary = item["summary"]
            return (
                item["selection_metrics"]["macro_f1"],
                summary["precision"],
                summary["fraud_alert_recall"],
                -summary["normal_alert_false_positive_rate"],
                -safe_float(summary.get("mean_detection_delay_sec"), 1e9),
            )
        selected_mainline = max(eligible, key=mainline_rank)
        decision = "selected_best_eligible_strategy"
    else:
        selected_mainline = "fixed_fusion_smoothing"
        decision = "no_candidate_passed_all_constraints_retain_fixed"
    return oof_records, {
        "protocol": "5-fold outer / 4-fold inner nested grouped CV; paired v2 scripts and all sample windows stay together",
        "threshold_selection": "label-only inner OOF lexicographic objective; case-specific constraints used only for final admission",
        "samples": len(records),
        "group_count": len({record["fusion_cv_group_id"] for record in records}),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "strategies_preregistered": list(STRATEGIES),
        "constraints_preregistered": CONSTRAINTS,
        "semantic_fraud_text_only_recall_reference": round(semantic_text_recall, 4),
        "aggregate": aggregate,
        "eligible_strategies": eligible,
        "mainline_decision": decision,
        "selected_mainline_strategy": selected_mainline,
        "fold_reports": fold_reports,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_csv(path: Path, report: Mapping[str, Any]) -> None:
    rows = []
    for strategy, item in report["aggregate"].items():
        summary = item["summary"]
        case = item["selection_metrics"]["case_final"]
        rows.append({
            "strategy": strategy,
            "eligible": item["all_constraints_passed"],
            "accuracy": summary["accuracy"],
            "precision": summary["precision"],
            "recall": summary["recall"],
            "f1": summary["f1"],
            "macro_f1": item["selection_metrics"]["macro_f1"],
            "fraud_alert_recall": summary["fraud_alert_recall"],
            "normal_final_fpr": summary["normal_final_false_positive_rate"],
            "normal_alert_fpr": summary["normal_alert_false_positive_rate"],
            "synthetic_voice_recall": case["synthetic_voice"],
            "semantic_fraud_recall": case["semantic_fraud"],
            "mixed_risk_recall": case["mixed_risk"],
            "normal_daily_fpr": case["normal_daily"],
            "normal_finance_fpr": case["normal_finance"],
            "mean_detection_delay_sec": summary["mean_detection_delay_sec"],
            "outer_fold_macro_f1_mean": item["outer_fold_macro_f1_mean"],
            "outer_fold_macro_f1_std": item["outer_fold_macro_f1_std"],
            "passing_thresholds_diagnostic": item["posthoc_threshold_sensitivity"].get("passing_threshold_count"),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-predictions", type=Path, default=FINAL_PREDICTIONS)
    parser.add_argument("--v2-predictions", type=Path, default=V2_PREDICTIONS)
    parser.add_argument("--v2-metadata", type=Path, default=V2_METADATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = prepare_records(
        args.final_predictions.resolve(),
        args.v2_predictions.resolve(),
        args.v2_metadata.resolve(),
    )
    oof_records, report = run_nested_cv(records, args.outer_splits, args.inner_splits, args.random_state)
    output_dir = args.output_dir.resolve()
    report_dir = args.report_dir.resolve()
    write_json(output_dir / "oof_predictions.json", {"records": oof_records})
    write_json(report_dir / "fusion_strategy_nested_cv_report.json", report)
    write_summary_csv(report_dir / "fusion_strategy_nested_cv_summary.csv", report)
    print(json.dumps({
        "samples": report["samples"],
        "groups": report["group_count"],
        "eligible_strategies": report["eligible_strategies"],
        "selected_mainline_strategy": report["selected_mainline_strategy"],
        "decision": report["mainline_decision"],
        "report": (report_dir / "fusion_strategy_nested_cv_report.json").as_posix(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
