from __future__ import annotations

import copy
import unittest

from evaluation import compare_fusion_strategies_nested_cv as comparison


class FusionStrategyNestedCVTests(unittest.TestCase):
    def test_paired_group_never_crosses_folds(self):
        records = []
        for index in range(20):
            for case_type in ("normal_daily", "synthetic_voice"):
                records.append({
                    "sample_id": f"{case_type}_{index}",
                    "case_type": case_type,
                    "dataset_origin": "external_frozen_v2",
                    "fusion_cv_group_id": f"v2:B{index:02d}",
                })
        folds = comparison.grouped_stratified_folds(records, 5)
        fold_by_group = {}
        for fold_index, fold in enumerate(folds):
            for record_index in fold:
                group = records[record_index]["fusion_cv_group_id"]
                fold_by_group.setdefault(group, set()).add(fold_index)
        self.assertTrue(all(len(value) == 1 for value in fold_by_group.values()))
        self.assertEqual([len(fold) for fold in folds], [8, 8, 8, 8, 8])

    def test_prefix_features_ignore_future_windows(self):
        timeline = [
            {"text_score": 10, "voice_score": 20, "fused_score": 12, "smoothed_score": 12, "end_sec": 10},
            {"text_score": 90, "voice_score": 99, "fused_score": 92, "smoothed_score": 80, "end_sec": 20},
        ]
        before = comparison.prefix_features(timeline, 0, "voice_reliability")
        changed = copy.deepcopy(timeline)
        changed[1].update({"voice_score": 0, "end_sec": 999})
        self.assertEqual(before, comparison.prefix_features(changed, 0, "voice_reliability"))

    def test_voice_reliability_features_exclude_duration_and_count(self):
        timeline = [
            {"text_score": 10, "voice_score": 80, "fused_score": 24, "smoothed_score": 24, "end_sec": 10},
            {"text_score": 10, "voice_score": 80, "fused_score": 24, "smoothed_score": 24, "end_sec": 20},
        ]
        features = comparison.prefix_features(timeline, 1, "voice_reliability")
        self.assertEqual(len(features), 9)
        self.assertNotIn(20.0, features)

    def test_monotonic_strategy_cannot_lower_text_probability(self):
        class VoiceModel:
            def predict_proba(self, values):
                return [[0.9, 0.1] for _ in values]

        record = {
            "timeline": [{"text_score": 95, "voice_score": 0, "fused_score": 76, "smoothed_score": 76, "end_sec": 10}]
        }
        models = comparison.FittedModels(voice_reliability=VoiceModel())
        probability = comparison.predict_probability_series(
            "monotonic_evidence_preserving", models, [record]
        )[0][0]
        self.assertGreaterEqual(probability, 0.95)

    def test_constraints_apply_semantic_reference(self):
        summary = {
            "case_final": {
                "synthetic_voice": 0.8,
                "semantic_fraud": 0.85,
                "mixed_risk": 0.9,
                "normal_daily": 0.1,
                "normal_finance": 0.1,
            }
        }
        result = comparison.constraint_results(summary, semantic_text_recall=0.9)
        self.assertTrue(all(result.values()))

    def test_inner_threshold_rank_does_not_use_case_constraints(self):
        summary = {
            "macro_f1": 0.8,
            "precision": 0.9,
            "recall": 0.7,
            "fraud_alert_recall": 0.75,
            "normal_alert_fpr": 0.1,
        }
        self.assertEqual(
            comparison.threshold_rank(summary, 0.6),
            (0.8, 0.9, 0.7, 0.75, -0.1, -0.6),
        )

    def test_grouped_calibration_splits_keep_paired_records_together(self):
        timeline = [{"text_score": 0, "voice_score": 0, "fused_score": 0, "smoothed_score": 0}]
        records = []
        for index in range(6):
            for case_type in ("normal_daily", "synthetic_voice"):
                records.append({
                    "case_type": case_type,
                    "dataset_origin": "external_frozen_v2",
                    "fusion_cv_group_id": f"v2:B{index}",
                    "timeline": timeline,
                })
        splits = comparison.grouped_calibration_splits(records, n_splits=3)
        for _training_rows, validation_rows in splits:
            validation_records = set(int(row) for row in validation_rows)
            for index in range(6):
                pair = {index * 2, index * 2 + 1}
                self.assertIn(len(pair & validation_records), (0, 2))


if __name__ == "__main__":
    unittest.main()
