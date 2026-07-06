from __future__ import annotations

import unittest

from evaluation import calibrated_late_fusion as clf


class CalibratedLateFusionFeatureTests(unittest.TestCase):
    def test_prefix_features_do_not_read_future_windows(self) -> None:
        timeline = [
            {"text_score": 10, "voice_score": 20, "fused_score": 12, "smoothed_score": 12},
            {"text_score": 30, "voice_score": 40, "fused_score": 34, "smoothed_score": 20},
            {"text_score": 90, "voice_score": 95, "fused_score": 91, "smoothed_score": 80},
        ]
        before = clf.extract_prefix_features(timeline, 0)
        timeline[2].update({"text_score": 0, "voice_score": 0, "fused_score": 0, "smoothed_score": 0})
        after = clf.extract_prefix_features(timeline, 0)
        self.assertEqual(before, after)

    def test_feature_names_exclude_forbidden_metadata(self) -> None:
        lowered = [name.lower() for name in clf.FEATURE_NAMES]
        for token in clf.FORBIDDEN_FEATURE_TOKENS:
            self.assertFalse(any(token in name for name in lowered), token)
        clf.assert_feature_contract()

    def test_empty_timeline_features_are_safe(self) -> None:
        features = clf.extract_prefix_features([], -1)
        self.assertEqual(len(features), len(clf.FEATURE_NAMES))
        self.assertTrue(all(isinstance(value, float) for value in features))
        self.assertTrue(all(value == 0.0 for value in features))

    def test_probability_score_stays_below_display_threshold_before_decision_threshold(self) -> None:
        score = clf.probability_to_score(probability=0.6099, decision_threshold=0.61, alert_threshold_score=70.0)
        self.assertLess(round(score, 2), 70.0)

    def test_frozen_artifact_rejects_feature_contract_mismatch(self) -> None:
        artifact = {
            "model": object(),
            "model_name": "test",
            "feature_names": ["wrong_feature"],
            "decision_threshold_probability": 0.61,
            "alert_threshold_score": 70.0,
        }
        with self.assertRaisesRegex(ValueError, "feature contract"):
            clf.validate_frozen_artifact(artifact)

    def test_frozen_artifact_accepts_saved_contract(self) -> None:
        artifact = {
            "model": object(),
            "model_name": "test",
            "feature_names": list(clf.FEATURE_NAMES),
            "decision_threshold_probability": 0.61,
            "alert_threshold_score": 70.0,
        }
        self.assertIs(clf.validate_frozen_artifact(artifact), artifact)

    def test_balanced_folds_are_deterministic(self) -> None:
        records = []
        for case_type, prefix in [
            ("normal_daily", "ND"),
            ("semantic_fraud", "SF"),
            ("mixed_risk", "MR"),
            ("synthetic_voice", "SV"),
        ]:
            for index in range(1, 6):
                records.append({"case_type": case_type, "sample_id": f"{prefix}_{index:02d}"})
        first = clf.balanced_fold_indices(records, n_splits=5)
        second = clf.balanced_fold_indices(records, n_splits=5)
        self.assertEqual(first, second)
        self.assertEqual([len(fold) for fold in first], [4, 4, 4, 4, 4])

    def test_balanced_folds_hold_out_each_case_type(self) -> None:
        records = []
        for case_type, prefix in [
            ("normal_daily", "ND"),
            ("semantic_fraud", "SF"),
            ("mixed_risk", "MR"),
            ("synthetic_voice", "SV"),
        ]:
            for index in range(1, 21):
                records.append({"case_type": case_type, "sample_id": f"{prefix}_{index:02d}"})

        folds = clf.balanced_fold_indices(records, n_splits=5)
        self.assertEqual([len(fold) for fold in folds], [16, 16, 16, 16, 16])
        for fold in folds:
            counts = {}
            for record_index in fold:
                case_type = records[record_index]["case_type"]
                counts[case_type] = counts.get(case_type, 0) + 1
            self.assertEqual(counts, {
                "normal_daily": 4,
                "semantic_fraud": 4,
                "mixed_risk": 4,
                "synthetic_voice": 4,
            })


if __name__ == "__main__":
    unittest.main()
