from __future__ import annotations

import copy
import unittest

from evaluation import causal_late_fusion_v2 as fusion


def timeline():
    return [
        {"start_sec": 0, "end_sec": 10, "text_score": 10, "voice_score": 20, "fused_score": 12, "smoothed_score": 12},
        {"start_sec": 5, "end_sec": 15, "text_score": 40, "voice_score": 80, "fused_score": 48, "smoothed_score": 30},
        {"start_sec": 10, "end_sec": 20, "text_score": 90, "voice_score": 95, "fused_score": 91, "smoothed_score": 80},
    ]


class CausalLateFusionV2Tests(unittest.TestCase):
    def test_prefix_features_ignore_future_values_and_total_length(self):
        short = timeline()[:1]
        extended = timeline()
        self.assertEqual(fusion.extract_prefix_features(short, 0), fusion.extract_prefix_features(extended, 0))
        changed = copy.deepcopy(extended)
        changed[2].update({"text_score": 0, "voice_score": 0, "end_sec": 999})
        self.assertEqual(fusion.extract_prefix_features(extended, 0), fusion.extract_prefix_features(changed, 0))

    def test_feature_contract_excludes_future_and_metadata_tokens(self):
        fusion.assert_feature_contract()
        self.assertNotIn("window_progress", fusion.FEATURE_NAMES)

    def test_training_uses_every_prefix_with_event_labels(self):
        records = [{"label": "fraud", "event_time_sec": 12, "timeline": timeline()}]
        features, labels, groups = fusion.build_prefix_training_matrix(records)
        self.assertEqual(features.shape[0], 3)
        self.assertEqual(labels.tolist(), [0, 1, 1])
        self.assertEqual(groups.tolist(), [0, 0, 0])

    def test_normal_prefixes_remain_negative(self):
        record = {"label": "normal", "event_time_sec": "", "timeline": timeline()}
        self.assertEqual([fusion.prefix_label(record, point) for point in record["timeline"]], [0, 0, 0])

    def test_threshold_objective_is_case_type_independent(self):
        records = [
            {"label": "normal", "case_type": "normal_daily"},
            {"label": "fraud", "case_type": "synthetic_voice"},
        ]
        series = [[0.1, 0.2], [0.4, 0.9]]
        first = fusion.select_threshold(records, series)
        mutated = copy.deepcopy(records)
        mutated[0]["case_type"] = "anything"
        mutated[1]["case_type"] = "anything_else"
        self.assertEqual(first, fusion.select_threshold(mutated, series))


if __name__ == "__main__":
    unittest.main()
