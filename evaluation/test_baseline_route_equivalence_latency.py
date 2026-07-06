from __future__ import annotations

import unittest

from evaluation import baseline_route_equivalence_latency as route_eval


class BaselineRouteEquivalenceLatencyTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(route_eval.percentile([], 0.95), None)
        self.assertEqual(route_eval.percentile([1.0], 0.95), 1.0)
        self.assertAlmostEqual(route_eval.percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)

    def test_route_and_browser_segment_contracts(self) -> None:
        route = route_eval.route_segments(duration_ms=12_000, window_ms=10_000, step_ms=5_000)
        browser = route_eval.browser_segments(duration_ms=12_000, chunk_ms=5_000)
        self.assertEqual(route, [(0, 10_000), (5_000, 12_000), (10_000, 12_000)])
        self.assertEqual(browser, [(0, 5_000), (5_000, 10_000), (10_000, 12_000)])

    def test_compare_identical_timelines(self) -> None:
        timeline = [
            {"start_sec": 0, "end_sec": 10, "text": "a", "text_score": 10, "voice_score": 20,
             "fused_score": 12, "smoothed_score": 12},
            {"start_sec": 5, "end_sec": 15, "text": "b", "text_score": 80, "voice_score": 40,
             "fused_score": 72, "smoothed_score": 70},
        ]
        summary, points = route_eval.compare_route_timelines("sample", timeline, timeline)
        self.assertEqual(summary["final_label_agreement"], 1)
        self.assertEqual(summary["first_alert_abs_delta_sec"], 0)
        self.assertEqual(summary["mae_smoothed_score"], 0)
        self.assertTrue(all(point["timing_exact"] for point in points))


if __name__ == "__main__":
    unittest.main()
