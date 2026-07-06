#!/usr/bin/env python3
"""Validate baseline route equivalence and browser-style chunk processing latency."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydub import AudioSegment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from streaming_analysis.window_pipeline import build_window_starts


DEFAULT_METADATA = PROJECT_ROOT / "test_samples/metadata_final.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation/reports/baseline_route_equivalence_latency"
DEFAULT_SAMPLE_IDS = [
    "ND_long_01", "ND_long_05", "ND_long_10",
    "SF_long_01", "SF_long_05", "SF_long_10",
    "MR_long_01", "MR_long_05", "MR_long_10",
    "SV_long_01", "SV_long_05", "SV_long_10",
]
SCORE_KEYS = ["text_score", "voice_score", "fused_score", "smoothed_score"]
ALERT_THRESHOLD = 70.0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_label(value: Any) -> int:
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "true", "fraud", "phishing", "positive", "risk"} else 0


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def percentile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def first_alert_time(timeline: Sequence[Dict[str, Any]]) -> Optional[float]:
    for point in timeline:
        if safe_float(point.get("smoothed_score")) >= ALERT_THRESHOLD:
            return safe_float(point.get("end_sec"))
    return None


def label_from_timeline(timeline: Sequence[Dict[str, Any]]) -> int:
    return int(bool(timeline) and safe_float(timeline[-1].get("smoothed_score")) >= ALERT_THRESHOLD)


def max_score(timeline: Sequence[Dict[str, Any]]) -> float:
    return max((safe_float(point.get("smoothed_score")) for point in timeline), default=0.0)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_ndjson(data: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for raw_line in data.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    error = next((event for event in events if event.get("event") == "error"), None)
    if error:
        raise RuntimeError(str(error.get("error") or error))
    done = next((event for event in reversed(events) if event.get("event") == "done"), {})
    timeline = done.get("timeline") or [
        event.get("point") for event in events if event.get("event") == "point" and event.get("point")
    ]
    return timeline, done


def wav_buffer(segment: AudioSegment) -> io.BytesIO:
    buffer = io.BytesIO()
    segment.export(buffer, format="wav")
    buffer.seek(0)
    return buffer


def load_selected_rows(metadata_path: Path, sample_ids: Sequence[str]) -> List[Dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["sample_id"]: row for row in rows}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Sample IDs missing from metadata: {missing}")
    selected = [by_id[sample_id] for sample_id in sample_ids]
    missing_audio = [row["sample_id"] for row in selected if not resolve_project_path(row["audio_path"]).is_file()]
    if missing_audio:
        raise FileNotFoundError(f"Audio files missing for: {missing_audio}")
    return selected


def run_uploaded_route(client: Any, row: Dict[str, str], window_seconds: float, step_seconds: float) -> Dict[str, Any]:
    audio_path = resolve_project_path(row["audio_path"])
    started = time.perf_counter()
    with audio_path.open("rb") as handle:
        response = client.post(
            "/api/stream_audio_analysis",
            data={
                "audio_file": (handle, audio_path.name),
                "window_seconds": str(window_seconds),
                "step_seconds": str(step_seconds),
                "text_weight": "0.8",
                "smoothing_previous_weight": "0.65",
                "scoring_mode": "baseline",
                "case_type": row["case_type"],
            },
            content_type="multipart/form-data",
            buffered=True,
        )
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(f"Uploaded route failed ({response.status_code}): {response.data[:300]!r}")
    timeline, done = parse_ndjson(response.data)
    return {"timeline": timeline, "done": done, "elapsed_sec": elapsed}


def post_live_segments(
    client: Any,
    row: Dict[str, str],
    audio: AudioSegment,
    segments: Sequence[Tuple[int, int]],
    protocol: str,
) -> Dict[str, Any]:
    session_id: Optional[str] = None
    timeline: List[Dict[str, Any]] = []
    chunk_rows: List[Dict[str, Any]] = []
    total_started = time.perf_counter()

    for index, (start_ms, end_ms) in enumerate(segments):
        segment = audio[start_ms:end_ms]
        duration_sec = max((end_ms - start_ms) / 1000.0, 0.001)
        payload: Dict[str, Any] = {
            "audio_chunk": (wav_buffer(segment), f"{row['sample_id']}_{protocol}_{index:04d}.wav"),
            "chunk_index": str(index),
            "chunk_seconds": str(duration_sec),
            "chunk_start_sec": str(start_ms / 1000.0),
            "chunk_end_sec": str(end_ms / 1000.0),
            "text_weight": "0.8",
            "smoothing_previous_weight": "0.65",
            "scoring_mode": "baseline",
            "case_type": row["case_type"],
        }
        if session_id:
            payload["session_id"] = session_id
        started = time.perf_counter()
        response = client.post(
            "/api/live_audio_chunk",
            data=payload,
            content_type="multipart/form-data",
        )
        latency = time.perf_counter() - started
        data = response.get_json(silent=True) or {}
        if response.status_code != 200 or data.get("error"):
            raise RuntimeError(
                f"Live route failed for {row['sample_id']} chunk {index} "
                f"({response.status_code}): {data or response.data[:300]!r}"
            )
        session_id = str(data.get("session_id") or session_id or "")
        point = data.get("point") or {}
        timeline.append(point)
        chunk_rows.append({
            "sample_id": row["sample_id"],
            "case_type": row["case_type"],
            "protocol": protocol,
            "chunk_index": index,
            "start_sec": start_ms / 1000.0,
            "end_sec": end_ms / 1000.0,
            "audio_duration_sec": duration_sec,
            "latency_sec": latency,
            "realtime_factor": latency / duration_sec,
            "within_audio_duration": int(latency <= duration_sec),
            "status_code": response.status_code,
            "text_score": safe_float(point.get("text_score")),
            "voice_score": safe_float(point.get("voice_score")),
            "fused_score": safe_float(point.get("fused_score")),
            "smoothed_score": safe_float(point.get("smoothed_score")),
        })

    finish_started = time.perf_counter()
    finish_response = client.post("/api/live_audio_finish", json={"session_id": session_id})
    finish_latency = time.perf_counter() - finish_started
    finish_data = finish_response.get_json(silent=True) or {}
    if finish_response.status_code != 200 or finish_data.get("error"):
        raise RuntimeError(f"Live finish failed ({finish_response.status_code}): {finish_data}")
    total_elapsed = time.perf_counter() - total_started
    return {
        "timeline": finish_data.get("timeline") or timeline,
        "done": finish_data,
        "chunk_rows": chunk_rows,
        "finish_latency_sec": finish_latency,
        "total_elapsed_sec": total_elapsed,
        "session_id": session_id,
    }


def route_segments(duration_ms: int, window_ms: int, step_ms: int) -> List[Tuple[int, int]]:
    return [(start, min(start + window_ms, duration_ms)) for start in build_window_starts(duration_ms, window_ms, step_ms)]


def browser_segments(duration_ms: int, chunk_ms: int) -> List[Tuple[int, int]]:
    return [(start, min(start + chunk_ms, duration_ms)) for start in range(0, duration_ms, chunk_ms)]


def compare_route_timelines(
    sample_id: str,
    offline: Sequence[Dict[str, Any]],
    live: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(offline) != len(live):
        raise ValueError(f"Window count mismatch for {sample_id}: offline={len(offline)} live={len(live)}")
    point_rows: List[Dict[str, Any]] = []
    for index, (offline_point, live_point) in enumerate(zip(offline, live)):
        row: Dict[str, Any] = {
            "sample_id": sample_id,
            "window_index": index,
            "offline_start_sec": safe_float(offline_point.get("start_sec")),
            "live_start_sec": safe_float(live_point.get("start_sec")),
            "offline_end_sec": safe_float(offline_point.get("end_sec")),
            "live_end_sec": safe_float(live_point.get("end_sec")),
            "timing_exact": int(
                safe_float(offline_point.get("start_sec")) == safe_float(live_point.get("start_sec"))
                and safe_float(offline_point.get("end_sec")) == safe_float(live_point.get("end_sec"))
            ),
            "transcript_exact": int(str(offline_point.get("text") or "") == str(live_point.get("text") or "")),
        }
        for key in SCORE_KEYS:
            offline_score = safe_float(offline_point.get(key))
            live_score = safe_float(live_point.get(key))
            row[f"offline_{key}"] = offline_score
            row[f"live_{key}"] = live_score
            row[f"abs_diff_{key}"] = abs(offline_score - live_score)
        point_rows.append(row)

    offline_alert = first_alert_time(offline)
    live_alert = first_alert_time(live)
    both_alert = offline_alert is not None and live_alert is not None
    sample_summary: Dict[str, Any] = {
        "sample_id": sample_id,
        "window_count": len(offline),
        "timing_exact_windows": sum(row["timing_exact"] for row in point_rows),
        "transcript_exact_windows": sum(row["transcript_exact"] for row in point_rows),
        "offline_final_score": safe_float(offline[-1].get("smoothed_score")) if offline else 0.0,
        "live_final_score": safe_float(live[-1].get("smoothed_score")) if live else 0.0,
        "offline_max_score": max_score(offline),
        "live_max_score": max_score(live),
        "offline_first_alert_sec": offline_alert,
        "live_first_alert_sec": live_alert,
        "first_alert_state_agreement": int((offline_alert is None) == (live_alert is None)),
        "first_alert_abs_delta_sec": abs(offline_alert - live_alert) if both_alert else None,
        "final_label_agreement": int(label_from_timeline(offline) == label_from_timeline(live)),
    }
    for key in SCORE_KEYS:
        differences = [row[f"abs_diff_{key}"] for row in point_rows]
        sample_summary[f"mae_{key}"] = statistics.mean(differences) if differences else 0.0
        sample_summary[f"max_abs_diff_{key}"] = max(differences, default=0.0)
    return sample_summary, point_rows


def aggregate_report(
    sample_rows: Sequence[Dict[str, Any]],
    point_rows: Sequence[Dict[str, Any]],
    chunk_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    route_rows = [row for row in chunk_rows if row["protocol"] == "route_equivalence_10s_5s"]
    browser_rows = [row for row in chunk_rows if row["protocol"] == "browser_style_5s"]
    comparable_alert_deltas = [
        float(row["first_alert_abs_delta_sec"])
        for row in sample_rows
        if row.get("first_alert_abs_delta_sec") is not None
    ]
    browser_alert_deltas = [
        float(row["browser_first_alert_abs_delta_sec"])
        for row in sample_rows
        if row.get("browser_first_alert_abs_delta_sec") is not None
    ]
    route_summary: Dict[str, Any] = {
        "samples": len(sample_rows),
        "windows": len(point_rows),
        "timing_exact_rate": sum(row["timing_exact"] for row in point_rows) / max(len(point_rows), 1),
        "transcript_exact_rate": sum(row["transcript_exact"] for row in point_rows) / max(len(point_rows), 1),
        "final_label_agreement_rate": sum(row["final_label_agreement"] for row in sample_rows) / max(len(sample_rows), 1),
        "alert_state_agreement_rate": sum(row["first_alert_state_agreement"] for row in sample_rows) / max(len(sample_rows), 1),
        "mean_first_alert_abs_delta_sec": statistics.mean(comparable_alert_deltas) if comparable_alert_deltas else None,
        "max_first_alert_abs_delta_sec": max(comparable_alert_deltas, default=None),
        "median_chunk_latency_sec": percentile([row["latency_sec"] for row in route_rows], 0.5),
        "p95_chunk_latency_sec": percentile([row["latency_sec"] for row in route_rows], 0.95),
    }
    for key in SCORE_KEYS:
        differences = [float(row[f"abs_diff_{key}"]) for row in point_rows]
        route_summary[f"mae_{key}"] = statistics.mean(differences) if differences else 0.0
        route_summary[f"max_abs_diff_{key}"] = max(differences, default=0.0)

    browser_summary = {
        "samples": len(sample_rows),
        "chunks": len(browser_rows),
        "successful_chunks": sum(int(row["status_code"] == 200) for row in browser_rows),
        "median_latency_sec": percentile([row["latency_sec"] for row in browser_rows], 0.5),
        "p95_latency_sec": percentile([row["latency_sec"] for row in browser_rows], 0.95),
        "max_latency_sec": max((row["latency_sec"] for row in browser_rows), default=None),
        "median_realtime_factor": percentile([row["realtime_factor"] for row in browser_rows], 0.5),
        "p95_realtime_factor": percentile([row["realtime_factor"] for row in browser_rows], 0.95),
        "chunks_within_audio_duration_rate": sum(row["within_audio_duration"] for row in browser_rows) / max(len(browser_rows), 1),
        "final_label_agreement_rate_vs_offline": sum(row["browser_final_label_agreement"] for row in sample_rows) / max(len(sample_rows), 1),
        "alert_state_agreement_rate_vs_offline": sum(row["browser_alert_state_agreement"] for row in sample_rows) / max(len(sample_rows), 1),
        "mean_first_alert_abs_delta_sec_vs_offline": statistics.mean(browser_alert_deltas) if browser_alert_deltas else None,
        "max_first_alert_abs_delta_sec_vs_offline": max(browser_alert_deltas, default=None),
        "measurement_scope": "In-process Flask endpoint latency; excludes network transport and MediaRecorder encoding time.",
    }
    return {
        "protocol": {
            "route_equivalence": "Identical 10 s windows at 5 s steps sent sequentially to the live chunk endpoint.",
            "browser_latency": "Accelerated sequential replay of non-overlapping 5 s WAV chunks through the browser chunk endpoint.",
            "baseline": "fixed 0.8 text + 0.2 voice fusion with 0.65 previous-score smoothing",
            "alert_threshold": ALERT_THRESHOLD,
        },
        "sample_case_type_counts": dict(sorted(Counter(row["case_type"] for row in sample_rows).items())),
        "route_equivalence": route_summary,
        "browser_latency": browser_summary,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from server import app

    selected_rows = load_selected_rows(args.metadata, args.sample_id or DEFAULT_SAMPLE_IDS)
    sample_rows: List[Dict[str, Any]] = []
    point_rows: List[Dict[str, Any]] = []
    chunk_rows: List[Dict[str, Any]] = []
    raw_records: List[Dict[str, Any]] = []
    window_ms = int(args.window_seconds * 1000)
    step_ms = int(args.step_seconds * 1000)
    browser_chunk_ms = int(args.browser_chunk_seconds * 1000)

    with app.test_client() as client:
        for index, row in enumerate(selected_rows, start=1):
            audio_path = resolve_project_path(row["audio_path"])
            audio = AudioSegment.from_file(audio_path.as_posix())
            duration_ms = len(audio)
            offline = run_uploaded_route(client, row, args.window_seconds, args.step_seconds)
            route_live = post_live_segments(
                client,
                row,
                audio,
                route_segments(duration_ms, window_ms, step_ms),
                protocol="route_equivalence_10s_5s",
            )
            browser_live = post_live_segments(
                client,
                row,
                audio,
                browser_segments(duration_ms, browser_chunk_ms),
                protocol="browser_style_5s",
            )

            comparison, sample_point_rows = compare_route_timelines(
                row["sample_id"], offline["timeline"], route_live["timeline"]
            )
            offline_alert = first_alert_time(offline["timeline"])
            browser_alert = first_alert_time(browser_live["timeline"])
            both_browser_alert = offline_alert is not None and browser_alert is not None
            browser_latencies = [item["latency_sec"] for item in browser_live["chunk_rows"]]
            browser_rtfs = [item["realtime_factor"] for item in browser_live["chunk_rows"]]
            comparison.update({
                "case_type": row["case_type"],
                "label": normalize_label(row["label"]),
                "audio_duration_sec": duration_ms / 1000.0,
                "offline_total_elapsed_sec": offline["elapsed_sec"],
                "route_live_total_elapsed_sec": route_live["total_elapsed_sec"],
                "browser_chunk_count": len(browser_live["timeline"]),
                "browser_final_score": safe_float(browser_live["timeline"][-1].get("smoothed_score")),
                "browser_max_score": max_score(browser_live["timeline"]),
                "browser_first_alert_sec": browser_alert,
                "browser_alert_state_agreement": int((offline_alert is None) == (browser_alert is None)),
                "browser_first_alert_abs_delta_sec": abs(offline_alert - browser_alert) if both_browser_alert else None,
                "browser_final_label_agreement": int(
                    label_from_timeline(offline["timeline"]) == label_from_timeline(browser_live["timeline"])
                ),
                "browser_median_latency_sec": percentile(browser_latencies, 0.5),
                "browser_p95_latency_sec": percentile(browser_latencies, 0.95),
                "browser_median_realtime_factor": percentile(browser_rtfs, 0.5),
                "browser_chunks_within_duration_rate": sum(
                    item["within_audio_duration"] for item in browser_live["chunk_rows"]
                ) / max(len(browser_live["chunk_rows"]), 1),
            })
            sample_rows.append(comparison)
            point_rows.extend(sample_point_rows)
            chunk_rows.extend(route_live["chunk_rows"])
            chunk_rows.extend(browser_live["chunk_rows"])
            raw_records.append({
                "sample_id": row["sample_id"],
                "case_type": row["case_type"],
                "audio_path": row["audio_path"],
                "offline": offline,
                "route_equivalence_live": route_live,
                "browser_style_live": browser_live,
            })
            print(
                f"[{index:02d}/{len(selected_rows)}] {row['sample_id']} {row['case_type']} "
                f"route_mae={comparison['mae_smoothed_score']:.4f} "
                f"browser_p95={comparison['browser_p95_latency_sec']:.3f}s",
                flush=True,
            )

    report = aggregate_report(sample_rows, point_rows, chunk_rows)
    report["sample_ids"] = [row["sample_id"] for row in selected_rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(args.output_dir / "baseline_route_equivalence_latency_report.json", report)
    json_dump(args.output_dir / "baseline_route_equivalence_latency_records.json", {"records": raw_records})
    write_csv(args.output_dir / "baseline_route_equivalence_latency_samples.csv", sample_rows)
    write_csv(args.output_dir / "baseline_route_equivalence_points.csv", point_rows)
    write_csv(args.output_dir / "baseline_browser_chunk_latency.csv", chunk_rows)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--step-seconds", type=float, default=5.0)
    parser.add_argument("--browser-chunk-seconds", type=float, default=5.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.metadata = args.metadata.resolve()
    args.output_dir = args.output_dir.resolve()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
