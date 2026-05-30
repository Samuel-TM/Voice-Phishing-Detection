#!/usr/bin/env python3
"""Generate dynamic audio predictions from test_samples/metadata.csv."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())
DEFAULT_METADATA = PROJECT_ROOT / "test_samples/metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".cache/evaluation_reports"
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "dynamic_predictions.json"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "dynamic_predictions_summary.csv"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value else 0
    text = str(value or "").strip().lower()
    return 1 if text in {"1", "true", "fraud", "phishing", "positive", "risk"} else 0


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def filter_rows(
    rows: Sequence[Dict[str, str]],
    sample_ids: set[str],
    case_types: set[str],
    limit: int | None,
) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    for row in rows:
        sample_id = row.get("sample_id", "").strip()
        case_type = row.get("case_type", "").strip()
        audio_path = resolve_project_path(row.get("audio_path", ""))
        if sample_ids and sample_id not in sample_ids:
            continue
        if case_types and case_type not in case_types:
            continue
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        if not audio_path.exists():
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def load_existing_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        records = data.get("records", [])
    else:
        records = data
    if not isinstance(records, list):
        return {}
    return {str(item.get("sample_id")): item for item in records if isinstance(item, dict)}


def write_json_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(list(records), handle, ensure_ascii=False, indent=2)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_stream_response(response_data: bytes) -> tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    events: List[Dict[str, Any]] = []
    error = ""
    for line in response_data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            error = line[:300]
    done = next((event for event in reversed(events) if event.get("event") == "done"), {})
    return events, done, error


def summarize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    timeline = record.get("timeline") or []
    highest = record.get("highest_risk_window") or {}
    text_scores = [safe_float(point.get("text_score")) for point in timeline]
    voice_scores = [safe_float(point.get("voice_score")) for point in timeline]
    fused_scores = [safe_float(point.get("fused_score")) for point in timeline]
    smoothed_scores = [safe_float(point.get("smoothed_score")) for point in timeline]
    return {
        "sample_id": record.get("sample_id", ""),
        "audio_path": record.get("audio_path", ""),
        "label": record.get("label", 0),
        "case_type": record.get("case_type", ""),
        "status_code": record.get("status_code", ""),
        "error": record.get("error", ""),
        "elapsed_sec": record.get("elapsed_sec", ""),
        "window_count": len(timeline),
        "final_score": round(safe_float(record.get("final_score")), 2),
        "max_score": round(safe_float(record.get("max_score")), 2),
        "false_positive_final": int(normalize_label(record.get("label")) == 0 and safe_float(record.get("final_score")) >= 70),
        "false_positive_alert": int(normalize_label(record.get("label")) == 0 and safe_float(record.get("max_score")) >= 70),
        "highest_index": highest.get("index", ""),
        "highest_start_sec": highest.get("start_sec", ""),
        "highest_end_sec": highest.get("end_sec", ""),
        "highest_text_score": round(safe_float(highest.get("text_score")), 2),
        "highest_voice_score": round(safe_float(highest.get("voice_score")), 2),
        "highest_fused_score": round(safe_float(highest.get("fused_score")), 2),
        "highest_smoothed_score": round(safe_float(highest.get("smoothed_score")), 2),
        "mean_text_score": round(statistics.mean(text_scores), 2) if text_scores else 0.0,
        "mean_voice_score": round(statistics.mean(voice_scores), 2) if voice_scores else 0.0,
        "mean_fused_score": round(statistics.mean(fused_scores), 2) if fused_scores else 0.0,
        "mean_smoothed_score": round(statistics.mean(smoothed_scores), 2) if smoothed_scores else 0.0,
        "highest_text": highest.get("text", ""),
        "transcript_preview": str(record.get("full_transcript", ""))[:160],
    }


def build_prediction_record(
    row: Dict[str, str],
    status_code: int,
    elapsed_sec: float,
    done: Dict[str, Any],
    timeline: List[Dict[str, Any]],
    error: str,
) -> Dict[str, Any]:
    max_point = max(timeline, key=lambda point: safe_float(point.get("smoothed_score")), default={})
    final_score = safe_float(done.get("final_score"), safe_float(timeline[-1].get("smoothed_score")) if timeline else 0.0)
    max_score = safe_float(done.get("max_score"), safe_float(max_point.get("smoothed_score")))
    event_time = row.get("event_time_sec", "")
    return {
        "sample_id": row.get("sample_id", ""),
        "label": normalize_label(row.get("label", 0)),
        "case_type": row.get("case_type", ""),
        "event_time_sec": None if event_time in ("", None) else safe_float(event_time),
        "audio_path": row.get("audio_path", ""),
        "source": row.get("source", ""),
        "window_seconds": done.get("window_seconds", 10),
        "step_seconds": done.get("step_seconds", 5),
        "scoring_mode": done.get("scoring_mode", ""),
        "timeline": timeline,
        "full_transcript": done.get("full_transcript", ""),
        "final_score": round(final_score, 4),
        "max_score": round(max_score, 4),
        "final_label": done.get("final_label", ""),
        "highest_risk_window": done.get("highest_risk_window") or max_point,
        "status_code": status_code,
        "elapsed_sec": round(elapsed_sec, 2),
        "error": error,
    }


def run_predictions(args: argparse.Namespace) -> List[Dict[str, Any]]:
    logging.getLogger().setLevel(logging.ERROR)
    for name in ["werkzeug", "speaker_analysis.whisper_stt", "ChineseBERTModel.ensemble_utils"]:
        logging.getLogger(name).setLevel(logging.ERROR)

    from server import app

    metadata_rows = load_metadata(args.metadata)
    selected = filter_rows(
        metadata_rows,
        sample_ids=set(args.sample_id or []),
        case_types=set(args.case_type or []),
        limit=args.limit,
    )

    existing = load_existing_predictions(args.predictions) if args.resume else {}
    records: Dict[str, Dict[str, Any]] = dict(existing)

    with app.test_client() as client:
        for index, row in enumerate(selected, start=1):
            sample_id = row.get("sample_id", "")
            if args.resume and sample_id in records and not records[sample_id].get("error"):
                print(f"[{index:02d}/{len(selected)}] {sample_id} skipped (resume)")
                continue

            audio_path = resolve_project_path(row.get("audio_path", ""))
            start = time.time()
            error = ""
            timeline: List[Dict[str, Any]] = []
            done: Dict[str, Any] = {}
            status_code = 0
            with audio_path.open("rb") as handle:
                response = client.post(
                    "/api/stream_audio_analysis",
                    data={
                        "audio_file": (handle, audio_path.name),
                        "window_seconds": str(args.window_seconds),
                        "step_seconds": str(args.step_seconds),
                        "scoring_mode": args.scoring_mode,
                        "case_type": row.get("case_type", ""),
                    },
                    content_type="multipart/form-data",
                    buffered=True,
                )
            status_code = response.status_code
            elapsed = time.time() - start
            if response.status_code == 200:
                events, done, parse_error = parse_stream_response(response.data)
                error = parse_error
                timeline = done.get("timeline") or [
                    event.get("point") for event in events if event.get("event") == "point"
                ]
                timeline = [point for point in timeline if point]
            else:
                error = response.data.decode("utf-8", errors="replace")[:500]

            records[sample_id] = build_prediction_record(row, status_code, elapsed, done, timeline, error)
            write_json_records(args.predictions, records.values())
            write_csv_rows(args.summary, [summarize_record(record) for record in records.values()])
            print(
                f"[{index:02d}/{len(selected)}] {sample_id} {row.get('case_type', '')} "
                f"final={records[sample_id]['final_score']:.2f} max={records[sample_id]['max_score']:.2f} "
                f"windows={len(timeline)} elapsed={elapsed:.1f}s error={bool(error)}",
                flush=True,
            )

    write_json_records(args.predictions, records.values())
    write_csv_rows(args.summary, [summarize_record(record) for record in records.values()])
    return list(records.values())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--window-seconds", type=float, default=10)
    parser.add_argument("--step-seconds", type=float, default=5)
    parser.add_argument("--scoring-mode", choices=["baseline", "gated_v1"], default="gated_v1")
    parser.add_argument("--sample-id", action="append", help="Run only this sample_id. Repeatable.")
    parser.add_argument("--case-type", action="append", help="Run only this case_type. Repeatable.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.metadata = args.metadata.resolve()
    args.predictions = args.predictions.resolve()
    args.summary = args.summary.resolve()
    records = run_predictions(args)
    print(f"wrote_predictions={args.predictions}")
    print(f"wrote_summary={args.summary}")
    print(f"records={len(records)}")


if __name__ == "__main__":
    main()
