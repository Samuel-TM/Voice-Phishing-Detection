#!/usr/bin/env python3
"""Generate final semantic-fraud TTS samples using Mimo voice design API."""

from __future__ import annotations

import argparse
import base64
import csv
import os
import re
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TTS_SPEC = PROJECT_ROOT / "test_samples/tts_final.md"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "test_samples/audio_final"
DEFAULT_SOURCE_METADATA = PROJECT_ROOT / "test_samples/metadata.csv"
DEFAULT_METADATA = PROJECT_ROOT / "test_samples/metadata_final.csv"
MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-tts-voicedesign"
ACTION_MARKER = "【ACTION_START】"
DEFAULT_CASE_TYPES = ("semantic_fraud",)
FINAL_SAMPLE_GROUPS = (
    ("normal_daily", "ND_long"),
    ("semantic_fraud", "SF_long"),
    ("mixed_risk", "MR_long"),
    ("synthetic_voice", "SV_long"),
)
FINAL_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

METADATA_FIELDS = [
    "sample_id",
    "audio_path",
    "label",
    "case_type",
    "event_time_sec",
    "source",
    "start_sec",
    "end_sec",
    "source_wav",
    "notes",
    "transcript_text",
]


@dataclass(frozen=True)
class TtsSample:
    case_type: str
    sample_id: str
    fraud_action_start: str
    voice_design_prompt: str
    text: str


def normalize_sample_id(raw_id: str) -> str:
    text = str(raw_id or "").strip()
    match = re.fullmatch(r"([A-Za-z]+(?:_[A-Za-z]+)*)_?(\d+)", text)
    if not match:
        return text
    return f"{match.group(1)}_{int(match.group(2)):02d}"


def metadata_label(case_type: str) -> str:
    return "fraud" if case_type in {"semantic_fraud", "synthetic_voice", "mixed_risk"} else "normal"


def clean_tts_text(text: str) -> str:
    return str(text or "").replace(ACTION_MARKER, "")


def estimate_event_time_sec(sample: TtsSample, duration_text: str) -> str:
    duration = 0.0
    try:
        duration = float(duration_text)
    except Exception:
        duration = 0.0

    if sample.case_type == "synthetic_voice":
        return "0"
    if ACTION_MARKER not in sample.text or duration <= 0:
        return ""

    before, after = sample.text.split(ACTION_MARKER, 1)
    before_len = len(before.strip())
    total_len = len((before + after).strip())
    if total_len <= 0:
        return ""
    return f"{duration * before_len / total_len:.2f}"


def split_markdown_row(line: str) -> List[str]:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return cells


def parse_tts_markdown(path: Path) -> List[TtsSample]:
    if not path.exists():
        raise FileNotFoundError(f"TTS spec not found: {path}")

    rows: List[TtsSample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = split_markdown_row(line)
        if len(cells) < 5:
            continue
        if cells[0] in {"case_type", "---------------"} or set(cells[0]) <= {"-"}:
            continue
        if cells[0].startswith("-"):
            continue
        normalized_header = cells[0].strip().strip("*").strip().lower()
        if normalized_header == "case_type":
            continue
        case_type, sample_id, fraud_action_start, voice_prompt, text = cells[:5]
        if not case_type or not sample_id:
            continue
        rows.append(
            TtsSample(
                case_type=case_type.strip(),
                sample_id=normalize_sample_id(sample_id),
                fraud_action_start=fraud_action_start.strip(),
                voice_design_prompt=voice_prompt.strip(),
                text=text.strip(),
            )
        )
    return rows


def filter_samples(
    samples: Sequence[TtsSample],
    sample_ids: set[str],
    case_types: set[str],
    limit: int | None,
    min_number: int | None,
    max_number: int | None,
) -> List[TtsSample]:
    selected: List[TtsSample] = []
    normalized_ids = {normalize_sample_id(item) for item in sample_ids}
    for sample in samples:
        number_match = re.fullmatch(r"[A-Za-z]+(?:_[A-Za-z]+)*_(\d+)", sample.sample_id)
        sample_number = int(number_match.group(1)) if number_match else None
        if normalized_ids and sample.sample_id not in normalized_ids:
            continue
        if case_types and sample.case_type not in case_types:
            continue
        if min_number is not None and sample_number is not None and sample_number < min_number:
            continue
        if max_number is not None and sample_number is not None and sample_number > max_number:
            continue
        selected.append(sample)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def require_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install it inside the dissertation environment first:\n"
            "  conda activate dissertation\n"
            "  pip install openai\n"
        ) from exc
    return OpenAI


def audio_duration_seconds(path: Path) -> str:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(path.as_posix(), "rb") as handle:
                duration = handle.getnframes() / float(handle.getframerate())
            return f"{duration:.3f}"
        except Exception:
            pass

    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(path.as_posix())
        return f"{len(audio) / 1000.0:.3f}"
    except Exception:
        return ""


def wav_duration_seconds(path: Path) -> str:
    try:
        with wave.open(path.as_posix(), "rb") as handle:
            duration = handle.getnframes() / float(handle.getframerate())
        return f"{duration:.3f}"
    except Exception:
        return ""


def read_metadata(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({field: row.get(field, "") for field in METADATA_FIELDS})
        return rows


def write_metadata(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_metadata_row(sample: TtsSample, audio_path: Path, duration: str) -> Dict[str, str]:
    source = "mimo_tts_voicedesign"
    source_wav = MIMO_MODEL
    suffix = audio_path.suffix.lower()
    if sample.case_type == "normal_daily":
        source = "real_recording"
        source_wav = audio_path.name
    elif suffix == ".mp3":
        source = "google_tts"
        source_wav = "gTTS zh-CN"

    return {
        "sample_id": sample.sample_id,
        "audio_path": str(audio_path.relative_to(PROJECT_ROOT)),
        "label": metadata_label(sample.case_type),
        "case_type": sample.case_type,
        "event_time_sec": estimate_event_time_sec(sample, duration),
        "source": source,
        "start_sec": "0.000",
        "end_sec": duration,
        "source_wav": source_wav,
        "notes": f"fraud_action_start={sample.fraud_action_start}; voice_design={sample.voice_design_prompt}",
        "transcript_text": clean_tts_text(sample.text),
    }


def make_final_sample_ids() -> List[str]:
    return [f"{prefix}_{index:02d}" for _case_type, prefix in FINAL_SAMPLE_GROUPS for index in range(1, 21)]


def find_final_audio_path(sample_id: str, audio_dir: Path) -> Path:
    for suffix in FINAL_AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    return audio_dir / f"{sample_id}.wav"


def build_final_metadata_rows(
    samples: Sequence[TtsSample],
    source_metadata_path: Path,
    audio_dir: Path,
) -> List[Dict[str, str]]:
    source_rows = {row.get("sample_id", ""): row for row in read_metadata(source_metadata_path)}
    spec_rows = {sample.sample_id: sample for sample in samples}

    rows: List[Dict[str, str]] = []
    missing_audio: List[str] = []
    missing_spec: List[str] = []
    for sample_id in make_final_sample_ids():
        sample = spec_rows.get(sample_id)
        if sample is None:
            missing_spec.append(sample_id)
            continue

        audio_path = find_final_audio_path(sample_id, audio_dir)
        if not audio_path.exists():
            missing_audio.append(sample_id)
            continue

        duration = audio_duration_seconds(audio_path)
        row = make_metadata_row(sample, audio_path, duration)
        base_row = source_rows.get(sample_id, {})
        if not row["end_sec"]:
            row["end_sec"] = base_row.get("end_sec", "")
        if not row["event_time_sec"]:
            row["event_time_sec"] = base_row.get("event_time_sec", "")
        rows.append(row)

    if missing_spec:
        raise SystemExit(f"Missing final sample specs in TTS markdown: {', '.join(missing_spec)}")
    if missing_audio:
        raise SystemExit(f"Missing final audio files under {audio_dir}: {', '.join(missing_audio)}")
    return rows


def generate_audio(client: Any, sample: TtsSample, output_path: Path, optimize_text_preview: bool) -> None:
    completion = client.chat.completions.create(
        model=MIMO_MODEL,
        messages=[
            {
                "role": "user",
                "content": sample.voice_design_prompt,
            },
            {
                "role": "assistant",
                "content": clean_tts_text(sample.text),
            },
        ],
        audio={
            "format": "wav",
            "optimize_text_preview": optimize_text_preview,
        },
    )
    message = completion.choices[0].message
    if not getattr(message, "audio", None) or not getattr(message.audio, "data", None):
        raise RuntimeError(f"No audio data returned for {sample.sample_id}")
    audio_bytes = base64.b64decode(message.audio.data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)


def generate_samples(args: argparse.Namespace) -> Dict[str, Any]:
    samples = parse_tts_markdown(args.tts_spec)
    selected = filter_samples(
        samples,
        sample_ids=set(args.sample_id or []),
        case_types=set(args.case_type or []),
        limit=args.limit,
        min_number=args.min_number,
        max_number=args.max_number,
    )

    if not selected:
        raise SystemExit("No samples selected.")

    if args.list:
        for sample in selected:
            print(
                f"{sample.sample_id},{sample.case_type},"
                f"fraud_action_start={sample.fraud_action_start},text={clean_tts_text(sample.text)[:50]}"
            )
        return {"generated_rows": [], "metadata_rows": []}

    client = None
    if not args.metadata_only:
        api_key = args.api_key or os.environ.get("MIMO_API_KEY")
        if not api_key:
            raise SystemExit(
                "MIMO_API_KEY is not set. Run one of these before generation:\n"
                "  export MIMO_API_KEY='your_api_key_here'\n"
                "or pass:\n"
                "  --api-key 'your_api_key_here'\n"
            )

        OpenAI = require_openai_client()
        client = OpenAI(api_key=api_key, base_url=args.base_url)

    generated_rows: List[Dict[str, str]] = []

    for index, sample in enumerate(selected, start=1):
        output_path = args.audio_dir / f"{sample.sample_id}.wav"
        if args.metadata_only and not output_path.exists():
            raise FileNotFoundError(f"Cannot update metadata; audio file does not exist: {output_path}")
        if args.metadata_only:
            print(f"[{index:02d}/{len(selected)}] metadata_only {sample.sample_id} {output_path}")
        elif output_path.exists() and not args.overwrite:
            print(f"[{index:02d}/{len(selected)}] skip_existing {sample.sample_id} {output_path}")
        else:
            print(f"[{index:02d}/{len(selected)}] generating {sample.sample_id} {sample.case_type}")
            generate_audio(client, sample, output_path, args.optimize_text_preview)
        duration = wav_duration_seconds(output_path)
        generated_rows.append(make_metadata_row(sample, output_path, duration))

    final_rows = build_final_metadata_rows(samples, args.source_metadata, args.audio_dir)
    write_metadata(args.metadata, final_rows)
    return {"generated_rows": generated_rows, "metadata_rows": final_rows}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-spec", type=Path, default=DEFAULT_TTS_SPEC)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--source-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--base-url", default=MIMO_BASE_URL)
    parser.add_argument("--api-key", help="Optional API key. Prefer MIMO_API_KEY env var.")
    parser.add_argument("--sample-id", action="append", help="Generate one sample_id, e.g. SF_long_01. Repeatable.")
    parser.add_argument(
        "--case-type",
        action="append",
        choices=DEFAULT_CASE_TYPES,
        help="Generate one case_type. Defaults to semantic_fraud.",
    )
    parser.add_argument("--min-number", type=int, help="Generate sample IDs with numeric suffix >= this value.")
    parser.add_argument("--max-number", type=int, help="Generate sample IDs with numeric suffix <= this value.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="Update metadata for existing generated audio only.")
    parser.add_argument(
        "--optimize-text-preview",
        action="store_true",
        help="Allow the TTS service to optimize text before synthesis. Off by default to preserve scripted content.",
    )
    parser.add_argument("--list", action="store_true", help="List selected samples without calling the API.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.case_type is None:
        args.case_type = list(DEFAULT_CASE_TYPES)
    args.tts_spec = args.tts_spec.resolve()
    args.audio_dir = args.audio_dir.resolve()
    args.source_metadata = args.source_metadata.resolve()
    args.metadata = args.metadata.resolve()
    result = generate_samples(args)
    if result["metadata_rows"]:
        print(f"metadata={args.metadata}")
        print(f"generated_count={len(result['generated_rows'])}")
        print(f"metadata_count={len(result['metadata_rows'])}")


if __name__ == "__main__":
    main()
