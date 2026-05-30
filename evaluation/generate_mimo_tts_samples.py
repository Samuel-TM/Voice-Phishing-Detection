#!/usr/bin/env python3
"""Generate TTS risk samples from test_samples/tts.md using Mimo voice design API."""

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
DEFAULT_TTS_SPEC = PROJECT_ROOT / "test_samples/tts.md"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "test_samples/audio"
DEFAULT_METADATA = PROJECT_ROOT / "test_samples/metadata.csv"
MIMO_BASE_URL = "https://token-plan-sgp.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-tts-voicedesign"
ACTION_MARKER = "【ACTION_START】"

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
]


@dataclass(frozen=True)
class TtsSample:
    case_type: str
    sample_id: str
    fraud_action_start: str
    voice_design_prompt: str
    text: str


def normalize_sample_id(raw_id: str) -> str:
    text = str(raw_id or "").strip().upper()
    match = re.fullmatch(r"([A-Z]+)_?(\d+)", text)
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
) -> List[TtsSample]:
    selected: List[TtsSample] = []
    normalized_ids = {normalize_sample_id(item) for item in sample_ids}
    for sample in samples:
        if normalized_ids and sample.sample_id not in normalized_ids:
            continue
        if case_types and sample.case_type not in case_types:
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
    return {
        "sample_id": sample.sample_id,
        "audio_path": str(audio_path.relative_to(PROJECT_ROOT)),
        "label": metadata_label(sample.case_type),
        "case_type": sample.case_type,
        "event_time_sec": estimate_event_time_sec(sample, duration),
        "source": "mimo_tts_voicedesign",
        "start_sec": "0.000",
        "end_sec": duration,
        "source_wav": MIMO_MODEL,
        "notes": f"fraud_action_start={sample.fraud_action_start}; voice_design={sample.voice_design_prompt}",
    }


def generate_audio(client: Any, sample: TtsSample, output_path: Path) -> None:
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
            "optimize_text_preview": True,
        },
    )
    message = completion.choices[0].message
    if not getattr(message, "audio", None) or not getattr(message.audio, "data", None):
        raise RuntimeError(f"No audio data returned for {sample.sample_id}")
    audio_bytes = base64.b64decode(message.audio.data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(audio_bytes)


def generate_samples(args: argparse.Namespace) -> List[Dict[str, str]]:
    samples = parse_tts_markdown(args.tts_spec)
    selected = filter_samples(
        samples,
        sample_ids=set(args.sample_id or []),
        case_types=set(args.case_type or []),
        limit=args.limit,
    )

    if args.list:
        for sample in selected:
            print(
                f"{sample.sample_id},{sample.case_type},"
                f"fraud_action_start={sample.fraud_action_start},text={clean_tts_text(sample.text)[:50]}"
            )
        return []

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

    existing_rows = read_metadata(args.metadata)
    selected_ids = {sample.sample_id for sample in selected}
    kept_rows = [row for row in existing_rows if row.get("sample_id") not in selected_ids]
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
            generate_audio(client, sample, output_path)
        duration = wav_duration_seconds(output_path)
        generated_rows.append(make_metadata_row(sample, output_path, duration))

    write_metadata(args.metadata, kept_rows + generated_rows)
    return generated_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-spec", type=Path, default=DEFAULT_TTS_SPEC)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--base-url", default=MIMO_BASE_URL)
    parser.add_argument("--api-key", help="Optional API key. Prefer MIMO_API_KEY env var.")
    parser.add_argument("--sample-id", action="append", help="Generate one sample_id, e.g. SF_01 or SF01. Repeatable.")
    parser.add_argument("--case-type", action="append", help="Generate one case_type. Repeatable.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-only", action="store_true", help="Update metadata for existing generated audio only.")
    parser.add_argument("--list", action="store_true", help="List selected samples without calling the API.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.tts_spec = args.tts_spec.resolve()
    args.audio_dir = args.audio_dir.resolve()
    args.metadata = args.metadata.resolve()
    rows = generate_samples(args)
    if rows:
        print(f"metadata={args.metadata}")
        print(f"generated_count={len(rows)}")


if __name__ == "__main__":
    main()
