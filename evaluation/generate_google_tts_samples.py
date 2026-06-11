#!/usr/bin/env python3
"""Generate Google TTS audio for selected long evaluation samples."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TTS_SPEC = PROJECT_ROOT / "test_samples/tts_final.md"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "test_samples/audio_final"
PROJECT_CACHE_DIR = PROJECT_ROOT / ".cache/google_tts"
ACTION_MARKER = "【ACTION_START】"
DEFAULT_CASE_TYPES = ("mixed_risk",)
DEFAULT_SAMPLE_IDS: tuple[str, ...] = ()


@dataclass(frozen=True)
class TtsSample:
    case_type: str
    sample_id: str
    fraud_action_start: str
    text: str


def normalize_sample_id(raw_id: str) -> str:
    text = str(raw_id or "").strip()
    match = re.fullmatch(r"([A-Za-z]+(?:_[A-Za-z]+)*)_?(\d+)", text)
    if not match:
        return text
    return f"{match.group(1)}_{int(match.group(2)):02d}"


def clean_tts_text(text: str) -> str:
    return str(text or "").replace(ACTION_MARKER, "")


def split_markdown_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_tts_markdown(path: Path) -> list[TtsSample]:
    if not path.exists():
        raise FileNotFoundError(f"TTS spec not found: {path}")

    rows: list[TtsSample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue

        cells = split_markdown_row(line)
        if len(cells) < 5:
            continue

        normalized_header = cells[0].strip().strip("*").strip().lower()
        if normalized_header == "case_type" or set(cells[0]) <= {"-"}:
            continue

        case_type, sample_id, fraud_action_start, _voice_design_prompt, text = cells[:5]
        if not case_type or not sample_id:
            continue

        rows.append(
            TtsSample(
                case_type=case_type.strip(),
                sample_id=normalize_sample_id(sample_id),
                fraud_action_start=fraud_action_start.strip(),
                text=text.strip(),
            )
        )
    return rows


def filter_samples(
    samples: Sequence[TtsSample],
    sample_ids: Sequence[str],
    case_types: Sequence[str],
    generate_all: bool,
    limit: int | None,
) -> list[TtsSample]:
    normalized_ids = {normalize_sample_id(item) for item in sample_ids}
    selected: list[TtsSample] = []

    for sample in samples:
        if sample.case_type not in set(case_types):
            continue
        if not generate_all and normalized_ids and sample.sample_id not in normalized_ids:
            continue
        selected.append(sample)
        if limit is not None and len(selected) >= limit:
            break

    return selected


def configure_project_cache() -> None:
    PROJECT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(PROJECT_CACHE_DIR)
    os.environ["TEMP"] = str(PROJECT_CACHE_DIR)
    os.environ["TMP"] = str(PROJECT_CACHE_DIR)


def require_gtts():
    try:
        from gtts import gTTS
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: gtts. Install it inside the dissertation environment first:\n"
            "  conda activate dissertation\n"
            "  python -m pip install gTTS\n"
        ) from exc
    return gTTS


def generate_audio(sample: TtsSample, output_path: Path, lang: str, tld: str, slow: bool) -> None:
    configure_project_cache()
    gTTS = require_gtts()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = clean_tts_text(sample.text)
    tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
    tts.save(output_path.as_posix())


def generate_samples(args: argparse.Namespace) -> list[Path]:
    samples = parse_tts_markdown(args.tts_spec)
    selected = filter_samples(
        samples=samples,
        sample_ids=args.sample_id,
        case_types=args.case_type,
        generate_all=args.all,
        limit=args.limit,
    )

    if not selected:
        raise SystemExit("No samples selected.")

    if args.list:
        for sample in selected:
            preview = clean_tts_text(sample.text)[:60]
            print(f"{sample.sample_id},{sample.case_type},fraud_action_start={sample.fraud_action_start},text={preview}")
        return []

    generated: list[Path] = []
    for index, sample in enumerate(selected, start=1):
        output_path = args.audio_dir / f"{sample.sample_id}.mp3"
        if output_path.exists() and not args.overwrite:
            print(f"[{index:02d}/{len(selected)}] skip_existing {sample.sample_id} {output_path}")
        else:
            print(f"[{index:02d}/{len(selected)}] generating {sample.sample_id} {sample.case_type}")
            generate_audio(sample, output_path, args.lang, args.tld, args.slow)
        generated.append(output_path)

    return generated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-spec", type=Path, default=DEFAULT_TTS_SPEC)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument(
        "--sample-id",
        action="append",
        help="Generate one sample_id, e.g. MR_long_01. Repeatable. Defaults to all mixed_risk samples.",
    )
    parser.add_argument(
        "--case-type",
        action="append",
        choices=DEFAULT_CASE_TYPES,
        help="Allowed case_type. Defaults to mixed_risk.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all selected case types even when --sample-id is provided.",
    )
    parser.add_argument("--limit", type=int, help="Limit generated samples after filtering.")
    parser.add_argument("--lang", default="zh-CN", help="gTTS language code.")
    parser.add_argument("--tld", default="com", help="Google Translate TTS top-level domain.")
    parser.add_argument("--slow", action="store_true", help="Use gTTS slow mode.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true", help="List selected samples without calling Google TTS.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.sample_id is None:
        args.sample_id = list(DEFAULT_SAMPLE_IDS)
    if args.case_type is None:
        args.case_type = list(DEFAULT_CASE_TYPES)
    args.tts_spec = args.tts_spec.resolve()
    args.audio_dir = args.audio_dir.resolve()
    generated = generate_samples(args)
    if generated:
        print(f"generated_count={len(generated)}")
        for path in generated:
            print(path)


if __name__ == "__main__":
    main()
