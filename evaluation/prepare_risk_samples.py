#!/usr/bin/env python3
"""Import local fraud-risk candidate audio into test_samples/audio."""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/Users/sunjiashan/Material/HKU/Dissertation/Material/Dataset")
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "test_samples/audio"
DEFAULT_METADATA = PROJECT_ROOT / "test_samples/metadata.csv"

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
class ImportCandidate:
    sample_id: str
    case_type: str
    source_path: Path
    event_time_sec: str
    source: str
    notes: str


DEFAULT_CANDIDATES = [
    ImportCandidate(
        "SF_01",
        "semantic_fraud",
        DATASET_ROOT / "TAF/POS-imitate-4-tts_test1.mp3",
        "0",
        "taf_positive_candidate",
        "TAF POS semantic-fraud candidate; review transcript before final thesis reporting.",
    ),
    ImportCandidate(
        "SF_02",
        "semantic_fraud",
        DATASET_ROOT / "TAF/POS-imitate-4-tts_test10.mp3",
        "0",
        "taf_positive_candidate",
        "TAF POS semantic-fraud candidate; review transcript before final thesis reporting.",
    ),
    ImportCandidate(
        "SF_03",
        "semantic_fraud",
        DATASET_ROOT / "TAF/POS-imitate-7-tts_test1502.mp3",
        "0",
        "taf_positive_candidate",
        "TAF POS semantic-fraud candidate; review transcript before final thesis reporting.",
    ),
    ImportCandidate(
        "SV_01",
        "synthetic_voice",
        DATASET_ROOT / "Audio-Deepfake/fake/10.wav",
        "0",
        "audio_deepfake_fake",
        "Synthetic/fake voice candidate; semantic content may be neutral or unknown.",
    ),
    ImportCandidate(
        "SV_02",
        "synthetic_voice",
        DATASET_ROOT / "Audio-Deepfake/fake/10000.wav",
        "0",
        "audio_deepfake_fake",
        "Synthetic/fake voice candidate; semantic content may be neutral or unknown.",
    ),
    ImportCandidate(
        "SV_03",
        "synthetic_voice",
        DATASET_ROOT / "Audio-Deepfake/fake/10002.wav",
        "0",
        "audio_deepfake_fake",
        "Synthetic/fake voice candidate; semantic content may be neutral or unknown.",
    ),
    ImportCandidate(
        "MR_01",
        "mixed_risk",
        DATASET_ROOT / "TAF/33tool_output_PN.mp3",
        "0",
        "taf_mixed_candidate",
        "Mixed-risk candidate from TAF; verify semantic event timing before final reporting.",
    ),
    ImportCandidate(
        "MR_02",
        "mixed_risk",
        DATASET_ROOT / "TAF/33tool_output_NP.mp3",
        "0",
        "taf_mixed_candidate",
        "Mixed-risk candidate from TAF; verify semantic event timing before final reporting.",
    ),
]


def ffprobe_duration(path: Path) -> str:
    try:
        output = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            text=True,
        ).strip()
        return f"{float(output):.3f}"
    except Exception:
        return ""


def export_mp3(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(destination),
        ],
        check=True,
    )


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


def import_candidates(candidates: Sequence[ImportCandidate], audio_dir: Path, metadata_path: Path) -> List[Dict[str, str]]:
    imported_rows: List[Dict[str, str]] = []
    existing_rows = read_metadata(metadata_path)
    replacing_ids = {candidate.sample_id for candidate in candidates}
    kept_rows = [row for row in existing_rows if row.get("sample_id") not in replacing_ids]

    for candidate in candidates:
        if not candidate.source_path.exists():
            print(f"skip_missing={candidate.sample_id} source={candidate.source_path}")
            continue
        output_path = audio_dir / f"{candidate.sample_id}.mp3"
        export_mp3(candidate.source_path, output_path)
        duration = ffprobe_duration(output_path)
        row = {
            "sample_id": candidate.sample_id,
            "audio_path": str(output_path.relative_to(PROJECT_ROOT)),
            "label": "fraud",
            "case_type": candidate.case_type,
            "event_time_sec": candidate.event_time_sec,
            "source": candidate.source,
            "start_sec": "0.000",
            "end_sec": duration,
            "source_wav": candidate.source_path.name,
            "notes": candidate.notes,
        }
        imported_rows.append(row)
        print(f"imported={candidate.sample_id} case_type={candidate.case_type} duration={duration}")

    write_metadata(metadata_path, kept_rows + imported_rows)
    return imported_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--list-defaults", action="store_true", help="List default candidates without importing.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.dataset_root != DATASET_ROOT:
        candidates = [
            ImportCandidate(
                c.sample_id,
                c.case_type,
                args.dataset_root / c.source_path.relative_to(DATASET_ROOT),
                c.event_time_sec,
                c.source,
                c.notes,
            )
            for c in DEFAULT_CANDIDATES
        ]
    else:
        candidates = DEFAULT_CANDIDATES

    if args.list_defaults:
        for candidate in candidates:
            print(f"{candidate.sample_id},{candidate.case_type},{candidate.source_path},{candidate.source_path.exists()}")
        return

    imported = import_candidates(candidates, args.audio_dir.resolve(), args.metadata.resolve())
    print(f"metadata={args.metadata.resolve()}")
    print(f"imported_count={len(imported)}")


if __name__ == "__main__":
    main()
