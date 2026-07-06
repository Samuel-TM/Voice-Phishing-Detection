#!/usr/bin/env python3
"""Prepare and audit the frozen external stress evaluation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_METADATA = PROJECT_ROOT / "test_samples/metadata.csv"
FINAL_METADATA = PROJECT_ROOT / "test_samples/metadata_final.csv"
OUTPUT_METADATA = PROJECT_ROOT / "test_samples/metadata_external_frozen_v1.csv"
MODEL_PATH = PROJECT_ROOT / "evaluation/models/calibrated_late_fusion_w10_s5.joblib"
MANIFEST_PATH = PROJECT_ROOT / "evaluation/external_frozen_v1_manifest.json"

CORE_IDS = (
    [f"MD_N{index:02d}" for index in range(1, 11)]
    + [f"RAMC_N{index:02d}" for index in range(1, 11)]
    + [f"SF_{index:02d}" for index in range(1, 21)]
    + [f"MR_{index:02d}" for index in range(1, 21)]
    + [f"SV_{index:02d}" for index in range(1, 21)]
)
HARD_NEGATIVE_IDS = (
    [f"MD_F{index:02d}" for index in range(1, 11)]
    + [f"RAMC_F{index:02d}" for index in range(1, 11)]
)
SELECTED_IDS = CORE_IDS + HARD_NEGATIVE_IDS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return " ".join((value or "").split())


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_audio_path(row: Dict[str, str]) -> Path:
    path = Path(row.get("audio_path", ""))
    return path if path.is_absolute() else PROJECT_ROOT / path


def count_by(rows: Iterable[Dict[str, str]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(row.get(key, "") for row in rows).items()))


def prepare() -> Dict[str, Any]:
    source_rows = load_csv(SOURCE_METADATA)
    final_rows = load_csv(FINAL_METADATA)
    source_by_id = {row["sample_id"]: row for row in source_rows}
    missing_ids = [sample_id for sample_id in SELECTED_IDS if sample_id not in source_by_id]
    if missing_ids:
        raise ValueError(f"Selected sample IDs missing from metadata.csv: {missing_ids}")

    final_ids = {row["sample_id"] for row in final_rows}
    final_texts = {
        normalized_text(row.get("transcript_text", ""))
        for row in final_rows
        if normalized_text(row.get("transcript_text", ""))
    }
    final_audio_hashes = {sha256_file(resolve_audio_path(row)) for row in final_rows}

    selected_rows: List[Dict[str, str]] = []
    overlap_findings: List[Dict[str, str]] = []
    for sample_id in SELECTED_IDS:
        row = dict(source_by_id[sample_id])
        audio_path = resolve_audio_path(row)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Selected audio file is missing: {audio_path}")
        audio_sha256 = sha256_file(audio_path)
        text = normalized_text(row.get("transcript_text", ""))
        reasons = []
        if sample_id in final_ids:
            reasons.append("sample_id")
        if text and text in final_texts:
            reasons.append("transcript_text")
        if audio_sha256 in final_audio_hashes:
            reasons.append("audio_sha256")
        if reasons:
            overlap_findings.append({"sample_id": sample_id, "reasons": ",".join(reasons)})
        row["external_partition"] = "core_matched" if sample_id in CORE_IDS else "hard_negative"
        row["audio_sha256"] = audio_sha256
        selected_rows.append(row)

    if overlap_findings:
        raise ValueError(f"Selected samples overlap metadata_final.csv: {overlap_findings}")
    if len(selected_rows) != 100 or len({row["sample_id"] for row in selected_rows}) != 100:
        raise ValueError("external_frozen_v1 must contain exactly 100 unique samples.")

    fieldnames = list(source_rows[0].keys()) + ["external_partition", "audio_sha256"]
    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_METADATA.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    manifest = {
        "name": "external_frozen_v1",
        "protocol": "Frozen non-overlapping legacy-pool stress evaluation",
        "source_metadata": SOURCE_METADATA.relative_to(PROJECT_ROOT).as_posix(),
        "source_metadata_sha256": sha256_file(SOURCE_METADATA),
        "excluded_training_metadata": FINAL_METADATA.relative_to(PROJECT_ROOT).as_posix(),
        "excluded_training_metadata_sha256": sha256_file(FINAL_METADATA),
        "output_metadata": OUTPUT_METADATA.relative_to(PROJECT_ROOT).as_posix(),
        "output_metadata_sha256": sha256_file(OUTPUT_METADATA),
        "frozen_model": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "frozen_model_sha256": sha256_file(MODEL_PATH),
        "selection": {
            "records": len(selected_rows),
            "core_matched_records": len(CORE_IDS),
            "hard_negative_records": len(HARD_NEGATIVE_IDS),
            "case_type_counts": count_by(selected_rows, "case_type"),
            "source_counts": count_by(selected_rows, "source"),
            "partition_counts": count_by(selected_rows, "external_partition"),
            "sample_ids": [row["sample_id"] for row in selected_rows],
        },
        "leakage_audit": {
            "sample_id_overlap_with_metadata_final": 0,
            "nonempty_transcript_overlap_with_metadata_final": 0,
            "audio_sha256_overlap_with_metadata_final": 0,
        },
        "freeze_rules": [
            "Do not fit or calibrate the learned late-fusion model on these samples.",
            "Do not scan or change the saved decision threshold on these samples.",
            "Do not alter sample membership after inspecting evaluation results.",
            "Report core_matched (80) and expanded_with_hard_negatives (100) separately.",
        ],
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def load_prediction_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Prediction file has no record list: {path}")
    return records


def finalize(learned_predictions: Path, core_output: Path, audit_output: Path) -> Dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current_model_sha256 = sha256_file(MODEL_PATH)
    if current_model_sha256 != manifest["frozen_model_sha256"]:
        raise ValueError("Frozen model hash differs from the prepared manifest.")
    records = load_prediction_records(learned_predictions)
    by_id = {str(record.get("sample_id")): record for record in records}
    missing = sorted(set(SELECTED_IDS) - set(by_id))
    unexpected = sorted(set(by_id) - set(SELECTED_IDS))
    errors = [
        {"sample_id": sample_id, "error": by_id[sample_id].get("error", "")}
        for sample_id in SELECTED_IDS
        if sample_id in by_id and by_id[sample_id].get("error")
    ]
    if missing or unexpected or errors:
        raise ValueError({"missing": missing, "unexpected": unexpected, "errors": errors})
    core_records = [by_id[sample_id] for sample_id in CORE_IDS]
    write_json(core_output, {"records": core_records})
    audit = {
        "status": "ready_for_metrics",
        "learned_predictions": learned_predictions.as_posix(),
        "learned_predictions_sha256": sha256_file(learned_predictions),
        "expanded_records": len(records),
        "core_records": len(core_records),
        "prediction_errors": 0,
        "frozen_model_sha256": current_model_sha256,
        "frozen_model_unchanged": True,
        "case_type_counts_expanded": count_by(records, "case_type"),
        "case_type_counts_core": count_by(core_records, "case_type"),
    }
    write_json(audit_output, audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--learned-predictions", required=True, type=Path)
    finalize_parser.add_argument("--core-output", required=True, type=Path)
    finalize_parser.add_argument("--audit-output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        payload = prepare()
    else:
        payload = finalize(
            learned_predictions=args.learned_predictions.resolve(),
            core_output=args.core_output.resolve(),
            audit_output=args.audit_output.resolve(),
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
