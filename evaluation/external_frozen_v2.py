#!/usr/bin/env python3
"""Prepare, audit, freeze, and verify the paired external_frozen_v2 benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "test_samples/tts_external_frozen_v2.md"
FINAL_SPEC_PATH = PROJECT_ROOT / "test_samples/tts_final.md"
AUDIO_DIR = PROJECT_ROOT / "test_samples/audio_external_frozen_v2"
METADATA_PATH = PROJECT_ROOT / "test_samples/metadata_external_frozen_v2.csv"
MODEL_PATH = PROJECT_ROOT / "evaluation/models/causal_late_fusion_v2_w10_s5.joblib"
MANIFEST_PATH = PROJECT_ROOT / "evaluation/external_frozen_v2_manifest.json"
CACHE_DIR = PROJECT_ROOT / ".cache/external_frozen_v2"
TEXT_AUDIT_PATH = CACHE_DIR / "text_audit.json"
AUDIO_QC_PATH = CACHE_DIR / "audio_technical_qc.json"
ASR_QC_PATH = CACHE_DIR / "asr_intelligibility_qc.json"
ACTION_MARKER = "【ACTION_START】"
EXPECTED_PER_CASE = 20
CASE_PREFIX = {
    "normal_daily": "EV2_ND",
    "normal_finance": "EV2_NF",
    "synthetic_voice": "EV2_SV",
    "mixed_risk": "EV2_MR",
    "semantic_fraud": "EV2_SF",
}
CASE_ENGINE = {
    "normal_daily": "mimo",
    "normal_finance": "mimo",
    "synthetic_voice": "google",
    "mixed_risk": "google",
    "semantic_fraud": "mimo",
}
PAIR_CASES = (("normal_daily", "synthetic_voice"), ("semantic_fraud", "mixed_risk"))
METADATA_FIELDS = [
    "sample_id", "audio_path", "label", "case_type", "event_time_sec", "source",
    "start_sec", "end_sec", "source_wav", "notes", "transcript_text",
    "script_id", "semantic_condition", "acoustic_condition", "tts_engine", "paired_sample_id",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(text or "").replace(ACTION_MARKER, "")).lower()


def ngrams(text: str, size: int = 5) -> set[str]:
    return {text[index:index + size] for index in range(max(0, len(text) - size + 1))}


def similarity(left: str, right: str) -> Tuple[float, float]:
    left_norm, right_norm = normalize_text(left), normalize_text(right)
    sequence = SequenceMatcher(None, left_norm, right_norm).ratio()
    left_grams, right_grams = ngrams(left_norm), ngrams(right_norm)
    union = left_grams | right_grams
    jaccard = len(left_grams & right_grams) / len(union) if union else 0.0
    return sequence, jaccard


def split_row(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_spec(path: Path) -> List[Dict[str, str]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = split_row(line)
        if len(cells) < 5 or cells[0].strip("*").lower() == "case_type" or set(cells[0]) <= {"-"}:
            continue
        case_type, sample_id, fraud_action_start, voice_prompt, text = cells[:5]
        if case_type in CASE_PREFIX:
            rows.append({
                "case_type": case_type,
                "sample_id": sample_id,
                "fraud_action_start": fraud_action_start,
                "voice_design_prompt": voice_prompt,
                "text": text,
            })
    return rows


def row_number(row: Dict[str, str]) -> int:
    match = re.fullmatch(r"[A-Za-z0-9_]+_(\d+)", row["sample_id"])
    if not match:
        raise ValueError(f"Invalid sample_id: {row['sample_id']}")
    return int(match.group(1))


def expected_sample_id(case_type: str, number: int) -> str:
    return f"{CASE_PREFIX[case_type]}_{number:02d}"


def audit_text(spec_path: Path = SPEC_PATH) -> Dict[str, Any]:
    rows = parse_spec(spec_path)
    errors: List[str] = []
    if len(rows) != 100:
        errors.append(f"expected 100 rows, found {len(rows)}")
    by_case = {case: [] for case in CASE_PREFIX}
    by_id = {}
    for row in rows:
        by_case[row["case_type"]].append(row)
        if row["sample_id"] in by_id:
            errors.append(f"duplicate sample_id: {row['sample_id']}")
        by_id[row["sample_id"]] = row
        if not normalize_text(row["text"]):
            errors.append(f"empty text: {row['sample_id']}")
    for case_type, case_rows in by_case.items():
        if len(case_rows) != EXPECTED_PER_CASE:
            errors.append(f"{case_type}: expected 20 rows, found {len(case_rows)}")
        expected = {expected_sample_id(case_type, number) for number in range(1, 21)}
        actual = {row["sample_id"] for row in case_rows}
        if expected != actual:
            errors.append(f"{case_type}: sample IDs do not match frozen naming contract")

    unique_scripts = list(by_case["normal_finance"])
    for first_case, second_case in PAIR_CASES:
        for number in range(1, 21):
            first = by_id.get(expected_sample_id(first_case, number))
            second = by_id.get(expected_sample_id(second_case, number))
            if not first or not second:
                continue
            if normalize_text(first["text"]) != normalize_text(second["text"]):
                errors.append(f"paired text mismatch: {first['sample_id']} vs {second['sample_id']}")
            is_fraud = first_case == "semantic_fraud"
            for row in (first, second):
                if is_fraud and ACTION_MARKER not in row["text"]:
                    errors.append(f"missing action marker: {row['sample_id']}")
                if not is_fraud and ACTION_MARKER in row["text"]:
                    errors.append(f"unexpected action marker: {row['sample_id']}")
            unique_scripts.append(first)

    near_duplicates = []
    for index, left in enumerate(unique_scripts):
        for right in unique_scripts[index + 1:]:
            sequence, jaccard = similarity(left["text"], right["text"])
            if sequence >= 0.82 or jaccard >= 0.62:
                near_duplicates.append({"left": left["sample_id"], "right": right["sample_id"], "sequence": round(sequence, 4), "jaccard_5gram": round(jaccard, 4)})
    if near_duplicates:
        errors.append(f"within-v2 near duplicates: {len(near_duplicates)}")

    final_rows = parse_spec(FINAL_SPEC_PATH)
    final_near_duplicates = []
    for current in unique_scripts:
        for prior in final_rows:
            sequence, jaccard = similarity(current["text"], prior["text"])
            if sequence >= 0.72 or jaccard >= 0.45:
                final_near_duplicates.append({"v2": current["sample_id"], "final": prior["sample_id"], "sequence": round(sequence, 4), "jaccard_5gram": round(jaccard, 4)})
    if final_near_duplicates:
        errors.append(f"near duplicates with tts_final.md: {len(final_near_duplicates)}")

    report = {
        "status": "pass" if not errors else "fail",
        "spec": spec_path.relative_to(PROJECT_ROOT).as_posix(),
        "rows": len(rows),
        "unique_scripts": len(unique_scripts),
        "case_counts": {case: len(case_rows) for case, case_rows in by_case.items()},
        "within_v2_near_duplicates": near_duplicates,
        "final_set_near_duplicates": final_near_duplicates,
        "errors": errors,
    }
    write_json(TEXT_AUDIT_PATH, report)
    return report


def audio_path_for(row: Dict[str, str]) -> Path:
    extension = ".wav" if CASE_ENGINE[row["case_type"]] == "mimo" else ".mp3"
    return AUDIO_DIR / f"{row['sample_id']}{extension}"


def audio_technical_qc(spec_path: Path = SPEC_PATH) -> Dict[str, Any]:
    rows = parse_spec(spec_path)
    results, errors = [], []
    for row in rows:
        path = audio_path_for(row)
        item: Dict[str, Any] = {"sample_id": row["sample_id"], "path": path.relative_to(PROJECT_ROOT).as_posix()}
        if not path.is_file() or path.stat().st_size == 0:
            item.update({"status": "fail", "error": "missing_or_empty"})
            errors.append(item)
            results.append(item)
            continue
        try:
            audio = AudioSegment.from_file(path)
            duration = len(audio) / 1000.0
            nonsilent = detect_nonsilent(audio, min_silence_len=500, silence_thresh=max(audio.dBFS - 18.0, -50.0))
            nonsilent_ms = sum(end - start for start, end in nonsilent)
            coverage = nonsilent_ms / max(len(audio), 1)
            leading = nonsilent[0][0] / 1000.0 if nonsilent else duration
            trailing = (len(audio) - nonsilent[-1][1]) / 1000.0 if nonsilent else duration
            reasons = []
            if not 25.0 <= duration <= 80.0:
                reasons.append("duration_out_of_range")
            if audio.dBFS < -48.0 or coverage < 0.35:
                reasons.append("mostly_silent_or_too_quiet")
            if leading > 4.0 or trailing > 4.0:
                reasons.append("excessive_edge_silence")
            item.update({
                "status": "pass" if not reasons else "fail",
                "duration_sec": round(duration, 3),
                "dbfs": round(audio.dBFS, 3),
                "nonsilent_coverage": round(coverage, 4),
                "leading_silence_sec": round(leading, 3),
                "trailing_silence_sec": round(trailing, 3),
                "reasons": reasons,
                "sha256": sha256_file(path),
            })
            if reasons:
                errors.append(item)
        except Exception as exc:
            item.update({"status": "fail", "error": str(exc)})
            errors.append(item)
        results.append(item)
    report = {"status": "pass" if len(results) == 100 and not errors else "fail", "samples": len(results), "failed": len(errors), "results": results}
    write_json(AUDIO_QC_PATH, report)
    return report


def event_time(row: Dict[str, str], duration: float) -> str:
    if row["case_type"] == "synthetic_voice":
        return "0.00"
    if row["case_type"] in {"normal_daily", "normal_finance"}:
        return ""
    before, marker, after = row["text"].partition(ACTION_MARKER)
    if not marker:
        return ""
    return f"{duration * len(normalize_text(before)) / max(len(normalize_text(before + after)), 1):.2f}"


def build_metadata(spec_path: Path = SPEC_PATH) -> Dict[str, Any]:
    text_report = audit_text(spec_path)
    audio_report = audio_technical_qc(spec_path)
    if text_report["status"] != "pass" or audio_report["status"] != "pass":
        raise ValueError("Text or technical audio QC failed; metadata was not frozen.")
    technical_by_id = {item["sample_id"]: item for item in audio_report["results"]}
    rows = parse_spec(spec_path)
    metadata = []
    for row in rows:
        number = row_number(row)
        case_type = row["case_type"]
        engine = CASE_ENGINE[case_type]
        paired_case = (
            next(
                second if case_type == first else first
                for first, second in PAIR_CASES
                if case_type in {first, second}
            )
            if case_type != "normal_finance"
            else ""
        )
        duration = float(technical_by_id[row["sample_id"]]["duration_sec"])
        semantic = "benign" if case_type in {"normal_daily", "normal_finance", "synthetic_voice"} else "fraud"
        metadata.append({
            "sample_id": row["sample_id"],
            "audio_path": audio_path_for(row).relative_to(PROJECT_ROOT).as_posix(),
            "label": "normal" if case_type in {"normal_daily", "normal_finance"} else "fraud",
            "case_type": case_type,
            "event_time_sec": event_time(row, duration),
            "source": "mimo_tts_voicedesign" if engine == "mimo" else "google_tts",
            "start_sec": "0.000",
            "end_sec": f"{duration:.3f}",
            "source_wav": "mimo-v2.5-tts-voicedesign" if engine == "mimo" else "gTTS zh-CN",
            "notes": f"frozen_v2 paired controlled holdout; fraud_action_start={row['fraud_action_start']}; voice_design={row['voice_design_prompt']}",
            "transcript_text": row["text"].replace(ACTION_MARKER, ""),
            "script_id": f"EV2_{'NF' if case_type == 'normal_finance' else ('B' if semantic == 'benign' else 'F')}_{number:02d}",
            "semantic_condition": semantic,
            "acoustic_condition": "naturalistic_tts" if engine == "mimo" else "obvious_synthetic",
            "tts_engine": engine,
            "paired_sample_id": expected_sample_id(paired_case, number) if paired_case else "",
        })
    with METADATA_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(metadata)
    return {"status": "ready_for_asr_qc", "metadata": METADATA_PATH.relative_to(PROJECT_ROOT).as_posix(), "records": len(metadata), "sha256": sha256_file(METADATA_PATH)}


def require_pass(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "pass":
        raise ValueError(f"{label} did not pass: {path}")
    return payload


def freeze() -> Dict[str, Any]:
    text_report = require_pass(TEXT_AUDIT_PATH, "text audit")
    audio_report = require_pass(AUDIO_QC_PATH, "technical audio QC")
    asr_report = require_pass(ASR_QC_PATH, "ASR intelligibility QC")
    rows = list(csv.DictReader(METADATA_PATH.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 100 or any(not row["transcript_text"].strip() for row in rows):
        raise ValueError("Metadata must contain 100 non-empty transcripts.")
    artifact = joblib.load(MODEL_PATH)
    threshold = float(artifact["decision_threshold_probability"])
    audio_hashes = {row["sample_id"]: sha256_file(PROJECT_ROOT / row["audio_path"]) for row in rows}
    manifest = {
        "name": "external_frozen_v2",
        "status": "frozen_pre_evaluation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "paired controlled holdout; no fitting, selection, or threshold scan on v2",
        "records": 100,
        "core_paired_records": 80,
        "normal_finance_hard_negative_records": 20,
        "spec": SPEC_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "spec_sha256": sha256_file(SPEC_PATH),
        "metadata": METADATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "metadata_sha256": sha256_file(METADATA_PATH),
        "audio_dir": AUDIO_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "audio_sha256": audio_hashes,
        "model": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "model_sha256": sha256_file(MODEL_PATH),
        "model_feature_contract_version": artifact.get("feature_contract_version"),
        "decision_threshold_probability": threshold,
        "alert_threshold_score": float(artifact["alert_threshold_score"]),
        "baseline_config": {"window_seconds": 10, "step_seconds": 5, "text_weight": 0.8, "smoothing_previous_weight": 0.65, "alert_threshold_score": 70.0},
        "qc": {
            "text_audit_sha256": sha256_file(TEXT_AUDIT_PATH),
            "technical_audio_qc_sha256": sha256_file(AUDIO_QC_PATH),
            "asr_intelligibility_qc_sha256": sha256_file(ASR_QC_PATH),
            "text_unique_scripts": text_report["unique_scripts"],
            "technical_audio_samples": audio_report["samples"],
            "asr_samples": asr_report["samples"],
        },
        "freeze_rules": [
            "Do not alter membership, audio, labels, model, or threshold after this manifest is written.",
            "Do not fit, calibrate, select, or tune on external_frozen_v2.",
            "Report the paired core 80 and expanded 100 with normal_finance separately.",
        ],
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def verify_frozen() -> Dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mismatches = []
    checks = ((SPEC_PATH, manifest["spec_sha256"], "spec"), (METADATA_PATH, manifest["metadata_sha256"], "metadata"), (MODEL_PATH, manifest["model_sha256"], "model"))
    for path, expected, label in checks:
        if sha256_file(path) != expected:
            mismatches.append(label)
    for sample_id, expected in manifest["audio_sha256"].items():
        row = next(row for row in csv.DictReader(METADATA_PATH.open(encoding="utf-8-sig")) if row["sample_id"] == sample_id)
        if sha256_file(PROJECT_ROOT / row["audio_path"]) != expected:
            mismatches.append(sample_id)
    if mismatches:
        raise ValueError(f"Frozen v2 mismatch: {mismatches}")
    return {"status": "pass", "verified": len(manifest["audio_sha256"]) + 3, "manifest_sha256": sha256_file(MANIFEST_PATH)}


def prepare_result_views(
    learned_predictions: Path,
    core_output: Path,
    audit_output: Path,
) -> Dict[str, Any]:
    """Validate a completed frozen run and write the predefined core-80 view."""
    verification = verify_frozen()
    payload = json.loads(learned_predictions.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Prediction file must contain a record list.")
    metadata_rows = list(csv.DictReader(METADATA_PATH.open(encoding="utf-8-sig", newline="")))
    expected_ids = {row["sample_id"] for row in metadata_rows}
    by_id = {str(record.get("sample_id") or ""): record for record in records}
    missing = sorted(expected_ids - set(by_id))
    unexpected = sorted(set(by_id) - expected_ids)
    errors = sorted(sample_id for sample_id, record in by_id.items() if record.get("error"))
    missing_scores = sorted(
        sample_id
        for sample_id, record in by_id.items()
        if not record.get("timeline")
        or any("causal_late_fusion_v2_score" not in point for point in record.get("timeline", []))
    )
    if len(records) != 100 or missing or unexpected or errors or missing_scores:
        raise ValueError({
            "records": len(records),
            "missing": missing,
            "unexpected": unexpected,
            "errors": errors,
            "missing_causal_scores": missing_scores,
        })
    core_records = [
        record for record in records
        if str(record.get("case_type") or "") != "normal_finance"
    ]
    if len(core_records) != 80:
        raise ValueError(f"Expected 80 core records, found {len(core_records)}")
    write_json(core_output, {"records": core_records})
    audit = {
        "status": "ready_for_metrics",
        "frozen_manifest_sha256": verification["manifest_sha256"],
        "learned_predictions": learned_predictions.relative_to(PROJECT_ROOT).as_posix(),
        "learned_predictions_sha256": sha256_file(learned_predictions),
        "expanded_records": len(records),
        "core_records": len(core_records),
        "normal_finance_records": len(records) - len(core_records),
        "prediction_errors": 0,
        "core_output": core_output.relative_to(PROJECT_ROOT).as_posix(),
        "core_output_sha256": sha256_file(core_output),
    }
    write_json(audit_output, audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("audit-text", "audio-qc", "build-metadata", "freeze", "verify", "prepare-result-views"),
    )
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--learned-predictions", type=Path)
    parser.add_argument("--core-output", type=Path)
    parser.add_argument("--audit-output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.spec = args.spec.resolve()
    functions = {
        "audit-text": lambda: audit_text(args.spec),
        "audio-qc": lambda: audio_technical_qc(args.spec),
        "build-metadata": lambda: build_metadata(args.spec),
        "freeze": freeze,
        "verify": verify_frozen,
        "prepare-result-views": lambda: prepare_result_views(
            args.learned_predictions.resolve(),
            args.core_output.resolve(),
            args.audit_output.resolve(),
        ),
    }
    if args.command == "prepare-result-views" and not all(
        (args.learned_predictions, args.core_output, args.audit_output)
    ):
        raise SystemExit("prepare-result-views requires --learned-predictions, --core-output, and --audit-output")
    print(json.dumps(functions[args.command](), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
