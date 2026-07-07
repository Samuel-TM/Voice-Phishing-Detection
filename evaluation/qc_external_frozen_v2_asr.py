#!/usr/bin/env python3
"""Run ASR-only intelligibility QC for external_frozen_v2 before freezing it."""

from __future__ import annotations

import csv
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())

from speaker_analysis.asr_backend import clean_stt_text, transcribe_segment_with_metadata


METADATA = PROJECT_ROOT / "test_samples/metadata_external_frozen_v2.csv"
OUTPUT = PROJECT_ROOT / ".cache/external_frozen_v2/asr_intelligibility_qc.json"


def normalize(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", clean_stt_text(text or "")).lower()


def main() -> None:
    rows = list(csv.DictReader(METADATA.open(encoding="utf-8-sig", newline="")))
    results = []
    for index, row in enumerate(rows, start=1):
        audio_path = PROJECT_ROOT / row["audio_path"]
        result = transcribe_segment_with_metadata(audio_path.as_posix())
        reference = normalize(row["transcript_text"])
        hypothesis = normalize(result.text)
        similarity = SequenceMatcher(None, reference, hypothesis).ratio() if reference and hypothesis else 0.0
        reasons = []
        if result.error:
            reasons.append("asr_error")
        if len(hypothesis) < max(20, int(len(reference) * 0.25)):
            reasons.append("asr_transcript_too_short")
        if similarity < 0.30:
            reasons.append("low_reference_similarity")
        item = {
            "sample_id": row["sample_id"],
            "status": "pass" if not reasons else "fail",
            "reference_chars": len(reference),
            "asr_chars": len(hypothesis),
            "sequence_similarity": round(similarity, 4),
            "backend": result.backend,
            "model_name": result.model_name,
            "reasons": reasons,
            "asr_text": result.text,
        }
        results.append(item)
        print(f"[{index:02d}/{len(rows)}] {row['sample_id']} {item['status']} similarity={similarity:.3f}")
    failed = [item for item in results if item["status"] != "pass"]
    payload = {"status": "pass" if len(results) == 100 and not failed else "fail", "samples": len(results), "failed": len(failed), "results": results}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "samples", "failed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
