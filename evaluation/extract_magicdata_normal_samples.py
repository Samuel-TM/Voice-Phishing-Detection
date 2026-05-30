#!/usr/bin/env python3
"""Extract normal MagicData meeting clips for dynamic-risk testing."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/Users/sunjiashan/Material/HKU/Dissertation/Material/Dataset")
DEFAULT_DATASET = DATASET_ROOT / "MagicData/A_CHINESE_CONVERSATIONAL_MEETING_SPEECH_CORPUS"
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

LINE_RE = re.compile(
    r"^\[(?P<start>[\d.]+),(?P<end>[\d.]+)\]\s+"
    r"(?P<speaker>\S+)\s+(?P<meta>\S+)\s+(?P<text>.*)$"
)

NOISE_TEXT = {"ok", "北京爱数智慧语音采集"}

FRAUD_TERMS = [
    "验证码",
    "安全账户",
    "账户冻结",
    "冒充公检法",
    "公检法",
    "远程控制",
    "屏幕共享",
    "紧急汇款",
    "汇款",
    "转账",
    "刷流水",
    "洗钱",
    "通缉",
    "公安局",
    "检察院",
    "法院",
    "银行卡异常",
]

FINANCE_TERMS = [
    "工资",
    "社保",
    "五险一金",
    "合同",
    "签约",
    "续签",
    "费用",
    "补贴",
    "金额",
    "付款",
    "支付",
    "提成",
    "租金",
    "分期",
    "保险",
    "成本",
    "价格",
    "底薪",
    "薪资",
    "福利",
    "报销",
    "罚款",
    "罚金",
    "块",
    "元",
    "万",
]

DAILY_TERMS = [
    "住宿",
    "着装",
    "员工",
    "管理",
    "制度",
    "工作",
    "安排",
    "上班",
    "宿舍",
    "骑手",
    "外卖",
    "衣服",
    "正装",
    "考勤",
    "公司",
    "会议",
    "方案",
    "培训",
    "人员",
    "部门",
    "规范",
    "吃住",
    "餐厅",
    "工装",
    "服务",
    "食堂",
    "疫情",
    "发型",
    "迟到",
    "请假",
    "企业文化",
]

QUESTION_HINTS = [
    "吗",
    "呢",
    "吧",
    "有没有",
    "如何",
    "怎么",
    "多少",
    "哪种",
    "什么",
    "觉得",
    "看法",
    "问题",
    "方案",
    "建议",
]


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    speaker: str
    text: str


@dataclass(frozen=True)
class Candidate:
    case_type: str
    score: float
    source_txt: Path
    source_wav: Path
    start: float
    end: float
    text: str
    notes: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def count_terms(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms)


def amount_mentions(text: str) -> int:
    return len(re.findall(r"[一二三四五六七八九十百千万两\d]+(?:块|元|万|千|百|%)", text))


def is_question_like(text: str) -> bool:
    tail = text[-18:]
    return text.endswith(("？", "?")) or any(hint in tail for hint in QUESTION_HINTS)


def parse_transcript(path: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = LINE_RE.match(raw.strip())
        if not match:
            continue
        text = match.group("text").strip().replace(" ", "")
        if not text or text in NOISE_TEXT:
            continue
        if text.startswith("[") and text.endswith("]"):
            continue
        utterances.append(
            Utterance(
                start=float(match.group("start")),
                end=float(match.group("end")),
                speaker=match.group("speaker"),
                text=text,
            )
        )
    return utterances


def natural_start_indexes(utterances: list[Utterance]) -> list[int]:
    starts: set[int] = set()
    for idx, current in enumerate(utterances):
        if idx == 0:
            starts.add(idx)
            continue
        previous = utterances[idx - 1]
        gap = current.start - previous.end
        if gap >= 1.7:
            starts.add(idx)
            continue
        if is_question_like(previous.text) and current.speaker != previous.speaker:
            starts.add(idx)
            continue
        if current.text.startswith(
            (
                "嗯但是",
                "但是",
                "那",
                "然后",
                "首先",
                "另外",
                "还有",
                "对然后",
                "我们现在",
                "今天",
                "目前",
                "关于",
            )
        ) and len(current.text) > 6:
            starts.add(idx)
    return sorted(starts)


def choose_topic_end(utterances: list[Utterance], start_idx: int) -> int | None:
    best: tuple[float, int] | None = None
    start_time = utterances[start_idx].start
    for end_idx in range(start_idx, len(utterances)):
        duration = utterances[end_idx].end - start_time
        if duration < 45:
            continue
        if duration > 75:
            break

        next_gap = 5.0
        next_switch = False
        if end_idx + 1 < len(utterances):
            next_gap = utterances[end_idx + 1].start - utterances[end_idx].end
            next_switch = utterances[end_idx + 1].speaker != utterances[end_idx].speaker

        complete_boundary = next_gap >= 0.8 or next_switch or is_question_like(utterances[end_idx].text)
        penalty = abs(duration - 60.0)
        if complete_boundary:
            penalty -= 3.0
        if next_gap >= 1.5:
            penalty -= 2.0
        if is_question_like(utterances[end_idx].text):
            penalty += 1.5

        if best is None or penalty < best[0]:
            best = (penalty, end_idx)
    return None if best is None else best[1]


def note_for(case_type: str, text: str) -> str:
    if case_type == "normal_daily":
        if any(term in text for term in ("着装", "正装", "衣服", "拖鞋", "发型")):
            return "dress-code discussion, normal meeting context"
        if any(term in text for term in ("住宿", "宿舍", "吃住")):
            return "employee accommodation discussion, normal meeting context"
        if "企业文化" in text:
            return "company culture and training discussion"
        if any(term in text for term in ("考勤", "管理制度", "偷懒")):
            return "staff management discussion, normal meeting context"
        if any(term in text for term in ("骑手", "外卖", "餐厅")):
            return "ordinary operations and staffing discussion"
        return "ordinary business meeting discussion"

    if any(term in text for term in ("合同", "续签", "签约")):
        return "contract renewal or business agreement discussion"
    if any(term in text for term in ("工资", "底薪", "薪资", "提成")):
        return "salary and commission discussion without fraud actions"
    if any(term in text for term in ("社保", "五险一金", "保险")):
        return "social insurance and benefits discussion without fraud actions"
    if any(term in text for term in ("租", "费用", "付款", "支付")):
        return "normal cost, rent, or payment discussion"
    return "normal finance-related business discussion"


def classify_candidate(source_txt: Path, utterances: list[Utterance], start_idx: int) -> Candidate | None:
    end_idx = choose_topic_end(utterances, start_idx)
    if end_idx is None:
        return None

    segment = utterances[start_idx : end_idx + 1]
    text = "".join(item.text for item in segment)
    if has_any(text, FRAUD_TERMS):
        return None

    finance_score = count_terms(text, FINANCE_TERMS)
    daily_score = count_terms(text, DAILY_TERMS)
    amounts = amount_mentions(text)
    duration = segment[-1].end - segment[0].start

    source_wav = source_txt.with_suffix(".wav")
    if finance_score >= 3 or amounts >= 2:
        case_type = "normal_finance"
        score = finance_score * 2.0 + amounts * 2.0 + daily_score * 0.2 - abs(duration - 60.0) * 0.1
    elif daily_score >= 3 and finance_score <= 2 and amounts <= 1:
        case_type = "normal_daily"
        score = daily_score * 1.5 - finance_score * 1.5 - amounts * 2.0 - abs(duration - 60.0) * 0.1
    else:
        return None

    return Candidate(
        case_type=case_type,
        score=score,
        source_txt=source_txt,
        source_wav=source_wav,
        start=segment[0].start,
        end=segment[-1].end,
        text=text,
        notes=note_for(case_type, text),
    )


def discover_candidates(dataset_dir: Path) -> list[Candidate]:
    wav_dir = dataset_dir / "wav"
    candidates: list[Candidate] = []
    for transcript in sorted(wav_dir.glob("*.txt")):
        utterances = parse_transcript(transcript)
        for start_idx in natural_start_indexes(utterances):
            candidate = classify_candidate(transcript, utterances, start_idx)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def overlaps(left: Candidate, right: Candidate) -> bool:
    if left.source_wav != right.source_wav:
        return False
    return not (left.end <= right.start or left.start >= right.end)


def select_candidates(candidates: list[Candidate], case_type: str, limit: int) -> list[Candidate]:
    selected: list[Candidate] = []
    ranked = sorted(
        (item for item in candidates if item.case_type == case_type),
        key=lambda item: (-item.score, abs(item.duration - 60.0), item.source_wav.name, item.start),
    )
    for candidate in ranked:
        if any(overlaps(candidate, item) for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def export_mp3(candidate: Candidate, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{candidate.start:.3f}",
        "-to",
        f"{candidate.end:.3f}",
        "-i",
        str(candidate.source_wav),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def read_existing_metadata(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            sample_id = row.get("sample_id", "")
            if sample_id.startswith(("MD_N", "MD_F")):
                continue
            rows.append({field: row.get(field, "") for field in METADATA_FIELDS})
        return rows


def write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_row(sample_id: str, output_path: Path, candidate: Candidate) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "audio_path": str(output_path.relative_to(PROJECT_ROOT)),
        "label": "normal",
        "case_type": candidate.case_type,
        "event_time_sec": "",
        "source": "magicdata_meeting",
        "start_sec": f"{candidate.start:.3f}",
        "end_sec": f"{candidate.end:.3f}",
        "source_wav": candidate.source_wav.name,
        "notes": candidate.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--max-per-type", type=int, default=10)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    audio_dir = args.audio_dir.resolve()
    metadata_path = args.metadata.resolve()

    candidates = discover_candidates(dataset)
    selected_daily = select_candidates(candidates, "normal_daily", args.max_per_type)
    selected_finance = select_candidates(candidates, "normal_finance", args.max_per_type)

    metadata_rows = read_existing_metadata(metadata_path)
    new_rows: list[dict[str, str]] = []

    for idx, candidate in enumerate(selected_daily, start=1):
        sample_id = f"MD_N{idx:02d}"
        output_path = audio_dir / f"{sample_id}.mp3"
        export_mp3(candidate, output_path)
        new_rows.append(make_row(sample_id, output_path, candidate))

    for idx, candidate in enumerate(selected_finance, start=1):
        sample_id = f"MD_F{idx:02d}"
        output_path = audio_dir / f"{sample_id}.mp3"
        export_mp3(candidate, output_path)
        new_rows.append(make_row(sample_id, output_path, candidate))

    write_metadata(metadata_path, metadata_rows + new_rows)

    print(f"normal_daily: {len(selected_daily)}")
    for row in new_rows[: len(selected_daily)]:
        print(
            f"  {row['sample_id']} {row['source_wav']} "
            f"{row['start_sec']}-{row['end_sec']} {row['notes']}"
        )
    print(f"normal_finance: {len(selected_finance)}")
    for row in new_rows[len(selected_daily) :]:
        print(
            f"  {row['sample_id']} {row['source_wav']} "
            f"{row['start_sec']}-{row['end_sec']} {row['notes']}"
        )


if __name__ == "__main__":
    main()
