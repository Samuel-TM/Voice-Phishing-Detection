#!/usr/bin/env python3
"""Extract normal MagicData-RAMC conversational clips for test samples."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "../../Material/Dataset/MagicData-RAMC/A_CHINESE_CONVERSATIONAL_SPEECH_CORPUS"
).resolve()
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
    "transcript_text",
]

LINE_RE = re.compile(
    r"^\[(?P<start>[\d.]+),(?P<end>[\d.]+)\]\s+"
    r"(?P<speaker>\S+)\s+(?P<meta>\S+)\s+(?P<text>.*)$"
)

NOISE_MARKERS = {"[*]", "[+]", "[SONANT]", "[LAUGHTER]"}

FRAUD_OR_UNSAFE_TERMS = [
    "验证码",
    "安全账户",
    "账户冻结",
    "冒充公检法",
    "公检法",
    "远程控制",
    "屏幕共享",
    "紧急汇款",
    "刷流水",
    "洗钱",
    "通缉",
    "公安局",
    "检察院",
    "银行卡异常",
    "偷东西",
    "抢劫",
    "毒品",
    "歹徒",
    "心生歹念",
]

FINANCE_STRONG_TERMS = [
    "支付宝",
    "微信支付",
    "支付",
    "付款",
    "花呗",
    "借呗",
    "贷款",
    "还款",
    "金融",
    "银行",
    "经济损失",
    "彩礼",
    "班费",
    "工资",
    "奖学金",
    "助学金",
    "生活费",
    "学费",
    "精品课",
    "会员",
    "保险",
    "合同",
    "费用",
    "补贴",
    "赔偿",
    "金额",
]

FINANCE_CONTEXT_TERMS = [
    "钱",
    "块钱",
    "挣钱",
    "赚钱",
    "兼职",
    "买",
    "卖",
    "贵",
    "便宜",
    "收费",
    "免费",
    "赞助",
    "消费",
    "车",
    "设备",
    "饭",
    "课程",
    "辅导",
]

FINANCE_FALSE_CONTEXT = [
    "元代",
    "元朝",
    "元曲",
    "元始",
    "元帅",
    "混元",
    "霍元甲",
    "状元",
    "一千米",
    "八百米",
    "四百米",
    "二百米",
    "一百米",
    "五千年",
    "两千年",
]

DAILY_TERMS = [
    "天气",
    "旅游",
    "学校",
    "上班",
    "工作",
    "朋友",
    "家里",
    "孩子",
    "父母",
    "手机",
    "电影",
    "电视",
    "运动",
    "衣服",
    "做饭",
    "火锅",
    "生活",
    "城市",
    "交通",
    "公交",
    "地铁",
    "学习",
    "老师",
    "同学",
    "健身",
    "节日",
    "国庆",
    "春节",
    "房子",
    "游戏",
    "音乐",
    "餐厅",
    "超市",
    "南方",
    "北方",
    "长城",
    "食物",
    "体育",
    "跑步",
    "大学",
    "宿舍",
    "校园",
    "软件",
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
    "是不是",
    "为什么",
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


def clean_text(text: str) -> str:
    return text.strip().replace(" ", "")


def parse_transcript(path: Path) -> list[Utterance]:
    utterances: list[Utterance] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = LINE_RE.match(raw.strip())
        if not match:
            continue
        text = clean_text(match.group("text"))
        if not text or text in NOISE_MARKERS or "爱数智慧语音采集" in text:
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


def is_question_like(text: str) -> bool:
    tail = text[-20:]
    return text.endswith(("？", "?")) or any(hint in tail for hint in QUESTION_HINTS)


def count_terms(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms)


def amount_mentions(text: str) -> int:
    return len(re.findall(r"[一二三四五六七八九十百千万两\d]+(?:块钱|块|元|万|千|百|%)", text))


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def natural_start_indexes(utterances: list[Utterance]) -> list[int]:
    starts: set[int] = set()
    for idx, current in enumerate(utterances):
        if idx == 0:
            starts.add(idx)
            continue

        previous = utterances[idx - 1]
        gap = current.start - previous.end
        if gap >= 1.8:
            starts.add(idx)
            continue
        if is_question_like(previous.text) and current.speaker != previous.speaker:
            starts.add(idx)
            continue
        if len(current.text) > 5 and current.text.startswith(
            (
                "那",
                "然后",
                "但是",
                "对",
                "嗯",
                "所以",
                "其实",
                "我觉得",
                "我们",
                "现在",
                "以前",
                "还有",
                "比如",
                "因为",
                "像",
                "说到",
            )
        ):
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

        penalty = abs(duration - 60.0)
        if next_gap >= 0.8 or next_switch or is_question_like(utterances[end_idx].text):
            penalty -= 3.0
        if next_gap >= 1.5:
            penalty -= 2.0
        if is_question_like(utterances[end_idx].text):
            penalty += 1.2

        if best is None or penalty < best[0]:
            best = (penalty, end_idx)
    return None if best is None else best[1]


def finance_score(text: str) -> int:
    score = count_terms(text, FINANCE_STRONG_TERMS) * 3
    score += count_terms(text, FINANCE_CONTEXT_TERMS)
    score += amount_mentions(text) * 2
    if has_any(text, FINANCE_FALSE_CONTEXT):
        score -= 10
    return score


def is_finance_candidate(text: str) -> bool:
    if has_any(text, FINANCE_FALSE_CONTEXT):
        return False
    strong_hits = count_terms(text, FINANCE_STRONG_TERMS)
    context_hits = count_terms(text, FINANCE_CONTEXT_TERMS)
    amounts = amount_mentions(text)
    if strong_hits >= 1 and (context_hits >= 1 or amounts >= 1):
        return True
    if strong_hits >= 2:
        return True
    if context_hits >= 4 and any(term in text for term in ("钱", "买", "贵", "便宜", "收费", "免费", "挣钱", "赚钱")):
        return True
    return amounts >= 1 and context_hits >= 3


def is_daily_candidate(text: str) -> bool:
    if is_finance_candidate(text):
        return False
    return count_terms(text, DAILY_TERMS) >= 4 and amount_mentions(text) <= 1


def note_for(case_type: str, text: str) -> str:
    if case_type == "normal_daily":
        if any(term in text for term in ("天气", "南方", "北方")):
            return "weather, region, and daily-life discussion"
        if any(term in text for term in ("体育", "运动", "健身", "跑步", "足球")):
            return "sports and healthy-lifestyle discussion"
        if any(term in text for term in ("学校", "老师", "学习", "同学", "大学")):
            return "school and study discussion"
        if any(term in text for term in ("手机", "软件", "电影", "电视")):
            return "technology or media discussion in normal daily context"
        return "ordinary daily conversation"

    if any(term in text for term in ("支付宝", "微信支付", "支付", "付款")):
        return "normal payment-app discussion without fraud actions"
    if any(term in text for term in ("花呗", "借呗", "贷款", "还款", "银行")):
        return "normal credit or lending discussion without fraud actions"
    if any(term in text for term in ("工资", "兼职", "挣钱", "赚钱")):
        return "normal income or part-time work discussion"
    if any(term in text for term in ("奖学金", "助学金", "班费", "学费", "生活费")):
        return "normal student finance or fee discussion"
    if any(term in text for term in ("买", "贵", "便宜", "费用", "收费", "免费")):
        return "normal purchase, cost, or pricing discussion"
    if "保险" in text:
        return "normal insurance-related discussion"
    return "normal finance-related conversation without fraud actions"


def classify_candidate(source_txt: Path, utterances: list[Utterance], start_idx: int) -> Candidate | None:
    end_idx = choose_topic_end(utterances, start_idx)
    if end_idx is None:
        return None

    segment = utterances[start_idx : end_idx + 1]
    text = "".join(item.text for item in segment)
    if has_any(text, FRAUD_OR_UNSAFE_TERMS):
        return None

    duration = segment[-1].end - segment[0].start
    source_wav = source_txt.with_suffix(".wav")

    if is_finance_candidate(text):
        case_type = "normal_finance"
        score = finance_score(text) + count_terms(text, DAILY_TERMS) * 0.05 - abs(duration - 60.0) * 0.1
    elif is_daily_candidate(text):
        case_type = "normal_daily"
        score = count_terms(text, DAILY_TERMS) * 1.4 - finance_score(text) * 0.4 - abs(duration - 60.0) * 0.1
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
    if case_type == "normal_finance":
        selected_by_source: list[Candidate] = []
        source_names = sorted({item.source_wav.name for item in candidates if item.case_type == case_type})
        for source_name in source_names:
            source_candidates = sorted(
                (
                    item
                    for item in candidates
                    if item.case_type == case_type and item.source_wav.name == source_name
                ),
                key=lambda item: (item.end, item.start),
            )
            if not source_candidates:
                continue

            dp: list[tuple[int, float, list[Candidate]]] = [(0, 0.0, [])]
            for idx, candidate in enumerate(source_candidates, start=1):
                previous_idx = 0
                for probe_idx in range(idx - 1, 0, -1):
                    if source_candidates[probe_idx - 1].end <= candidate.start:
                        previous_idx = probe_idx
                        break

                without_candidate = dp[idx - 1]
                previous_count, previous_score, previous_items = dp[previous_idx]
                with_candidate = (
                    previous_count + 1,
                    previous_score + candidate.score,
                    previous_items + [candidate],
                )
                if (with_candidate[0], with_candidate[1]) > (
                    without_candidate[0],
                    without_candidate[1],
                ):
                    dp.append(with_candidate)
                else:
                    dp.append(without_candidate)
            selected_by_source.extend(dp[-1][2])

        return sorted(
            selected_by_source,
            key=lambda item: (-item.score, abs(item.duration - 60.0), item.source_wav.name, item.start),
        )[:limit]

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
            if sample_id.startswith(("RAMC_N", "RAMC_F")):
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
        "source": "magicdata_ramc",
        "start_sec": f"{candidate.start:.3f}",
        "end_sec": f"{candidate.end:.3f}",
        "source_wav": candidate.source_wav.name,
        "notes": candidate.notes,
        "transcript_text": candidate.text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--max-per-type", type=int, default=20)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    audio_dir = args.audio_dir.resolve()
    metadata_path = args.metadata.resolve()

    candidates = discover_candidates(dataset)
    selected_daily = select_candidates(candidates, "normal_daily", args.max_per_type)
    selected_finance = select_candidates(candidates, "normal_finance", args.max_per_type)

    existing_rows = read_existing_metadata(metadata_path)
    new_rows: list[dict[str, str]] = []

    for idx, candidate in enumerate(selected_daily, start=1):
        sample_id = f"RAMC_N{idx:02d}"
        output_path = audio_dir / f"{sample_id}.mp3"
        export_mp3(candidate, output_path)
        new_rows.append(make_row(sample_id, output_path, candidate))

    for idx, candidate in enumerate(selected_finance, start=1):
        sample_id = f"RAMC_F{idx:02d}"
        output_path = audio_dir / f"{sample_id}.mp3"
        export_mp3(candidate, output_path)
        new_rows.append(make_row(sample_id, output_path, candidate))

    write_metadata(metadata_path, existing_rows + new_rows)

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
