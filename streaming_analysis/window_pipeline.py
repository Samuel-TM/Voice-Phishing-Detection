# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from pydub import AudioSegment

from .risk_scoring import (
    RiskScoringState,
    deduplicate_recent_context,
    final_label_from_score,
    normalize_scoring_mode,
    score_window,
)

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]
STREAM_WINDOW_CACHE_DIR = PROJECT_DIR / ".cache" / "stream_windows"
MODEL_MAX_LEN = 256
MODEL_RESERVED_SPECIAL_TOKENS = 26
CURRENT_WINDOW_TOKEN_BUDGET = 140
PREVIOUS_CONTEXT_TOKEN_BUDGET = 90
RECENT_CONTEXT_WINDOW_COUNT = 3
MIN_ANALYZABLE_WINDOW_MS = 1000


def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换分数，避免 NumPy 标量或异常值影响 JSON 输出。"""
    try:
        return float(value)
    except Exception:
        return default


def _normalize_audio_risk_result(result: Any) -> Dict[str, Any]:
    """兼容旧的 float 概率输出和新的校准风险输出。"""
    if isinstance(result, dict):
        probability = _safe_float(
            result.get("deepfake_probability", result.get("deepfake_score", 0.0))
        )
        return {
            "deepfake_probability": probability,
            "voice_score": _safe_float(result.get("voice_score"), round(probability * 100.0, 2)),
            "deepfake_detected_voice": bool(result.get("deepfake_detected_voice", False)),
            "decision_threshold": result.get("decision_threshold"),
            "threshold_source": result.get("threshold_source"),
        }

    probability = _safe_float(result)
    return {
        "deepfake_probability": probability,
        "voice_score": round(probability * 100.0, 2),
        "deepfake_detected_voice": probability > 0.5,
        "decision_threshold": 0.5,
        "threshold_source": "default_0_5",
    }


def _clamp_seconds(value: Any, default: float, minimum: float, maximum: float) -> float:
    """限制窗口参数范围，避免过小窗口或过大窗口拖垮演示流程。"""
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return max(minimum, min(maximum, numeric))


def build_window_starts(duration_ms: int, window_ms: int, step_ms: int) -> List[int]:
    if duration_ms <= 0:
        return []

    raw_starts = list(range(0, duration_ms, step_ms))
    if raw_starts and raw_starts[-1] + window_ms < duration_ms:
        raw_starts.append(max(0, duration_ms - window_ms))
    elif not raw_starts:
        raw_starts = [0]

    starts: List[int] = []
    seen: set[int] = set()
    for raw_start in raw_starts:
        start_ms = max(0, min(int(raw_start), duration_ms))
        end_ms = min(start_ms + window_ms, duration_ms)
        segment_ms = end_ms - start_ms
        if segment_ms <= 0:
            continue
        if segment_ms < MIN_ANALYZABLE_WINDOW_MS and starts:
            logger.info(
                "Skipping too-short trailing stream window: start=%.3fs end=%.3fs duration_ms=%s",
                start_ms / 1000.0,
                end_ms / 1000.0,
                segment_ms,
            )
            continue
        if start_ms in seen:
            continue
        starts.append(start_ms)
        seen.add(start_ms)
    return starts


def _count_text_tokens(text: str) -> int:
    try:
        from ChineseBERTModel.ensemble_utils import count_text_tokens
    except Exception:
        try:
            from ML.ChineseBERTModel.ensemble_utils import count_text_tokens
        except Exception:
            return len(str(text or ""))
    return count_text_tokens(text)


def _trim_text_to_token_budget(text: str, token_budget: int, keep: str = "first") -> str:
    try:
        from ChineseBERTModel.ensemble_utils import trim_text_to_token_budget
    except Exception:
        try:
            from ML.ChineseBERTModel.ensemble_utils import trim_text_to_token_budget
        except Exception:
            text = str(text or "").strip()
            return text[:token_budget] if keep == "first" else text[-token_budget:]
    return trim_text_to_token_budget(text, token_budget, keep=keep)


def build_rolling_context_fields(
    previous_window_texts: Sequence[str],
    current_window_text: str,
    max_len: int = MODEL_MAX_LEN,
    current_window_budget: int = CURRENT_WINDOW_TOKEN_BUDGET,
    previous_context_budget: int = PREVIOUS_CONTEXT_TOKEN_BUDGET,
) -> Dict[str, Any]:
    current_window_text = str(current_window_text or "").strip()
    if not current_window_text:
        return {
            "current_window_text": "",
            "recent_context_text": "",
            "rolling_context_text": "",
            "rolling_context_previous_budget": 0,
            "rolling_context_current_budget": current_window_budget,
        }

    recent_texts = list(previous_window_texts)[-RECENT_CONTEXT_WINDOW_COUNT:]
    recent_context_text = deduplicate_recent_context(recent_texts)
    model_input_budget = max(1, max_len - MODEL_RESERVED_SPECIAL_TOKENS)
    current_text_trimmed = _trim_text_to_token_budget(
        current_window_text,
        min(current_window_budget, model_input_budget),
        keep="first",
    )
    current_tokens = _count_text_tokens(current_text_trimmed)
    current_spare = max(0, current_window_budget - current_tokens)
    previous_budget = min(
        previous_context_budget + current_spare,
        max(0, model_input_budget - current_tokens),
    )
    previous_tail_text = _trim_text_to_token_budget(
        recent_context_text,
        previous_budget,
        keep="last",
    )
    rolling_context_text = (
        f"{previous_tail_text}\n{current_text_trimmed}".strip()
        if previous_tail_text
        else current_text_trimmed
    )
    return {
        "current_window_text": current_window_text,
        "recent_context_text": recent_context_text,
        "rolling_context_text": rolling_context_text,
        "rolling_context_previous_budget": previous_budget,
        "rolling_context_current_budget": current_window_budget,
    }


def analyze_audio_stream(
    audio_path: str,
    audio_risk_model_path: Optional[str],
    audio_risk_config_path: Optional[str],
    text_inference: Callable[[str], Dict[str, Any]],
    audio_risk_inference: Callable[[str, str, str], Any],
    transcribe_segment: Callable[[str], str],
    window_seconds: Any = 10,
    step_seconds: Any = 5,
    text_weight: float = 0.8,
    voice_weight: float = 0.2,
    smoothing_previous_weight: float = 0.65,
    scoring_mode: Any = None,
    case_type: str = "",
) -> Dict[str, Any]:
    """
    使用滑动窗口模拟实时通话流分析。

    兼容旧调用：内部消费流式窗口结果并一次性返回完整 timeline。
    需要实时展示时应使用 iter_audio_stream_analysis。
    """
    timeline: List[Dict[str, Any]] = []
    full_transcript = ""
    metadata: Dict[str, Any] = {}

    for event in iter_audio_stream_analysis(
        audio_path=audio_path,
        audio_risk_model_path=audio_risk_model_path,
        audio_risk_config_path=audio_risk_config_path,
        text_inference=text_inference,
        audio_risk_inference=audio_risk_inference,
        transcribe_segment=transcribe_segment,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        text_weight=text_weight,
        voice_weight=voice_weight,
        smoothing_previous_weight=smoothing_previous_weight,
        scoring_mode=scoring_mode,
        case_type=case_type,
    ):
        if event.get("event") == "point":
            timeline.append(event["point"])
            full_transcript = event.get("full_transcript", full_transcript)
        elif event.get("event") == "done":
            metadata = event

    return {
        "timeline": timeline,
        "full_transcript": full_transcript,
        "final_score": metadata.get("final_score", 0.0),
        "max_score": metadata.get("max_score", 0.0),
        "final_label": metadata.get("final_label", final_label_from_score(0.0)),
        "highest_risk_window": metadata.get("highest_risk_window"),
        "scoring_mode": metadata.get("scoring_mode", "baseline"),
        "case_type": metadata.get("case_type", case_type),
        "window_seconds": metadata.get("window_seconds", window_seconds),
        "step_seconds": metadata.get("step_seconds", step_seconds),
        "weights": metadata.get("weights", {
            "text": text_weight,
            "voice": voice_weight,
            "smoothing_previous": smoothing_previous_weight,
            "smoothing_current": 1.0 - smoothing_previous_weight,
        }),
    }


def iter_audio_stream_analysis(
    audio_path: str,
    audio_risk_model_path: Optional[str],
    audio_risk_config_path: Optional[str],
    text_inference: Callable[[str], Dict[str, Any]],
    audio_risk_inference: Callable[[str, str, str], Any],
    transcribe_segment: Callable[[str], str],
    window_seconds: Any = 10,
    step_seconds: Any = 5,
    text_weight: float = 0.8,
    voice_weight: float = 0.2,
    smoothing_previous_weight: float = 0.65,
    scoring_mode: Any = None,
    case_type: str = "",
) -> Iterator[Dict[str, Any]]:
    """逐窗口分析音频，并在每个窗口完成后立即 yield 风险点。"""
    source_path = Path(audio_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    window_seconds = _clamp_seconds(window_seconds, default=10, minimum=2, maximum=60)
    step_seconds = _clamp_seconds(step_seconds, default=5, minimum=1, maximum=window_seconds)
    scoring_mode = normalize_scoring_mode(scoring_mode or "baseline")

    audio = AudioSegment.from_file(source_path.as_posix())
    duration_ms = len(audio)
    if duration_ms <= 0:
        yield {
            "event": "done",
            "timeline": [],
            "full_transcript": "",
            "final_score": 0.0,
            "max_score": 0.0,
            "final_label": "No Analyzable Audio",
            "highest_risk_window": None,
            "window_seconds": window_seconds,
            "step_seconds": step_seconds,
        }
        return

    window_ms = int(window_seconds * 1000)
    step_ms = int(step_seconds * 1000)
    starts = build_window_starts(duration_ms, window_ms, step_ms)

    timeline: List[Dict[str, Any]] = []
    transcript_parts: List[str] = []
    scoring_state = RiskScoringState()

    STREAM_WINDOW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stream_windows_", dir=STREAM_WINDOW_CACHE_DIR.as_posix()) as temp_dir:
        temp_path = Path(temp_dir)

        for index, start_ms in enumerate(starts):
            end_ms = min(start_ms + window_ms, duration_ms)
            if end_ms <= start_ms:
                continue

            segment = audio[start_ms:end_ms]
            segment_path = temp_path / f"window_{index:04d}.wav"
            segment.export(segment_path.as_posix(), format="wav")

            window_text = ""
            text_error = None
            try:
                window_text = transcribe_segment(segment_path.as_posix()) or ""
                if window_text.startswith("(STT"):
                    text_error = window_text
                    window_text = ""
            except Exception as exc:
                logger.error("窗口 STT 失败: %s", exc, exc_info=True)
                text_error = "STT failed for this window."

            previous_window_texts = transcript_parts[-RECENT_CONTEXT_WINDOW_COUNT:]
            rolling_context = build_rolling_context_fields(
                previous_window_texts=previous_window_texts,
                current_window_text=window_text,
            )

            text_score = 0.0
            text_result: Dict[str, Any] = {}
            text_model_input = window_text.strip()
            if text_model_input:
                try:
                    text_result = text_inference(text_model_input) or {}
                    text_score = _safe_float(text_result.get("llm_score", 0.0))
                except Exception as exc:
                    logger.error("窗口文本风险推理失败: %s", exc, exc_info=True)
                    text_result = {"error": "Text inference failed."}

            context_text_score = 0.0
            context_text_result: Dict[str, Any] = {}
            rolling_context_text = rolling_context["rolling_context_text"]
            if text_model_input and rolling_context_text:
                if rolling_context_text == text_model_input:
                    context_text_score = text_score
                    context_text_result = dict(text_result)
                else:
                    try:
                        context_text_result = text_inference(rolling_context_text) or {}
                        context_text_score = _safe_float(context_text_result.get("llm_score", 0.0))
                    except Exception as exc:
                        logger.error("滚动上下文文本风险推理失败: %s", exc, exc_info=True)
                        context_text_result = {"error": "Rolling-context text inference failed."}

            if window_text.strip():
                transcript_parts.append(window_text.strip())
            cumulative_text = " ".join(transcript_parts).strip()

            deepfake_probability = 0.0
            voice_score = 0.0
            audio_decision: Dict[str, Any] = {}
            voice_error = None
            if audio_risk_model_path and audio_risk_config_path:
                try:
                    audio_decision = _normalize_audio_risk_result(
                        audio_risk_inference(
                            segment_path.as_posix(),
                            audio_risk_model_path,
                            audio_risk_config_path,
                        )
                    )
                    deepfake_probability = _safe_float(audio_decision.get("deepfake_probability"))
                    voice_score = _safe_float(audio_decision.get("voice_score"))
                except Exception as exc:
                    logger.error("窗口音频风险推理失败: %s", exc, exc_info=True)
                    voice_error = "Voice inference failed."
            else:
                voice_error = "Voice model or config is missing."

            scoring = score_window(
                raw_text_score=text_score,
                context_text_score=context_text_score,
                voice_score=voice_score,
                text=window_text,
                state=scoring_state,
                scoring_mode=scoring_mode,
                case_type=case_type,
                text_weight=text_weight,
                voice_weight=voice_weight,
                smoothing_previous_weight=smoothing_previous_weight,
            )

            point: Dict[str, Any] = {
                "index": index,
                "start_sec": round(start_ms / 1000.0, 2),
                "end_sec": round(end_ms / 1000.0, 2),
                "text": window_text,
                "current_window_text": rolling_context["current_window_text"],
                "recent_context_text": rolling_context["recent_context_text"],
                "rolling_context_text": rolling_context["rolling_context_text"],
                "cumulative_text": cumulative_text,
                "case_type": case_type,
                "text_score": scoring["text_score"],
                "raw_text_score": scoring["raw_text_score"],
                "raw_window_text_score": scoring["raw_window_text_score"],
                "context_text_score": scoring["context_text_score"],
                "voice_score": scoring["voice_score"],
                "deepfake_score": round(deepfake_probability, 4),
                "deepfake_detected_voice": bool(audio_decision.get("deepfake_detected_voice", False)),
                "fused_score": scoring["fused_score"],
                "smoothed_score": scoring["smoothed_score"],
                "risk_level": scoring["risk_level"],
            }
            point.update({key: value for key, value in scoring.items() if key not in point})
            if audio_decision.get("decision_threshold") is not None:
                point["voice_decision_threshold"] = round(_safe_float(audio_decision.get("decision_threshold")), 4)
            if audio_decision.get("threshold_source"):
                point["voice_threshold_source"] = audio_decision["threshold_source"]
            if text_error:
                point["text_error"] = text_error
            if voice_error:
                point["voice_error"] = voice_error
            if text_result.get("error"):
                point["text_model_error"] = text_result["error"]
            if context_text_result.get("error"):
                point["context_text_model_error"] = context_text_result["error"]

            timeline.append(point)
            yield {
                "event": "point",
                "point": point,
                "full_transcript": " ".join(transcript_parts).strip(),
            }

    max_point = max(timeline, key=lambda item: item["smoothed_score"], default=None)
    final_score = timeline[-1]["smoothed_score"] if timeline else 0.0

    yield {
        "event": "done",
        "timeline": timeline,
        "full_transcript": " ".join(transcript_parts).strip(),
        "final_score": round(final_score, 2),
        "max_score": round(max_point["smoothed_score"], 2) if max_point else 0.0,
        "final_label": final_label_from_score(final_score),
        "highest_risk_window": max_point,
        "scoring_mode": scoring_mode,
        "case_type": case_type,
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "weights": {
            "text": text_weight,
            "voice": voice_weight,
            "smoothing_previous": smoothing_previous_weight,
            "smoothing_current": 1.0 - smoothing_previous_weight,
        },
    }
