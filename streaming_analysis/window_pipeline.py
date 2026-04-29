# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from pydub import AudioSegment

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]
STREAM_WINDOW_CACHE_DIR = PROJECT_DIR / ".cache" / "stream_windows"


def risk_level(score: float) -> str:
    """将 0-100 风险分数映射为英文前端展示等级。"""
    if score >= 90:
        return "Critical"
    if score >= 70:
        return "High Risk"
    if score >= 50:
        return "Suspicious"
    return "Normal"


def final_label_from_score(score: float) -> str:
    """根据最终平滑风险生成英文判定。"""
    return "Fraud Risk Detected" if score >= 70 else "No High Risk Detected"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换分数，避免 NumPy 标量或异常值影响 JSON 输出。"""
    try:
        return float(value)
    except Exception:
        return default


def _clamp_seconds(value: Any, default: float, minimum: float, maximum: float) -> float:
    """限制窗口参数范围，避免过小窗口或过大窗口拖垮演示流程。"""
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return max(minimum, min(maximum, numeric))


def analyze_audio_stream(
    audio_path: str,
    deepvoice_model_path: Optional[str],
    deepvoice_config_path: Optional[str],
    text_inference: Callable[[str], Dict[str, Any]],
    deepvoice_inference: Callable[[str, str, str], float],
    transcribe_segment: Callable[[str], str],
    window_seconds: Any = 10,
    step_seconds: Any = 5,
    text_weight: float = 0.8,
    voice_weight: float = 0.2,
    smoothing_previous_weight: float = 0.65,
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
        deepvoice_model_path=deepvoice_model_path,
        deepvoice_config_path=deepvoice_config_path,
        text_inference=text_inference,
        deepvoice_inference=deepvoice_inference,
        transcribe_segment=transcribe_segment,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        text_weight=text_weight,
        voice_weight=voice_weight,
        smoothing_previous_weight=smoothing_previous_weight,
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
    deepvoice_model_path: Optional[str],
    deepvoice_config_path: Optional[str],
    text_inference: Callable[[str], Dict[str, Any]],
    deepvoice_inference: Callable[[str, str, str], float],
    transcribe_segment: Callable[[str], str],
    window_seconds: Any = 10,
    step_seconds: Any = 5,
    text_weight: float = 0.8,
    voice_weight: float = 0.2,
    smoothing_previous_weight: float = 0.65,
) -> Iterator[Dict[str, Any]]:
    """逐窗口分析音频，并在每个窗口完成后立即 yield 风险点。"""
    source_path = Path(audio_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    window_seconds = _clamp_seconds(window_seconds, default=10, minimum=2, maximum=60)
    step_seconds = _clamp_seconds(step_seconds, default=5, minimum=1, maximum=window_seconds)
    current_weight = 1.0 - smoothing_previous_weight

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
    starts = list(range(0, max(duration_ms - 1, 0), step_ms))
    if starts and starts[-1] + window_ms < duration_ms:
        starts.append(max(0, duration_ms - window_ms))
    elif not starts:
        starts = [0]

    timeline: List[Dict[str, Any]] = []
    transcript_parts: List[str] = []
    previous_smoothed: Optional[float] = None

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

            if window_text.strip():
                transcript_parts.append(window_text.strip())
            cumulative_text = " ".join(transcript_parts).strip()

            text_score = 0.0
            text_result: Dict[str, Any] = {}
            if cumulative_text:
                try:
                    text_result = text_inference(cumulative_text) or {}
                    text_score = _safe_float(text_result.get("llm_score", 0.0))
                except Exception as exc:
                    logger.error("窗口文本风险推理失败: %s", exc, exc_info=True)
                    text_result = {"error": "Text inference failed."}

            deepfake_probability = 0.0
            voice_score = 0.0
            voice_error = None
            if deepvoice_model_path and deepvoice_config_path:
                try:
                    deepfake_probability = _safe_float(
                        deepvoice_inference(
                            segment_path.as_posix(),
                            deepvoice_model_path,
                            deepvoice_config_path,
                        )
                    )
                    voice_score = round(deepfake_probability * 100.0, 2)
                except Exception as exc:
                    logger.error("窗口音频风险推理失败: %s", exc, exc_info=True)
                    voice_error = "Voice inference failed."
            else:
                voice_error = "Voice model or config is missing."

            fused_score = round((text_weight * text_score) + (voice_weight * voice_score), 2)
            if previous_smoothed is None:
                smoothed_score = fused_score
            else:
                smoothed_score = round(
                    smoothing_previous_weight * previous_smoothed + current_weight * fused_score,
                    2,
                )
            previous_smoothed = smoothed_score

            point: Dict[str, Any] = {
                "index": index,
                "start_sec": round(start_ms / 1000.0, 2),
                "end_sec": round(end_ms / 1000.0, 2),
                "text": window_text,
                "cumulative_text": cumulative_text,
                "text_score": round(text_score, 2),
                "voice_score": voice_score,
                "deepfake_score": round(deepfake_probability, 4),
                "fused_score": fused_score,
                "smoothed_score": smoothed_score,
                "risk_level": risk_level(smoothed_score),
            }
            if text_error:
                point["text_error"] = text_error
            if voice_error:
                point["voice_error"] = voice_error
            if text_result.get("error"):
                point["text_model_error"] = text_result["error"]

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
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "weights": {
            "text": text_weight,
            "voice": voice_weight,
            "smoothing_previous": smoothing_previous_weight,
            "smoothing_current": current_weight,
        },
    }
