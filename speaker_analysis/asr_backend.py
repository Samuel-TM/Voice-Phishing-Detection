"""Unified ASR backend selection for project transcription.

Set ASR_BACKEND to switch engines without changing the streaming pipeline.
The project default is FunASR paraformer-zh because the thesis/demo data is
primarily Mandarin phone-call speech:

- funasr_paraformer: paraformer-zh through FunASR.
- whisper: existing OpenAI Whisper path, controlled by WHISPER_MODEL_NAME.
- funasr_sensevoice: FunAudioLLM/SenseVoiceSmall through FunASR.

FunASR must be installed before using the default backend. If it is unavailable,
this module returns the same STT failure marker style used by the current
Whisper implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional
import wave

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_CACHE_DIR = PROJECT_DIR / ".cache"
ASR_CACHE_DIR = PROJECT_CACHE_DIR / "asr"

os.environ.setdefault("HF_HOME", (PROJECT_CACHE_DIR / "huggingface").as_posix())
os.environ.setdefault("TRANSFORMERS_CACHE", (PROJECT_CACHE_DIR / "huggingface" / "hub").as_posix())
os.environ.setdefault("XDG_CACHE_HOME", PROJECT_CACHE_DIR.as_posix())
os.environ.setdefault("MODELSCOPE_CACHE", (PROJECT_CACHE_DIR / "modelscope").as_posix())

DEFAULT_SENSEVOICE_MODEL = "FunAudioLLM/SenseVoiceSmall"
DEFAULT_PARAFORMER_MODEL = "paraformer-zh"
DEFAULT_ASR_BACKEND = "funasr_paraformer"
MIN_ASR_AUDIO_DURATION_MS = 1000
LOCAL_MODELSCOPE_DIR = PROJECT_CACHE_DIR / "modelscope" / "models"
LOCAL_MODELSCOPE_ALIASES = {
    "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    "ct-punc-c": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
}

FUNASR_BACKENDS = {"funasr_sensevoice", "sensevoice", "funasr_paraformer", "paraformer"}
WHISPER_BACKENDS = {"whisper", "openai_whisper"}

_funasr_model: Optional[Any] = None
_funasr_model_key: Optional[tuple[str, str, str]] = None
_funasr_load_error: Optional[str] = None
_last_asr_result: Optional["ASRResult"] = None


@dataclass
class ASRResult:
    text: str
    backend: str
    model_name: str
    raw_text: str = ""
    language: Optional[str] = None
    emotion: Optional[str] = None
    event: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def get_asr_backend() -> str:
    return os.environ.get("ASR_BACKEND", DEFAULT_ASR_BACKEND).strip().lower() or DEFAULT_ASR_BACKEND


def clean_stt_text(text: str) -> str:
    """Normalize ASR text for the Chinese text-risk model."""
    if not text or not isinstance(text, str):
        return ""
    cleaned_text = re.sub(r"[^\u4e00-\u9fff0-9\s]", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return re.sub(r"(?<=[\u4e00-\u9fff0-9])\s+(?=[\u4e00-\u9fff0-9])", "", cleaned_text)


def get_last_asr_result() -> Optional[ASRResult]:
    return _last_asr_result


def _set_last_result(result: ASRResult) -> ASRResult:
    global _last_asr_result
    _last_asr_result = result
    return result


def _failure_result(audio_path: str, backend: str, model_name: str, message: str) -> ASRResult:
    logger.error("ASR failed for '%s' with backend '%s': %s", audio_path, backend, message)
    return _set_last_result(
        ASRResult(
            text=f"(STT failed - {message})",
            backend=backend,
            model_name=model_name,
            error=message,
        )
    )


def _empty_audio_result(audio_path: str, backend: str, model_name: str, duration_ms: float) -> ASRResult:
    logger.info(
        "Skipping ASR for too-short audio segment '%s' with backend '%s': duration_ms=%.2f",
        audio_path,
        backend,
        duration_ms,
    )
    return _set_last_result(
        ASRResult(
            text="",
            backend=backend,
            model_name=model_name,
            metadata={
                "skipped": "audio_too_short",
                "duration_ms": round(duration_ms, 2),
            },
        )
    )


def _wav_duration_ms(audio_path: str) -> Optional[float]:
    try:
        with wave.open(audio_path, "rb") as handle:
            frame_rate = float(handle.getframerate())
            if frame_rate <= 0:
                return None
            return handle.getnframes() / frame_rate * 1000.0
    except Exception:
        return None


def _min_asr_audio_duration_ms() -> int:
    try:
        return int(os.environ.get("ASR_MIN_AUDIO_MS", MIN_ASR_AUDIO_DURATION_MS))
    except Exception:
        return MIN_ASR_AUDIO_DURATION_MS


def _parse_sensevoice_tokens(raw_text: str) -> Dict[str, str]:
    tokens = re.findall(r"<\|([^|]+)\|>", raw_text or "")
    metadata: Dict[str, str] = {}
    if not tokens:
        return metadata

    known_languages = {"zh", "zn", "en", "yue", "ja", "ko", "nospeech"}
    known_events = {"speech", "music", "applause", "laughter", "crying", "cough", "sneeze"}
    known_text_norm = {"withitn", "woitn"}

    for token in tokens:
        lowered = token.lower()
        if lowered in known_languages:
            metadata["language"] = token
        elif lowered in known_events:
            metadata["event"] = token
        elif lowered in known_text_norm:
            metadata["text_normalization"] = token
        elif lowered not in known_text_norm:
            metadata.setdefault("emotion", token)
    return metadata


def _funasr_model_name(backend: str) -> str:
    if backend in {"funasr_paraformer", "paraformer"}:
        return os.environ.get("ASR_MODEL_NAME", DEFAULT_PARAFORMER_MODEL).strip() or DEFAULT_PARAFORMER_MODEL
    return os.environ.get("ASR_MODEL_NAME", DEFAULT_SENSEVOICE_MODEL).strip() or DEFAULT_SENSEVOICE_MODEL


def _resolve_local_model_path(model_name: str) -> str:
    """Prefer project-local ModelScope cache to avoid hub checks after download."""
    if not model_name:
        return model_name

    explicit_path = Path(model_name).expanduser()
    if explicit_path.exists():
        return explicit_path.as_posix()

    repo_id = LOCAL_MODELSCOPE_ALIASES.get(model_name, model_name)
    if "/" not in repo_id:
        return model_name

    local_dir = LOCAL_MODELSCOPE_DIR / repo_id
    has_config = (local_dir / "configuration.json").exists() or (local_dir / "config.yaml").exists()
    has_weights = any((local_dir / filename).exists() for filename in ("model.pt", "pytorch_model.bin", "model.onnx"))
    if has_config and has_weights:
        return local_dir.as_posix()

    return model_name


def _load_funasr_model(backend: str, model_name: str) -> Any:
    global _funasr_model, _funasr_model_key, _funasr_load_error

    device = os.environ.get("ASR_DEVICE", "cpu").strip() or "cpu"
    resolved_model_name = _resolve_local_model_path(model_name)
    model_key = (backend, resolved_model_name, device)
    if _funasr_model is not None and _funasr_model_key == model_key:
        return _funasr_model
    if _funasr_load_error is not None:
        raise RuntimeError(_funasr_load_error)

    try:
        from funasr import AutoModel
    except Exception as exc:
        _funasr_load_error = f"FunASR is not installed ({exc})"
        raise RuntimeError(_funasr_load_error) from exc

    ASR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading ASR backend '%s' model '%s' on %s", backend, resolved_model_name, device)

    try:
        common_kwargs: Dict[str, Any] = {
            "model": resolved_model_name,
            "device": device,
            "disable_update": True,
            "disable_pbar": os.environ.get("ASR_DISABLE_PBAR", "true").lower() != "false",
            "check_latest": False,
        }
        if backend in {"funasr_sensevoice", "sensevoice"}:
            common_kwargs.update(
                {
                    "vad_model": os.environ.get("ASR_VAD_MODEL", "fsmn-vad"),
                    "vad_kwargs": {
                        "max_single_segment_time": int(os.environ.get("ASR_VAD_MAX_SEGMENT_MS", "30000"))
                    },
                }
            )
            if "/" in model_name:
                common_kwargs["hub"] = os.environ.get("ASR_HUB", "hf")
        else:
            common_kwargs["vad_model"] = _resolve_local_model_path(os.environ.get("ASR_VAD_MODEL", "fsmn-vad"))
            punc_model = os.environ.get("ASR_PUNC_MODEL", "").strip()
            if punc_model:
                common_kwargs["punc_model"] = _resolve_local_model_path(punc_model)

        _funasr_model = AutoModel(**common_kwargs)
        _funasr_model_key = model_key
        return _funasr_model
    except Exception as exc:
        _funasr_load_error = str(exc)
        raise


def _transcribe_with_funasr(audio_path: str, backend: str) -> ASRResult:
    model_name = _funasr_model_name(backend)
    if not os.path.exists(audio_path):
        return _failure_result(audio_path, backend, model_name, "file not found")

    duration_ms = _wav_duration_ms(audio_path)
    if duration_ms is not None and duration_ms < _min_asr_audio_duration_ms():
        return _empty_audio_result(audio_path, backend, model_name, duration_ms)

    try:
        model = _load_funasr_model(backend, model_name)
        logger.info("Transcribing audio segment: %s with ASR backend '%s' model '%s'", audio_path, backend, model_name)
        generate_kwargs: Dict[str, Any] = {
            "input": audio_path,
            "cache": {},
            "batch_size_s": int(os.environ.get("ASR_BATCH_SIZE_S", "60")),
            "disable_pbar": os.environ.get("ASR_DISABLE_PBAR", "true").lower() != "false",
        }
        if backend in {"funasr_sensevoice", "sensevoice"}:
            generate_kwargs.update(
                {
                    "language": os.environ.get("ASR_LANGUAGE", "auto"),
                    "use_itn": os.environ.get("ASR_USE_ITN", "true").lower() != "false",
                    "merge_vad": os.environ.get("ASR_MERGE_VAD", "true").lower() != "false",
                    "merge_length_s": int(os.environ.get("ASR_MERGE_LENGTH_S", "15")),
                }
            )

        result = model.generate(**generate_kwargs)
        raw_text = ""
        if isinstance(result, list) and result:
            raw_text = str(result[0].get("text", ""))
        elif isinstance(result, dict):
            raw_text = str(result.get("text", ""))

        metadata = _parse_sensevoice_tokens(raw_text) if backend in {"funasr_sensevoice", "sensevoice"} else {}
        normalized_text = raw_text
        if backend in {"funasr_sensevoice", "sensevoice"}:
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess

                normalized_text = rich_transcription_postprocess(raw_text)
            except Exception:
                normalized_text = re.sub(r"<\|[^|]+\|>", "", raw_text)

        cleaned_text = clean_stt_text(normalized_text)
        logger.info("Raw ASR for '%s' (first 100 chars): %s...", audio_path, raw_text[:100])
        logger.info("Cleaned ASR for '%s' (first 100 chars): %s...", audio_path, cleaned_text[:100])
        return _set_last_result(
            ASRResult(
                text=cleaned_text,
                backend=backend,
                model_name=model_name,
                raw_text=raw_text,
                language=metadata.get("language"),
                emotion=metadata.get("emotion"),
                event=metadata.get("event"),
                metadata=metadata,
            )
        )
    except Exception as exc:
        logger.error("FunASR transcription failed for '%s': %s", audio_path, exc, exc_info=True)
        return _failure_result(audio_path, backend, model_name, "funasr transcription failed")


def transcribe_segment_with_metadata(audio_path: str) -> ASRResult:
    backend = get_asr_backend()
    if backend in WHISPER_BACKENDS:
        try:
            try:
                from .whisper_stt import MODEL_NAME, transcribe_segment as whisper_transcribe
            except Exception:
                from speaker_analysis.whisper_stt import MODEL_NAME, transcribe_segment as whisper_transcribe

            text = whisper_transcribe(audio_path) or ""
            return _set_last_result(ASRResult(text=text, backend="whisper", model_name=MODEL_NAME))
        except Exception as exc:
            return _failure_result(audio_path, "whisper", os.environ.get("WHISPER_MODEL_NAME", "small"), str(exc))

    if backend in FUNASR_BACKENDS:
        return _transcribe_with_funasr(audio_path, backend)

    logger.warning("Unknown ASR_BACKEND '%s'; falling back to Whisper.", backend)
    try:
        try:
            from .whisper_stt import MODEL_NAME, transcribe_segment as whisper_transcribe
        except Exception:
            from speaker_analysis.whisper_stt import MODEL_NAME, transcribe_segment as whisper_transcribe

        text = whisper_transcribe(audio_path) or ""
        return _set_last_result(ASRResult(text=text, backend="whisper", model_name=MODEL_NAME))
    except Exception as exc:
        return _failure_result(audio_path, backend, os.environ.get("WHISPER_MODEL_NAME", "small"), str(exc))


def transcribe_segment(audio_path: str) -> str:
    """Backward-compatible string-only transcription entrypoint."""
    return transcribe_segment_with_metadata(audio_path).text
