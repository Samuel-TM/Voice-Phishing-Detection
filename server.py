# server.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import json
import logging
import time
import uuid
import warnings
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

THIS_FILE = Path(__file__).resolve()
ML_DIR = THIS_FILE.parent
PROJECT_CACHE_DIR = ML_DIR / ".cache"
os.environ.setdefault("HF_HOME", (PROJECT_CACHE_DIR / "huggingface").as_posix())
os.environ.setdefault("TRANSFORMERS_CACHE", (PROJECT_CACHE_DIR / "huggingface" / "hub").as_posix())
os.environ.setdefault("XDG_CACHE_HOME", PROJECT_CACHE_DIR.as_posix())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import numpy as np  # NumPy 类型检查用
from flask import Flask, Response, jsonify, request, render_template, stream_with_context
from flask_cors import CORS
from pydub import AudioSegment
from werkzeug.utils import secure_filename

warnings.filterwarnings(action="ignore")

# -----------------------------------------------------------------------------
# 运行上下文兼容：既支持包方式，也支持在项目根目录直接运行 server.py
# -----------------------------------------------------------------------------
PROJECT_ROOT = ML_DIR.parent

if PROJECT_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, PROJECT_ROOT.as_posix())
if ML_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ML_DIR.as_posix())

# 包导入和直接脚本导入都做兼容
try:
    from .ChineseBERTModel.ensemble_utils import ensemble_inference
    from .audio_risk_detection.predict_audio_risk import (
        audio_risk_predict,
        audio_risk_predict_with_decision,
        get_decision_threshold,
        probability_to_voice_score,
    )
    from .speaker_analysis.speaker_pipeline import analyze_multi_speaker_audio
    from .speaker_analysis.asr_backend import transcribe_segment
    from .streaming_analysis.risk_scoring import (
        RiskScoringState,
        final_label_from_score,
        normalize_scoring_mode,
        score_window,
    )
    from .streaming_analysis.window_pipeline import (
        analyze_audio_stream,
        build_rolling_context_fields,
        iter_audio_stream_analysis,
    )
except Exception:
    try:
        from ML.ChineseBERTModel.ensemble_utils import ensemble_inference
        from ML.audio_risk_detection.predict_audio_risk import (
            audio_risk_predict,
            audio_risk_predict_with_decision,
            get_decision_threshold,
            probability_to_voice_score,
        )
        from ML.speaker_analysis.speaker_pipeline import analyze_multi_speaker_audio
        from ML.speaker_analysis.asr_backend import transcribe_segment
        from ML.streaming_analysis.risk_scoring import (
            RiskScoringState,
            final_label_from_score,
            normalize_scoring_mode,
            score_window,
        )
        from ML.streaming_analysis.window_pipeline import (
            analyze_audio_stream,
            build_rolling_context_fields,
            iter_audio_stream_analysis,
        )
    except Exception:
        from ChineseBERTModel.ensemble_utils import ensemble_inference
        from audio_risk_detection.predict_audio_risk import (
            audio_risk_predict,
            audio_risk_predict_with_decision,
            get_decision_threshold,
            probability_to_voice_score,
        )
        from speaker_analysis.speaker_pipeline import analyze_multi_speaker_audio
        from speaker_analysis.asr_backend import transcribe_segment
        from streaming_analysis.risk_scoring import (
            RiskScoringState,
            final_label_from_score,
            normalize_scoring_mode,
            score_window,
        )
        from streaming_analysis.window_pipeline import (
            analyze_audio_stream,
            build_rolling_context_fields,
            iter_audio_stream_analysis,
        )

# -----------------------------------------------------------------------------
# Flask APP
# -----------------------------------------------------------------------------
TEMPLATE_DIR = (ML_DIR / "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR.as_posix())
CORS(app)

log_format = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "[%(module)s:%(funcName)s:%(lineno)d] - %(message)s"
)
logging.basicConfig(level=logging.INFO, format=log_format)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
app.logger.info("Flask App Logger initialized.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app.logger.info(f"Using device: {device} in server.py")

# -----------------------------------------------------------------------------
# 路径/模型文件
# -----------------------------------------------------------------------------
AUDIO_RISK_MODEL_FILENAME = "best_f1_model.pt"
AUDIO_RISK_MODEL_PATH = ML_DIR / "audio_risk_detection" / "model" / AUDIO_RISK_MODEL_FILENAME
AUDIO_RISK_CONFIG_PATH = ML_DIR / "audio_risk_detection" / "model" / "audio_risk_config.json"
LEGACY_AUDIO_RISK_CONFIG_PATH = ML_DIR / "audio_risk_detection" / "audio_risk_config.json"
OLD_AUDIO_RISK_MODEL_PATH = ML_DIR / "deepvoice_detection" / "model" / AUDIO_RISK_MODEL_FILENAME
OLD_AUDIO_RISK_CONFIG_PATH = ML_DIR / "deepvoice_detection" / "model" / "deepvoice_config.json"
OLD_LEGACY_AUDIO_RISK_CONFIG_PATH = ML_DIR / "deepvoice_detection" / "deepvoice_config.json"
if not AUDIO_RISK_MODEL_PATH.exists() and OLD_AUDIO_RISK_MODEL_PATH.exists():
    AUDIO_RISK_MODEL_PATH = OLD_AUDIO_RISK_MODEL_PATH
if not AUDIO_RISK_CONFIG_PATH.exists() and LEGACY_AUDIO_RISK_CONFIG_PATH.exists():
    AUDIO_RISK_CONFIG_PATH = LEGACY_AUDIO_RISK_CONFIG_PATH
if not AUDIO_RISK_CONFIG_PATH.exists() and OLD_AUDIO_RISK_CONFIG_PATH.exists():
    AUDIO_RISK_CONFIG_PATH = OLD_AUDIO_RISK_CONFIG_PATH
if not AUDIO_RISK_CONFIG_PATH.exists() and OLD_LEGACY_AUDIO_RISK_CONFIG_PATH.exists():
    AUDIO_RISK_CONFIG_PATH = OLD_LEGACY_AUDIO_RISK_CONFIG_PATH

critical_error = False
if not AUDIO_RISK_MODEL_PATH.exists():
    app.logger.critical(f"Audio risk model NOT FOUND at {AUDIO_RISK_MODEL_PATH}")
    critical_error = True
if not AUDIO_RISK_CONFIG_PATH.exists():
    app.logger.critical(f"Audio risk config NOT FOUND at {AUDIO_RISK_CONFIG_PATH}")
    critical_error = True
if critical_error:
    app.logger.error("Essential model or config files are missing (audio risk). "
                     "Audio-related features may be disabled.")

AUDIO_RISK_DECISION_THRESHOLD = 0.5
AUDIO_RISK_THRESHOLD_SOURCE = "default_0_5"
if AUDIO_RISK_CONFIG_PATH.exists():
    try:
        AUDIO_RISK_DECISION_THRESHOLD = get_decision_threshold(AUDIO_RISK_CONFIG_PATH.as_posix())
        with AUDIO_RISK_CONFIG_PATH.open("r", encoding="utf-8") as config_handle:
            config_payload = json.load(config_handle)
        AUDIO_RISK_THRESHOLD_SOURCE = (
            config_payload.get("decision_params", {}).get("threshold_source")
            or AUDIO_RISK_THRESHOLD_SOURCE
        )
        app.logger.info(
            "Audio risk model configured: model=%s config=%s decision_threshold=%.4f source=%s",
            AUDIO_RISK_MODEL_PATH,
            AUDIO_RISK_CONFIG_PATH,
            AUDIO_RISK_DECISION_THRESHOLD,
            AUDIO_RISK_THRESHOLD_SOURCE,
        )
    except Exception:
        app.logger.warning("Failed to load audio risk decision threshold; using 0.5.", exc_info=True)

UPLOAD_DIR = ML_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LIVE_STREAM_CACHE_DIR = PROJECT_CACHE_DIR / "live_audio_streams"
LIVE_STREAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LIVE_STREAM_SESSIONS: Dict[str, Dict[str, Any]] = {}
LIVE_STREAM_LOCK = Lock()

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def safe_float(x: Any, default: float = 0.0) -> float:
    """数字/字符串/NumPy 标量统一安全转换为 float。"""
    try:
        # 防御 NumPy 标量类型
        if isinstance(x, (np.generic,)):
            return float(np.asarray(x))
        return float(x)
    except Exception:
        return default


def build_text_response(analysis_result: Dict[str, Any], text_fallback: str) -> Dict[str, Any]:
    """/predict 响应 JSON，保留旧接口字段。"""
    llm_score = safe_float(analysis_result.get("llm_score", 0))
    final_label = analysis_result.get("final_label", "Analysis Failed")
    text_out = analysis_result.get("text", text_fallback)

    return {
        "final_label": final_label,
        "text": text_out,
        "llm_score": round(llm_score, 2),
        "voice_score": 0,
        "deepfake_score": "N/A",
        "total_score": round(llm_score, 2),
    }


def clamp_number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return max(minimum, min(maximum, numeric))


def live_extension_from_upload(filename: str, mime_type: str) -> str:
    mime_type = str(mime_type or "").lower()
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    allowed = {"wav", "mp3", "flac", "m4a", "ogg", "webm", "mp4"}

    def ext_from_mime(mime: str) -> str:
        if "m4a" in mime:
            return "m4a"
        if "mp4" in mime:
            return "mp4"
        if "webm" in mime:
            return "webm"
        if "ogg" in mime:
            return "ogg"
        if "mpeg" in mime or "mp3" in mime:
            return "mp3"
        if "wav" in mime:
            return "wav"
        return ""

    mime_ext = ext_from_mime(mime_type)
    if suffix in allowed:
        if mime_ext and mime_ext != suffix:
            return mime_ext
        return suffix
    if mime_ext:
        return mime_ext
    return "webm"


def get_live_session(session_id: Optional[str] = None) -> Dict[str, Any]:
    now = time.time()
    with LIVE_STREAM_LOCK:
        if session_id and session_id in LIVE_STREAM_SESSIONS:
            session = LIVE_STREAM_SESSIONS[session_id]
            session["updated_at"] = now
            return session

        new_id = session_id or uuid.uuid4().hex
        session = {
            "session_id": new_id,
            "created_at": now,
            "updated_at": now,
            "transcript_parts": [],
            "previous_smoothed": None,
            "consecutive_risk": 0,
            "consecutive_suspicious_evidence": 0,
            "alert_latched": False,
            "scoring_mode": "baseline",
            "timeline": [],
        }
        LIVE_STREAM_SESSIONS[new_id] = session
        return session


def update_live_session_point(
    session_id: str,
    point: Dict[str, Any],
    transcript_parts: List[str],
    scoring_state: RiskScoringState,
) -> None:
    with LIVE_STREAM_LOCK:
        session = LIVE_STREAM_SESSIONS[session_id]
        session["timeline"].append(point)
        session["transcript_parts"] = transcript_parts
        session["previous_smoothed"] = scoring_state.previous_smoothed
        session["consecutive_risk"] = scoring_state.consecutive_risk
        session["consecutive_suspicious_evidence"] = scoring_state.consecutive_suspicious_evidence
        session["alert_latched"] = scoring_state.alert_latched
        session["scoring_mode"] = point.get("scoring_mode")
        session["case_type"] = point.get("case_type", session.get("case_type", ""))
        session["updated_at"] = time.time()


def convert_live_chunk_to_wav(chunk_path: Path) -> Path:
    wav_path = chunk_path.with_suffix(".wav")
    if chunk_path.suffix.lower() == ".wav":
        return chunk_path
    audio = AudioSegment.from_file(chunk_path.as_posix())
    audio.export(wav_path.as_posix(), format="wav")
    return wav_path


def prepare_live_chunk_wav(
    session_id: str,
    chunk_path: Path,
    chunk_index: int,
    start_sec: float,
    end_sec: float,
    chunk_seconds: float,
) -> Path:
    suffix = chunk_path.suffix.lower().lstrip(".")
    if suffix not in {"webm", "ogg", "mp4", "m4a"}:
        return convert_live_chunk_to_wav(chunk_path)

    session_dir = LIVE_STREAM_CACHE_DIR / session_id
    stream_path = session_dir / f"live_recording.{suffix}"
    mode = "wb" if chunk_index == 0 or not stream_path.exists() else "ab"
    with LIVE_STREAM_LOCK:
        with chunk_path.open("rb") as source, stream_path.open(mode) as target:
            target.write(source.read())

    audio = AudioSegment.from_file(stream_path.as_posix())
    audio_len_ms = len(audio)
    if audio_len_ms <= 0:
        raise ValueError("Decoded live audio stream is empty.")

    start_ms = int(max(0.0, start_sec) * 1000)
    end_ms = int(max(start_sec, end_sec) * 1000)
    if end_ms <= start_ms:
        end_ms = start_ms + int(max(1.0, chunk_seconds) * 1000)

    start_ms = min(start_ms, audio_len_ms)
    end_ms = min(max(end_ms, start_ms + 100), audio_len_ms)
    if end_ms <= start_ms:
        start_ms = max(0, audio_len_ms - int(max(1.0, chunk_seconds) * 1000))
        end_ms = audio_len_ms

    wav_path = chunk_path.with_suffix(".wav")
    audio[start_ms:end_ms].export(wav_path.as_posix(), format="wav")
    return wav_path

# -----------------------------------------------------------------------------
# 路由
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def main_page():
    if TEMPLATE_DIR.exists() and (TEMPLATE_DIR / "main.html").exists():
        return render_template("main.html")
    return "Server is running. (No main.html found)", 200


@app.route("/predict", methods=["POST"])
def predict_text_route():
    """
    文本单次检测接口。
    保证旧前端需要的字段 final_label/text/llm_score/voice_score/deepfake_score/total_score。
    """
    try:
        data = request.get_json(silent=True) or {}
        text_input = (data.get("text") or "").strip()

        if not text_input:
            return jsonify({
                "final_label": "No Input",
                "text": "",
                "llm_score": 0,
                "voice_score": 0,
                "deepfake_score": "N/A",
                "total_score": 0,
                "error": "Text input is required."
            }), 400

        analysis_result = ensemble_inference(text_input) or {}
        response_data = build_text_response(analysis_result, text_input)
        app.logger.debug(f"Text prediction API response data: {response_data}")
        return jsonify(response_data), 200

    except Exception:
        app.logger.error("文本预测 API 错误", exc_info=True)
        return jsonify({
            "final_label": "Error",
            "text": "",
            "llm_score": 0,
            "voice_score": 0,
            "deepfake_score": "N/A",
            "total_score": 0,
            "error": "A server error occurred during text analysis."
        }), 500


@app.route("/api/stream_audio_analysis", methods=["POST"])
def api_stream_audio_analysis():
    """
    上传录音后按滑动窗口模拟实时分析，以 NDJSON 逐窗口返回风险点。
    第一版不做实时说话人分离，符合 simulated streaming 原型范围。
    """
    if "audio_file" not in request.files:
        return jsonify({"error": "Audio file is required."}), 400

    audio_file = request.files["audio_file"]
    if not audio_file.filename or not audio_file.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
        return jsonify({"error": "Please upload a valid audio file (.wav, .mp3, .flac, .m4a, .ogg)."}), 400

    filename = secure_filename(audio_file.filename)
    audio_path = (UPLOAD_DIR / f"stream_{filename}").resolve()
    audio_file.save(audio_path.as_posix())

    audio_risk_ok = AUDIO_RISK_MODEL_PATH.exists() and AUDIO_RISK_CONFIG_PATH.exists()
    window_seconds = request.form.get("window_seconds", 10)
    step_seconds = request.form.get("step_seconds", 5)
    text_weight = clamp_number(request.form.get("text_weight"), default=0.8, minimum=0.0, maximum=1.0)
    voice_weight = round(1.0 - text_weight, 4)
    smoothing_previous_weight = clamp_number(
        request.form.get("smoothing_previous_weight"),
        default=0.65,
        minimum=0.0,
        maximum=0.95,
    )
    scoring_mode = normalize_scoring_mode(request.form.get("scoring_mode") or "baseline")
    case_type = (request.form.get("case_type") or "").strip()

    def encode_event(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False) + "\n"

    @stream_with_context
    def generate_events():
        try:
            if not audio_risk_ok:
                yield encode_event({
                    "event": "warning",
                    "warning": "Voice model or config is missing. Voice scores are set to zero.",
                })

            for event in iter_audio_stream_analysis(
                audio_path=audio_path.as_posix(),
                audio_risk_model_path=AUDIO_RISK_MODEL_PATH.as_posix() if audio_risk_ok else None,
                audio_risk_config_path=AUDIO_RISK_CONFIG_PATH.as_posix() if audio_risk_ok else None,
                text_inference=ensemble_inference,
                audio_risk_inference=audio_risk_predict_with_decision,
                transcribe_segment=transcribe_segment,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                text_weight=text_weight,
                voice_weight=voice_weight,
                smoothing_previous_weight=smoothing_previous_weight,
                scoring_mode=scoring_mode,
                case_type=case_type,
            ):
                yield encode_event(event)
        except Exception:
            app.logger.error("滑动窗口音频分析失败", exc_info=True)
            yield encode_event({
                "event": "error",
                "error": "A server error occurred during streaming audio analysis.",
            })
        finally:
            try:
                if audio_path.exists():
                    audio_path.unlink(missing_ok=True)
            except Exception as e_remove:
                app.logger.error(f"Error cleaning up audio file {audio_path}: {e_remove}")

    return Response(
        generate_events(),
        mimetype="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/live_audio_chunk", methods=["POST"])
def api_live_audio_chunk():
    """
    浏览器麦克风实时分片分析接口。
    每个 MediaRecorder chunk 单独上传，后端维护 session 级累计文本和平滑风险。
    """
    if "audio_chunk" not in request.files:
        return jsonify({"error": "Audio chunk is required."}), 400

    audio_chunk = request.files["audio_chunk"]
    session_id = (request.form.get("session_id") or "").strip() or None
    session = get_live_session(session_id)
    session_id = session["session_id"]

    chunk_index = safe_float(request.form.get("chunk_index"), len(session["timeline"]))
    chunk_seconds = clamp_number(request.form.get("chunk_seconds"), default=5.0, minimum=1.0, maximum=30.0)
    start_sec = safe_float(request.form.get("chunk_start_sec"), chunk_index * chunk_seconds)
    end_sec = safe_float(request.form.get("chunk_end_sec"), start_sec + chunk_seconds)
    text_weight = clamp_number(request.form.get("text_weight"), default=0.8, minimum=0.0, maximum=1.0)
    voice_weight = round(1.0 - text_weight, 4)
    smoothing_previous_weight = clamp_number(
        request.form.get("smoothing_previous_weight"),
        default=0.65,
        minimum=0.0,
        maximum=0.95,
    )
    smoothing_current_weight = 1.0 - smoothing_previous_weight
    scoring_mode = normalize_scoring_mode(request.form.get("scoring_mode") or session.get("scoring_mode"))
    case_type = (request.form.get("case_type") or session.get("case_type") or "").strip()

    filename = secure_filename(audio_chunk.filename or f"chunk_{int(chunk_index):04d}.webm")
    ext = live_extension_from_upload(filename, audio_chunk.mimetype)
    session_dir = LIVE_STREAM_CACHE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = (session_dir / f"chunk_{int(chunk_index):04d}.{ext}").resolve()
    audio_chunk.save(chunk_path.as_posix())

    audio_risk_ok = AUDIO_RISK_MODEL_PATH.exists() and AUDIO_RISK_CONFIG_PATH.exists()
    transcript_parts: List[str] = list(session.get("transcript_parts", []))
    scoring_state = RiskScoringState(
        previous_smoothed=session.get("previous_smoothed"),
        consecutive_risk=int(safe_float(session.get("consecutive_risk"), 0)),
        consecutive_suspicious_evidence=int(
            safe_float(session.get("consecutive_suspicious_evidence"), 0)
        ),
        alert_latched=bool(session.get("alert_latched", False)),
    )

    try:
        wav_path = prepare_live_chunk_wav(
            session_id=session_id,
            chunk_path=chunk_path,
            chunk_index=int(chunk_index),
            start_sec=start_sec,
            end_sec=end_sec,
            chunk_seconds=chunk_seconds,
        )

        window_text = ""
        text_error = None
        try:
            window_text = transcribe_segment(wav_path.as_posix()) or ""
            if window_text.startswith("(STT"):
                text_error = window_text
                window_text = ""
        except Exception:
            app.logger.error("实时音频分片 STT 失败", exc_info=True)
            text_error = "STT failed for this chunk."

        previous_window_texts = transcript_parts[-3:]
        rolling_context = build_rolling_context_fields(
            previous_window_texts=previous_window_texts,
            current_window_text=window_text,
        )
        text_score = 0.0
        text_result: Dict[str, Any] = {}
        text_model_input = window_text.strip()
        if text_model_input:
            try:
                text_result = ensemble_inference(text_model_input) or {}
                text_score = safe_float(text_result.get("llm_score", 0.0))
            except Exception:
                app.logger.error("实时音频分片文本风险推理失败", exc_info=True)
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
                    context_text_result = ensemble_inference(rolling_context_text) or {}
                    context_text_score = safe_float(context_text_result.get("llm_score", 0.0))
                except Exception:
                    app.logger.error("实时音频分片滚动上下文文本风险推理失败", exc_info=True)
                    context_text_result = {"error": "Rolling-context text inference failed."}

        if window_text.strip():
            transcript_parts.append(window_text.strip())
        cumulative_text = " ".join(transcript_parts).strip()

        deepfake_probability = 0.0
        voice_score = 0.0
        voice_decision: Dict[str, Any] = {}
        voice_error = None
        if audio_risk_ok:
            try:
                voice_decision = audio_risk_predict_with_decision(
                    wav_path.as_posix(),
                    AUDIO_RISK_MODEL_PATH.as_posix(),
                    AUDIO_RISK_CONFIG_PATH.as_posix(),
                )
                deepfake_probability = safe_float(voice_decision.get("deepfake_probability", 0.0))
                voice_score = safe_float(voice_decision.get("voice_score", 0.0))
            except Exception:
                app.logger.error("实时音频分片声学风险推理失败", exc_info=True)
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
            "index": int(chunk_index),
            "start_sec": round(start_sec, 2),
            "end_sec": round(max(end_sec, start_sec), 2),
            "text": window_text,
            "current_window_text": rolling_context["current_window_text"],
            "recent_context_text": rolling_context["recent_context_text"],
            "rolling_context_text": rolling_context["rolling_context_text"],
            "cumulative_text": cumulative_text,
            "text_score": scoring["text_score"],
            "raw_text_score": scoring["raw_text_score"],
            "raw_window_text_score": scoring["raw_window_text_score"],
            "context_text_score": scoring["context_text_score"],
            "voice_score": scoring["voice_score"],
            "deepfake_score": round(deepfake_probability, 4),
            "deepfake_detected_voice": bool(voice_decision.get("deepfake_detected_voice", False)),
            "fused_score": scoring["fused_score"],
            "smoothed_score": scoring["smoothed_score"],
            "risk_level": scoring["risk_level"],
            "source": "microphone",
            "case_type": case_type,
        }
        point.update({key: value for key, value in scoring.items() if key not in point})
        if voice_decision.get("decision_threshold") is not None:
            point["voice_decision_threshold"] = round(safe_float(voice_decision.get("decision_threshold")), 4)
        if voice_decision.get("threshold_source"):
            point["voice_threshold_source"] = voice_decision["threshold_source"]
        if text_error:
            point["text_error"] = text_error
        if voice_error:
            point["voice_error"] = voice_error
        if text_result.get("error"):
            point["text_model_error"] = text_result["error"]
        if context_text_result.get("error"):
            point["context_text_model_error"] = context_text_result["error"]

        update_live_session_point(session_id, point, transcript_parts, scoring_state)
        return jsonify({
            "event": "point",
            "session_id": session_id,
            "point": point,
            "full_transcript": cumulative_text,
            "weights": {
                "text": text_weight,
                "voice": voice_weight,
                "smoothing_previous": smoothing_previous_weight,
                "smoothing_current": smoothing_current_weight,
            },
            "scoring_mode": scoring_mode,
            "warning": None if audio_risk_ok else "Voice model or config is missing. Voice scores are set to zero.",
        }), 200
    except Exception:
        app.logger.error("实时音频分片分析失败", exc_info=True)
        return jsonify({
            "event": "error",
            "session_id": session_id,
            "error": "A server error occurred during live audio analysis.",
        }), 500


@app.route("/api/live_audio_finish", methods=["POST"])
def api_live_audio_finish():
    """结束浏览器实时麦克风分析，并返回 session 汇总。"""
    data = request.get_json(silent=True) or request.form or {}
    session_id = (data.get("session_id") or "").strip()
    if not session_id or session_id not in LIVE_STREAM_SESSIONS:
        return jsonify({"error": "Live audio session not found."}), 404

    with LIVE_STREAM_LOCK:
        session = LIVE_STREAM_SESSIONS[session_id]
        timeline = list(session.get("timeline", []))
        full_transcript = " ".join(session.get("transcript_parts", [])).strip()

    max_point = max(timeline, key=lambda item: safe_float(item.get("smoothed_score")), default=None)
    final_score = safe_float(timeline[-1].get("smoothed_score")) if timeline else 0.0
    return jsonify({
        "event": "done",
        "session_id": session_id,
        "timeline": timeline,
        "full_transcript": full_transcript,
        "final_score": round(final_score, 2),
        "max_score": round(safe_float(max_point.get("smoothed_score")), 2) if max_point else 0.0,
        "final_label": final_label_from_score(final_score),
        "highest_risk_window": max_point,
        "mode": "browser_microphone",
    }), 200


@app.route("/api/audio_result", methods=["POST"])
def api_audio_result():
    """
    旧版整段音频分析接口。
    - 多说话人：复用原 speaker pipeline，并规整为统一分数字段
    - 单说话人：STT → 文本风险分析 + 深伪语音概率 → 融合输出
    """
    app.logger.debug("Received request for /api/audio_result")

    # 检查上传文件和扩展名
    if "audio_file" not in request.files:
        app.logger.warning("Audio file not in request.files")
        return jsonify({"error": "Audio file is required."}), 400

    audio_file = request.files["audio_file"]
    if not audio_file.filename or not audio_file.filename.lower().endswith((".wav", ".mp3", ".flac")):
        app.logger.warning(f"Invalid audio file: {audio_file.filename}")
        return jsonify({"error": "Please upload a valid audio file (.wav, .mp3, .flac)."}), 400

    # 保存到临时上传目录
    filename = secure_filename(audio_file.filename)
    audio_path = (UPLOAD_DIR / filename).resolve()
    audio_file.save(audio_path.as_posix())
    app.logger.debug(f"Audio file saved to {audio_path}")

    # 兼容旧版多说话人/单说话人选项
    multi_speaker = True
    try:
        multi_speaker_form_value = request.form.get("multi_speaker", "true")
        multi_speaker = multi_speaker_form_value.lower() == "true"
    except Exception:
        pass
    app.logger.debug(f"Multi-speaker mode: {multi_speaker}")

    # 默认响应骨架，便于异常时仍返回稳定结构
    final_response_data: Dict[str, Any] = {
        "speaker_0": {
            "deepfake_score": 0.0,
            "final_label": "Analysis Failed",
            "llm_score": 0.0,
            "text": "",
            "total_score": 0.0,
            "voice_score": 0.0,
            "phishing": False,
            "final_decision": "Analysis Failed",
            "phishing_detected_text": False,
            "deepfake_detected_voice": False,
        }
    }

    try:
        # Audio risk 文件缺失时禁用音频风险评分
        audio_risk_ok = AUDIO_RISK_MODEL_PATH.exists() and AUDIO_RISK_CONFIG_PATH.exists()
        if not audio_risk_ok:
            app.logger.warning("Audio risk model/config is missing. Voice-related scores will be zeros.")

        if multi_speaker:
            app.logger.debug(f"Analyzing multi-speaker audio: {audio_path}")
            # 原 pipeline 结果示例：
            # {
            #   "speaker_me":   {"text": ..., "text_score": 0~100, "phishing_detected_text": bool,
            #                    "deepfake_score": 0~1, "deepfake_detected_voice": bool, ...},
            #   "speaker_other": {...}
            # }
            raw_results = analyze_multi_speaker_audio(
                audio_path.as_posix(),
                AUDIO_RISK_MODEL_PATH.as_posix(),
                AUDIO_RISK_CONFIG_PATH.as_posix()
            ) if audio_risk_ok else analyze_multi_speaker_audio(
                audio_path.as_posix(), None, None
            )
            app.logger.debug(f"Multi-speaker analysis raw result: {raw_results}")

            if isinstance(raw_results, dict) and raw_results:
                processed: Dict[str, Any] = {}
                for speaker_id, data in raw_results.items():
                    llm_s  = safe_float(data.get("text_score", 0.0))     # 0~100
                    dv_prob = safe_float(data.get("deepfake_score", 0.0)) # 0~1

                    # 统一到校准后的 0-100 风险分并做 8:2 融合
                    voice_s = safe_float(
                        data.get("voice_score"),
                        probability_to_voice_score(dv_prob, AUDIO_RISK_DECISION_THRESHOLD),
                    )
                    total_s = round((0.8 * llm_s) + (0.2 * voice_s), 2)

                    # 优先使用 pipeline 原始标记，缺失时按阈值补齐
                    text_flag  = bool(data.get("phishing_detected_text", llm_s >= 70))
                    voice_flag = bool(data.get("deepfake_detected_voice", dv_prob > AUDIO_RISK_DECISION_THRESHOLD))

                    # 最终二分类判定：70 分以上视为高风险
                    final_label = "Fraud Risk Detected" if total_s >= 70 else "No High Risk Detected"

                    processed[speaker_id] = {
                        "text": data.get("text", ""),
                        "phishing_detected_text": text_flag,
                        "text_score": round(llm_s, 2),
                        "deepfake_score": round(dv_prob, 4),
                        "deepfake_detected_voice": voice_flag,
                        "voice_decision_threshold": safe_float(
                            data.get("voice_decision_threshold"),
                            AUDIO_RISK_DECISION_THRESHOLD,
                        ),
                        "voice_threshold_source": data.get("voice_threshold_source", AUDIO_RISK_THRESHOLD_SOURCE),
                        "phishing": (total_s >= 70),
                        "final_decision": final_label,

                        # 旧 UI 兼容键
                        "final_label": final_label,
                        "llm_score": round(llm_s, 2),
                        "voice_score": voice_s,     # 0~100
                        "total_score": total_s,     # 0~100
                    }
                final_response_data = processed
            else:
                app.logger.warning("No valid multi-speaker results. Returning default error payload.")
                final_response_data["speaker_0"]["error"] = "Speaker analysis result could not be processed."
                final_response_data = {"speaker_0": final_response_data["speaker_0"]}

        else:
            app.logger.debug(f"Analyzing single-speaker audio: {audio_path}")
            # --- STT ---
            try:
                # 延迟导入，减少非单说话人路径的初始化成本
                from .speaker_analysis.asr_backend import transcribe_segment
            except Exception:
                from ML.speaker_analysis.asr_backend import transcribe_segment  # fallback

            try:
                text = transcribe_segment(audio_path.as_posix())
            except Exception as e_stt:
                app.logger.error(f"STT Error for single speaker: {e_stt}", exc_info=True)
                text = ""

            # --- 文本风险 ---
            text_risk_result = ensemble_inference(text) or {}
            llm_s = safe_float(text_risk_result.get("llm_score", 0.0))
            is_text_phishing = bool(text_risk_result.get("phishing_detected", llm_s > 50))

            # --- 音频深伪风险 ---
            voice_s = 0.0
            is_voice_deepfake = False
            voice_decision: Dict[str, Any] = {}
            if audio_risk_ok:
                voice_decision = audio_risk_predict_with_decision(
                    audio_path.as_posix(),
                    AUDIO_RISK_MODEL_PATH.as_posix(),
                    AUDIO_RISK_CONFIG_PATH.as_posix()
                )
                deep_prob = safe_float(voice_decision.get("deepfake_probability", 0.0))
                voice_s = safe_float(voice_decision.get("voice_score", 0.0))
                is_voice_deepfake = bool(voice_decision.get("deepfake_detected_voice", False))
            else:
                deep_prob = 0.0

            # 0-100 分制 + 8:2 融合
            total_s = round((0.8 * llm_s) + (0.2 * voice_s), 2) # 0~100

            # 最终标签，70 分为高风险阈值
            final_label = "Fraud Risk Detected" if total_s >= 70 else "No High Risk Detected"

            final_response_data = {
                "speaker_0": {
                    "deepfake_score": round(deep_prob, 4),
                    "final_label": final_label,
                    "llm_score": round(llm_s, 2),
                    "text": text,
                    "total_score": total_s,
                    "voice_score": voice_s,
                    "phishing_detected_text": is_text_phishing,
                    "deepfake_detected_voice": is_voice_deepfake,
                    "voice_decision_threshold": safe_float(
                        voice_decision.get("decision_threshold"),
                        AUDIO_RISK_DECISION_THRESHOLD,
                    ),
                    "voice_threshold_source": voice_decision.get("threshold_source", AUDIO_RISK_THRESHOLD_SOURCE),
                    "phishing": (total_s >= 70),
                    "final_decision": final_label,
                }
            }

        app.logger.debug(f"Final response data for audio API: {final_response_data}")
        return jsonify(final_response_data), 200

    except Exception:
        app.logger.error("整段音频 API 处理失败", exc_info=True)
        error_response_key = "error_info"
        final_response_data = {
            error_response_key: {
                "error": "A server error occurred during audio processing.",
                "final_label": "Error",
                "text": "",
                "llm_score": 0,
                "voice_score": 0,
                "deepfake_score": 0.0,
                "total_score": 0,
            }
        }
        return jsonify(final_response_data), 500

    finally:
        # 清理上传文件
        try:
            if audio_path.exists():
                audio_path.unlink(missing_ok=True)
        except Exception as e_remove:
            app.logger.error(f"Error cleaning up audio file {audio_path}: {e_remove}")

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 开发环境启动方式
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=True, use_reloader=False)
    # 部署环境示例
    # app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
