# server.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Any, Dict

THIS_FILE = Path(__file__).resolve()
ML_DIR = THIS_FILE.parent
PROJECT_CACHE_DIR = ML_DIR / ".cache"
os.environ.setdefault("HF_HOME", (PROJECT_CACHE_DIR / "huggingface").as_posix())
os.environ.setdefault("TRANSFORMERS_CACHE", (PROJECT_CACHE_DIR / "huggingface" / "hub").as_posix())
os.environ.setdefault("XDG_CACHE_HOME", PROJECT_CACHE_DIR.as_posix())
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import numpy as np  # NumPy 类型检查用
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
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
    from .KoBERTModel.ensemble_utils import ensemble_inference
    from .deepvoice_detection.predict_deepvoice import deepvoice_predict
    from .speaker_analysis.speaker_pipeline import analyze_multi_speaker_audio
    from .speaker_analysis.whisper_stt import transcribe_segment
    from .streaming_analysis.window_pipeline import analyze_audio_stream
except Exception:
    try:
        from ML.KoBERTModel.ensemble_utils import ensemble_inference
        from ML.deepvoice_detection.predict_deepvoice import deepvoice_predict
        from ML.speaker_analysis.speaker_pipeline import analyze_multi_speaker_audio
        from ML.speaker_analysis.whisper_stt import transcribe_segment
        from ML.streaming_analysis.window_pipeline import analyze_audio_stream
    except Exception:
        from KoBERTModel.ensemble_utils import ensemble_inference
        from deepvoice_detection.predict_deepvoice import deepvoice_predict
        from speaker_analysis.speaker_pipeline import analyze_multi_speaker_audio
        from speaker_analysis.whisper_stt import transcribe_segment
        from streaming_analysis.window_pipeline import analyze_audio_stream

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
DEEPVOICE_MODEL_FILENAME = "best_f1_model.pt"
DEEPVOICE_MODEL_PATH = ML_DIR / "deepvoice_detection" / "model" / DEEPVOICE_MODEL_FILENAME
DEEPVOICE_CONFIG_PATH = ML_DIR / "deepvoice_detection" / "model" / "deepvoice_config.json"
LEGACY_DEEPVOICE_CONFIG_PATH = ML_DIR / "deepvoice_detection" / "deepvoice_config.json"
if not DEEPVOICE_CONFIG_PATH.exists() and LEGACY_DEEPVOICE_CONFIG_PATH.exists():
    DEEPVOICE_CONFIG_PATH = LEGACY_DEEPVOICE_CONFIG_PATH

critical_error = False
if not DEEPVOICE_MODEL_PATH.exists():
    app.logger.critical(f"Deepvoice model NOT FOUND at {DEEPVOICE_MODEL_PATH}")
    critical_error = True
if not DEEPVOICE_CONFIG_PATH.exists():
    app.logger.critical(f"Deepvoice config NOT FOUND at {DEEPVOICE_CONFIG_PATH}")
    critical_error = True
if critical_error:
    app.logger.error("Essential model or config files are missing (deepvoice). "
                     "Audio-related features may be disabled.")

UPLOAD_DIR = ML_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    上传录音后按滑动窗口模拟实时分析，返回风险时间线。
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

    try:
        deepvoice_ok = DEEPVOICE_MODEL_PATH.exists() and DEEPVOICE_CONFIG_PATH.exists()
        result = analyze_audio_stream(
            audio_path=audio_path.as_posix(),
            deepvoice_model_path=DEEPVOICE_MODEL_PATH.as_posix() if deepvoice_ok else None,
            deepvoice_config_path=DEEPVOICE_CONFIG_PATH.as_posix() if deepvoice_ok else None,
            text_inference=ensemble_inference,
            deepvoice_inference=deepvoice_predict,
            transcribe_segment=transcribe_segment,
            window_seconds=request.form.get("window_seconds", 10),
            step_seconds=request.form.get("step_seconds", 5),
        )
        if not deepvoice_ok:
            result["warning"] = "Voice model or config is missing. Voice scores are set to zero."
        return jsonify(result), 200
    except Exception:
        app.logger.error("滑动窗口音频分析失败", exc_info=True)
        return jsonify({"error": "A server error occurred during streaming audio analysis."}), 500
    finally:
        try:
            if audio_path.exists():
                audio_path.unlink(missing_ok=True)
        except Exception as e_remove:
            app.logger.error(f"Error cleaning up audio file {audio_path}: {e_remove}")


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
        # Deepvoice 文件缺失时禁用音频风险评分
        deepvoice_ok = DEEPVOICE_MODEL_PATH.exists() and DEEPVOICE_CONFIG_PATH.exists()
        if not deepvoice_ok:
            app.logger.warning("Deepvoice model/config is missing. Voice-related scores will be zeros.")

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
                DEEPVOICE_MODEL_PATH.as_posix(),
                DEEPVOICE_CONFIG_PATH.as_posix()
            ) if deepvoice_ok else analyze_multi_speaker_audio(
                audio_path.as_posix(), None, None
            )
            app.logger.debug(f"Multi-speaker analysis raw result: {raw_results}")

            if isinstance(raw_results, dict) and raw_results:
                processed: Dict[str, Any] = {}
                for speaker_id, data in raw_results.items():
                    llm_s  = safe_float(data.get("text_score", 0.0))     # 0~100
                    dv_prob = safe_float(data.get("deepfake_score", 0.0)) # 0~1

                    # 统一到 0-100 分制并做 8:2 融合
                    voice_s = round(dv_prob * 100, 2)
                    total_s = round((0.8 * llm_s) + (0.2 * voice_s), 2)

                    # 优先使用 pipeline 原始标记，缺失时按阈值补齐
                    text_flag  = bool(data.get("phishing_detected_text", llm_s >= 70))
                    voice_flag = bool(data.get("deepfake_detected_voice", dv_prob >= 0.5))

                    # 最终二分类判定：70 分以上视为高风险
                    final_label = "Fraud Risk Detected" if total_s >= 70 else "No High Risk Detected"

                    processed[speaker_id] = {
                        "text": data.get("text", ""),
                        "phishing_detected_text": text_flag,
                        "text_score": round(llm_s, 2),
                        "deepfake_score": round(dv_prob, 4),
                        "deepfake_detected_voice": voice_flag,
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
                from .speaker_analysis.whisper_stt import transcribe_segment
            except Exception:
                from ML.speaker_analysis.whisper_stt import transcribe_segment  # fallback

            try:
                text = transcribe_segment(audio_path.as_posix())
            except Exception as e_stt:
                app.logger.error(f"STT Error for single speaker: {e_stt}", exc_info=True)
                text = ""

            # --- 文本风险 ---
            kobert_text_result = ensemble_inference(text) or {}
            llm_s = safe_float(kobert_text_result.get("llm_score", 0.0))
            is_text_phishing = bool(kobert_text_result.get("phishing_detected", llm_s > 50))

            # --- 音频深伪风险 ---
            if deepvoice_ok:
                deep_prob = deepvoice_predict(
                    audio_path.as_posix(),
                    DEEPVOICE_MODEL_PATH.as_posix(),
                    DEEPVOICE_CONFIG_PATH.as_posix()
                )
                deep_prob = 0.0 if deep_prob is None else safe_float(deep_prob, 0.0)
            else:
                deep_prob = 0.0

            is_voice_deepfake = deep_prob > 0.5

            # 0-100 分制 + 8:2 融合
            voice_s = round(deep_prob * 100, 2)                 # 0~100
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
