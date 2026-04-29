import whisper
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Whisper 模型名称可选 tiny/base/small/medium/large；small 在准确率和速度之间相对均衡。
PROJECT_DIR = Path(__file__).resolve().parents[1]
WHISPER_CACHE_DIR = PROJECT_DIR / ".cache" / "whisper"
MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "small")
model = None
model_load_error_message = None

def load_whisper_model_once():
    """
    首次 STT 调用时懒加载 Whisper，避免服务启动阶段下载/加载模型。
    成功返回 True，失败返回 False。
    """
    global model, model_load_error_message
    if model is not None:
        logger.info(f"Whisper model '{MODEL_NAME}' is already loaded.")
        return True
    
    if model_load_error_message is not None:
        logger.error(f"Skipping Whisper model load attempt due to previous error: {model_load_error_message}")
        return False

    try:
        logger.info(f"Attempting to load Whisper model: {MODEL_NAME}...")
        WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model = whisper.load_model(MODEL_NAME, device="cpu", download_root=WHISPER_CACHE_DIR.as_posix())
        logger.info(f"Whisper model '{MODEL_NAME}' loaded successfully onto CPU.")
        return True
    except Exception as e:
        model_load_error_message = str(e)
        logger.critical(f"CRITICAL: Failed to load Whisper model '{MODEL_NAME}': {e}", exc_info=True)
        return False

def clean_stt_text(text: str) -> str:
    """
    清理 STT 文本：当前系统面向中文诈骗对话，保留中文、数字和空白。
    """
    if not text or not isinstance(text, str):
        return ""
    
    cleaned_text = re.sub(r'[^\u4e00-\u9fff0-9\s]', '', text)
    
    # 合并连续空白并去除首尾空白
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def transcribe_segment(audio_path: str) -> str:
    """将指定音频片段转写为中文文本并做基础清理。"""
    global model, model_load_error_message

    # 1. 检查 Whisper 模型状态
    if model_load_error_message:
        logger.error(f"STT failed for '{audio_path}': Whisper model previously failed to load ({model_load_error_message}).")
        return "(STT failed - initial model load error)"
    
    if model is None:
        logger.warning(f"STT warning for '{audio_path}': Whisper model was None. Attempting to reload...")
        if not load_whisper_model_once() or model is None:
             logger.error(f"STT failed for '{audio_path}': Whisper model reload failed or still None.")
             return "(STT failed - model reload failed)"
        logger.info(f"Whisper model '{MODEL_NAME}' reloaded successfully for STT on '{audio_path}'.")

    # 2. 检查音频文件
    if not os.path.exists(audio_path):
        logger.error(f"STT failed: Audio file not found at '{audio_path}'")
        return "(STT failed - file not found)"
    
    try:
        file_size = os.path.getsize(audio_path)
        if file_size < 1024:
            logger.warning(f"STT warning: Audio file at '{audio_path}' is very small ({file_size} bytes). Transcription may be empty or inaccurate.")
    except OSError as e_size:
        logger.error(f"STT failed: Could not get size of audio file at '{audio_path}': {e_size}")
        return "(STT failed - file size unavailable)"

    # 3. Whisper 中文转写
    try:
        logger.info(f"Transcribing audio segment: {audio_path} with model '{MODEL_NAME}'")
        
        transcription_result = model.transcribe(audio_path, language="zh", fp16=False)
        
        raw_text = transcription_result["text"]
        logger.debug(f"Raw STT for '{audio_path}' (first 100 chars): {raw_text[:100]}...")
        
        # 4. 清理转写文本
        cleaned_text = clean_stt_text(raw_text)
        logger.info(f"Cleaned STT for '{audio_path}' (first 100 chars): {cleaned_text[:100]}...")
        
        if not cleaned_text.strip() and raw_text.strip():
             logger.warning(f"STT warning: Cleaned text is empty for '{audio_path}', but raw text was not. Raw: '{raw_text[:50]}...'")
        
        return cleaned_text
    except Exception as e_transcribe:
        logger.error(f"STT error during transcription for '{audio_path}': {e_transcribe}", exc_info=True)
        return "(STT failed)"
