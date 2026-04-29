# speaker_analysis/speaker_pipeline.py

try:
    from .diarization_utils import split_speakers
    from .whisper_stt import transcribe_segment
    from ..deepvoice_detection.predict_deepvoice import deepvoice_predict
    from ..KoBERTModel.ensemble_utils import ensemble_inference
except Exception:
    from speaker_analysis.diarization_utils import split_speakers
    from speaker_analysis.whisper_stt import transcribe_segment
    from deepvoice_detection.predict_deepvoice import deepvoice_predict
    from KoBERTModel.ensemble_utils import ensemble_inference

import os
import logging

logger = logging.getLogger(__name__)

# server.py 调用的旧版整段音频分析主函数。
def analyze_multi_speaker_audio(audio_path: str, dv_model_path: str, dv_config_path: str):
    """
    对完整音频执行说话人分离、STT、文本风险分析和深伪语音检测，
    返回按说话人聚合的分析结果。
    """
    logger.info(f"Starting audio analysis pipeline for: {audio_path}")
    logger.debug(f"Using DV Model: {dv_model_path}, DV Config: {dv_config_path}")
    
    diarization_temp_output_dir = "" # 说话人分离产生的临时音频片段目录
    speaker_segments = {} # 说话人分离结果
    
    try:
        # 在上传文件同级目录下创建临时说话人片段目录
        diarization_temp_output_dir = os.path.join(os.path.dirname(audio_path), "temp_diarization_segments")
        os.makedirs(diarization_temp_output_dir, exist_ok=True)
        
        # 调用说话人分离工具
        speaker_segments = split_speakers(audio_path, out_dir=diarization_temp_output_dir)
        logger.info(f"Diarization result for {audio_path}: {speaker_segments}")
    except Exception as e:
        logger.error(f"Diarization process critically failed for {audio_path}: {e}", exc_info=True)
        # 分离失败时将整段音频视为单说话人
        speaker_segments = {"speaker_me": audio_path} 

    analysis_results = {}
    segment_files_to_delete_finally = []

    # 防御说话人分离返回空结果或非字典结果
    if not speaker_segments or not isinstance(speaker_segments, dict):
        logger.error(f"No speaker segments returned from diarization for {audio_path}. Cannot proceed with analysis.")
        return {"error": "Speaker diarization failed; audio analysis cannot continue."}


    # 对分离出的每个说话人片段执行分析
    for speaker_id, segment_audio_path in speaker_segments.items():
        # 检查片段路径有效性
        if not segment_audio_path or not os.path.exists(segment_audio_path):
            logger.error(f"Invalid or non-existent segment path for speaker {speaker_id}: '{segment_audio_path}'. Skipping this segment.")
            analysis_results[speaker_id] = {"error": "Segment file error", "final_decision": "Analysis Unavailable"}
            continue

        logger.info(f"Processing segment for {speaker_id} from path: {segment_audio_path}")
        
        # 只清理说话人分离生成的临时片段，不删除原始上传文件
        if segment_audio_path != audio_path and diarization_temp_output_dir and segment_audio_path.startswith(diarization_temp_output_dir):
            segment_files_to_delete_finally.append(segment_audio_path)

        # STT：语音转文本
        transcribed_text = ""
        try:
            transcribed_text = transcribe_segment(segment_audio_path)
            logger.info(f"STT result for {speaker_id} (first 50 chars): {transcribed_text[:50]}...")
        except Exception as e_stt:
            logger.error(f"STT failed for segment {segment_audio_path} of speaker {speaker_id}: {e_stt}", exc_info=True)

        # 中文 BERT 文本风险分析
        kobert_analysis_result = {"phishing_detected": False, "llm_score": 0.0, "final_label": "Normal"}
        try:
            if transcribed_text and not transcribed_text.startswith("(STT") and transcribed_text.strip():
                kobert_analysis_result = ensemble_inference(transcribed_text)
                logger.info(f"KoBERT analysis result for {speaker_id}: {kobert_analysis_result}")
            else:
                logger.warning(f"Skipping KoBERT analysis for {speaker_id} due to STT failure or empty text.")
        except Exception as e_kobert:
            logger.error(f"KoBERT analysis failed for text from speaker {speaker_id}: {e_kobert}", exc_info=True)

        # 深伪语音检测
        deepfake_probability = 0.0
        if dv_model_path and dv_config_path:
            try:
                logger.debug(f"Predicting deepvoice for {segment_audio_path} with model {dv_model_path}")
                deepfake_probability = deepvoice_predict(segment_audio_path, dv_model_path, dv_config_path)
                if deepfake_probability is None:
                    deepfake_probability = 0.0
                logger.info(f"Deepvoice score (probability) for {speaker_id}: {deepfake_probability:.4f}")
            except Exception as e_dv:
                logger.error(f"Deepvoice prediction failed for segment {segment_audio_path} of speaker {speaker_id}: {e_dv}", exc_info=True)
        else:
            logger.warning("Deepvoice model or config path is missing; voice risk score is set to 0.")

        # 汇总单说话人的多模态结果
        is_text_phishing = bool(kobert_analysis_result.get("phishing_detected", False))
        deepfake_detection_threshold = 0.5 
        is_voice_deepfake = deepfake_probability > deepfake_detection_threshold

        is_overall_phishing = is_text_phishing or is_voice_deepfake
        final_decision_message = "Fraud Risk Detected" if is_overall_phishing else "No High Risk Detected"

        analysis_results[speaker_id] = {
            "text": transcribed_text,
            "phishing_detected_text": is_text_phishing,
            "text_score": kobert_analysis_result.get("llm_score", 0.0),
            "deepfake_score": round(deepfake_probability, 4),
            "deepfake_detected_voice": is_voice_deepfake,
            "phishing": is_overall_phishing,
            "final_decision": final_decision_message,
        }
    
    # 清理说话人分离临时片段
    for path_to_delete in segment_files_to_delete_finally:
        if os.path.exists(path_to_delete):
            try:
                os.remove(path_to_delete)
                logger.debug(f"Successfully removed temporary segment file: {path_to_delete}")
            except OSError as e_remove:
                logger.error(f"Error removing temporary segment file {path_to_delete}: {e_remove}", exc_info=True)
    
    # 如果临时目录为空，则清理目录
    if diarization_temp_output_dir and os.path.exists(diarization_temp_output_dir):
        try:
            if not os.listdir(diarization_temp_output_dir):
                os.rmdir(diarization_temp_output_dir)
                logger.debug(f"Successfully removed empty temporary directory: {diarization_temp_output_dir}")
        except OSError as e_rmdir:
            logger.warning(f"Could not remove temporary directory {diarization_temp_output_dir} (it might not be empty or other issues): {e_rmdir}")
            
    logger.info(f"Audio analysis pipeline finished for: {audio_path}. Results: {analysis_results}")
    return analysis_results
