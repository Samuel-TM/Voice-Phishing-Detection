import torch
import torch.nn as nn
import numpy as np
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", (PROJECT_DIR / ".cache" / "numba").as_posix())

import librosa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)
_MODEL_CACHE: Dict[Tuple[str, str], Tuple["AudioCNNLSTM", dict]] = {}

class AudioCNNLSTM(nn.Module): 
    def __init__(self, model_config):
        super(AudioCNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=model_config['conv_in_channels'],
            out_channels=model_config['conv_out_channels'],
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        lstm_input_actual_size = model_config['conv_out_channels']
        self.lstm = nn.LSTM(
            input_size=lstm_input_actual_size,
            hidden_size=model_config['lstm_hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc1 = nn.Linear(model_config['lstm_hidden_size'] * 2, model_config['fc1_out_features'])
        self.fc2 = nn.Linear(model_config['fc1_out_features'], model_config['num_classes'])

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def extract_mfcc_features(audio_path, feature_config):
    n_mfcc = feature_config['n_mfcc']
    n_fft = feature_config['n_fft']
    hop_length = feature_config['hop_length']
    max_length = feature_config['max_length']
    try:
        # 提取 MFCC 特征，供 CNN-BiLSTM 推理使用
        logger.debug(f"Extracting MFCC for: {audio_path}")
        audio_data, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length).T
        if mfccs.shape[0] > max_length:
            mfccs = mfccs[:max_length, :]
        else:
            pad_width = max_length - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant', constant_values=(0,))
        logger.debug(f"MFCC extracted successfully for {audio_path}, shape: {mfccs.shape}")
        return mfccs
    except Exception as e:
        logger.error(f"Error processing audio file {audio_path} for MFCC: {e}", exc_info=True)
        return None

def load_predict_config(config_path):
    # 加载训练时保存的特征参数和模型结构参数
    logger.debug(f"Loading config from: {config_path}")
    if not os.path.exists(config_path):
        logger.critical(f"Config file NOT FOUND at {config_path}")
        return None, None, None
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if config['model_params']['conv_in_channels'] != config['feature_params']['n_mfcc']:
        logger.warning(f"model_params.conv_in_channels adjusted to feature_params.n_mfcc for {config_path}.")
        config['model_params']['conv_in_channels'] = config['feature_params']['n_mfcc']
    return config.get('feature_params'), config.get('model_params'), config.get('output_paths')


def load_deepvoice_model_once(model_path: str, config_path: str) -> Tuple[Optional[AudioCNNLSTM], Optional[dict]]:
    """按模型路径和配置路径缓存 CNN-BiLSTM，滑动窗口分析时避免重复加载权重。"""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return None, None

    cache_key = (os.path.abspath(model_path), os.path.abspath(config_path))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    feature_params, model_params_from_config, _ = load_predict_config(config_path)
    if feature_params is None or model_params_from_config is None:
        logger.error(f"Failed to load feature_params or model_params from config {config_path}")
        return None, None

    model = AudioCNNLSTM(model_params_from_config).to(device)
    try:
        logger.debug(f"Loading model state_dict from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.debug(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model state_dict from {model_path}: {e}", exc_info=True)
        return None, None
    model.eval()
    _MODEL_CACHE[cache_key] = (model, feature_params)
    return model, feature_params


def deepvoice_predict(audio_path: str, model_path: str, config_path: str) -> float:
    logger.debug(f"Attempting prediction for: {audio_path} using model: {model_path}, config: {config_path}")

    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found at {audio_path}")
        return 0.0

    model, feature_params = load_deepvoice_model_once(model_path, config_path)
    if model is None or feature_params is None:
        return 0.0

    features = extract_mfcc_features(audio_path, feature_params)
    if features is None:
        logger.warning(f"Feature extraction returned None for {audio_path}")
        return 0.0 

    features_tensor = torch.tensor(features, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)

    deepfake_probability = 0.0
    try:
        with torch.no_grad():
            outputs = model(features_tensor) 
            probabilities = torch.softmax(outputs, dim=1)
            deepfake_probability = probabilities[0, 1].item()
            logger.debug(f"Raw logits for {audio_path}: {outputs.cpu().numpy()}")
            logger.debug(f"Probabilities for {audio_path}: {probabilities.cpu().numpy()}, DF Prob: {deepfake_probability}")
    except Exception as e:
        logger.error(f"Error during model inference for {audio_path}: {e}", exc_info=True)
        return 0.0
    
    return deepfake_probability

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict if an audio file is a deepvoice.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--model_path", type=str, help="Path to the trained model file (.pt). Defaults to best_f1_model.pt from config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the JSON configuration file used for training.")
    
    args = parser.parse_args()

    used_model_path = args.model_path
    if used_model_path is None:
        _, _, output_paths_config = load_predict_config(args.config_path)
        if output_paths_config and 'best_model_save_path' in output_paths_config:
            used_model_path = output_paths_config['best_model_save_path']
            logger.info(f"Model path not provided, using default from config: {used_model_path}")
        else:
            print("Error: Model path not provided and 'best_model_save_path' not found in config.")
            logger.critical("Model path not provided and 'best_model_save_path' not found in config.")
            exit()
            
    if not used_model_path or not os.path.exists(used_model_path):
        print(f"Error: Effective model path is invalid or model file does not exist: {used_model_path}")
        logger.critical(f"Effective model path is invalid or model file does not exist: {used_model_path}")
        exit()

    score = deepvoice_predict(args.audio_path, used_model_path, args.config_path)
    
    print(f"--- Prediction Result ---")
    print(f"Audio: {args.audio_path}")
    print(f"Using model: {used_model_path}")
    print(f"Using config: {args.config_path}")
    print(f"Deepfake Probability (Class 1): {score:.4f}")

    if score > 0.5: 
        print("Prediction: Deepvoice / synthetic speech risk")
    else:
        print("Prediction: Real voice / low synthetic speech risk")
