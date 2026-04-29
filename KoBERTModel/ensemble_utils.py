# 文件名: KoBERTModel/ensemble_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# 路径固定：以当前文件为基准定位权重，避免受启动目录影响
# ---------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
os.environ.setdefault("HF_HOME", (PROJECT_DIR / ".cache" / "huggingface").as_posix())
os.environ.setdefault("TRANSFORMERS_CACHE", (PROJECT_DIR / ".cache" / "huggingface" / "hub").as_posix())
MODEL_PATH = THIS_DIR / "model" / "train.pt"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
CHINESE_BERT_MODEL_NAME = "bert-base-chinese"
MAX_LEN = 256

# ---------------------------------------------------------------------
# 与中文 BERT 训练脚本兼容的分类器定义
# ---------------------------------------------------------------------
class ChineseBERTClassifier(nn.Module):
    def __init__(self, bert_model, hidden_size: int = 768, num_classes: int = 2, dr_rate: float = 0.3):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        # 部分模型可能没有 pooler_output，这里回退到 CLS 向量
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            last_hidden = outputs.last_hidden_state  # [B, T, H]
            pooled_output = last_hidden[:, 0, :]     # [B, H]
        attentions = outputs.attentions

        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits, attentions

class ChineseBERTDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_len: int = MAX_LEN):
        self.texts = [str(t) for t in texts]
        self.labels = labels if labels is not None else [0] * len(self.texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        # BERT 系列通常有 token_type_ids；保留防御逻辑便于替换模型
        if "token_type_ids" in encoding:
            result["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        else:
            result["token_type_ids"] = torch.zeros_like(result["input_ids"])
        return result

# ---------------------------------------------------------------------
# 全局加载：中文 BERT tokenizer / base model / 已训练分类器
# ---------------------------------------------------------------------
bert_tokenizer_global: Optional[AutoTokenizer] = None
bert_trained_classifier_global: Optional[ChineseBERTClassifier] = None
bert_model_load_error_global: Optional[str] = None


def load_text_model_once() -> bool:
    """懒加载中文 BERT 模型，避免服务导入时因依赖或权重问题直接崩溃。"""
    global bert_tokenizer_global, bert_trained_classifier_global, bert_model_load_error_global

    if bert_tokenizer_global is not None and bert_trained_classifier_global is not None:
        return True
    if bert_model_load_error_global:
        return False

    try:
        logger.info("[ensemble_utils] Loading Chinese BERT tokenizer...")
        try:
            bert_tokenizer_global = AutoTokenizer.from_pretrained(CHINESE_BERT_MODEL_NAME, local_files_only=True)
        except Exception:
            logger.warning("Local Chinese BERT tokenizer cache is missing; falling back to online download.")
            bert_tokenizer_global = AutoTokenizer.from_pretrained(CHINESE_BERT_MODEL_NAME)

        logger.info("[ensemble_utils] Loading Chinese BERT base...")
        try:
            bert_base_for_classifier = AutoModel.from_pretrained(
                CHINESE_BERT_MODEL_NAME, output_attentions=True, local_files_only=True
            )
        except Exception:
            logger.warning("Local Chinese BERT base cache is missing; falling back to online download.")
            bert_base_for_classifier = AutoModel.from_pretrained(
                CHINESE_BERT_MODEL_NAME, output_attentions=True
            )

        if not MODEL_PATH.exists():
            bert_model_load_error_global = f"Chinese BERT classifier file not found at: {MODEL_PATH}"
            logger.critical(bert_model_load_error_global)
            return False

        logger.info(f"[ensemble_utils] Loading trained Chinese BERT classifier: {MODEL_PATH}")
        model = ChineseBERTClassifier(bert_base_for_classifier).to(device)
        state = torch.load(MODEL_PATH.as_posix(), map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        if missing_keys:
            logger.warning(f"Chinese BERT state_dict missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Chinese BERT state_dict unexpected keys: {unexpected_keys}")
        model.eval()
        bert_trained_classifier_global = model
        logger.info("[ensemble_utils] Chinese BERT classifier loaded successfully.")
        return True
    except Exception as e_load:
        bert_model_load_error_global = f"Error loading Chinese BERT model: {e_load}"
        logger.critical(bert_model_load_error_global, exc_info=True)
        return False

# ---------------------------------------------------------------------
# 内部预测函数
# ---------------------------------------------------------------------
@torch.no_grad()
def bert_predict_internal(text: str) -> Tuple[float, Any]:
    """
    Returns:
        phishing_probability (float in [0,1]),
        attentions (Any or None)
    """
    if not load_text_model_once():
        logger.error(f"Chinese BERT model not loaded. Error: {bert_model_load_error_global}")
        return 0.0, None

    dataset = ChineseBERTDataset([text], [0], bert_tokenizer_global, max_len=MAX_LEN)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    logits, attentions = bert_trained_classifier_global(input_ids, attention_mask, token_type_ids)
    probs = F.softmax(logits, dim=-1)
    phishing_prob = float(probs[0, 1].detach().cpu().item())  # class 1 = phishing
    return phishing_prob, attentions

# ---------------------------------------------------------------------
# 服务端使用的公开函数
# ---------------------------------------------------------------------
def _label_from_score(score_0_100: float) -> str:
    """按照统一风险等级生成中文标签，供旧接口兼容使用。"""
    if score_0_100 >= 90:
        return "极高危"
    if score_0_100 >= 70:
        return "高危"
    if score_0_100 > 50:
        return "可疑"
    return "正常"

def ensemble_inference(text: str) -> Dict[str, Any]:
    """
    返回 server.py 和滑动窗口管线需要的字典格式：
    {
        "text": str,
        "llm_score": float(0~100),
        "final_label": str,
        "phishing_detected": bool,
        "attentions": Any
    }
    """
    phishing_prob, attentions = bert_predict_internal(text)
    llm_score = round(phishing_prob * 100.0, 2)
    final_label = _label_from_score(llm_score)
    phishing_detected = llm_score > 50.0

    # 模型未加载等根因错误透传给调用方，便于前端展示和排查
    payload: Dict[str, Any] = {
        "text": text,
        "llm_score": llm_score,
        "final_label": final_label,
        "phishing_detected": phishing_detected,
        "attentions": attentions,
    }
    if bert_model_load_error_global:
        payload["error"] = bert_model_load_error_global
    return payload
