# External normal-finance hard-negative stress test for Chinese-BERT.
#
# Purpose:
#   Evaluate an already trained text-risk model on an independent external
#   normal_finance / fraud semantics holdout. This script does not train,
#   does not sample from TeleAntiFraud, and does not generate paraphrases.
#
# Expected input:
#   Upload a CSV/JSON/JSONL dataset to Kaggle with at least a text column.
#   Recommended columns:
#     sample_id, case_type, text_label, text
#
# Label convention:
#   text_label=1 means the text semantics are phishing/fraud.
#   text_label=0 means the text is benign, even if audio/global label is risky.
#   If text_label is missing, the script derives it from case_type.

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


CONFIG: Dict[str, Any] = {
    # Set these paths in Kaggle if your dataset/model names differ.
    "external_data_path": os.environ.get(
        "EXTERNAL_STRESS_DATA_PATH",
        "/kaggle/input/normal-finance-hard-negative-stress",
    ),
    "model_checkpoint_path": os.environ.get("CHINESE_BERT_CHECKPOINT", ""),
    "model_name": os.environ.get("CHINESE_BERT_MODEL_NAME", "bert-base-chinese"),
    "output_dir": os.environ.get(
        "STRESS_OUTPUT_DIR",
        "/kaggle/working/external_normal_finance_stress_test",
    ),
    "max_len": int(os.environ.get("CHINESE_BERT_MAX_LEN", "256")),
    "batch_size": int(os.environ.get("CHINESE_BERT_BATCH_SIZE", "32")),
    "thresholds": [50.0, 70.0],
    "seed": 42,
}

TEXT_COLUMNS = (
    "text",
    "full_asr_text",
    "metadata_text",
    "transcript_text",
    "full_text",
    "content",
    "utterance",
)
TEXT_LABEL_COLUMNS = ("text_label", "semantic_label")
GLOBAL_LABEL_COLUMNS = ("label", "global_label")
CASE_TYPE_TEXT_LABEL = {
    "normal_daily": 0,
    "normal_finance": 0,
    "synthetic_voice": 0,
    "benign": 0,
    "benign_finance": 0,
    "hardneg_benign": 0,
    "semantic_fraud": 1,
    "mixed_risk": 1,
    "fraud": 1,
    "phishing": 1,
    "adv_phish": 1,
}
CHECKPOINT_CANDIDATES = (
    "/kaggle/input/notebooks/samueltmsun/fork-of-chinese-bert/model/best_model.pt",
    "/kaggle/input/notebooks/samueltmsun/fork-of-chinese-bert/model/train.pt",
    "/kaggle/input/chinese-bert-model/best_model.pt",
    "/kaggle/input/chinese-bert-model/train.pt",
    "/kaggle/working/model/best_model.pt",
    "/kaggle/working/model/train.pt",
    "ChineseBERTModel/model/train.pt",
    "ChineseBERTModel/model/best_model.pt",
)

HARD_KEYWORDS = (
    "验证码",
    "短信验证码",
    "动态码",
    "校验码",
    "otp",
    "转账",
    "汇款",
    "打款",
    "付款",
    "充值",
    "刷流水",
    "链接",
    "网址",
    "点击",
    "扫码",
    "二维码",
    "下载",
    "安装",
    "app",
    "远程",
    "共享屏幕",
    "冻结",
    "解冻",
    "风控",
    "账户异常",
    "异常交易",
    "贷款",
    "额度",
    "征信",
    "逾期",
    "退款",
    "理赔",
    "赔付",
    "保证金",
    "手续费",
    "公安",
    "检察院",
    "法院",
    "警察",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text_for_dedup(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；：,.!?;:（）()\[\]{}【】\"'“”‘’·…—\-_/\\|<>~`@#$%^&*+=]", "", text)
    return text.lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def parse_binary_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "fraud", "phishing", "scam", "risk", "risky"}:
        return 1
    if text in {"0", "0.0", "false", "normal", "benign", "safe"}:
        return 0
    return int(safe_float(value))


def find_first_existing_file(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path
    if path.is_dir():
        candidates: List[Path] = []
        for pattern in ("*.csv", "*.jsonl", "*.json"):
            candidates.extend(sorted(path.glob(pattern)))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        "External stress dataset not found. Set EXTERNAL_STRESS_DATA_PATH to a CSV/JSON/JSONL file "
        "or to a Kaggle input directory containing one."
    )


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        raw = handle.read().strip()
    if not raw:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    payload = json.loads(raw)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [row for row in payload["records"] if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON schema: {path}")


def load_external_dataset(path_value: str) -> pd.DataFrame:
    path = find_first_existing_file(path_value)
    print(f"Loading external stress dataset: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        df = pd.DataFrame(load_json_records(path))
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if df.empty:
        raise ValueError(f"No rows loaded from {path}")

    text_col = next((col for col in TEXT_COLUMNS if col in df.columns), None)
    if text_col is None:
        raise ValueError(f"No text column found. Expected one of: {TEXT_COLUMNS}")

    out = pd.DataFrame()
    out["text"] = df[text_col].fillna("").astype(str).str.strip()
    out["case_type"] = df.get("case_type", "unknown")
    out["case_type"] = out["case_type"].fillna("unknown").astype(str).str.strip()
    out["sample_id"] = df.get("sample_id", [f"sample_{idx:05d}" for idx in range(len(df))])
    out["sample_id"] = out["sample_id"].fillna("").astype(str)
    out.loc[out["sample_id"].str.len() == 0, "sample_id"] = [
        f"sample_{idx:05d}" for idx in out.index[out["sample_id"].str.len() == 0]
    ]

    text_label_col = next((col for col in TEXT_LABEL_COLUMNS if col in df.columns), None)
    global_label_col = next((col for col in GLOBAL_LABEL_COLUMNS if col in df.columns), None)
    normalized_case = out["case_type"].str.lower()

    if text_label_col:
        out["text_label"] = df[text_label_col].apply(parse_binary_label)
        out["label_source"] = text_label_col
    elif normalized_case.isin(CASE_TYPE_TEXT_LABEL).all():
        out["text_label"] = normalized_case.map(CASE_TYPE_TEXT_LABEL).astype(int)
        out["label_source"] = "case_type_mapping"
    elif global_label_col:
        out["text_label"] = df[global_label_col].apply(parse_binary_label)
        out["label_source"] = global_label_col
        print(
            "Warning: using global label as text_label. For multimodal sets, prefer explicit "
            "text_label so synthetic_voice benign text is not counted as text fraud."
        )
    else:
        missing = sorted(out.loc[~normalized_case.isin(CASE_TYPE_TEXT_LABEL), "case_type"].unique())
        raise ValueError(
            "Rows are missing text_label/semantic_label and case_type cannot be mapped: "
            f"{missing}. Add text_label where benign text is 0 and phishing text is 1."
        )

    optional_columns = ("source", "audio_path", "input_view", "notes")
    for col in optional_columns:
        if col in df.columns:
            out[col] = df[col]

    out = out[out["text"].str.len() > 0].copy()
    out["norm"] = out["text"].apply(normalize_text_for_dedup)
    before = len(out)
    out = out.drop_duplicates(subset=["norm"], keep="first").drop(columns=["norm"]).reset_index(drop=True)
    print(f"Rows loaded: {len(out)} (deduplicated {before - len(out)})")
    print("Case distribution:")
    print(out["case_type"].value_counts().to_string())
    print("Text-label distribution:")
    print(out["text_label"].value_counts().sort_index().to_string())
    return out


class ChineseBERTClassifier(nn.Module):
    def __init__(self, bert_model: nn.Module, hidden_size: int = 768, num_classes: int = 2, dr_rate: float = 0.3):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(p=dr_rate)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled_output))
        return logits


class TextDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer: Any, max_len: int):
        self.texts = [str(text) for text in texts]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        token_type_ids = encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"]))
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
        }


def resolve_checkpoint(path_value: str) -> Path:
    if path_value:
        path = Path(path_value)
        if path.exists():
            return path
        raise FileNotFoundError(f"Configured checkpoint does not exist: {path}")
    for candidate in CHECKPOINT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "No Chinese-BERT checkpoint found. Set CHINESE_BERT_CHECKPOINT to best_model.pt or train.pt."
    )


def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[ChineseBERTClassifier, Any, torch.device, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(str(config.get("model_checkpoint_path", "")))
    print(f"Using device: {device}")
    print(f"Loading tokenizer/base model: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    bert = AutoModel.from_pretrained(config["model_name"])
    model = ChineseBERTClassifier(bert, hidden_size=768, num_classes=2, dr_rate=0.3).to(device)
    print(f"Loading checkpoint: {checkpoint}")
    state = torch.load(checkpoint.as_posix(), map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing state_dict keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected state_dict keys: {unexpected}")
    model.eval()
    return model, tokenizer, device, checkpoint


@torch.no_grad()
def predict_scores(
    texts: Sequence[str],
    model: ChineseBERTClassifier,
    tokenizer: Any,
    device: torch.device,
    max_len: int,
    batch_size: int,
) -> np.ndarray:
    dataset = TextDataset(texts, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores: List[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        logits = model(input_ids, attention_mask, token_type_ids)
        probs = F.softmax(logits, dim=-1)[:, 1]
        scores.extend((probs.detach().cpu().numpy() * 100.0).tolist())
    return np.array(scores, dtype=float)


def keyword_scores(texts: Sequence[str]) -> np.ndarray:
    pattern = re.compile("|".join(re.escape(keyword.lower()) for keyword in HARD_KEYWORDS))
    scores: List[float] = []
    for text in texts:
        normalized = str(text or "").lower().replace(" ", "")
        scores.append(99.0 if pattern.search(normalized) else 1.0)
    return np.array(scores, dtype=float)


def binary_metrics(y_true: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, Any]:
    y_true_arr = np.array(y_true, dtype=int)
    score_arr = np.array(scores, dtype=float)
    y_pred = (score_arr >= threshold).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=labels).ravel()
    auc_value: Optional[float]
    try:
        auc_value = float(roc_auc_score(y_true_arr, score_arr)) if len(set(y_true_arr.tolist())) > 1 else None
    except Exception:
        auc_value = None
    return {
        "threshold": threshold,
        "samples": int(len(y_true_arr)),
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred)), 4),
        "precision": round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
        "roc_auc": round(auc_value, 4) if auc_value is not None else "",
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "normal_samples": int((y_true_arr == 0).sum()),
        "normal_fp_rate": round(float(fp / max(1, (y_true_arr == 0).sum())), 4),
        "fraud_samples": int((y_true_arr == 1).sum()),
        "fraud_recall": round(float(tp / max(1, (y_true_arr == 1).sum())), 4),
        "mean_score": round(float(np.mean(score_arr)), 4) if len(score_arr) else 0.0,
    }


def build_metrics_table(df: pd.DataFrame, score_field: str, thresholds: Iterable[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        row = binary_metrics(df["text_label"], df[score_field], threshold)
        row.update({"scope": "overall", "case_type": "all", "score_field": score_field})
        rows.append(row)
        for case_type, group in df.groupby("case_type", dropna=False):
            case_row = binary_metrics(group["text_label"], group[score_field], threshold)
            case_row.update({"scope": "case_type", "case_type": case_type, "score_field": score_field})
            rows.append(case_row)
    return pd.DataFrame(rows)


def build_distribution_table(df: pd.DataFrame, score_field: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for case_type, group in df.groupby("case_type", dropna=False):
        scores = group[score_field].astype(float)
        rows.append(
            {
                "case_type": case_type,
                "score_field": score_field,
                "samples": len(group),
                "mean": round(float(scores.mean()), 4),
                "std": round(float(scores.std(ddof=0)), 4),
                "min": round(float(scores.min()), 4),
                "p25": round(float(scores.quantile(0.25)), 4),
                "median": round(float(scores.median()), 4),
                "p75": round(float(scores.quantile(0.75)), 4),
                "max": round(float(scores.max()), 4),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame, output_dir: Path, thresholds: Sequence[float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "external_stress_predictions.csv"
    metrics_path = output_dir / "external_stress_metrics.csv"
    distribution_path = output_dir / "score_distribution_by_case.csv"
    normal_fp_path = output_dir / "normal_finance_false_positives.csv"
    fn_path = output_dir / "false_negatives.csv"

    metrics = pd.concat(
        [
            build_metrics_table(df, "bert_score", thresholds),
            build_metrics_table(df, "keyword_score", thresholds),
        ],
        ignore_index=True,
    )
    distribution = pd.concat(
        [
            build_distribution_table(df, "bert_score"),
            build_distribution_table(df, "keyword_score"),
        ],
        ignore_index=True,
    )

    main_threshold = float(thresholds[0])
    df["bert_prediction"] = (df["bert_score"] >= main_threshold).astype(int)
    df["keyword_prediction"] = (df["keyword_score"] >= main_threshold).astype(int)
    df["bert_false_positive"] = ((df["text_label"] == 0) & (df["bert_prediction"] == 1)).astype(int)
    df["bert_false_negative"] = ((df["text_label"] == 1) & (df["bert_prediction"] == 0)).astype(int)

    df.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    distribution.to_csv(distribution_path, index=False, encoding="utf-8-sig")

    normal_finance_fp = df[
        (df["case_type"].astype(str).str.lower() == "normal_finance") & (df["bert_false_positive"] == 1)
    ].sort_values("bert_score", ascending=False)
    normal_finance_fp.to_csv(normal_fp_path, index=False, encoding="utf-8-sig")

    false_negatives = df[df["bert_false_negative"] == 1].sort_values("bert_score", ascending=True)
    false_negatives.to_csv(fn_path, index=False, encoding="utf-8-sig")

    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote score distribution: {distribution_path}")
    print(f"Wrote normal_finance false positives: {normal_fp_path}")
    print(f"Wrote false negatives: {fn_path}")
    print("\nBERT metrics:")
    print(metrics[metrics["score_field"] == "bert_score"].to_string(index=False))


def main() -> None:
    set_seed(int(CONFIG["seed"]))
    output_dir = Path(CONFIG["output_dir"])
    df = load_external_dataset(str(CONFIG["external_data_path"]))
    model, tokenizer, device, checkpoint = load_model_and_tokenizer(CONFIG)
    df["bert_score"] = predict_scores(
        df["text"].tolist(),
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_len=int(CONFIG["max_len"]),
        batch_size=int(CONFIG["batch_size"]),
    )
    df["keyword_score"] = keyword_scores(df["text"].tolist())
    df["checkpoint"] = checkpoint.as_posix()
    df["max_len"] = int(CONFIG["max_len"])
    write_outputs(df, output_dir=output_dir, thresholds=[float(x) for x in CONFIG["thresholds"]])


if __name__ == "__main__":
    main()
