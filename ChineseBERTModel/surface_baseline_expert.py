# Compare a surface-feature baseline against the trained Chinese-BERT model.
#
# This script is intentionally close to baseline_of_chinese_bert_kaggle.py:
# it uses the same dataset schema, label convention, ChineseBERTClassifier,
# checkpoint loading, and score threshold style. The added part is a
# deliberately shallow SurfaceFeatureBaseline plus thesis-friendly tables
# and figures.
#
# Recommended thesis usage:
#   1. Primary comparison:
#      evaluation/external_normal_finance_hard_negative_stress.jsonl
#   2. Secondary final-system text diagnostic:
#      evaluation/final_text_label_diagnostic_metadata_final.jsonl
#
# Do not use fine-tuning train/dev files as final comparison evidence.

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path("/kaggle/working/.cache") if Path("/kaggle/working").exists() else PROJECT_ROOT / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", CACHE_ROOT.as_posix())
os.environ.setdefault("MPLCONFIGDIR", (CACHE_ROOT / "matplotlib").as_posix())
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:
    sns = None
    HAS_SEABORN = False


CONFIG: Dict[str, Any] = {
    "data_path": os.environ.get(
        "COMPARISON_DATA_PATH",
        os.environ.get("EXTERNAL_STRESS_DATA_PATH", "/kaggle/input/normal-finance-hard-negative-stress"),
    ),
    "model_checkpoint_path": os.environ.get("CHINESE_BERT_CHECKPOINT", ""),
    "model_name": os.environ.get("CHINESE_BERT_MODEL_NAME", "bert-base-chinese"),
    "output_dir": os.environ.get("COMPARISON_OUTPUT_DIR", "/kaggle/working/surface_vs_chinese_bert"),
    "max_len": int(os.environ.get("CHINESE_BERT_MAX_LEN", "256")),
    "batch_size": int(os.environ.get("CHINESE_BERT_BATCH_SIZE", "32")),
    "threshold": float(os.environ.get("COMPARISON_THRESHOLD", "50")),
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
    "normal_finance_hard_negative": 0,
    "semantic_fraud": 1,
    "mixed_risk": 1,
    "finance_semantic_contrast_fraud": 1,
    "fraud": 1,
    "phishing": 1,
    "adv_phish": 1,
}
CHECKPOINT_CANDIDATES = (
    "/kaggle/input/notebooks/samueltmsun/fork-of-chinese-bert/model/best_model.pt",
    "/kaggle/input/notebooks/samueltmsun/fork-of-chinese-bert/model/train.pt",
    "/kaggle/input/chinese-bert-model/best_model.pt",
    "/kaggle/input/chinese-bert-model/train.pt",
    "/kaggle/input/chinese-bert-baseline-model/best_model.pt",
    "/kaggle/input/chinese-bert-baseline-model/train.pt",
    "/kaggle/working/model/best_model.pt",
    "/kaggle/working/model/train.pt",
    "ChineseBERTModel/model/train.pt",
    "ChineseBERTModel/model/best_model.pt",
)

MODEL_DISPLAY_NAMES = {
    "surface_score": "Surface Baseline",
    "bert_score": "Proposed Chinese-BERT",
}


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


def parse_binary_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "fraud", "phishing", "scam", "risk", "risky"}:
        return 1
    if text in {"0", "0.0", "false", "normal", "benign", "safe"}:
        return 0
    return int(float(value))


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
        "Comparison dataset not found. Set COMPARISON_DATA_PATH to a CSV/JSON/JSONL file "
        "or to a Kaggle input directory containing one."
    )


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
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


def load_comparison_dataset(path_value: str) -> Tuple[pd.DataFrame, Path]:
    path = find_first_existing_file(path_value)
    print(f"Loading comparison dataset: {path}")
    if path.suffix.lower() == ".csv":
        raw_df = pd.read_csv(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        raw_df = pd.DataFrame(load_json_records(path))
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if raw_df.empty:
        raise ValueError(f"No rows loaded from {path}")

    text_col = next((col for col in TEXT_COLUMNS if col in raw_df.columns), None)
    if text_col is None:
        raise ValueError(f"No text column found. Expected one of: {TEXT_COLUMNS}")

    out = pd.DataFrame()
    out["text"] = raw_df[text_col].fillna("").astype(str).str.strip()
    out["case_type"] = raw_df.get("case_type", "unknown")
    out["case_type"] = out["case_type"].fillna("unknown").astype(str).str.strip()
    out["sample_id"] = raw_df.get("sample_id", [f"sample_{idx:05d}" for idx in range(len(raw_df))])
    out["sample_id"] = out["sample_id"].fillna("").astype(str)
    out.loc[out["sample_id"].str.len() == 0, "sample_id"] = [
        f"sample_{idx:05d}" for idx in out.index[out["sample_id"].str.len() == 0]
    ]

    text_label_col = next((col for col in TEXT_LABEL_COLUMNS if col in raw_df.columns), None)
    global_label_col = next((col for col in GLOBAL_LABEL_COLUMNS if col in raw_df.columns), None)
    normalized_case = out["case_type"].str.lower()
    if text_label_col:
        out["text_label"] = raw_df[text_label_col].apply(parse_binary_label)
        out["label_source"] = text_label_col
    elif normalized_case.isin(CASE_TYPE_TEXT_LABEL).all():
        out["text_label"] = normalized_case.map(CASE_TYPE_TEXT_LABEL).astype(int)
        out["label_source"] = "case_type_mapping"
    elif global_label_col:
        out["text_label"] = raw_df[global_label_col].apply(parse_binary_label)
        out["label_source"] = global_label_col
        print("Warning: using global label as text_label. Prefer explicit text_label for multimodal sets.")
    else:
        missing = sorted(out.loc[~normalized_case.isin(CASE_TYPE_TEXT_LABEL), "case_type"].unique())
        raise ValueError(f"Rows are missing text_label and case_type cannot be mapped: {missing}")

    for col in ("source", "audio_path", "input_view", "notes"):
        if col in raw_df.columns:
            out[col] = raw_df[col]

    out = out[out["text"].str.len() > 0].copy()
    out["norm"] = out["text"].apply(normalize_text_for_dedup)
    before = len(out)
    out = out.drop_duplicates(subset=["norm"], keep="first").drop(columns=["norm"]).reset_index(drop=True)
    print(f"Rows loaded: {len(out)} (deduplicated {before - len(out)})")
    print("Case distribution:")
    print(out["case_type"].value_counts().to_string())
    print("Text-label distribution:")
    print(out["text_label"].value_counts().sort_index().to_string())
    return out, path


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
        return self.classifier(self.dropout(pooled_output))


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
    raise FileNotFoundError("No Chinese-BERT checkpoint found. Set CHINESE_BERT_CHECKPOINT to best_model.pt or train.pt.")


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


class SurfaceFeatureBaseline:
    """A shallow, keyword-driven baseline for thesis comparison."""

    DIRECTIVE = (
        ("验证码", 5.0),
        ("短信验证码", 5.0),
        ("动态码", 5.0),
        ("支付密码", 5.0),
        ("转账", 4.5),
        ("汇款", 4.0),
        ("打款", 4.0),
        ("扫码", 4.0),
        ("二维码", 4.0),
        ("链接", 3.5),
        ("下载", 3.5),
        ("共享屏幕", 5.0),
        ("远程", 4.0),
        ("安全账户", 5.0),
    )
    PRESSURE = (
        ("马上", 2.5),
        ("立即", 2.5),
        ("现在", 1.5),
        ("必须", 2.5),
        ("不要挂断", 3.0),
        ("不要告诉", 3.0),
        ("保密", 3.0),
        ("否则", 2.5),
        ("十分钟", 2.0),
        ("倒计时", 2.5),
    )
    FINANCE = (
        ("银行", 1.8),
        ("银行卡", 2.0),
        ("账户", 1.8),
        ("账号", 1.5),
        ("支付宝", 1.6),
        ("微信支付", 1.6),
        ("花呗", 1.8),
        ("借呗", 1.8),
        ("贷款", 1.8),
        ("额度", 1.6),
        ("征信", 1.8),
        ("退款", 1.8),
        ("理赔", 1.8),
        ("保证金", 2.5),
        ("手续费", 2.0),
        ("合同", 1.4),
        ("金额", 1.4),
        ("彩礼", 1.2),
    )
    AUTHORITY = (
        ("公安", 2.5),
        ("警察", 2.5),
        ("检察院", 2.5),
        ("法院", 2.5),
        ("客服", 1.5),
        ("专员", 1.5),
        ("安全中心", 2.0),
        ("审核", 1.5),
        ("风控", 2.0),
    )
    NEGATION = (
        ("不会索要", 2.5),
        ("不要相信", 2.0),
        ("不要把验证码", 2.5),
        ("不需要验证码", 2.0),
        ("官方渠道", 1.5),
        ("自己登录", 1.5),
        ("不转账", 2.0),
    )

    def _weighted_hits(self, text: str, patterns: Sequence[Tuple[str, float]]) -> Tuple[float, int]:
        score = 0.0
        count = 0
        for term, weight in patterns:
            hits = text.count(term.lower())
            if hits:
                score += min(hits, 3) * weight
                count += hits
        return score, count

    def raw_score(self, text: str) -> Dict[str, float]:
        normalized = re.sub(r"\s+", "", str(text or "").lower())
        directive_score, directive_hits = self._weighted_hits(normalized, self.DIRECTIVE)
        pressure_score, pressure_hits = self._weighted_hits(normalized, self.PRESSURE)
        finance_score, finance_hits = self._weighted_hits(normalized, self.FINANCE)
        authority_score, authority_hits = self._weighted_hits(normalized, self.AUTHORITY)
        negation_score, negation_hits = self._weighted_hits(normalized, self.NEGATION)

        coupling = 0.0
        if directive_hits and finance_hits:
            coupling += 5.0
        if pressure_hits and directive_hits:
            coupling += 4.0
        if authority_hits and directive_hits:
            coupling += 3.0
        if pressure_hits and finance_hits:
            coupling += 2.0

        length_bonus = min(len(normalized) / 500.0, 1.0)
        raw = directive_score + pressure_score + finance_score + authority_score + coupling + length_bonus - negation_score
        return {
            "raw_surface_score": raw,
            "directive_hits": directive_hits,
            "pressure_hits": pressure_hits,
            "finance_hits": finance_hits,
            "authority_hits": authority_hits,
            "negation_hits": negation_hits,
        }

    def score_many(self, texts: Sequence[str]) -> Tuple[np.ndarray, pd.DataFrame]:
        rows = [self.raw_score(text) for text in texts]
        feature_df = pd.DataFrame(rows)
        raw = feature_df["raw_surface_score"].astype(float).to_numpy()
        # Fixed monotonic mapping: deliberately not learned from test data.
        scores = 100.0 / (1.0 + np.exp(-(raw - 9.0) / 4.0))
        return scores.astype(float), feature_df


def binary_metrics(y_true: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, Any]:
    y_true_arr = np.array(y_true, dtype=int)
    score_arr = np.array(scores, dtype=float)
    y_pred = (score_arr >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
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


def build_metrics_table(df: pd.DataFrame, score_fields: Sequence[str], threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for score_field in score_fields:
        model_name = MODEL_DISPLAY_NAMES.get(score_field, score_field)
        row = binary_metrics(df["text_label"], df[score_field], threshold)
        row.update({"scope": "overall", "case_type": "all", "score_field": score_field, "model": model_name})
        rows.append(row)
        for case_type, group in df.groupby("case_type", dropna=False):
            case_row = binary_metrics(group["text_label"], group[score_field], threshold)
            case_row.update({"scope": "case_type", "case_type": case_type, "score_field": score_field, "model": model_name})
            rows.append(case_row)
    return pd.DataFrame(rows)


def build_thesis_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    overall = metrics_df[metrics_df["scope"].eq("overall")].copy()
    rows = []
    for _, row in overall.iterrows():
        rows.append(
            {
                "Model": row["model"],
                "Accuracy": row["accuracy"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1"],
                "ROC-AUC": row["roc_auc"],
                "FP": row["fp"],
                "FN": row["fn"],
                "Normal FP Rate": row["normal_fp_rate"],
                "Fraud Recall": row["fraud_recall"],
            }
        )
    return pd.DataFrame(rows)


def build_case_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    case_df = metrics_df[metrics_df["scope"].eq("case_type")].copy()
    return case_df[
        [
            "model",
            "case_type",
            "samples",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "tn",
            "fp",
            "fn",
            "tp",
            "normal_fp_rate",
            "fraud_recall",
            "mean_score",
        ]
    ].sort_values(["case_type", "model"])


def plot_metric_bars(thesis_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = thesis_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1", "Fraud Recall"], var_name="Metric", value_name="Value")
    plt.figure(figsize=(9, 5))
    if HAS_SEABORN:
        sns.barplot(data=plot_df, x="Metric", y="Value", hue="Model", palette=["#9aa0a6", "#1f77b4"])
    else:
        pivot = plot_df.pivot(index="Metric", columns="Model", values="Value")
        pivot.plot(kind="bar", color=["#9aa0a6", "#1f77b4"], ax=plt.gca())
    plt.ylim(0, 1.05)
    plt.title("Overall Model Comparison")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_overall_metric_bars.png", dpi=220)
    plt.savefig(output_dir / "figure_overall_metric_bars.pdf")
    plt.close()


def plot_confusion_matrices(df: pd.DataFrame, score_fields: Sequence[str], threshold: float, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(score_fields), figsize=(5 * len(score_fields), 4))
    if len(score_fields) == 1:
        axes = [axes]
    for ax, score_field in zip(axes, score_fields):
        pred = (df[score_field].astype(float) >= threshold).astype(int)
        cm = confusion_matrix(df["text_label"].astype(int), pred, labels=[0, 1])
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
        else:
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1], labels=["Normal", "Fraud"])
            ax.set_yticks([0, 1], labels=["Normal", "Fraud"])
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="black")
        ax.set_title(MODEL_DISPLAY_NAMES.get(score_field, score_field))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "figure_confusion_matrices.png", dpi=220)
    plt.savefig(output_dir / "figure_confusion_matrices.pdf")
    plt.close()


def plot_case_performance(case_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = case_df.copy()
    plot_df["Normal Specificity"] = 1.0 - plot_df["normal_fp_rate"].astype(float)
    normal_df = plot_df[plot_df["normal_samples"].fillna(0).astype(int) > 0] if "normal_samples" in plot_df.columns else pd.DataFrame()
    if normal_df.empty:
        normal_df = plot_df[plot_df["case_type"].str.contains("normal", case=False, na=False)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if HAS_SEABORN:
        sns.barplot(data=plot_df, x="case_type", y="fraud_recall", hue="model", ax=axes[0], palette=["#9aa0a6", "#1f77b4"])
    else:
        plot_df.pivot(index="case_type", columns="model", values="fraud_recall").plot(kind="bar", color=["#9aa0a6", "#1f77b4"], ax=axes[0])
    axes[0].set_title("Fraud Recall by Case")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Recall")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylim(0, 1.05)

    if HAS_SEABORN:
        sns.barplot(data=normal_df, x="case_type", y="normal_fp_rate", hue="model", ax=axes[1], palette=["#9aa0a6", "#1f77b4"])
    elif not normal_df.empty:
        normal_df.pivot(index="case_type", columns="model", values="normal_fp_rate").plot(kind="bar", color=["#9aa0a6", "#1f77b4"], ax=axes[1])
    axes[1].set_title("False Positive Rate on Normal Cases")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("FP Rate")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(0, max(0.2, float(normal_df["normal_fp_rate"].max()) + 0.08) if not normal_df.empty else 1)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_case_performance.png", dpi=220)
    plt.savefig(output_dir / "figure_case_performance.pdf")
    plt.close()


def plot_score_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = pd.concat(
        [
            pd.DataFrame({"case_type": df["case_type"], "score": df["surface_score"], "Model": "Surface Baseline"}),
            pd.DataFrame({"case_type": df["case_type"], "score": df["bert_score"], "Model": "Proposed Chinese-BERT"}),
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(12, 5))
    if HAS_SEABORN:
        sns.boxplot(data=plot_df, x="case_type", y="score", hue="Model", palette=["#9aa0a6", "#1f77b4"], showfliers=True)
    else:
        ax = plt.gca()
        case_types = list(df["case_type"].drop_duplicates())
        positions = []
        labels = []
        data = []
        for idx, case_type in enumerate(case_types):
            subset = plot_df[plot_df["case_type"].eq(case_type)]
            for offset, model_name in [(-0.18, "Surface Baseline"), (0.18, "Proposed Chinese-BERT")]:
                positions.append(idx + 1 + offset)
                labels.append(model_name)
                data.append(subset[subset["Model"].eq(model_name)]["score"].astype(float).tolist())
        bp = ax.boxplot(data, positions=positions, widths=0.28, patch_artist=True, showfliers=True)
        colors = ["#9aa0a6", "#1f77b4"] * len(case_types)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(case_types) + 1), labels=case_types)
    plt.axhline(float(CONFIG["threshold"]), color="#d62728", linestyle="--", linewidth=1, label="Decision threshold")
    plt.title("Risk Score Distribution by Case")
    plt.xlabel("")
    plt.ylabel("Risk Score")
    plt.xticks(rotation=25)
    plt.ylim(-3, 103)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_score_distribution_by_case.png", dpi=220)
    plt.savefig(output_dir / "figure_score_distribution_by_case.pdf")
    plt.close()


def plot_curves(df: pd.DataFrame, score_fields: Sequence[str], output_dir: Path) -> None:
    y_true = df["text_label"].astype(int).to_numpy()
    if len(set(y_true.tolist())) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for score_field in score_fields:
        scores = df[score_field].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        axes[0].plot(fpr, tpr, label=f"{MODEL_DISPLAY_NAMES[score_field]} (AUC={auc(fpr, tpr):.3f})")
        axes[1].plot(recall, precision, label=f"{MODEL_DISPLAY_NAMES[score_field]} (AUC={auc(recall, precision):.3f})")
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_dir / "figure_roc_pr_curves.png", dpi=220)
    plt.savefig(output_dir / "figure_roc_pr_curves.pdf")
    plt.close()


def write_outputs(df: pd.DataFrame, output_dir: Path, checkpoint: Path, data_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(CONFIG["threshold"])
    score_fields = ["surface_score", "bert_score"]

    for score_field in score_fields:
        pred_col = f"{score_field.replace('_score', '')}_prediction"
        fp_col = f"{score_field.replace('_score', '')}_false_positive"
        fn_col = f"{score_field.replace('_score', '')}_false_negative"
        df[pred_col] = (df[score_field].astype(float) >= threshold).astype(int)
        df[fp_col] = ((df["text_label"] == 0) & (df[pred_col] == 1)).astype(int)
        df[fn_col] = ((df["text_label"] == 1) & (df[pred_col] == 0)).astype(int)

    metrics_df = build_metrics_table(df, score_fields=score_fields, threshold=threshold)
    thesis_df = build_thesis_summary(metrics_df)
    case_df = build_case_summary(metrics_df)
    error_df = df[
        [
            "sample_id",
            "case_type",
            "text_label",
            "surface_score",
            "bert_score",
            "surface_prediction",
            "bert_prediction",
            "surface_false_positive",
            "bert_false_positive",
            "surface_false_negative",
            "bert_false_negative",
            "text",
        ]
        + [col for col in ("source", "notes") if col in df.columns]
    ].copy()
    error_df = error_df[(error_df["surface_false_positive"] == 1) | (error_df["surface_false_negative"] == 1) | (error_df["bert_false_positive"] == 1) | (error_df["bert_false_negative"] == 1)]

    df.to_csv(output_dir / "comparison_predictions.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(output_dir / "comparison_metrics_long.csv", index=False, encoding="utf-8-sig")
    thesis_df.to_csv(output_dir / "table_overall_model_comparison.csv", index=False, encoding="utf-8-sig")
    case_df.to_csv(output_dir / "table_case_type_comparison.csv", index=False, encoding="utf-8-sig")
    error_df.to_csv(output_dir / "table_error_examples.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "dataset": data_path.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "threshold": threshold,
        "model_name": CONFIG["model_name"],
        "max_len": CONFIG["max_len"],
        "note": "Surface Baseline is a shallow keyword/surface-feature model; Proposed Chinese-BERT is the trained checkpoint.",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", font_scale=1.0)
    else:
        plt.style.use("default")
    plot_metric_bars(thesis_df, output_dir)
    plot_confusion_matrices(df, score_fields, threshold, output_dir)
    plot_case_performance(case_df, output_dir)
    plot_score_distributions(df, output_dir)
    plot_curves(df, score_fields, output_dir)

    print(f"Wrote outputs to: {output_dir}")
    print("\nOverall thesis table:")
    print(thesis_df.to_string(index=False))
    print("\nCase comparison:")
    print(case_df.to_string(index=False))


def main() -> None:
    set_seed(int(CONFIG["seed"]))
    output_dir = Path(CONFIG["output_dir"])
    df, data_path = load_comparison_dataset(str(CONFIG["data_path"]))
    model, tokenizer, device, checkpoint = load_model_and_tokenizer(CONFIG)

    surface = SurfaceFeatureBaseline()
    surface_scores, feature_df = surface.score_many(df["text"].tolist())
    df["surface_score"] = surface_scores
    for col in feature_df.columns:
        df[f"surface_{col}"] = feature_df[col]

    df["bert_score"] = predict_scores(
        df["text"].tolist(),
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_len=int(CONFIG["max_len"]),
        batch_size=int(CONFIG["batch_size"]),
    )
    df["checkpoint"] = checkpoint.as_posix()
    df["max_len"] = int(CONFIG["max_len"])
    write_outputs(df, output_dir=output_dir, checkpoint=checkpoint, data_path=data_path)


if __name__ == "__main__":
    main()
