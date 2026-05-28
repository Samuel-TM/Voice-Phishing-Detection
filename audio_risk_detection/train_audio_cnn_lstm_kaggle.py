# %%
# =============================================================================
# 🎙️ 音频深度伪造检测模型训练 - CNN-BiLSTM + ASVspoof 2019 LA
# =============================================================================
# 完全复用原项目 audio_risk_detection/train_audio_risk.py 的模型架构
# 核心改动:
#   1. 数据加载: 原项目 real_audio_dir/deepfake_audio_dir 两个平级目录
#      → 改为解析 ASVspoof 2019 LA 的 protocol 文件 + flac 目录
#   2. 模型/特征/训练逻辑: 100% 原样复用, 零改动
# =============================================================================

# ===================== Cell 1: 环境安装 =====================
# !pip install librosa soundfile scikit-learn tqdm matplotlib seaborn -q

# %%
# ===================== Cell 2: 导入库 =====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import librosa
import numpy as np
import os
import csv
import json
import random
import warnings
import logging
import hashlib
import gc
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== Cell 3: 全局配置 =====================
# ──────────── 你需要修改的唯一部分: Kaggle 数据路径 ────────────
# ASVspoof 2019 数据集在 Kaggle 上的根目录
# 在 Kaggle 中添加数据集后, 确认路径是否正确
ASVSPOOF_ROOT = "/kaggle/input/asvpoof-2019-dataset"  # ← 根据你的Kaggle数据集名称调整
LA_ROOT = "/kaggle/input/asvpoof-2019-dataset/LA"
CONFIG = {
    # ────── ASVspoof 2019 LA 数据路径 ──────
    "asvspoof_root": ASVSPOOF_ROOT,
    "la_root": os.path.join(ASVSPOOF_ROOT, "LA"),

    # Protocol 文件 (标注真伪标签)
    "train_protocol": os.path.join(LA_ROOT, "LA", "ASVspoof2019_LA_cm_protocols",
                                   "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev_protocol": os.path.join(LA_ROOT, "LA", "ASVspoof2019_LA_cm_protocols",
                                 "ASVspoof2019.LA.cm.dev.trl.txt"),

    # 音频文件目录
    "train_flac_dir": os.path.join(LA_ROOT, "LA", "ASVspoof2019_LA_train", "flac"),
    "dev_flac_dir": os.path.join(LA_ROOT, "LA", "ASVspoof2019_LA_dev", "flac"),

    # ────── 特征参数 (与原项目 audio_risk_config.json 一致) ──────
    "feature_params": {
        "n_mfcc": 13,
        "n_fft": 2048,
        "hop_length": 512,
        "max_length": 500,   # 约 16 秒 @ 16kHz, hop=512
    },

    # ────── 模型参数 (与原项目完全一致) ──────
    "model_params": {
        "num_classes": 2,        # 0=bonafide, 1=spoof
        "conv_in_channels": 13,  # 必须 == n_mfcc
        "conv_out_channels": 32,
        "lstm_hidden_size": 64,
        "fc1_out_features": 128,
    },

    # ────── 训练参数 (基于原项目, 适当调优) ──────
    "training_params": {
        "num_epochs": 6,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_workers": 4,
        "validation_split": 0.0,
        "random_seed": 42,
        "lr_scheduler_patience": 2,
        "lr_scheduler_factor": 0.5,
        "early_stopping_patience": 3,
    },

    # ────── 输出路径 ──────
    "output_paths": {
        "log_csv_path": "/kaggle/working/train_deepfake_metrics_log.csv",
        "model_save_path": "/kaggle/working/model/last_epoch_model.pt",
        "best_model_save_path": "/kaggle/working/model/best_f1_model.pt",
    },

    # ────── In-the-Wild Audio Deepfake Dataset ──────
    # This target-domain dataset is split into train / validation / OOD test.
    # The OOD test split is never used for training, model selection, or threshold calibration.
    "in_the_wild": {
        "enabled": True,
        "real_dir": "/kaggle/input/datasets/abdallamohamed312/in-the-wild-audio-deepfake/release_in_the_wild/real",
        "fake_dir": "/kaggle/input/datasets/abdallamohamed312/in-the-wild-audio-deepfake/release_in_the_wild/fake",
        "train_ratio": 0.70,
        "val_ratio": 0.10,
        "ood_test_ratio": 0.20,
        # Caps keep Kaggle iteration time predictable while preserving a held-out OOD test.
        # Set any of these to None if you want to use the full split.
        "max_train_per_class": 3000,
        "max_val_per_class": 500,
        "max_ood_test_per_class": 1000,
    },

    "seed": 42,
}

def set_seed(seed_value=42):
    """与原项目 set_seed() 完全一致"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(os.path.dirname(CONFIG["output_paths"]["best_model_save_path"]), exist_ok=True)

# %%
# # ===================== Cell 4: 数据集探索 =====================
# print("=" * 70)
# print("📂 Step 1: 探索 ASVspoof 2019 LA 数据集结构")
# print("=" * 70)

# la_root = Path(CONFIG["la_root"])
# print(f"\nLA 根目录: {la_root}")
# print(f"目录是否存在: {la_root.exists()}")

# if la_root.exists():
#     print(f"\nLA 目录结构:")
#     for item in sorted(la_root.iterdir()):
#         if item.is_dir():
#             # 计算目录下文件数
#             file_count = sum(1 for f in item.rglob("*") if f.is_file())
#             print(f"  📁 {item.name}/ ({file_count} files)")
#         else:
#             size_kb = item.stat().st_size / 1024
#             print(f"  📄 {item.name} ({size_kb:.1f} KB)")

# # 检查关键文件
# for key, path in [("train_protocol", CONFIG["train_protocol"]),
#                    ("dev_protocol", CONFIG["dev_protocol"]),
#                    ("train_flac_dir", CONFIG["train_flac_dir"]),
#                    ("dev_flac_dir", CONFIG["dev_flac_dir"])]:
#     exists = os.path.exists(path)
#     print(f"\n  {'✅' if exists else '❌'} {key}: {path}")
#     if exists and os.path.isdir(path):
#         files = os.listdir(path)
#         print(f"     文件数: {len(files)}, 示例: {files[:3]}")

# %%
# ===================== Cell 5: ASVspoof 2019 Protocol 解析器 =====================
print("\n" + "=" * 70)
print("📊 Step 2: 解析 ASVspoof 2019 LA Protocol 文件")
print("=" * 70)

class ASVspoofDataLoader:
    """
    ASVspoof 2019 LA 数据集解析器

    适配原项目 train_audio_risk.py 的数据加载逻辑:
    原项目: real_audio_dir → label=0, deepfake_audio_dir → label=1
    本解析器: protocol 文件中 bonafide → label=0, spoof → label=1

    Protocol 文件格式 (每行, 空格分隔):
    LA_0079 LA_T_1138215 - A01 spoof
    LA_0079 LA_T_1138220 - - bonafide

    列含义:
    [0] SPEAKER_ID   - 说话人ID
    [1] AUDIO_NAME   - 音频文件名 (不含扩展名)
    [2] -            - 占位符
    [3] ATTACK_ID    - 攻击算法 (A01-A19) 或 "-" (bonafide)
    [4] LABEL        - "bonafide" 或 "spoof"
    """

    def __init__(self, config: dict):
        self.config = config

    def parse_protocol(self, protocol_path: str, flac_dir: str) -> Tuple[List[str], List[int], List[dict]]:
        """
        解析单个 protocol 文件

        Returns:
            file_paths: 音频文件完整路径列表
            labels: 标签列表 (0=bonafide, 1=spoof)
            metadata: 元数据列表 (speaker_id, attack_id 等)
        """
        file_paths = []
        labels = []
        metadata = []

        skipped = 0

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    skipped += 1
                    continue

                speaker_id = parts[0]
                audio_name = parts[1]
                attack_id = parts[3]
                label_str = parts[4]

                # 构建完整音频路径
                audio_path = os.path.join(flac_dir, f"{audio_name}.flac")

                # 检查文件是否存在
                if not os.path.exists(audio_path):
                    skipped += 1
                    continue

                # 标签: bonafide=0 (real), spoof=1 (deepfake)
                # 与原项目 train_audio_risk.py 一致: 0=real, 1=deepfake
                if label_str == "bonafide":
                    label = 0
                elif label_str == "spoof":
                    label = 1
                else:
                    skipped += 1
                    continue

                file_paths.append(audio_path)
                labels.append(label)
                metadata.append({
                    "speaker_id": speaker_id,
                    "audio_name": audio_name,
                    "attack_id": attack_id,
                    "label_str": label_str,
                })

        if skipped > 0:
            logger.info(f"  跳过 {skipped} 条记录 (文件不存在或格式异常)")

        return file_paths, labels, metadata

    def load_train_dev(self) -> Dict:
        """
        加载训练集和开发集

        Returns:
            dict: 包含 train_files, train_labels, dev_files, dev_labels 等
        """
        result = {}

        # ── 训练集 ──
        print(f"\n  解析训练集 protocol: {self.config['train_protocol']}")
        train_files, train_labels, train_meta = self.parse_protocol(
            self.config["train_protocol"], self.config["train_flac_dir"]
        )
        result["train_files"] = train_files
        result["train_labels"] = train_labels
        result["train_meta"] = train_meta

        train_counts = Counter(train_labels)
        print(f"  ✅ 训练集: {len(train_files)} 样本 "
              f"(bonafide={train_counts.get(0,0)}, spoof={train_counts.get(1,0)})")

        # ── 开发/验证集 ──
        print(f"\n  解析开发集 protocol: {self.config['dev_protocol']}")
        dev_files, dev_labels, dev_meta = self.parse_protocol(
            self.config["dev_protocol"], self.config["dev_flac_dir"]
        )
        result["dev_files"] = dev_files
        result["dev_labels"] = dev_labels
        result["dev_meta"] = dev_meta

        dev_counts = Counter(dev_labels)
        print(f"  ✅ 开发集: {len(dev_files)} 样本 "
              f"(bonafide={dev_counts.get(0,0)}, spoof={dev_counts.get(1,0)})")

        # ── 攻击类型分布 ──
        print(f"\n  ── 训练集攻击类型分布 ──")
        train_attacks = [m["attack_id"] for m, l in zip(train_meta, train_labels) if l == 1]
        for atk, cnt in sorted(Counter(train_attacks).items()):
            print(f"    {atk}: {cnt} 样本")

        return result

# 执行加载
asvspoof_loader = ASVspoofDataLoader(CONFIG)
data = asvspoof_loader.load_train_dev()
data["asvspoof_train_files"] = list(data["train_files"])
data["asvspoof_train_labels"] = list(data["train_labels"])
data["asvspoof_dev_files"] = list(data["dev_files"])
data["asvspoof_dev_labels"] = list(data["dev_labels"])


def collect_audio_files(audio_dir: str) -> List[str]:
    audio_extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    root = Path(audio_dir)
    if not root.exists():
        logger.warning(f"In-the-Wild audio directory not found: {audio_dir}")
        return []
    return sorted(
        path.as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in audio_extensions
    )


def limit_per_class(files: List[str], limit: Optional[int], seed: int) -> List[str]:
    if limit is None or limit <= 0 or len(files) <= limit:
        return files
    rng = random.Random(seed)
    sampled = list(files)
    rng.shuffle(sampled)
    return sorted(sampled[:limit])


def split_class_files(
    files: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if not files:
        return [], [], []
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    ratios = ratios / ratios.sum()
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_test = max(1, int(round(n_total * ratios[2]))) if n_total >= 3 else 0
    n_val = max(1, int(round(n_total * ratios[1]))) if n_total >= 4 else 0
    if n_val + n_test >= n_total:
        n_val = 1 if n_total >= 3 else 0
        n_test = 1 if n_total >= 3 else 0
    test_files = shuffled[:n_test]
    val_files = shuffled[n_test:n_test + n_val]
    train_files = shuffled[n_test + n_val:]
    return sorted(train_files), sorted(val_files), sorted(test_files)


def load_in_the_wild_splits(config: dict, seed: int) -> Dict[str, List]:
    itw_config = config.get("in_the_wild", {})
    empty = {
        "train_files": [], "train_labels": [],
        "val_files": [], "val_labels": [],
        "ood_test_files": [], "ood_test_labels": [],
    }
    if not itw_config.get("enabled", False):
        return empty

    real_files = collect_audio_files(itw_config["real_dir"])
    fake_files = collect_audio_files(itw_config["fake_dir"])
    if not real_files or not fake_files:
        logger.warning("In-the-Wild dataset is incomplete; continuing with ASVspoof only.")
        return empty

    ratios = (
        float(itw_config.get("train_ratio", 0.70)),
        float(itw_config.get("val_ratio", 0.10)),
        float(itw_config.get("ood_test_ratio", 0.20)),
    )
    real_train, real_val, real_test = split_class_files(real_files, *ratios, seed=seed)
    fake_train, fake_val, fake_test = split_class_files(fake_files, *ratios, seed=seed + 1)

    real_train = limit_per_class(real_train, itw_config.get("max_train_per_class"), seed)
    fake_train = limit_per_class(fake_train, itw_config.get("max_train_per_class"), seed + 1)
    real_val = limit_per_class(real_val, itw_config.get("max_val_per_class"), seed + 2)
    fake_val = limit_per_class(fake_val, itw_config.get("max_val_per_class"), seed + 3)
    real_test = limit_per_class(real_test, itw_config.get("max_ood_test_per_class"), seed + 4)
    fake_test = limit_per_class(fake_test, itw_config.get("max_ood_test_per_class"), seed + 5)

    def pack(real_part: List[str], fake_part: List[str]) -> Tuple[List[str], List[int]]:
        files = real_part + fake_part
        labels = [0] * len(real_part) + [1] * len(fake_part)
        combined = list(zip(files, labels))
        random.Random(seed).shuffle(combined)
        if not combined:
            return [], []
        files, labels = zip(*combined)
        return list(files), list(labels)

    train_files, train_labels = pack(real_train, fake_train)
    val_files, val_labels = pack(real_val, fake_val)
    test_files, test_labels = pack(real_test, fake_test)

    print("\n  ✅ In-the-Wild dataset split:")
    print(f"     train: {len(train_files)} samples "
          f"(real={train_labels.count(0)}, fake={train_labels.count(1)})")
    print(f"     val:   {len(val_files)} samples "
          f"(real={val_labels.count(0)}, fake={val_labels.count(1)})")
    print(f"     OOD test (held out): {len(test_files)} samples "
          f"(real={test_labels.count(0)}, fake={test_labels.count(1)})")

    return {
        "train_files": train_files, "train_labels": train_labels,
        "val_files": val_files, "val_labels": val_labels,
        "ood_test_files": test_files, "ood_test_labels": test_labels,
    }


in_the_wild_splits = load_in_the_wild_splits(CONFIG, CONFIG["seed"])
data["in_the_wild_train_files"] = in_the_wild_splits["train_files"]
data["in_the_wild_train_labels"] = in_the_wild_splits["train_labels"]
data["in_the_wild_val_files"] = in_the_wild_splits["val_files"]
data["in_the_wild_val_labels"] = in_the_wild_splits["val_labels"]
data["in_the_wild_ood_test_files"] = in_the_wild_splits["ood_test_files"]
data["in_the_wild_ood_test_labels"] = in_the_wild_splits["ood_test_labels"]

# ASVspoof remains the main source domain. In-the-Wild train/val are used for
# target-domain adaptation and model selection; OOD test remains untouched.
data["train_files"] = data["asvspoof_train_files"] + data["in_the_wild_train_files"]
data["train_labels"] = data["asvspoof_train_labels"] + data["in_the_wild_train_labels"]
data["dev_files"] = data["asvspoof_dev_files"] + data["in_the_wild_val_files"]
data["dev_labels"] = data["asvspoof_dev_labels"] + data["in_the_wild_val_labels"]

# %%
# ===================== Cell 6: 数据可视化 =====================
print("\n" + "=" * 70)
print("📈 Step 3: 数据统计与可视化")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 训练集标签分布
train_counts = Counter(data["train_labels"])
ax1 = axes[0]
bars = ax1.bar(["Bonafide (0)", "Spoof (1)"],
               [train_counts.get(0, 0), train_counts.get(1, 0)],
               color=["#2ecc71", "#e74c3c"])
ax1.set_title("Training set label distribution", fontsize=14)
ax1.set_ylabel("Sample size")
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=11)

# 开发集标签分布
dev_counts = Counter(data["dev_labels"])
ax2 = axes[1]
bars = ax2.bar(["Bonafide (0)", "Spoof (1)"],
               [dev_counts.get(0, 0), dev_counts.get(1, 0)],
               color=["#2ecc71", "#e74c3c"], alpha=0.8)
ax2.set_title("Development set (validation set) label distribution", fontsize=14)
ax2.set_ylabel("Sample size")
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=11)

# 训练集攻击算法分布
ax3 = axes[2]
train_attacks = [m["attack_id"] for m, l in zip(data["train_meta"], data["train_labels"]) if l == 1]
atk_counts = Counter(train_attacks)
atk_sorted = sorted(atk_counts.items())
if atk_sorted:
    atk_names, atk_vals = zip(*atk_sorted)
    ax3.bar(atk_names, atk_vals, color="#9b59b6", alpha=0.7)
    ax3.set_title("Distribution of Spoof Attack Algorithms in the Training Set", fontsize=14)
    ax3.set_xlabel("Attack algorithm")
    ax3.set_ylabel("Sample size")
    ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("/kaggle/working/asvspoof_data_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# 音频时长抽样统计
print("\n🎵 音频时长抽样统计 (随机抽取 200 个文件):")
sample_files = random.sample(data["train_files"], min(200, len(data["train_files"])))
durations = []
for sf in sample_files:
    try:
        y, sr = librosa.load(sf, sr=None, duration=30)
        durations.append(len(y) / sr)
    except:
        pass
if durations:
    print(f"  采样率: {sr} Hz")
    print(f"  最短: {min(durations):.2f}s, 最长: {max(durations):.2f}s")
    print(f"  平均: {np.mean(durations):.2f}s, 中位数: {np.median(durations):.2f}s")

# %%
import os
import csv
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import librosa
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)

# ===================== 0. 固定随机种子 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===================== 1. 模型定义 (原 AudioCNNLSTM) =====================
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

        self.lstm = nn.LSTM(
            input_size=model_config['conv_out_channels'],
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


# ===================== 2. MFCC 提取 (原 extract_mfcc_features) =====================
def extract_mfcc_features(audio_path, feature_config):
    n_mfcc = feature_config['n_mfcc']
    n_fft = feature_config['n_fft']
    hop_length = feature_config['hop_length']
    max_length = feature_config['max_length']

    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(
            y=audio_data, sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        ).T

        if mfccs.shape[0] > max_length:
            mfccs = mfccs[:max_length, :]
        else:
            pad_width = max_length - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')

        return mfccs
    except:
        return None


# ===================== 3. Dataset（带缓存，大幅加速） =====================
class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, feature_config, cache_dir="/kaggle/working/mfcc_cache"):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_config = feature_config
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.file_paths)

    def _cache_path(self, audio_path):
        stem = Path(audio_path).stem
        path_hash = hashlib.md5(os.path.abspath(audio_path).encode("utf-8")).hexdigest()[:12]
        return os.path.join(self.cache_dir, f"{stem}_{path_hash}.npy")

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        cache_path = self._cache_path(audio_path)

        if os.path.exists(cache_path):
            try:
                mfccs = np.load(cache_path)
            except:
                mfccs = None
        else:
            mfccs = extract_mfcc_features(audio_path, self.feature_config)
            if mfccs is not None:
                np.save(cache_path, mfccs)

        if mfccs is None:
            return None

        return (
            torch.tensor(mfccs, dtype=torch.float32).transpose(0, 1),
            torch.tensor(label, dtype=torch.long)
        )


def collate_fn_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ===================== 4. 评估函数 =====================
def compute_eer(y_true, y_score):
    """
    Equal Error Rate (EER)
    y_true: 0/1 标签（0=Bonafide, 1=Spoof）
    y_score: 正类(1=Spoof)的概率/分数（越大越像 Spoof）
    return: eer(float), eer_threshold(float)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5

    # roc_curve 会返回 fpr, tpr, thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    # 找到 |FPR - FNR| 最小点作为 EER 近似
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_thr = float(thresholds[idx])
    return eer, eer_thr


def compute_binary_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0:
        return {
            "threshold": float(threshold),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "balanced_acc": 0.0,
            "real_precision": 0.0,
            "fake_precision": 0.0,
            "real_recall": 0.0,
            "fake_recall": 0.0,
            "real_f1": 0.0,
            "fake_f1": 0.0,
            "min_class_precision": 0.0,
            "min_class_recall": 0.0,
            "min_class_f1": 0.0,
            "recall_gap": 0.0,
            "precision_gap": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "all_preds": [],
        }
    y_pred = (y_score > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    real_recall = tn / max(1, tn + fp)
    fake_recall = tp / max(1, tp + fn)
    real_precision = tn / max(1, tn + fn)
    fake_precision = tp / max(1, tp + fp)
    real_f1 = 2.0 * real_precision * real_recall / max(1e-12, real_precision + real_recall)
    fake_f1 = 2.0 * fake_precision * fake_recall / max(1e-12, fake_precision + fake_recall)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(fake_precision),
        "recall": float(fake_recall),
        "f1": float(fake_f1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "real_precision": float(real_precision),
        "fake_precision": float(fake_precision),
        "real_recall": float(real_recall),
        "fake_recall": float(fake_recall),
        "real_f1": float(real_f1),
        "fake_f1": float(fake_f1),
        "min_class_precision": float(min(real_precision, fake_precision)),
        "min_class_recall": float(min(real_recall, fake_recall)),
        "min_class_f1": float(min(real_f1, fake_f1)),
        "recall_gap": float(abs(real_recall - fake_recall)),
        "precision_gap": float(abs(real_precision - fake_precision)),
        "confusion_matrix": cm.tolist(),
        "all_preds": y_pred.tolist(),
    }


def choose_operating_threshold(
    y_true,
    y_score,
    min_real_recall=0.85,
    min_fake_recall=0.85,
    max_recall_gap=0.10,
    top_k=5,
):
    """Pick a balanced deployment threshold instead of optimizing majority-class accuracy."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_score) == 0:
        return 0.5, compute_binary_metrics(y_true, y_score, threshold=0.5)
    if len(np.unique(y_true)) < 2:
        return 0.5, compute_binary_metrics(y_true, y_score, threshold=0.5)

    thresholds = np.unique(y_score)
    thresholds = np.unique(np.concatenate(([0.0, 1.0], thresholds)))
    sorted_idx = np.argsort(y_score)
    sorted_scores = y_score[sorted_idx]
    sorted_labels = y_true[sorted_idx]
    real_prefix = np.cumsum(sorted_labels == 0)
    fake_prefix = np.cumsum(sorted_labels == 1)

    right_idx = np.searchsorted(sorted_scores, thresholds, side="right") - 1
    valid = right_idx >= 0
    real_le = np.zeros_like(thresholds, dtype=float)
    fake_le = np.zeros_like(thresholds, dtype=float)
    real_le[valid] = real_prefix[right_idx[valid]]
    fake_le[valid] = fake_prefix[right_idx[valid]]

    real_total = float(np.sum(y_true == 0))
    fake_total = float(np.sum(y_true == 1))
    tn = real_le
    fp = real_total - real_le
    fn = fake_le
    tp = fake_total - fake_le

    real_recall = tn / max(1.0, real_total)
    fake_recall = tp / max(1.0, fake_total)
    balanced_acc = 0.5 * (real_recall + fake_recall)
    accuracy = (tp + tn) / max(1.0, real_total + fake_total)
    precision = tp / np.maximum(1.0, tp + fp)
    fake_f1 = (2.0 * precision * fake_recall) / np.maximum(1e-12, precision + fake_recall)
    real_precision = tn / np.maximum(1.0, tn + fn)
    real_f1 = (2.0 * real_precision * real_recall) / np.maximum(1e-12, real_precision + real_recall)
    macro_f1 = 0.5 * (real_f1 + fake_f1)
    min_class_f1 = np.minimum(real_f1, fake_f1)
    min_class_precision = np.minimum(real_precision, precision)
    min_class_recall = np.minimum(real_recall, fake_recall)
    recall_gap = np.abs(real_recall - fake_recall)

    eligible = (
        (real_recall >= min_real_recall)
        & (fake_recall >= min_fake_recall)
        & (recall_gap <= max_recall_gap)
    )
    constraints_satisfied = bool(np.any(eligible))
    if not np.any(eligible):
        eligible = np.ones_like(thresholds, dtype=bool)

    objective = (
        0.35 * macro_f1
        + 0.20 * min_class_f1
        + 0.15 * min_class_precision
        + 0.15 * min_class_recall
        + 0.10 * balanced_acc
        - 0.05 * recall_gap
    )
    objective[~eligible] = -np.inf
    best_idx = int(np.nanargmax(objective))
    threshold = float(thresholds[best_idx])

    candidate_order = np.argsort(objective)[::-1]
    candidates = []
    for idx in candidate_order[:top_k]:
        if not np.isfinite(objective[idx]):
            continue
        candidates.append({
            "threshold": float(thresholds[idx]),
            "objective": float(objective[idx]),
            "macro_f1": float(macro_f1[idx]),
            "min_class_f1": float(min_class_f1[idx]),
            "real_precision": float(real_precision[idx]),
            "fake_precision": float(precision[idx]),
            "real_recall": float(real_recall[idx]),
            "fake_recall": float(fake_recall[idx]),
            "recall_gap": float(recall_gap[idx]),
            "balanced_acc": float(balanced_acc[idx]),
            "constraints_satisfied": constraints_satisfied,
        })

    threshold_metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)
    threshold_metrics["threshold_constraints"] = {
        "min_real_recall": float(min_real_recall),
        "min_fake_recall": float(min_fake_recall),
        "max_recall_gap": float(max_recall_gap),
        "satisfied": constraints_satisfied,
    }
    threshold_metrics["threshold_candidates"] = candidates
    return threshold, threshold_metrics


def evaluate_model(
    model,
    dataloader,
    criterion,
    threshold=None,
    min_real_recall=0.85,
    min_fake_recall=0.85,
    max_recall_gap=0.10,
):
    model.eval()
    total_loss = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_data in dataloader:
            if batch_data is None:
                continue

            inputs, targets = batch_data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

    if len(all_targets) == 0:
        return {
            "loss": 0,
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "macro_f1": 0,
            "balanced_acc": 0,
            "real_recall": 0,
            "fake_recall": 0,
            "eer": 0,
            "eer_threshold": 0,
            "threshold": 0.5,
            "threshold_0_5": {},
            "confusion_matrix": [[0, 0], [0, 0]],
            "all_preds": [],
            "all_targets": [],
            "all_probs": []
        }

    avg_loss = total_loss / max(1, len(dataloader))

    eer, eer_thr = compute_eer(all_targets, all_probs)
    if threshold is None:
        threshold, threshold_metrics = choose_operating_threshold(
            all_targets,
            all_probs,
            min_real_recall=min_real_recall,
            min_fake_recall=min_fake_recall,
            max_recall_gap=max_recall_gap,
        )
    else:
        threshold_metrics = compute_binary_metrics(all_targets, all_probs, threshold=threshold)
    default_metrics = compute_binary_metrics(all_targets, all_probs, threshold=0.5)

    result = {
        "loss": avg_loss,
        "eer": eer,
        "eer_threshold": eer_thr,
        "threshold_0_5": default_metrics,
        "all_targets": all_targets,
        "all_probs": all_probs
    }
    result.update(threshold_metrics)
    return result


# ===================== 5. Collapse 检测（全预测0或全预测1都停） =====================
def detect_collapse(all_preds, threshold=0.98):
    if len(all_preds) == 0:
        return False, "No predictions"

    all_preds = np.array(all_preds)
    ratio_spoof = np.mean(all_preds == 1)
    ratio_bona = np.mean(all_preds == 0)

    msg = f"[Collapse Check] Spoof-ratio={ratio_spoof:.4f}, Bonafide-ratio={ratio_bona:.4f}"

    if ratio_spoof >= threshold:
        return True, msg + " => COLLAPSE to Spoof"
    if ratio_bona >= threshold:
        return True, msg + " => COLLAPSE to Bonafide"

    return False, msg


# ===================== 6. 平衡训练集（关键！借鉴 Keras notebook 的成功经验） =====================
def build_balanced_training_set(train_files, train_labels, ratio=1.0, seed=42):
    """
    ratio=1.0 -> spoof 数量 = bonafide 数量（完全平衡）
    ratio=2.0 -> spoof 数量 = 2 * bonafide（轻微不平衡）
    """
    random.seed(seed)

    bonafide_files = [f for f, y in zip(train_files, train_labels) if y == 0]
    spoof_files = [f for f, y in zip(train_files, train_labels) if y == 1]

    bonafide_labels = [0] * len(bonafide_files)

    target_spoof_num = int(len(bonafide_files) * ratio)
    spoof_files_sampled = random.sample(spoof_files, min(target_spoof_num, len(spoof_files)))
    spoof_labels_sampled = [1] * len(spoof_files_sampled)

    new_files = bonafide_files + spoof_files_sampled
    new_labels = bonafide_labels + spoof_labels_sampled

    combined = list(zip(new_files, new_labels))
    random.shuffle(combined)

    new_files, new_labels = zip(*combined)

    return list(new_files), list(new_labels)


def stratified_limit(files, labels, max_per_class=None, seed=42):
    if max_per_class is None or max_per_class <= 0:
        return list(files), list(labels)

    rng = random.Random(seed)
    selected = []
    for label in sorted(set(labels)):
        class_files = [path for path, y in zip(files, labels) if y == label]
        rng.shuffle(class_files)
        selected.extend((path, label) for path in class_files[:max_per_class])

    rng.shuffle(selected)
    if not selected:
        return [], []
    selected_files, selected_labels = zip(*selected)
    return list(selected_files), list(selected_labels)


# ===================== 7. 你的 CONFIG（你只需保证这些字段存在） =====================
CONFIG = {
    "model_params": {
        "conv_in_channels": 20,
        "conv_out_channels": 64,
        "lstm_hidden_size": 64,
        "fc1_out_features": 64,
        "num_classes": 2
    },
    "feature_params": {
        "n_mfcc": 20,
        "n_fft": 512,
        "hop_length": 160,
        "max_length": 300
    },
    "training_params": {
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "num_workers": 4,
        "lr_scheduler_patience": 2,
        "lr_scheduler_factor": 0.5,
        "early_stopping_patience": 3,
        "min_real_recall_for_threshold": 0.85,
        "min_fake_recall_for_threshold": 0.85,
        "max_recall_gap_for_threshold": 0.10,
        "deployment_min_macro_f1": 0.82,
        "deployment_min_class_f1": 0.75,
        "deployment_min_ood_macro_f1": 0.86,
        "deployment_min_ood_real_recall": 0.86,
        "deployment_min_balanced_fake_recall": 0.85,
        "deployment_min_asvspoof_fake_recall": 0.84,
        "checkpoint_min_inwild_real_recall": 0.85,
        "checkpoint_min_asvspoof_fake_recall": 0.84,
        "asvspoof_spoof_sampling_ratio": 1.25,
        "in_the_wild_spoof_sampling_ratio": 1.0,
        # Per-epoch model selection uses a fixed ASVspoof subset to avoid spending
        # most of the runtime on full validation. Full reports still run at the end.
        "selection_asvspoof_val_per_class": 1200,
    },
    "output_paths": {
        "model_save_path": "/kaggle/working/model/last_epoch_model.pt",
        "best_model_save_path": "/kaggle/working/model/best_f1_model.pt",
        "log_csv_path": "/kaggle/working/model/training_log.csv"
    }
}

os.makedirs("/kaggle/working/model", exist_ok=True)


# ===================== 8. data 必须存在（你原脚本已经有） =====================
# data = {
#     "train_files": [...],
#     "train_labels": [...],
#     "dev_files": [...],
#     "dev_labels": [...]
# }

# ===================== 9. 构建平衡训练集（核心！） =====================
# Balance inside each domain first, then merge. This avoids ASVspoof's large spoof
# pool overwhelming the In-the-Wild fake samples during pooled downsampling.
asvspoof_train_files_bal, asvspoof_train_labels_bal = build_balanced_training_set(
    data["asvspoof_train_files"],
    data["asvspoof_train_labels"],
    ratio=CONFIG["training_params"]["asvspoof_spoof_sampling_ratio"],
    seed=42,
)
in_the_wild_train_files_bal, in_the_wild_train_labels_bal = build_balanced_training_set(
    data["in_the_wild_train_files"],
    data["in_the_wild_train_labels"],
    ratio=CONFIG["training_params"]["in_the_wild_spoof_sampling_ratio"],
    seed=43,
) if data["in_the_wild_train_files"] else ([], [])

combined_balanced = list(zip(
    asvspoof_train_files_bal + in_the_wild_train_files_bal,
    asvspoof_train_labels_bal + in_the_wild_train_labels_bal,
))
random.Random(42).shuffle(combined_balanced)
train_files_bal, train_labels_bal = zip(*combined_balanced)
train_files_bal = list(train_files_bal)
train_labels_bal = list(train_labels_bal)

print("Original train size:", len(data["train_files"]))
print("Balanced train size:", len(train_files_bal))
print("Balanced label counts:", np.bincount(np.array(train_labels_bal)))
print(
    "Sampling ratios: "
    f"ASVspoof spoof={CONFIG['training_params']['asvspoof_spoof_sampling_ratio']}, "
    f"In-the-Wild spoof={CONFIG['training_params']['in_the_wild_spoof_sampling_ratio']}"
)
print("ASVspoof balanced label counts:", np.bincount(np.array(asvspoof_train_labels_bal)))
if in_the_wild_train_labels_bal:
    print("In-the-Wild balanced label counts:", np.bincount(np.array(in_the_wild_train_labels_bal)))

# ===================== 10. Dataset & DataLoader =====================
feature_config = CONFIG["feature_params"]
training_config = CONFIG["training_params"]
output_config = CONFIG["output_paths"]

selection_asvspoof_val_files, selection_asvspoof_val_labels = stratified_limit(
    data["asvspoof_dev_files"],
    data["asvspoof_dev_labels"],
    max_per_class=training_config["selection_asvspoof_val_per_class"],
    seed=42,
)
selection_val_files = selection_asvspoof_val_files + data["in_the_wild_val_files"]
selection_val_labels = selection_asvspoof_val_labels + data["in_the_wild_val_labels"]

train_dataset = AudioDataset(train_files_bal, train_labels_bal, feature_config, cache_dir="/kaggle/working/mfcc_cache/train")
selection_val_dataset = AudioDataset(
    selection_val_files,
    selection_val_labels,
    feature_config,
    cache_dir="/kaggle/working/mfcc_cache/selection_val",
)
selection_asvspoof_val_dataset = AudioDataset(
    selection_asvspoof_val_files,
    selection_asvspoof_val_labels,
    feature_config,
    cache_dir="/kaggle/working/mfcc_cache/selection_asvspoof_dev",
)
full_val_dataset = AudioDataset(data["dev_files"], data["dev_labels"], feature_config, cache_dir="/kaggle/working/mfcc_cache/full_dev")
full_asvspoof_val_dataset = AudioDataset(
    data["asvspoof_dev_files"],
    data["asvspoof_dev_labels"],
    feature_config,
    cache_dir="/kaggle/working/mfcc_cache/full_asvspoof_dev",
)
in_the_wild_val_dataset = AudioDataset(
    data["in_the_wild_val_files"],
    data["in_the_wild_val_labels"],
    feature_config,
    cache_dir="/kaggle/working/mfcc_cache/in_the_wild_val",
)
in_the_wild_ood_test_dataset = AudioDataset(
    data["in_the_wild_ood_test_files"],
    data["in_the_wild_ood_test_labels"],
    feature_config,
    cache_dir="/kaggle/working/mfcc_cache/in_the_wild_ood_test",
)


def make_dataloader(dataset, shuffle=False):
    loader_kwargs = {
        "batch_size": training_config["batch_size"],
        "shuffle": shuffle,
        "num_workers": training_config["num_workers"],
        "collate_fn": collate_fn_skip_none,
        "pin_memory": True,
    }
    if training_config["num_workers"] > 0:
        # Keep this False in Kaggle notebooks. Persistent workers can keep the
        # kernel alive after the final cell has printed "Notebook finished".
        loader_kwargs["persistent_workers"] = False
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


train_dataloader = make_dataloader(train_dataset, shuffle=True)
selection_val_dataloader = make_dataloader(selection_val_dataset, shuffle=False)
selection_asvspoof_val_dataloader = make_dataloader(selection_asvspoof_val_dataset, shuffle=False)
full_val_dataloader = make_dataloader(full_val_dataset, shuffle=False)
full_asvspoof_val_dataloader = make_dataloader(full_asvspoof_val_dataset, shuffle=False)
in_the_wild_val_dataloader = make_dataloader(in_the_wild_val_dataset, shuffle=False)
in_the_wild_ood_test_dataloader = make_dataloader(in_the_wild_ood_test_dataset, shuffle=False)

print("Train batches:", len(train_dataloader), "Selection val batches:", len(selection_val_dataloader))
print("Selection ASVspoof val batches:", len(selection_asvspoof_val_dataloader),
      "In-the-Wild val batches:", len(in_the_wild_val_dataloader),
      "In-the-Wild OOD test batches:", len(in_the_wild_ood_test_dataloader))
print("Full combined val batches:", len(full_val_dataloader),
      "Full ASVspoof val batches:", len(full_asvspoof_val_dataloader))

# ===================== 11. 训练初始化 =====================
model = AudioCNNLSTM(CONFIG["model_params"]).to(device)

criterion = nn.CrossEntropyLoss()   # ⭐ 重要：平衡采样后不要再加权
optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

def checkpoint_gate_status(combined_metrics, asvspoof_metrics, in_the_wild_metrics, training_config):
    blockers = []
    if combined_metrics["fake_recall"] < training_config["deployment_min_balanced_fake_recall"]:
        blockers.append(
            f"balanced_fake_recall={combined_metrics['fake_recall']:.4f} "
            f"< {training_config['deployment_min_balanced_fake_recall']:.4f}"
        )
    if combined_metrics["recall_gap"] > training_config["max_recall_gap_for_threshold"]:
        blockers.append(
            f"balanced_recall_gap={combined_metrics['recall_gap']:.4f} "
            f"> {training_config['max_recall_gap_for_threshold']:.4f}"
        )
    if asvspoof_metrics["fake_recall"] < training_config["checkpoint_min_asvspoof_fake_recall"]:
        blockers.append(
            f"asvspoof_fake_recall={asvspoof_metrics['fake_recall']:.4f} "
            f"< {training_config['checkpoint_min_asvspoof_fake_recall']:.4f}"
        )
    if in_the_wild_metrics and len(in_the_wild_metrics.get("all_targets", [])) > 0:
        if in_the_wild_metrics["real_recall"] < training_config["checkpoint_min_inwild_real_recall"]:
            blockers.append(
                f"inwild_real_recall={in_the_wild_metrics['real_recall']:.4f} "
                f"< {training_config['checkpoint_min_inwild_real_recall']:.4f}"
            )
    else:
        blockers.append("inwild_validation_unavailable")
    return len(blockers) == 0, blockers


def model_selection_score(combined_metrics, asvspoof_metrics, in_the_wild_metrics=None, training_config=None):
    metric_sets = [combined_metrics, asvspoof_metrics]
    if in_the_wild_metrics and len(in_the_wild_metrics.get("all_targets", [])) > 0:
        metric_sets.append(in_the_wild_metrics)
    balanced_acc = np.mean([m["balanced_acc"] for m in metric_sets])
    macro_f1 = np.mean([m["macro_f1"] for m in metric_sets])
    min_class_f1 = np.mean([m["min_class_f1"] for m in metric_sets])
    min_class_precision = np.mean([m["min_class_precision"] for m in metric_sets])
    min_class_recall = np.mean([m["min_class_recall"] for m in metric_sets])
    recall_gap = np.mean([m["recall_gap"] for m in metric_sets])
    score = float(
        0.35 * macro_f1
        + 0.20 * min_class_f1
        + 0.15 * min_class_precision
        + 0.15 * min_class_recall
        + 0.10 * balanced_acc
        - 0.05 * recall_gap
    )
    if training_config:
        gate_ok, blockers = checkpoint_gate_status(
            combined_metrics,
            asvspoof_metrics,
            in_the_wild_metrics,
            training_config,
        )
        deficit_penalty = 0.0
        deficit_penalty += max(0.0, training_config["deployment_min_balanced_fake_recall"] - combined_metrics["fake_recall"])
        deficit_penalty += max(0.0, training_config["checkpoint_min_asvspoof_fake_recall"] - asvspoof_metrics["fake_recall"])
        if in_the_wild_metrics and len(in_the_wild_metrics.get("all_targets", [])) > 0:
            deficit_penalty += max(0.0, training_config["checkpoint_min_inwild_real_recall"] - in_the_wild_metrics["real_recall"])
        score -= 0.60 * deficit_penalty
        if gate_ok:
            score += 0.03
    return float(score)


best_selection_score = -1.0
best_epoch = -1
best_decision_threshold = 0.5
best_threshold_candidates = []
best_checkpoint_gate_passed = False
best_checkpoint_blockers = []
epochs_no_improve = 0
training_logs = []

log_path = output_config["log_csv_path"]
with open(log_path, mode="w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Date", "Epoch", "TrainLoss", "ValLoss",
        "DecisionThreshold", "SelectionScore",
        "ValAcc", "ValMacroF1", "ValFakeF1", "ValMinClassF1",
        "ValRealPrecision", "ValFakePrecision", "ValBalancedAcc",
        "ValRealRecall", "ValFakeRecall", "ValRecallGap",
        "ASVspoofBalancedAcc", "ASVspoofRealRecall", "ASVspoofFakeRecall",
        "InWildValBalancedAcc", "InWildValRealRecall", "InWildValFakeRecall",
        "CheckpointGatePassed", "CheckpointBlockers",
        "ValEER", "EER_Thr",
    ])

print("\n🚀 Start training...")

# ===================== 12. 训练循环 =====================
for epoch in range(training_config["num_epochs"]):
    model.train()
    total_train_loss = 0
    train_preds, train_targets = [], []

    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{training_config['num_epochs']} [Train]", unit="batch")

    for batch_data in loop:
        if batch_data is None:
            continue

        inputs, targets = batch_data
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)

        train_preds.extend(preds.detach().cpu().numpy())
        train_targets.extend(targets.detach().cpu().numpy())

        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = total_train_loss / max(1, len(train_dataloader))
    train_acc = accuracy_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds, zero_division=0)

    # ===== 验证 =====
    val_metrics = evaluate_model(
        model,
        selection_val_dataloader,
        criterion,
        threshold=None,
        min_real_recall=training_config["min_real_recall_for_threshold"],
        min_fake_recall=training_config["min_fake_recall_for_threshold"],
        max_recall_gap=training_config["max_recall_gap_for_threshold"],
    )
    decision_threshold = val_metrics["threshold"]
    asvspoof_val_metrics = evaluate_model(
        model,
        selection_asvspoof_val_dataloader,
        criterion,
        threshold=decision_threshold,
    )
    in_the_wild_val_metrics = evaluate_model(
        model,
        in_the_wild_val_dataloader,
        criterion,
        threshold=decision_threshold,
    )
    selection_score = model_selection_score(
        val_metrics,
        asvspoof_val_metrics,
        in_the_wild_val_metrics,
        training_config,
    )
    checkpoint_gate_passed, checkpoint_blockers = checkpoint_gate_status(
        val_metrics,
        asvspoof_val_metrics,
        in_the_wild_val_metrics,
        training_config,
    )

    print(
        f"\nEpoch [{epoch+1}] "
        f"TrainLoss={avg_train_loss:.4f} TrainAcc={train_acc:.4f} TrainF1={train_f1:.4f} | "
        f"ValLoss={val_metrics['loss']:.4f} ValAcc={val_metrics['accuracy']:.4f} "
        f"ValMacroF1={val_metrics['macro_f1']:.4f} ValFakeF1={val_metrics['f1']:.4f} "
        f"ValMinClassF1={val_metrics['min_class_f1']:.4f} "
        f"BalancedAcc={val_metrics['balanced_acc']:.4f} "
        f"RealPrecision={val_metrics['real_precision']:.4f} FakePrecision={val_metrics['fake_precision']:.4f} "
        f"RealRecall={val_metrics['real_recall']:.4f} FakeRecall={val_metrics['fake_recall']:.4f} "
        f"RecallGap={val_metrics['recall_gap']:.4f} "
        f"DecisionThr={decision_threshold:.4f} SelectionScore={selection_score:.4f} "
        f"ValEER={val_metrics['eer']*100:.2f}% (eer_thr={val_metrics['eer_threshold']:.4f})"
    )
    if checkpoint_gate_passed:
        print("  Checkpoint gate: PASS")
    else:
        print("  Checkpoint gate: BLOCKED | " + "; ".join(checkpoint_blockers))
    print(
        f"  ASVspoof Val: BalancedAcc={asvspoof_val_metrics['balanced_acc']:.4f} "
        f"RealRecall={asvspoof_val_metrics['real_recall']:.4f} "
        f"FakeRecall={asvspoof_val_metrics['fake_recall']:.4f}"
    )
    if len(in_the_wild_val_metrics["all_targets"]) > 0:
        print(
            f"  In-the-Wild Val: BalancedAcc={in_the_wild_val_metrics['balanced_acc']:.4f} "
            f"RealRecall={in_the_wild_val_metrics['real_recall']:.4f} "
            f"FakeRecall={in_the_wild_val_metrics['fake_recall']:.4f}"
        )
    else:
        print("  In-the-Wild Val: unavailable")

    print(
        f"  Val@0.5: BalancedAcc={val_metrics['threshold_0_5']['balanced_acc']:.4f} "
        f"RealRecall={val_metrics['threshold_0_5']['real_recall']:.4f} "
        f"FakeRecall={val_metrics['threshold_0_5']['fake_recall']:.4f}"
    )

    # ===== collapse check =====
    is_collapse, msg = detect_collapse(val_metrics["all_preds"], threshold=0.98)
    print(msg)
    if is_collapse:
        print("⚠️ Collapse-like prediction distribution at calibrated threshold; this epoch will not be saved as best.")

    # ===== 保存 best =====
    if (not is_collapse) and selection_score > best_selection_score:
        best_selection_score = selection_score
        best_epoch = epoch + 1
        best_decision_threshold = decision_threshold
        best_threshold_candidates = val_metrics.get("threshold_candidates", [])
        best_checkpoint_gate_passed = checkpoint_gate_passed
        best_checkpoint_blockers = checkpoint_blockers
        torch.save(model.state_dict(), output_config["best_model_save_path"])
        print(
            f"✅ Best model saved at epoch {best_epoch} | "
            f"SelectionScore={best_selection_score:.4f} DecisionThr={best_decision_threshold:.4f}"
        )
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # ===== early stopping =====
    if epochs_no_improve >= training_config["early_stopping_patience"]:
        print("⏹️ Early stopping triggered.")
        break

    # training_logs
    training_logs.append({
        "Epoch": epoch + 1,
        "Train Loss": float(avg_train_loss),
        "Val Loss": float(val_metrics["loss"]),
        "Train Acc": float(train_acc),
        "Val Acc": float(val_metrics["accuracy"]),
        "Decision Threshold": float(decision_threshold),
        "Selection Score": float(selection_score),
        "Train F1": float(train_f1),
        "Val Fake F1": float(val_metrics["f1"]),
        "Val Macro F1": float(val_metrics["macro_f1"]),
        "Val Min Class F1": float(val_metrics["min_class_f1"]),
        "Val Precision": float(val_metrics["precision"]),
        "Val Real Precision": float(val_metrics["real_precision"]),
        "Val Fake Precision": float(val_metrics["fake_precision"]),
        "Val Fake Recall": float(val_metrics["fake_recall"]),
        "Val Real Recall": float(val_metrics["real_recall"]),
        "Val Recall Gap": float(val_metrics["recall_gap"]),
        "Val Balanced Acc": float(val_metrics["balanced_acc"]),
        "ASVspoof Balanced Acc": float(asvspoof_val_metrics["balanced_acc"]),
        "ASVspoof Real Recall": float(asvspoof_val_metrics["real_recall"]),
        "ASVspoof Fake Recall": float(asvspoof_val_metrics["fake_recall"]),
        "InWild Val Balanced Acc": float(in_the_wild_val_metrics["balanced_acc"]),
        "InWild Val Real Recall": float(in_the_wild_val_metrics["real_recall"]),
        "InWild Val Fake Recall": float(in_the_wild_val_metrics["fake_recall"]),
        "Checkpoint Gate Passed": bool(checkpoint_gate_passed),
        "Checkpoint Blockers": "; ".join(checkpoint_blockers),
        "Val EER": float(val_metrics["eer"]),
        "EER Thr": float(val_metrics["eer_threshold"]),
    })

    # ===== log =====
    with open(log_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            epoch + 1,
            round(avg_train_loss, 4),
            round(val_metrics["loss"], 4),
            round(decision_threshold, 6),
            round(selection_score, 6),
            round(val_metrics["accuracy"], 4),
            round(val_metrics["macro_f1"], 4),
            round(val_metrics["f1"], 4),
            round(val_metrics["min_class_f1"], 4),
            round(val_metrics["real_precision"], 4),
            round(val_metrics["fake_precision"], 4),
            round(val_metrics["balanced_acc"], 4),
            round(val_metrics["real_recall"], 4),
            round(val_metrics["fake_recall"], 4),
            round(val_metrics["recall_gap"], 4),
            round(asvspoof_val_metrics["balanced_acc"], 4),
            round(asvspoof_val_metrics["real_recall"], 4),
            round(asvspoof_val_metrics["fake_recall"], 4),
            round(in_the_wild_val_metrics["balanced_acc"], 4),
            round(in_the_wild_val_metrics["real_recall"], 4),
            round(in_the_wild_val_metrics["fake_recall"], 4),
            int(checkpoint_gate_passed),
            "; ".join(checkpoint_blockers),
            round(val_metrics["eer"], 6),
            round(val_metrics["eer_threshold"], 6),
        ])

# ===== 保存最后模型 =====
torch.save(model.state_dict(), output_config["model_save_path"])
print(f"\n💾 Last model saved: {output_config['model_save_path']}")

if best_epoch != -1:
    print(
        f"💾 Best model saved: {output_config['best_model_save_path']} "
        f"(Epoch {best_epoch}, SelectionScore={best_selection_score:.4f}, "
        f"DecisionThr={best_decision_threshold:.4f})"
    )
    model.load_state_dict(torch.load(output_config["best_model_save_path"], map_location=device))
else:
    print("⚠️ No best model saved (training stopped too early).")

# ===== 最终报告 =====
final_balanced_val = evaluate_model(model, selection_val_dataloader, criterion, threshold=best_decision_threshold)
final_balanced_asvspoof_val = evaluate_model(
    model,
    selection_asvspoof_val_dataloader,
    criterion,
    threshold=best_decision_threshold,
)
final_val = evaluate_model(model, full_val_dataloader, criterion, threshold=best_decision_threshold)
final_asvspoof_val = evaluate_model(model, full_asvspoof_val_dataloader, criterion, threshold=best_decision_threshold)
final_in_the_wild_val = evaluate_model(model, in_the_wild_val_dataloader, criterion, threshold=best_decision_threshold)
final_in_the_wild_ood = evaluate_model(model, in_the_wild_ood_test_dataloader, criterion, threshold=best_decision_threshold)

print("\n📋 Final Balanced Validation Report (model selection distribution):")
print(classification_report(final_balanced_val["all_targets"], final_balanced_val["all_preds"], digits=4))
print("Confusion Matrix:\n", confusion_matrix(final_balanced_val["all_targets"], final_balanced_val["all_preds"]))
print("Balanced Accuracy:", final_balanced_val["balanced_acc"])
print("Macro F1:", final_balanced_val["macro_f1"])
print("Min Class F1:", final_balanced_val["min_class_f1"])
print("Decision Threshold:", best_decision_threshold)

print("\n📋 Final Balanced ASVspoof Validation Report:")
print(classification_report(final_balanced_asvspoof_val["all_targets"], final_balanced_asvspoof_val["all_preds"], digits=4))
print("Confusion Matrix:\n", confusion_matrix(final_balanced_asvspoof_val["all_targets"], final_balanced_asvspoof_val["all_preds"]))
print("Balanced Accuracy:", final_balanced_asvspoof_val["balanced_acc"])

print("\n📋 Final Combined Validation Report (best model, calibrated threshold):")
print(classification_report(final_val["all_targets"], final_val["all_preds"], digits=4))
print("Confusion Matrix:\n", confusion_matrix(final_val["all_targets"], final_val["all_preds"]))
print("Balanced Accuracy:", final_val["balanced_acc"])
print("Decision Threshold:", best_decision_threshold)

print("\n📋 Final ASVspoof Validation Report:")
print(classification_report(final_asvspoof_val["all_targets"], final_asvspoof_val["all_preds"], digits=4))
print("Confusion Matrix:\n", confusion_matrix(final_asvspoof_val["all_targets"], final_asvspoof_val["all_preds"]))
print("Balanced Accuracy:", final_asvspoof_val["balanced_acc"])

if len(final_in_the_wild_val["all_targets"]) > 0:
    print("\n📋 Final In-the-Wild Validation Report:")
    print(classification_report(final_in_the_wild_val["all_targets"], final_in_the_wild_val["all_preds"], digits=4))
    print("Confusion Matrix:\n", confusion_matrix(final_in_the_wild_val["all_targets"], final_in_the_wild_val["all_preds"]))
    print("Balanced Accuracy:", final_in_the_wild_val["balanced_acc"])

if len(final_in_the_wild_ood["all_targets"]) > 0:
    print("\n🧪 Held-out In-the-Wild OOD Test Report (never used for training/model selection):")
    print(classification_report(final_in_the_wild_ood["all_targets"], final_in_the_wild_ood["all_preds"], digits=4))
    print("Confusion Matrix:\n", confusion_matrix(final_in_the_wild_ood["all_targets"], final_in_the_wild_ood["all_preds"]))
    print("Balanced Accuracy:", final_in_the_wild_ood["balanced_acc"])

# %%
# ===================== Cell 10: 训练结果可视化 =====================
print("\n" + "=" * 70)
print("📉 Step 7: 训练结果可视化")
print("=" * 70)

fig, axes = plt.subplots(2, 4, figsize=(20, 12))
epochs_range = range(1, len(training_logs) + 1)

# Loss
axes[0, 0].plot(epochs_range, [l["Train Loss"] for l in training_logs], "b-o", label="Train", markersize=4)
axes[0, 0].plot(epochs_range, [l["Val Loss"] for l in training_logs], "r-o", label="Val", markersize=4)
axes[0, 0].set_title("Loss"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")

# Accuracy
axes[0, 1].plot(epochs_range, [l["Train Acc"] for l in training_logs], "b-o", label="Train", markersize=4)
axes[0, 1].plot(epochs_range, [l["Val Acc"] for l in training_logs], "r-o", label="Val", markersize=4)
axes[0, 1].set_title("Accuracy"); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy")

# F1 Score
axes[0, 2].plot(epochs_range, [l["Train F1"] for l in training_logs], "b-o", label="Train F1", markersize=4)
axes[0, 2].plot(epochs_range, [l["Val Macro F1"] for l in training_logs], "r-o", label="Val Macro F1", markersize=4)
axes[0, 2].plot(epochs_range, [l["Val Fake F1"] for l in training_logs], "m-o", label="Val Fake F1", markersize=4)
axes[0, 2].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5, label=f"Best (E{best_epoch})")
axes[0, 2].set_title("F1 Score"); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_xlabel("Epoch"); axes[0, 2].set_ylabel("F1")

# EER
axes[0, 3].plot(epochs_range, [l["Val EER"] for l in training_logs], "k-o", label="Val EER", markersize=4)
axes[0, 3].set_title("EER (Validation)"); axes[0, 3].legend(); axes[0, 3].grid(True, alpha=0.3)
axes[0, 3].set_xlabel("Epoch"); axes[0, 3].set_ylabel("EER")

# Real/Fake recall
axes[1, 0].plot(epochs_range, [l["Val Real Recall"] for l in training_logs], "g-o", label="Real Recall", markersize=4)
axes[1, 0].plot(epochs_range, [l["Val Fake Recall"] for l in training_logs], "m-o", label="Fake Recall", markersize=4)
axes[1, 0].set_title("Val Real/Fake Recall"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlabel("Epoch")

# 最终验证集混淆矩阵
final_val = evaluate_model(model, full_val_dataloader, criterion, threshold=best_decision_threshold)
cm = confusion_matrix(final_val["all_targets"], final_val["all_preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1],
            xticklabels=["Bonafide", "Spoof"], yticklabels=["Bonafide", "Spoof"])
axes[1, 1].set_title("Confusion Matrix (Val)"); axes[1, 1].set_ylabel("True"); axes[1, 1].set_xlabel("Predict")

# ROC 曲线
if final_val["all_probs"]:
    fpr, tpr, _ = roc_curve(final_val["all_targets"], final_val["all_probs"])
    roc_auc = auc(fpr, tpr)
    axes[1, 2].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    axes[1, 2].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    axes[1, 2].set_title("ROC Curve"); axes[1, 2].legend()
    axes[1, 2].set_xlabel("False Positive Rate"); axes[1, 2].set_ylabel("True Positive Rate")
    axes[1, 2].grid(True, alpha=0.3)

# 右下角：显示最终 EER 文本（也可以改成画阈值曲线）
axes[1, 3].axis("off")
axes[1, 3].text(
    0.02, 0.8,
    f"Final Val EER: {final_val['eer']*100:.2f}%\nEER Thr: {final_val['eer_threshold']:.4f}",
    fontsize=14
)

plt.suptitle("CNN-BiLSTM (ASVspoof 2019 LA)", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("/kaggle/working/training_results.png", dpi=150, bbox_inches="tight")
plt.show()

# 详细分类报告
print("\n📋 详细分类报告 (验证集):")
print(classification_report(final_val["all_targets"], final_val["all_preds"],
                            target_names=["Bonafide", "Spoof"]))

# ===================== Cell 11: 导出兼容原项目的配置文件 =====================
print("\n" + "=" * 70)
print("📦 Step 8: 导出与原项目兼容的 audio_risk_config.json")
print("=" * 70)


def metrics_for_export(metrics):
    keys = [
        "threshold", "accuracy", "precision", "recall", "f1", "macro_f1",
        "balanced_acc", "real_precision", "fake_precision", "real_recall",
        "fake_recall", "real_f1", "fake_f1", "min_class_precision",
        "min_class_recall", "min_class_f1", "recall_gap", "precision_gap",
        "eer", "eer_threshold", "confusion_matrix",
    ]
    return {key: metrics.get(key) for key in keys if key in metrics}


def collect_error_examples(split_name, dataset, metrics, max_examples=200):
    paths = list(getattr(dataset, "file_paths", []))
    targets = list(metrics.get("all_targets", []))
    preds = list(metrics.get("all_preds", []))
    probs = list(metrics.get("all_probs", []))
    rows = []
    usable_len = min(len(paths), len(targets), len(preds), len(probs))
    for idx in range(usable_len):
        target = int(targets[idx])
        pred = int(preds[idx])
        if target == pred:
            continue
        probability = float(probs[idx])
        error_type = "false_positive_real_as_fake" if target == 0 else "false_negative_fake_as_real"
        rows.append({
            "split": split_name,
            "error_type": error_type,
            "target": target,
            "prediction": pred,
            "deepfake_probability": probability,
            "audio_path": paths[idx],
        })

    rows.sort(
        key=lambda row: (
            row["error_type"],
            -row["deepfake_probability"] if row["error_type"] == "false_positive_real_as_fake"
            else row["deepfake_probability"],
        )
    )
    return rows[:max_examples]


def write_error_examples_csv(path, rows):
    fieldnames = ["split", "error_type", "target", "prediction", "deepfake_probability", "audio_path"]
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


deployment_criteria = {
    "balanced_min_macro_f1": training_config["deployment_min_macro_f1"],
    "balanced_min_class_f1": training_config["deployment_min_class_f1"],
    "min_real_recall": training_config["min_real_recall_for_threshold"],
    "min_fake_recall": training_config["min_fake_recall_for_threshold"],
    "max_recall_gap": training_config["max_recall_gap_for_threshold"],
    "ood_min_macro_f1": training_config["deployment_min_ood_macro_f1"],
    "ood_min_real_recall": training_config["deployment_min_ood_real_recall"],
    "asvspoof_min_fake_recall": training_config["deployment_min_asvspoof_fake_recall"],
}


def build_deployment_blockers(balanced_metrics, asvspoof_metrics, ood_metrics, threshold_candidates, criteria):
    blockers = []
    threshold_ok = bool(threshold_candidates and threshold_candidates[0].get("constraints_satisfied"))
    if not threshold_ok:
        blockers.append("No threshold satisfied balanced recall and recall-gap constraints.")
    if balanced_metrics["macro_f1"] < criteria["balanced_min_macro_f1"]:
        blockers.append(
            f"Balanced validation macro_f1 {balanced_metrics['macro_f1']:.4f} "
            f"< {criteria['balanced_min_macro_f1']:.4f}."
        )
    if balanced_metrics["min_class_f1"] < criteria["balanced_min_class_f1"]:
        blockers.append(
            f"Balanced validation min_class_f1 {balanced_metrics['min_class_f1']:.4f} "
            f"< {criteria['balanced_min_class_f1']:.4f}."
        )
    if balanced_metrics["real_recall"] < criteria["min_real_recall"]:
        blockers.append(
            f"Balanced validation real_recall {balanced_metrics['real_recall']:.4f} "
            f"< {criteria['min_real_recall']:.4f}."
        )
    if balanced_metrics["fake_recall"] < criteria["min_fake_recall"]:
        blockers.append(
            f"Balanced validation fake_recall {balanced_metrics['fake_recall']:.4f} "
            f"< {criteria['min_fake_recall']:.4f}."
        )
    if balanced_metrics["recall_gap"] > criteria["max_recall_gap"]:
        blockers.append(
            f"Balanced validation recall_gap {balanced_metrics['recall_gap']:.4f} "
            f"> {criteria['max_recall_gap']:.4f}."
        )
    if asvspoof_metrics["fake_recall"] < criteria["asvspoof_min_fake_recall"]:
        blockers.append(
            f"ASVspoof balanced fake_recall {asvspoof_metrics['fake_recall']:.4f} "
            f"< {criteria['asvspoof_min_fake_recall']:.4f}."
        )
    if len(ood_metrics.get("all_targets", [])) == 0:
        blockers.append("In-the-Wild OOD test is unavailable.")
    else:
        if ood_metrics["macro_f1"] < criteria["ood_min_macro_f1"]:
            blockers.append(
                f"In-the-Wild OOD macro_f1 {ood_metrics['macro_f1']:.4f} "
                f"< {criteria['ood_min_macro_f1']:.4f}."
            )
        if ood_metrics["real_recall"] < criteria["ood_min_real_recall"]:
            blockers.append(
                f"In-the-Wild OOD real_recall {ood_metrics['real_recall']:.4f} "
                f"< {criteria['ood_min_real_recall']:.4f}."
            )
        if ood_metrics["fake_recall"] < criteria["min_fake_recall"]:
            blockers.append(
                f"In-the-Wild OOD fake_recall {ood_metrics['fake_recall']:.4f} "
                f"< {criteria['min_fake_recall']:.4f}."
            )
    return blockers


deployment_blockers = build_deployment_blockers(
    final_balanced_val,
    final_balanced_asvspoof_val,
    final_in_the_wild_ood,
    best_threshold_candidates,
    deployment_criteria,
)
deployment_ready = len(deployment_blockers) == 0

error_examples = []
error_examples.extend(collect_error_examples("balanced_combined", selection_val_dataset, final_balanced_val))
error_examples.extend(collect_error_examples("balanced_asvspoof", selection_asvspoof_val_dataset, final_balanced_asvspoof_val))
error_examples.extend(collect_error_examples("in_the_wild_val", in_the_wild_val_dataset, final_in_the_wild_val))
error_examples.extend(collect_error_examples("in_the_wild_ood", in_the_wild_ood_test_dataset, final_in_the_wild_ood))
error_examples_path = "/kaggle/working/model/error_examples.csv"
write_error_examples_csv(error_examples_path, error_examples)

# 生成与原项目 audio_risk_config.json 完全兼容的配置文件
# 这样本地的 predict_audio_risk.py 可以直接加载使用
export_config = {
    "data_paths": {
        "real_audio_dir": "audio_risk_detection/real_audio",
        "deepfake_audio_dir": "audio_risk_detection/deepfake_audio"
    },
    "feature_params": CONFIG["feature_params"],
    "model_params": CONFIG["model_params"],
    "training_params": {
        "num_epochs": training_config["num_epochs"],
        "batch_size": training_config["batch_size"],
        "learning_rate": training_config["learning_rate"],
        "num_workers": 0,
        "validation_split": 0.2,
        "random_seed": 42,
        "lr_scheduler_patience": training_config["lr_scheduler_patience"],
        "lr_scheduler_factor": training_config["lr_scheduler_factor"],
        "early_stopping_patience": training_config["early_stopping_patience"],
        "min_real_recall_for_threshold": training_config["min_real_recall_for_threshold"],
        "min_fake_recall_for_threshold": training_config["min_fake_recall_for_threshold"],
        "max_recall_gap_for_threshold": training_config["max_recall_gap_for_threshold"],
        "deployment_min_macro_f1": training_config["deployment_min_macro_f1"],
        "deployment_min_class_f1": training_config["deployment_min_class_f1"],
        "deployment_min_ood_macro_f1": training_config["deployment_min_ood_macro_f1"],
        "deployment_min_ood_real_recall": training_config["deployment_min_ood_real_recall"],
        "deployment_min_balanced_fake_recall": training_config["deployment_min_balanced_fake_recall"],
        "deployment_min_asvspoof_fake_recall": training_config["deployment_min_asvspoof_fake_recall"],
        "checkpoint_min_inwild_real_recall": training_config["checkpoint_min_inwild_real_recall"],
        "checkpoint_min_asvspoof_fake_recall": training_config["checkpoint_min_asvspoof_fake_recall"],
        "asvspoof_spoof_sampling_ratio": training_config["asvspoof_spoof_sampling_ratio"],
        "in_the_wild_spoof_sampling_ratio": training_config["in_the_wild_spoof_sampling_ratio"],
        "selection_asvspoof_val_per_class": training_config["selection_asvspoof_val_per_class"],
    },
    "decision_params": {
        "decision_threshold": best_decision_threshold,
        "threshold_source": "balanced_validation_macro_f1_constraints",
        "min_real_recall_for_threshold": training_config["min_real_recall_for_threshold"],
        "min_fake_recall_for_threshold": training_config["min_fake_recall_for_threshold"],
        "max_recall_gap_for_threshold": training_config["max_recall_gap_for_threshold"],
        "class_0": "real/bonafide",
        "class_1": "fake/spoof",
        "score_mapping": "threshold_linear_50_at_decision_threshold",
    },
    "output_paths": {
        "log_csv_path": "static/csv/train_deepfake_metrics_log.csv",
        "model_save_path": "audio_risk_detection/model/last_epoch_model.pt",
        "best_model_save_path": "audio_risk_detection/model/best_f1_model.pt"
    }
}

config_export_path = "/kaggle/working/model/audio_risk_config.json"
with open(config_export_path, "w", encoding="utf-8") as f:
    json.dump(export_config, f, indent=2, ensure_ascii=False)

# 保存训练元信息
meta_info = {
    "dataset": "ASVspoof 2019 LA + In-the-Wild Audio Deepfake",
    "deployment_ready": deployment_ready,
    "deployment_blockers": deployment_blockers,
    "deployment_criteria": deployment_criteria,
    "asvspoof_train_samples": len(data["asvspoof_train_files"]),
    "asvspoof_val_samples": len(data["asvspoof_dev_files"]),
    "in_the_wild_train_samples": len(data["in_the_wild_train_files"]),
    "in_the_wild_val_samples": len(data["in_the_wild_val_files"]),
    "in_the_wild_ood_test_samples": len(data["in_the_wild_ood_test_files"]),
    "combined_train_samples_before_balancing": len(data["train_files"]),
    "balanced_train_samples": len(train_files_bal),
    "asvspoof_balanced_train_samples": len(asvspoof_train_files_bal),
    "in_the_wild_balanced_train_samples": len(in_the_wild_train_files_bal),
    "combined_val_samples": len(data["dev_files"]),
    "selection_asvspoof_val_samples": len(selection_asvspoof_val_files),
    "selection_combined_val_samples": len(selection_val_files),
    "in_the_wild_caps": CONFIG.get("in_the_wild", {}),
    "best_selection_score": best_selection_score,
    "best_epoch": best_epoch,
    "decision_threshold": best_decision_threshold,
    "selected_threshold_candidates": best_threshold_candidates,
    "best_checkpoint_gate_passed": best_checkpoint_gate_passed,
    "best_checkpoint_blockers": best_checkpoint_blockers,
    "sampling_strategy": {
        "asvspoof_spoof_sampling_ratio": training_config["asvspoof_spoof_sampling_ratio"],
        "in_the_wild_spoof_sampling_ratio": training_config["in_the_wild_spoof_sampling_ratio"],
        "intent": "version3 prioritizes ASVspoof fake recall while preserving In-the-Wild real recall.",
    },
    "error_examples_csv": "model/error_examples.csv",
    "total_epochs_run": len(training_logs),
    "model_architecture": "AudioCNNLSTM (CNN-BiLSTM)",
    "feature": f"MFCC-{CONFIG['feature_params']['n_mfcc']}",
    "balanced_validation_metrics": {
        "combined": metrics_for_export(final_balanced_val),
        "asvspoof": metrics_for_export(final_balanced_asvspoof_val),
        "in_the_wild": metrics_for_export(final_in_the_wild_val),
    },
    "full_combined_validation_metrics": metrics_for_export(final_val),
    "in_the_wild_ood_metrics": metrics_for_export(final_in_the_wild_ood),
    "final_combined_val": {
        "balanced_acc": final_val["balanced_acc"],
        "macro_f1": final_val["macro_f1"],
        "real_recall": final_val["real_recall"],
        "fake_recall": final_val["fake_recall"],
        "confusion_matrix": final_val["confusion_matrix"],
    },
    "final_in_the_wild_ood_test": {
        "balanced_acc": final_in_the_wild_ood["balanced_acc"],
        "macro_f1": final_in_the_wild_ood["macro_f1"],
        "real_recall": final_in_the_wild_ood["real_recall"],
        "fake_recall": final_in_the_wild_ood["fake_recall"],
        "confusion_matrix": final_in_the_wild_ood["confusion_matrix"],
    },
}
with open("/kaggle/working/model/training_meta.json", "w") as f:
    json.dump(meta_info, f, indent=2)

evaluation_report = {
    "decision_threshold": best_decision_threshold,
    "threshold_source": export_config["decision_params"]["threshold_source"],
    "deployment_ready": deployment_ready,
    "deployment_blockers": deployment_blockers,
    "selected_threshold_candidates": best_threshold_candidates,
    "best_checkpoint_gate_passed": best_checkpoint_gate_passed,
    "best_checkpoint_blockers": best_checkpoint_blockers,
    "error_examples_csv": "model/error_examples.csv",
    "balanced_validation": meta_info["balanced_validation_metrics"],
    "full_combined_validation": meta_info["full_combined_validation_metrics"],
    "in_the_wild_ood": meta_info["in_the_wild_ood_metrics"],
}
with open("/kaggle/working/model/evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

print(f"✅ 配置文件已导出: {config_export_path}")
print(f"✅ 训练元信息已导出: /kaggle/working/model/training_meta.json")
print(f"✅ 评估报告已导出: /kaggle/working/model/evaluation_report.json")
print(f"✅ 误报/漏报样本清单已导出: {error_examples_path}")

# ===================== Cell 12: 文件清单与下载指引 =====================
print("\n" + "=" * 70)
print("📁 输出文件清单")
print("=" * 70)

# for item in Path("/kaggle/working").rglob("*"):
#     if item.is_file():
#         size_mb = item.stat().st_size / (1024 * 1024)
#         print(f"  {item.relative_to('/kaggle/working')} ({size_mb:.2f} MB)")

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  🎯 训练完成! 请下载以下文件到本地项目:                              ║
║                                                                      ║
║  1. model/best_f1_model.pt                                           ║
║     → audio_risk_detection/model/best_f1_model.pt                     ║
║                                                                      ║
║  2. model/audio_risk_config.json                                      ║
║     → audio_risk_detection/model/audio_risk_config.json                 ║
║     (包含 decision_params.decision_threshold, 供本地推理读取)           ║
║                                                                      ║
║  3. model/last_epoch_model.pt (备用)                                 ║
║     → audio_risk_detection/model/last_epoch_model.pt                  ║
║                                                                      ║
║  4. model/evaluation_report.json                                      ║
║     → 记录 balanced/full/OOD 指标与 deployment_ready                   ║
║                                                                      ║
║  5. model/error_examples.csv                                          ║
║     → 记录 false positive / false negative 样本，供 version4 定向分析 ║
║                                                                      ║
║  本地 predict_audio_risk.py 会读取校准阈值和分数映射配置。             ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print("🏁 Notebook 运行完毕!")

# Explicit cleanup for Kaggle notebooks. This avoids dangling DataLoader workers,
# open matplotlib figures, and cached CUDA memory after all artifacts are written.
for _name in [
    "train_dataloader",
    "selection_val_dataloader",
    "selection_asvspoof_val_dataloader",
    "full_val_dataloader",
    "full_asvspoof_val_dataloader",
    "in_the_wild_val_dataloader",
    "in_the_wild_ood_test_dataloader",
    "train_dataset",
    "selection_val_dataset",
    "selection_asvspoof_val_dataset",
    "full_val_dataset",
    "full_asvspoof_val_dataset",
    "in_the_wild_val_dataset",
    "in_the_wild_ood_test_dataset",
]:
    if _name in globals():
        del globals()[_name]

plt.close("all")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
