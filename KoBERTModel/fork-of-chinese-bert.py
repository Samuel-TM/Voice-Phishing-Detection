# %%
# =============================================================================
# 📞 电信诈骗文本分类模型训练 - Chinese BERT + TeleAntiFraud-28k
# =============================================================================
# 改造自原项目 KoBERTModel/train.py
# 核心改动:
#   1. monologg/kobert → bert-base-chinese
#   2. 韩文CSV数据集 → TeleAntiFraud-28k (JSONL 多轮对话)
#   3. 新增数据探索、多策略数据加载、可视化
# =============================================================================

# ===================== Cell 1: 环境安装 =====================
# !pip install transformers datasets scikit-learn tqdm matplotlib seaborn -q

# %%
# ===================== Cell 2: 导入库 =====================
import re
import os
import sys
import json
import glob
import random
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ===================== Cell 3: 配置与种子 =====================
# ──────────────── 全局配置 ────────────────
CONFIG = {
    # 数据集根目录 (你在Kaggle中的路径)
    "data_root": "/kaggle/input/teleantifraud-28k/TeleAntiFraud/JimmyMa99/TeleAntiFraud-28k/TeleAntiFraud-28k/TeleAntiFraud-28k",
    
    # 模型配置
    "model_name": "bert-base-chinese",
    "max_len": 256,           # 对话文本较长，适当增大 (原项目KoBERT用64)
    "hidden_size": 768,
    "num_classes": 2,         # 0=正常, 1=诈骗
    "dr_rate": 0.3,
    
    # 训练配置
    "batch_size": 16,         # Kaggle GPU 内存适配
    "num_epochs": 10,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "test_size": 0.15,
    "seed": 42,
    
    # 输出
    "model_save_dir": "/kaggle/working/model",
    "log_csv_path": "/kaggle/working/training_log.csv",
}

def set_seed(seed: int = 42):
    """固定随机种子，确保可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    logger.info(f"GPU: {props.name}")
    logger.info(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")

os.makedirs(CONFIG["model_save_dir"], exist_ok=True)

# %%
# # ===================== Cell 4: 数据集探索 =====================
# print("=" * 70)
# print("📂 Step 1: 探索 TeleAntiFraud-28k 数据集结构")
# print("=" * 70)

# data_root = Path(CONFIG["data_root"])
# print(f"\n数据集根目录: {data_root}")
# print(f"目录是否存在: {data_root.exists()}")

# if data_root.exists():
#     print(f"\n根目录下的文件和文件夹:")
#     for item in sorted(data_root.iterdir()):
#         if item.is_dir():
#             sub_count = len(list(item.iterdir()))
#             print(f"  📁 {item.name}/ ({sub_count} items)")
#         else:
#             size_mb = item.stat().st_size / (1024 * 1024)
#             print(f"  📄 {item.name} ({size_mb:.2f} MB)")
    
#     # 递归探索子目录 (最多2层)
#     print(f"\n递归目录结构 (前2层):")
#     for root_dir, dirs, files in os.walk(data_root):
#         depth = str(root_dir).replace(str(data_root), "").count(os.sep)
#         if depth > 2:
#             continue
#         indent = "  " * depth
#         dir_name = os.path.basename(root_dir)
#         print(f"{indent}📁 {dir_name}/")
#         if depth <= 1:
#             for f in sorted(files)[:10]:
#                 fpath = os.path.join(root_dir, f)
#                 size_mb = os.path.getsize(fpath) / (1024 * 1024)
#                 print(f"{indent}  📄 {f} ({size_mb:.2f} MB)")
#             if len(files) > 10:
#                 print(f"{indent}  ... 还有 {len(files) - 10} 个文件")

# %%
# ===================== Cell 5: 智能数据加载器 (重写) =====================
print("\n" + "=" * 70)
print("📊 Step 2: 加载和解析 TeleAntiFraud-28k 数据")
print("=" * 70)

class TeleAntiFraudDataLoader:
    """
    TeleAntiFraud-28k 数据集加载器 (适配真实数据结构)
    
    数据集结构:
    ├── merged_result/
    │   ├── NEG-xxx/             ← NEG开头 = 诈骗 (label=1)
    │   │   ├── tts_fraud_xxxxx/
    │   │   │   └── config.jsonl ← 单个JSON对象，包含 audio_segments 数组
    │   │   └── ...
    │   ├── POS-xxx/             ← POS开头 = 正常 (label=0)
    │   │   ├── tts_normal_xxxxx/
    │   │   │   └── config.jsonl
    │   │   └── ...
    ├── ASRtotal_train_clear_swift3.jsonl  ← 顶层训练JSONL文件
    ├── ASRtotal_test_swift3_demo.jsonl
    └── ...
    
    config.jsonl 格式 (虽然扩展名是.jsonl，但实际是单个JSON对象):
    {
      "audio_segments": [
        {"role": "left", "content": "...", "start_time": "...", "end_time": "..."},
        {"role": "right", "content": "...", ...},
        ...
      ],
      "terminated_by_manager": false,
      "termination_reason": "...",
      "terminator": "left"
    }
    """
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.merged_result_dir = self.data_root / "merged_result"
        self.all_texts = []
        self.all_labels = []
        self.metadata = []
        
    def _label_from_dirname(self, dirname: str) -> Optional[int]:
        """
        根据目录名判定标签:
        NEG-imitate-10, NEG-multi-agent-1 等 → 1 (诈骗)
        POS-imitate-4, POS-multi-agent-8 等 → 0 (正常)
        """
        upper = dirname.upper()
        if upper.startswith("NEG"):
            return 1  # 诈骗
        elif upper.startswith("POS"):
            return 0  # 正常
        return None
    
    def _parse_config_jsonl(self, filepath: Path) -> Optional[Tuple[str, dict]]:
        """
        解析单个 config.jsonl 文件
        
        将 audio_segments 中的对话按时间顺序拼接为完整文本。
        使用 "甲：" / "乙：" 标注不同说话人 (left/right)。
        
        Returns:
            (拼接后的对话文本, 元数据字典) 或 None
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            
            if not raw:
                return None
            
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # 极少数文件可能格式异常，跳过
            return None
        
        # 提取 audio_segments
        segments = data.get("audio_segments", [])
        if not segments:
            return None
        
        # 按对话顺序拼接文本
        text_parts = []
        total_duration = 0.0
        num_turns = 0
        
        for seg in segments:
            role = seg.get("role", "unknown")
            content = seg.get("content", "").strip()
            if not content:
                continue
            
            # 统一标注说话人
            if role == "left":
                speaker = "甲"
            elif role == "right":
                speaker = "乙"
            else:
                speaker = "未知"
            
            text_parts.append(f"{speaker}：{content}")
            num_turns += 1
            
            # 记录时长
            end_sec = seg.get("end_time_seconds", 0)
            if end_sec > total_duration:
                total_duration = end_sec
        
        if not text_parts:
            return None
        
        full_text = "\n".join(text_parts)
        
        # 过滤过短的文本 (< 10个字符的对话可能是噪声)
        if len(full_text) < 10:
            return None
        
        # 元数据
        meta = {
            "num_turns": num_turns,
            "duration_seconds": round(total_duration, 2),
            "text_length": len(full_text),
            "terminator": data.get("terminator", ""),
            "termination_reason": data.get("termination_reason", "")[:100],
            "terminated_by_manager": data.get("terminated_by_manager", False),
            "source_file": str(filepath),
        }
        
        return full_text, meta
    
    def load_merged_result(self) -> int:
        """
        加载 merged_result 目录下的所有 config.jsonl 文件
        
        目录结构:
        merged_result/
        ├── NEG-imitate-10/tts_fraud_00001/config.jsonl
        ├── NEG-imitate-10/tts_fraud_00002/config.jsonl
        ├── POS-imitate-4/tts_normal_00001/config.jsonl
        └── ...
        
        Returns:
            成功加载的Sample size
        """
        if not self.merged_result_dir.exists():
            logger.warning(f"merged_result 目录不存在: {self.merged_result_dir}")
            return 0
        
        count = 0
        category_stats = {}  # 统计每个子目录的加载情况
        
        # 遍历 merged_result 下的所有子目录 (NEG-xxx, POS-xxx)
        category_dirs = sorted([
            d for d in self.merged_result_dir.iterdir() 
            if d.is_dir()
        ])
        
        for cat_dir in category_dirs:
            label = self._label_from_dirname(cat_dir.name)
            if label is None:
                logger.warning(f"  ⚠️ 无法判定目录标签: {cat_dir.name}，跳过")
                continue
            
            label_text = "诈骗" if label == 1 else "正常"
            cat_count = 0
            cat_errors = 0
            
            # 遍历子目录下的所有样本目录 (tts_fraud_xxxxx, tts_normal_xxxxx)
            sample_dirs = sorted([
                d for d in cat_dir.iterdir()
                if d.is_dir()
            ])
            
            for sample_dir in sample_dirs:
                config_file = sample_dir / "config.jsonl"
                if not config_file.exists():
                    # 也尝试其他可能的文件名
                    config_candidates = list(sample_dir.glob("*.jsonl")) + list(sample_dir.glob("*.json"))
                    if config_candidates:
                        config_file = config_candidates[0]
                    else:
                        cat_errors += 1
                        continue
                
                result = self._parse_config_jsonl(config_file)
                if result is None:
                    cat_errors += 1
                    continue
                
                text, meta = result
                meta["category_dir"] = cat_dir.name
                meta["sample_id"] = sample_dir.name
                meta["label_source"] = f"目录名推断({cat_dir.name})"
                
                self.all_texts.append(text)
                self.all_labels.append(label)
                self.metadata.append(meta)
                cat_count += 1
                count += 1
            
            category_stats[cat_dir.name] = {
                "label": label_text,
                "loaded": cat_count,
                "errors": cat_errors,
                "total": len(sample_dirs)
            }
            logger.info(
                f"  📁 {cat_dir.name} [{label_text}]: "
                f"{cat_count}/{len(sample_dirs)} 样本加载成功"
                + (f" ({cat_errors} 个跳过)" if cat_errors else "")
            )
        
        # 汇总
        print(f"\n  ── merged_result 加载汇总 ──")
        for name, stats in category_stats.items():
            print(f"    {stats['label']} | {name}: {stats['loaded']} 样本")
        
        return count
    
    def load_top_level_jsonl(self) -> int:
        """
        加载顶层的 JSONL 文件 (如 ASRtotal_train_clear_swift3.jsonl 等)
        
        这些文件可能是为 Qwen2-Audio SFT 训练准备的，格式可能不同。
        我们尝试解析并提取可用的文本数据。
        
        Returns:
            成功加载的Sample size
        """
        count = 0
        jsonl_files = sorted(self.data_root.glob("*.jsonl"))
        
        if not jsonl_files:
            logger.info("  未发现顶层 JSONL 文件")
            return 0
        
        for jf in jsonl_files:
            file_count = 0
            size_mb = jf.stat().st_size / (1024 * 1024)
            logger.info(f"\n  📄 尝试加载: {jf.name} ({size_mb:.1f} MB)")
            
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                
                if not first_line:
                    continue
                
                # 预览第一行的结构
                try:
                    sample = json.loads(first_line)
                    logger.info(f"     字段: {list(sample.keys())[:10]}")
                except json.JSONDecodeError:
                    logger.warning(f"     首行不是合法JSON，跳过")
                    continue
                
                # 逐行解析
                with open(jf, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            
                            # 策略1: 如果有 audio_segments (与config.jsonl相同格式)
                            if "audio_segments" in data:
                                segments = data["audio_segments"]
                                parts = []
                                for seg in segments:
                                    role = seg.get("role", "unknown")
                                    content = seg.get("content", "").strip()
                                    if content:
                                        speaker = "甲" if role == "left" else "乙"
                                        parts.append(f"{speaker}：{content}")
                                if parts:
                                    text = "\n".join(parts)
                                    # 标签推断
                                    label = None
                                    tts_id = data.get("tts_id", "")
                                    if "fraud" in str(tts_id).lower():
                                        label = 1
                                    elif "normal" in str(tts_id).lower():
                                        label = 0
                                    # 从文件名推断
                                    if label is None:
                                        fname_lower = jf.name.lower()
                                        if "neg" in fname_lower:
                                            label = 1
                                        elif "pos" in fname_lower:
                                            label = 0
                                    if label is not None and len(text) >= 10:
                                        self.all_texts.append(text)
                                        self.all_labels.append(label)
                                        self.metadata.append({
                                            "source_file": jf.name,
                                            "line_num": line_num
                                        })
                                        file_count += 1
                                continue
                            
                            # 策略2: SFT训练格式 (可能有 query/response 或 messages)
                            # 这些文件可能是给LLM微调用的，格式类似:
                            # {"query": "分析以下对话...", "response": "这是诈骗...", ...}
                            # 或 {"messages": [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
                            
                            messages = data.get("messages", [])
                            query = data.get("query", "")
                            response = data.get("response", "")
                            
                            # 从 query/response 中提取对话文本
                            text_to_use = ""
                            label = None
                            
                            if messages:
                                # messages 格式
                                for msg in messages:
                                    content = msg.get("content", "")
                                    if len(content) > len(text_to_use):
                                        text_to_use = content
                            elif query:
                                text_to_use = query
                            
                            # 从response中推断标签
                            if response:
                                resp_lower = response.lower()
                                if any(kw in resp_lower for kw in ["诈骗", "fraud", "骗", "异常", "可疑"]):
                                    label = 1
                                elif any(kw in resp_lower for kw in ["正常", "normal", "合法", "日常"]):
                                    label = 0
                            
                            if text_to_use and label is not None and len(text_to_use) >= 10:
                                self.all_texts.append(text_to_use)
                                self.all_labels.append(label)
                                self.metadata.append({
                                    "source_file": jf.name,
                                    "line_num": line_num
                                })
                                file_count += 1
                                
                        except json.JSONDecodeError:
                            continue
                
                logger.info(f"     → 成功提取 {file_count} 个样本")
                count += file_count
                
            except Exception as e:
                logger.warning(f"     加载失败: {e}")
        
        return count
    
    def load_all(self) -> Tuple[List[str], List[int]]:
        """
        自动加载所有数据 (优先 merged_result, 然后顶层JSONL)
        """
        logger.info(f"数据集根目录: {self.data_root}")
        logger.info(f"merged_result 目录: {self.merged_result_dir}")
        logger.info(f"merged_result 存在: {self.merged_result_dir.exists()}")
        
        # ── 主要数据源: merged_result 目录 ──
        print(f"\n{'─'*50}")
        print(f"📁 加载 merged_result 目录 (主要数据源)")
        print(f"{'─'*50}")
        merged_count = self.load_merged_result()
        
        # ── 补充数据源: 顶层JSONL文件 ──
        print(f"\n{'─'*50}")
        print(f"📄 加载顶层 JSONL 文件 (补充数据源)")
        print(f"{'─'*50}")
        top_count = self.load_top_level_jsonl()
        
        # ── 最终汇总 ──
        total = len(self.all_texts)
        n_pos = self.all_labels.count(0)  # POS = 正常
        n_neg = self.all_labels.count(1)  # NEG = 诈骗
        
        print(f"\n{'='*50}")
        logger.info(f"✅ 数据加载完成!")
        logger.info(f"   merged_result: {merged_count} 个样本")
        logger.info(f"   顶层JSONL:     {top_count} 个样本")
        logger.info(f"   总计:          {total} 个样本")
        logger.info(f"   正常 (POS):    {n_pos}")
        logger.info(f"   诈骗 (NEG):    {n_neg}")
        
        if total == 0:
            logger.error("⚠️ 未加载到任何数据! 下面进行诊断...")
            self._diagnose()
        
        return self.all_texts, self.all_labels
    
    def _diagnose(self):
        """数据加载失败时的诊断函数"""
        print(f"\n🔍 诊断信息:")
        print(f"  数据根目录: {self.data_root}")
        print(f"  目录存在: {self.data_root.exists()}")
        
        if self.data_root.exists():
            print(f"  根目录内容:")
            for item in sorted(self.data_root.iterdir()):
                item_type = "📁" if item.is_dir() else "📄"
                print(f"    {item_type} {item.name}")
        
        if self.merged_result_dir.exists():
            print(f"\n  merged_result 目录内容:")
            for item in sorted(self.merged_result_dir.iterdir()):
                if item.is_dir():
                    sub_count = len(list(item.iterdir()))
                    label = self._label_from_dirname(item.name)
                    label_text = {0: "正常", 1: "诈骗", None: "未知"}.get(label, "?")
                    print(f"    📁 {item.name} [{label_text}] ({sub_count} 子目录)")
                    # 检查第一个子目录的结构
                    first_sub = next(item.iterdir(), None)
                    if first_sub and first_sub.is_dir():
                        sub_files = list(first_sub.iterdir())
                        print(f"       └─ 示例: {first_sub.name}/ → {[f.name for f in sub_files[:5]]}")
                        # 尝试读取第一个文件
                        for sf in sub_files:
                            if sf.name.endswith((".jsonl", ".json")):
                                try:
                                    with open(sf, "r", encoding="utf-8") as fh:
                                        content = fh.read()[:500]
                                    print(f"          内容预览: {content[:200]}...")
                                except Exception as e:
                                    print(f"          读取失败: {e}")
                                break

    def get_sample_preview(self, n: int = 3):
        """预览已加载的样本"""
        if not self.all_texts:
            print("  还没有加载任何数据")
            return
        
        for label_val, label_name in [(0, "正常"), (1, "诈骗")]:
            indices = [i for i, l in enumerate(self.all_labels) if l == label_val]
            print(f"\n  ── {label_name}样本预览 ({len(indices)} 个) ──")
            for idx in indices[:n]:
                text_preview = self.all_texts[idx][:200].replace("\n", " | ")
                meta = self.metadata[idx] if idx < len(self.metadata) else {}
                print(f"    [{idx}] {text_preview}...")
                if meta:
                    print(f"         目录: {meta.get('category_dir', 'N/A')} | "
                          f"样本: {meta.get('sample_id', 'N/A')} | "
                          f"轮次: {meta.get('num_turns', 'N/A')} | "
                          f"时长: {meta.get('duration_seconds', 'N/A')}s")


# ===================== Cell 6: 执行数据加载 =====================
loader = TeleAntiFraudDataLoader(CONFIG["data_root"])
texts, labels = loader.load_all()

# ===================== Cell 7: 数据统计与可视化 =====================
print("\n" + "=" * 70)
print("📈 Step 3: 数据统计与可视化")
print("=" * 70)

if len(texts) > 0:
    label_counts = Counter(labels)
    print(f"\n样本总数: {len(texts)}")
    print(f"正常样本 (0): {label_counts.get(0, 0)}")
    print(f"诈骗样本 (1): {label_counts.get(1, 0)}")
    
    # 文本长度分析
    text_lengths = [len(t) for t in texts]
    print(f"\n文本长度统计:")
    print(f"  最短: {min(text_lengths)} 字符")
    print(f"  最长: {max(text_lengths)} 字符")
    print(f"  平均: {np.mean(text_lengths):.0f} 字符")
    print(f"  中位数: {np.median(text_lengths):.0f} 字符")
    
    # 展示样本
    print(f"\n─── 正常对话样本 ───")
    normal_samples = [t for t, l in zip(texts, labels) if l == 0]
    if normal_samples:
        print(normal_samples[0][:300] + "..." if len(normal_samples[0]) > 300 else normal_samples[0])
    
    print(f"\n─── 诈骗对话样本 ───")
    fraud_samples = [t for t, l in zip(texts, labels) if l == 1]
    if fraud_samples:
        print(fraud_samples[0][:300] + "..." if len(fraud_samples[0]) > 300 else fraud_samples[0])
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 标签分布
    ax1 = axes[0]
    bars = ax1.bar(["Normal (0)", "Scam (1)"], 
                   [label_counts.get(0, 0), label_counts.get(1, 0)],
                   color=["#2ecc71", "#e74c3c"])
    ax1.set_title("Label distribution", fontsize=14)
    ax1.set_ylabel("Sample size")
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12)
    
    # 文本长度分布
    ax2 = axes[1]
    ax2.hist(text_lengths, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    ax2.set_title("Distribution of text length", fontsize=14)
    ax2.set_xlabel("Character count")
    ax2.set_ylabel("Sample size")
    ax2.axvline(x=CONFIG["max_len"] * 2, color="red", linestyle="--", 
                label=f"~max_len*2={CONFIG['max_len']*2}")
    ax2.legend()
    
    # 各类Distribution of Fraud Types (如果有metadata)
    ax3 = axes[2]
    if loader.metadata:
        fraud_types = [m.get("fraud_type", "unknown") for m, l in zip(loader.metadata, labels) if l == 1]
        fraud_types = [ft for ft in fraud_types if ft and str(ft) not in ("", "None", "unknown")]
        if fraud_types:
            ft_counts = Counter(fraud_types).most_common(10)
            ft_names, ft_vals = zip(*ft_counts)
            ax3.barh(range(len(ft_names)), ft_vals, color="#9b59b6", alpha=0.7)
            ax3.set_yticks(range(len(ft_names)))
            ax3.set_yticklabels(ft_names)
            ax3.set_title("Distribution of Fraud Types (Top 10)", fontsize=14)
            ax3.set_xlabel("Sample size")
        else:
            ax3.text(0.5, 0.5, "No fraud type metadata", ha="center", va="center", fontsize=12)
            ax3.set_title("Distribution of Fraud Types", fontsize=14)
    else:
        ax3.text(0.5, 0.5, "No metadata", ha="center", va="center", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("/kaggle/working/data_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 分布图已保存到 /kaggle/working/data_distribution.png")
else:
    print("⚠️ 未能加载到任何数据！请检查数据集路径和格式。")
    print("尝试手动检查数据目录...")
    for root, dirs, files in os.walk(CONFIG["data_root"]):
        for f in files[:5]:
            full_path = os.path.join(root, f)
            print(f"  文件: {full_path}")
            if f.endswith((".jsonl", ".json")):
                with open(full_path, "r", encoding="utf-8") as fh:
                    first_line = fh.readline().strip()
                    print(f"    首行: {first_line[:200]}")

# %%
# ===================== Cell 8: 数据集划分 =====================
print("\n" + "=" * 70)
print("✂️ Step 4: 数据集划分 (训练集/验证集)")
print("=" * 70)

assert len(texts) > 0, "没有加载到任何数据！请检查上方的数据探索和加载步骤。"

# ===================== Patch: 去重（防止 train/val 重复） =====================
def normalize_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[，。！？；：,.!?;:（）()\[\]{}【】\"“”‘’·…—\-_/\\|<>~`@#$%^&*+=]", "", s)
    # 可选：弱化说话人标记差异（保留内容）
    s = re.sub(r"(甲|乙)[:：]", "", s)
    return s

df = pd.DataFrame({"text": texts, "label": labels})
df["norm"] = df["text"].apply(normalize_text)

# 检查是否存在“同一norm对应不同label”（若有，建议直接丢弃这些冲突样本）
conflict_norms = df.groupby("norm")["label"].nunique()
conflict_norms = set(conflict_norms[conflict_norms > 1].index)
conflict_cnt = len(conflict_norms)

if conflict_cnt > 0:
    print(f"⚠️ 发现 {conflict_cnt} 个 norm-label 冲突组（同一文本归一化后对应多个标签），将丢弃这些组以避免脏标注。")
    df = df[~df["norm"].isin(conflict_norms)].copy()

before = len(df)
df = df.drop_duplicates(subset=["norm"], keep="first").reset_index(drop=True)
after = len(df)
print(f"🧹 Dedup: {before} -> {after}（移除重复 {before-after} 条），冲突组丢弃：{conflict_cnt}")

# 覆盖回训练数据
texts = df["text"].tolist()
labels = df["label"].tolist()
# ===================== Patch End =====================

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["seed"],
    stratify=labels  # 保持正负样本比例
)

print(f"训练集: {len(train_texts)} 样本 (正常: {train_labels.count(0)}, 诈骗: {train_labels.count(1)})")
print(f"验证集: {len(val_texts)} 样本 (正常: {val_labels.count(0)}, 诈骗: {val_labels.count(1)})")

# ===================== Cell 9: 模型定义 =====================
print("\n" + "=" * 70)
print("🏗️ Step 5: 模型定义 (改造自原项目 KoBERTClassifier)")
print("=" * 70)

# ───── 改造自原项目 KoBERTModel/train.py 中的 KoBERTClassifier ─────
# 唯一区别: 类名改为 ChineseBERTClassifier, 注释改为中文
# 模型结构完全一致: BERT → Dropout → Linear(768, 2)

class ChineseBERTClassifier(nn.Module):
    """
    中文BERT诈骗分类器
    结构与原项目 KoBERTClassifier 完全一致:
    BERT pooler_output → Dropout → Linear(hidden_size, num_classes)
    """
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
            return_dict=True
        )
        # 原项目同样的防御式写法
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS]
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


# ───── 改造自原项目 KoBERTModel/train.py 中的 KoBERTDataset ─────
# 区别: 适配 bert-base-chinese 的 tokenizer

class FraudTextDataset(Dataset):
    """
    文本分类数据集
    改造自原项目 KoBERTDataset, 适配 bert-base-chinese tokenizer
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 256):
        self.texts = [str(t) for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

print("✅ ChineseBERTClassifier 模型定义完成 (结构与原KoBERTClassifier一致)")
print("✅ FraudTextDataset 数据集类定义完成")

# ===================== Cell 10: 初始化模型和数据 =====================
print("\n" + "=" * 70)
print("⚙️ Step 6: 初始化模型、Tokenizer、DataLoader")
print("=" * 70)

# 加载 bert-base-chinese (替代原项目的 monologg/kobert)
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
bert = AutoModel.from_pretrained(CONFIG["model_name"])

model = ChineseBERTClassifier(
    bert,
    hidden_size=CONFIG["hidden_size"],
    num_classes=CONFIG["num_classes"],
    dr_rate=CONFIG["dr_rate"]
).to(device)

# 统计模型参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")

# 创建数据集和DataLoader
train_dataset = FraudTextDataset(train_texts, train_labels, tokenizer, CONFIG["max_len"])
val_dataset = FraudTextDataset(val_texts, val_labels, tokenizer, CONFIG["max_len"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

print(f"训练 DataLoader: {len(train_loader)} batches")
print(f"验证 DataLoader: {len(val_loader)} batches")

# 处理类别不平衡
label_counts = Counter(train_labels)
if label_counts[0] > 0 and label_counts[1] > 0:
    weight_for_0 = len(train_labels) / (2 * label_counts[0])
    weight_for_1 = len(train_labels) / (2 * label_counts[1])
    class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"类别权重: 正常={weight_for_0:.4f}, 诈骗={weight_for_1:.4f}")
else:
    criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

# 学习率调度器
total_steps = len(train_loader) * CONFIG["num_epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=CONFIG["learning_rate"],
    total_steps=total_steps, pct_start=CONFIG["warmup_ratio"]
)
print(f"学习率调度: OneCycleLR (warmup {warmup_steps} steps, total {total_steps} steps)")

# ===================== Cell 11: 训练循环 =====================
print("\n" + "=" * 70)
print("🚀 Step 7: 开始训练!")
print("=" * 70)

training_logs = []
best_val_f1 = -1.0
best_epoch = -1

for epoch in range(CONFIG["num_epochs"]):
    # ─── 训练阶段 ───
    model.train()
    total_loss = 0.0
    train_preds, train_targets = [], []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [训练]", unit="batch")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels_t = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels_t)
        loss.backward()
        
        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        train_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
        train_targets.extend(labels_t.cpu().tolist())
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})

    train_loss = total_loss / max(1, len(train_loader))
    train_acc = accuracy_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds, zero_division=0)

    # ─── 验证阶段 ───
    model.eval()
    val_preds, val_targets, val_probs = [], [], []
    val_loss_total = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']} [验证]", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels_t = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels_t)
            val_loss_total += loss.item()
            
            probs = F.softmax(logits, dim=-1)
            val_probs.extend(probs[:, 1].cpu().tolist())
            val_preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            val_targets.extend(labels_t.cpu().tolist())

    val_loss = val_loss_total / max(1, len(val_loader))
    val_acc = accuracy_score(val_targets, val_preds)
    val_precision = precision_score(val_targets, val_preds, zero_division=0)
    val_recall = recall_score(val_targets, val_preds, zero_division=0)
    val_f1 = f1_score(val_targets, val_preds, zero_division=0)

    # 日志
    current_lr = scheduler.get_last_lr()[0]
    log_entry = {
        "Epoch": epoch + 1,
        "Train Loss": f"{train_loss:.4f}",
        "Train Acc": f"{train_acc:.4f}",
        "Train F1": f"{train_f1:.4f}",
        "Val Loss": f"{val_loss:.4f}",
        "Val Acc": f"{val_acc:.4f}",
        "Val Precision": f"{val_precision:.4f}",
        "Val Recall": f"{val_recall:.4f}",
        "Val F1": f"{val_f1:.4f}",
        "LR": f"{current_lr:.6f}",
    }
    training_logs.append(log_entry)

    print(f"\n📊 Epoch {epoch+1}/{CONFIG['num_epochs']}:")
    print(f"   Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"   Valid - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | P: {val_precision:.4f} | R: {val_recall:.4f} | F1: {val_f1:.4f}")

    # 保存最佳模型 (基于 Val F1, 与原项目 train_deepvoice.py 一致的策略)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        best_model_path = os.path.join(CONFIG["model_save_dir"], "best_model.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"   ✅ 最佳模型已保存! Val F1: {val_f1:.4f} (Epoch {epoch+1})")

# 保存最终模型 (与原项目 KoBERTModel/train.py 对应的 train.pt)
final_model_path = os.path.join(CONFIG["model_save_dir"], "train.pt")
torch.save(model.state_dict(), final_model_path)
print(f"\n💾 最终模型已保存: {final_model_path}")
print(f"💾 最佳模型: {best_model_path} (Epoch {best_epoch}, F1: {best_val_f1:.4f})")

# 保存训练日志
log_df = pd.DataFrame(training_logs)
log_df.to_csv(CONFIG["log_csv_path"], index=False, encoding="utf-8-sig")
print(f"📝 训练日志已保存: {CONFIG['log_csv_path']}")

# %%
# ===================== Cell 12: 训练结果可视化 =====================
print("\n" + "=" * 70)
print("📉 Step 8: 训练结果可视化")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

epochs_range = range(1, len(training_logs) + 1)
train_losses = [float(l["Train Loss"]) for l in training_logs]
val_losses = [float(l["Val Loss"]) for l in training_logs]
train_accs = [float(l["Train Acc"]) for l in training_logs]
val_accs = [float(l["Val Acc"]) for l in training_logs]
train_f1s = [float(l["Train F1"]) for l in training_logs]
val_f1s = [float(l["Val F1"]) for l in training_logs]

# Loss 曲线
axes[0, 0].plot(epochs_range, train_losses, "b-o", label="Train Loss")
axes[0, 0].plot(epochs_range, val_losses, "r-o", label="Val Loss")
axes[0, 0].set_title("Loss Curve")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy 曲线
axes[0, 1].plot(epochs_range, train_accs, "b-o", label="Train Acc")
axes[0, 1].plot(epochs_range, val_accs, "r-o", label="Val Acc")
axes[0, 1].set_title("Accuracy Curve")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# F1 Score 曲线
axes[1, 0].plot(epochs_range, train_f1s, "b-o", label="Train F1")
axes[1, 0].plot(epochs_range, val_f1s, "r-o", label="Val F1")
axes[1, 0].axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5, label=f"Best (Epoch {best_epoch})")
axes[1, 0].set_title("F1 Score Curve")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("F1")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 混淆矩阵 (最后一个epoch的验证集)
cm = confusion_matrix(val_targets, val_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1],
            xticklabels=["Normal", "Scam"], yticklabels=["Normal", "Scam"])
axes[1, 1].set_title("Confusion matrix (Val)")
axes[1, 1].set_ylabel("True label")
axes[1, 1].set_xlabel("Predict label")

plt.suptitle("Chinese BERT", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("/kaggle/working/training_results.png", dpi=150, bbox_inches="tight")
plt.show()

# 详细分类报告
print("\n📋 详细分类报告 (验证集):")
print(classification_report(val_targets, val_preds, target_names=["Normal", "Scam"]))

# ===================== Cell 13: 推理测试 =====================
print("\n" + "=" * 70)
print("🧪 Step 9: 推理测试 (模拟本地 predict.py 的功能)")
print("=" * 70)

def predict_single(text: str, model, tokenizer, device, max_len=256):
    """
    单文本推理函数
    改造自原项目 KoBERTModel/predict.py 的 predict() 函数
    返回格式与原项目完全兼容: {"text", "llm_score", "phishing_detected"}
    """
    model.eval()
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).to(device)
        
        logits = model(input_ids, attention_mask, token_type_ids)
        probs = F.softmax(logits, dim=-1)
        phishing_prob = probs[0][1].item()
    
    score = round(phishing_prob * 100, 2)
    is_phishing = score > 50
    
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "llm_score": score,           # 0~100, 与原项目一致
        "phishing_detected": is_phishing,
        "final_label": "诈骗" if is_phishing else "正常"
    }


# 测试样本
test_samples = [
    "甲：喂，你好，这里是银行信贷中心，最近我们推出了一个特别的无抵押贷款，利率非常低，您有兴趣了解一下吗？\n乙：哦，是吗？说来听听。\n甲：只需要您提供身份证号和银行卡号，我们就能帮您快速审批。",
    "甲：你好，我是你们小区物业的，请问今天下午方便维修水管吗？\n乙：可以的，大概几点来？\n甲：下午两点左右，麻烦在家等一下。",
    "甲：你好，你的手机号码中了我们的大奖，奖金50万元，需要你先缴纳5000元手续费。\n乙：真的假的？怎么缴费？\n甲：请转账到这个账户...",
    "甲：喂，妈，我手机掉水里了，用同事的手机给你打的，你先转两万块钱到这个卡号上，我急用。",
]

print("\n推理结果:")
for i, sample in enumerate(test_samples, 1):
    result = predict_single(sample, model, tokenizer, device, CONFIG["max_len"])
    emoji = "🚨" if result["phishing_detected"] else "✅"
    print(f"\n{emoji} 测试 {i}: {result['final_label']} (分数: {result['llm_score']})")
    print(f"   文本: {result['text']}")

# ===================== OOD 测试模板 =====================
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# 你需要手工补充更多样例：建议每类 30~100 条
# 关键：尽量避免直球关键词（贷款/转账/验证码/链接/中奖/APP/下载…），用更隐蔽表达
# 每类 50 条 OOD 样例（label: 1=诈骗, 0=正常）
OOD_SAMPLES = [
    # ===================== 诈骗（label=1）50条：隐蔽话术/弱关键词 =====================
    {"label": 1, "text": "甲：您好，这里是账户核验中心，需要您配合做一次身份确认流程。\n乙：为什么要确认？\n甲：系统提示异常使用痕迹，不确认会影响后续服务。"},
    {"label": 1, "text": "甲：我们这边收到一条风险提示，您的资料需要补全核验。\n乙：我没办过什么。\n甲：可能信息被冒用，您先按流程处理，避免后续麻烦。"},
    {"label": 1, "text": "甲：您好，您名下有一项待确认事项，今天必须完成。\n乙：什么事项？\n甲：线上合规校验，拖延会触发限制。"},
    {"label": 1, "text": "甲：这边是服务安全专员，检测到您近期操作与常用习惯不一致。\n乙：那怎么办？\n甲：需要您立即配合一次验证。"},
    {"label": 1, "text": "甲：您好，我们要对您的信息做例行复核。\n乙：怎么复核？\n甲：按照我说的步骤逐项确认即可，过程很快。"},
    {"label": 1, "text": "甲：您现在方便吗？我们这边有一条紧急提示需要当场处理。\n乙：什么提示？\n甲：涉及您名下服务状态，稍后可能会被暂停。"},
    {"label": 1, "text": "甲：您好，您名下有一笔异常关联记录。\n乙：我不清楚。\n甲：先别着急，我们引导您做一次线上排查。"},
    {"label": 1, "text": "甲：我们检测到您资料存在缺失项。\n乙：缺什么？\n甲：需要您补充确认，否则影响后续办理。"},
    {"label": 1, "text": "甲：这边是合规复核岗，今天要完成一次抽检。\n乙：抽检什么？\n甲：核对您本人意愿与信息一致性。"},
    {"label": 1, "text": "甲：您好，您近期有一条可疑行为记录。\n乙：我没有啊。\n甲：有可能是他人代操作，您配合核对就能排除。"},
    {"label": 1, "text": "甲：请您确认一下最近是否委托他人处理过相关事项。\n乙：没有。\n甲：那需要立即做一次保护性确认，防止误判。"},
    {"label": 1, "text": "甲：您好，我们这边要对您名下资料做一次升级校验。\n乙：升级校验是什么？\n甲：把关键项重新确认一遍，避免系统锁定。"},
    {"label": 1, "text": "甲：您这边收到提示了吗？\n乙：什么提示？\n甲：我们刚发起了安全核验，您需要当场配合完成。"},
    {"label": 1, "text": "甲：您好，您名下出现一条跨地区使用记录。\n乙：我没去过。\n甲：那就是高风险信号，必须马上确认本人状态。"},
    {"label": 1, "text": "甲：我们需要您提供一次实时确认。\n乙：确认什么？\n甲：确认您本人正在接听并同意本次处理。"},
    {"label": 1, "text": "甲：这边提示您服务将进入限制状态。\n乙：为什么？\n甲：未完成核验会触发保护机制。"},
    {"label": 1, "text": "甲：您方便开一下免提吗？我们要做语音一致性确认。\n乙：这也要确认？\n甲：是的，合规要求。"},
    {"label": 1, "text": "甲：我们这边需要您复述一段确认语。\n乙：复述什么？\n甲：按我说的念一遍即可，用于留痕。"},
    {"label": 1, "text": "甲：您好，您名下资料被标记为需复核。\n乙：谁标记的？\n甲：系统自动标记，我们只负责协助处理。"},
    {"label": 1, "text": "甲：您现在不要挂断，我们要完成最后一步确认。\n乙：我有事。\n甲：这一步不做会直接进入限制流程。"},
    {"label": 1, "text": "甲：您好，我们将对您的信息做一次紧急更正。\n乙：更正什么？\n甲：系统里有不一致项，需要您口头确认。"},
    {"label": 1, "text": "甲：您是否授权他人使用过您的资料？\n乙：没有。\n甲：那我们需要立即启动保护性核验。"},
    {"label": 1, "text": "甲：您好，您这边出现异常投诉记录。\n乙：我没投诉。\n甲：可能被冒用，需要您配合确认来源。"},
    {"label": 1, "text": "甲：我们这边需要核对您近期是否进行过线下操作。\n乙：没有。\n甲：那就很可疑，必须按流程排查。"},
    {"label": 1, "text": "甲：您好，我们要求您在电话里完成一次确认，不然无法解除提示。\n乙：我不太明白。\n甲：您按步骤做就行。"},
    {"label": 1, "text": "甲：这边是风险处理专员，已为您开启临时处置通道。\n乙：处置什么？\n甲：排除误触发的限制。"},
    {"label": 1, "text": "甲：您好，您这边有一条待处理的安全事件。\n乙：什么事件？\n甲：涉及资料一致性，我们需要快速核验。"},
    {"label": 1, "text": "甲：我们要确认您本人是否知情。\n乙：知情什么？\n甲：知情并同意本次线上处理流程。"},
    {"label": 1, "text": "甲：请您把刚才的提示内容读出来。\n乙：我没看到。\n甲：那我们需要重新触发一次核验步骤。"},
    {"label": 1, "text": "甲：您好，我们这边要完成一次“本人确认”。\n乙：怎么确认？\n甲：回答几个问题并按指引操作即可。"},
    {"label": 1, "text": "甲：您名下服务状态异常。\n乙：异常是什么？\n甲：可能被他人触发，需要您马上自证。"},
    {"label": 1, "text": "甲：为避免误伤，我们需要您提供关键项核对。\n乙：我不想提供。\n甲：不配合会被系统判定为高风险。"},
    {"label": 1, "text": "甲：这边要做一次“快速确认”。\n乙：确认什么？\n甲：确认您本人信息与记录一致。"},
    {"label": 1, "text": "甲：您好，我们要对您的资料进行重新登记。\n乙：为什么要登记？\n甲：系统升级，未登记会影响使用。"},
    {"label": 1, "text": "甲：我们这边提示您存在异常关联。\n乙：关联什么？\n甲：关联到一条风险事件，需要您说明。"},
    {"label": 1, "text": "甲：您好，您近期操作触发了保护机制。\n乙：我做了什么？\n甲：先别纠结原因，先完成确认流程。"},
    {"label": 1, "text": "甲：请您保持通话，我们要远程协助完成排查。\n乙：排查什么？\n甲：排查是否存在被冒用风险。"},
    {"label": 1, "text": "甲：我们会引导您做一次‘自检’。\n乙：自检怎么做？\n甲：按步骤逐项核对即可。"},
    {"label": 1, "text": "甲：您好，您名下有一条待签收的线上确认。\n乙：我没签过。\n甲：现在电话里做一下确认就行。"},
    {"label": 1, "text": "甲：我们要求您在十分钟内完成确认。\n乙：这么急？\n甲：超过时限会进入限制流程。"},
    {"label": 1, "text": "甲：您好，您是否收到系统提示“需复核”？\n乙：没有。\n甲：那可能延迟了，我们这边先做口头核对。"},
    {"label": 1, "text": "甲：您名下记录显示有人尝试更改资料。\n乙：不是我。\n甲：那就需要您立刻配合确认并阻断。"},
    {"label": 1, "text": "甲：我们需要确认您是否本人在场。\n乙：我本人。\n甲：那请按我说的流程完成验证。"},
    {"label": 1, "text": "甲：您好，您这边存在被误判风险。\n乙：误判什么？\n甲：误判为异常用户，您配合核验即可解除。"},
    {"label": 1, "text": "甲：这边是后台处理岗，已为您开通临时通道。\n乙：通道做什么？\n甲：用于完成一次紧急确认。"},
    {"label": 1, "text": "甲：您好，您需要确认一份线上声明。\n乙：声明是什么？\n甲：声明由您本人发起并同意处理。"},
    {"label": 1, "text": "甲：请您把身份信息按顺序核对一下。\n乙：为什么要核对？\n甲：不核对会导致服务异常持续。"},
    {"label": 1, "text": "甲：您好，我们提示您资料存在风险。\n乙：怎么解决？\n甲：按引导完成确认，风险会自动解除。"},
    {"label": 1, "text": "甲：这边要做一次合规回访。\n乙：回访什么？\n甲：确认您是否授权过相关处理。"},
    {"label": 1, "text": "甲：您好，我们检测到您名下有未完成的核验步骤。\n乙：我没做过。\n甲：那就更要尽快完成，避免受限。"},
    {"label": 1, "text": "甲：请您准备好进行信息核对。\n乙：现在吗？\n甲：是的，今天必须完成，否则系统自动限制。"},
    {"label": 1, "text": "甲：您好，我们这边提示需要立即更新您的资料。\n乙：更新什么？\n甲：关键项重新确认，防止被冒用。"},
    {"label": 1, "text": "甲：我们这边将进行一次安全确认录音。\n乙：录音干嘛？\n甲：用于留存处理依据，您配合回答即可。"},

    # ===================== 正常（label=0）50条：日常对话 =====================
    {"label": 0, "text": "甲：你好，我是物业，明天上午要检查消防栓。\n乙：大概几点？\n甲：九点半左右，会提前敲门。"},
    {"label": 0, "text": "甲：你下班了吗？\n乙：快了。\n甲：那晚饭想吃什么，我路上买点。"},
    {"label": 0, "text": "甲：你好，这里是快递站，您的包裹到啦。\n乙：我现在不在家。\n甲：那我帮您放到柜子里。"},
    {"label": 0, "text": "甲：老师说周五要交作业。\n乙：我还没写完。\n甲：我们晚上一起对一下思路。"},
    {"label": 0, "text": "甲：我把会议纪要整理好了。\n乙：发我一下。\n甲：好，我稍后同步到群里。"},
    {"label": 0, "text": "甲：明天你有空吗？\n乙：上午可以。\n甲：那我们去医院复查一下。"},
    {"label": 0, "text": "甲：你的车明天保养？\n乙：对，约了十点。\n甲：我送你过去吧。"},
    {"label": 0, "text": "甲：你好，我是维修师傅，今天能上门吗？\n乙：可以，下午两点。\n甲：好的我准时到。"},
    {"label": 0, "text": "甲：你看下我发的文档有没有问题。\n乙：我现在打开。\n甲：主要看第二页的数据。"},
    {"label": 0, "text": "甲：晚点一起跑步吗？\n乙：可以，七点楼下。\n甲：好，那我提前热身。"},
    {"label": 0, "text": "甲：你今天加班不？\n乙：不加。\n甲：那我订个位置，我们吃火锅。"},
    {"label": 0, "text": "甲：这周末回家吗？\n乙：回，周六早上走。\n甲：我帮你看看车次。"},
    {"label": 0, "text": "甲：你好，这里是门诊，您预约的号需要改期吗？\n乙：我想改到下周。\n甲：好的，我帮您登记。"},
    {"label": 0, "text": "甲：你记得带伞，今天可能下雨。\n乙：好，我出门前拿。\n甲：路上慢点。"},
    {"label": 0, "text": "甲：我把照片都传上去了。\n乙：我看到了。\n甲：你挑几张发朋友圈吧。"},
    {"label": 0, "text": "甲：你好，老师通知下周一测验。\n乙：测什么内容？\n甲：主要是前两章。"},
    {"label": 0, "text": "甲：你中午吃啥？\n乙：食堂随便吃点。\n甲：我给你带杯咖啡。"},
    {"label": 0, "text": "甲：我到楼下了。\n乙：你上来吧。\n甲：好的，我进电梯了。"},
    {"label": 0, "text": "甲：你昨天说的那个问题我查到了。\n乙：怎么解决？\n甲：把参数改一下就行。"},
    {"label": 0, "text": "甲：你好，今天能来装宽带吗？\n乙：可以，下午三点。\n甲：我会在家等。"},
    {"label": 0, "text": "甲：这周项目进度怎么样？\n乙：差不多完成一半。\n甲：周五前我们再对齐一次。"},
    {"label": 0, "text": "甲：明天一起去图书馆吗？\n乙：可以，下午两点。\n甲：那门口见。"},
    {"label": 0, "text": "甲：我把发票整理好了。\n乙：辛苦了。\n甲：你看下金额对不对。"},
    {"label": 0, "text": "甲：你到了没？\n乙：快到了，堵车。\n甲：不急，到了发我。"},
    {"label": 0, "text": "甲：晚上要不要看电影？\n乙：可以。\n甲：你选个场次。"},
    {"label": 0, "text": "甲：你好，我是社区志愿者，明天有垃圾分类宣传。\n乙：几点开始？\n甲：九点到十一点。"},
    {"label": 0, "text": "甲：你能帮我看一下简历吗？\n乙：可以。\n甲：我把文件发你。"},
    {"label": 0, "text": "甲：我今天忘带钥匙了。\n乙：我在家，你回来敲门。\n甲：好，我十分钟到。"},
    {"label": 0, "text": "甲：今天课程作业要交到哪？\n乙：交到系统里。\n甲：我晚点提交。"},
    {"label": 0, "text": "甲：你好，您订的外卖到门口了。\n乙：放门口就行。\n甲：好的，祝您用餐愉快。"},
    {"label": 0, "text": "甲：你把那份表格发我一下。\n乙：马上。\n甲：我需要核对一下数字。"},
    {"label": 0, "text": "甲：周末去爬山吗？\n乙：可以。\n甲：那我准备点水和零食。"},
    {"label": 0, "text": "甲：你今天体检结果出来了吗？\n乙：出来了，没啥大问题。\n甲：那就好。"},
    {"label": 0, "text": "甲：你好，明天能上门清洗空调吗？\n乙：可以，上午十点。\n甲：好的，我在家等。"},
    {"label": 0, "text": "甲：我把资料都整理在共享盘了。\n乙：收到。\n甲：你看完给我反馈。"},
    {"label": 0, "text": "甲：你周五晚上有安排吗？\n乙：没有。\n甲：那一起吃饭聊聊近况。"},
    {"label": 0, "text": "甲：你好，您的挂号需要确认一下时间。\n乙：我想改到周三。\n甲：好的我帮您改。"},
    {"label": 0, "text": "甲：我刚到地铁站。\n乙：我也快到了。\n甲：那我们出口见。"},
    {"label": 0, "text": "甲：你今天记得带学生证。\n乙：带了。\n甲：进馆要用。"},
    {"label": 0, "text": "甲：我把上次的笔记发你了。\n乙：看到了。\n甲：有不懂的我们再讨论。"},
    {"label": 0, "text": "甲：明天早上几点出发？\n乙：七点半。\n甲：那我七点来接你。"},
    {"label": 0, "text": "甲：你好，这里是客服回访，想确认一下你上次的工单是否解决。\n乙：解决了。\n甲：好的，祝您生活愉快。"},
    {"label": 0, "text": "甲：你把门口的快递拿一下。\n乙：好。\n甲：小心别压坏。"},
    {"label": 0, "text": "甲：今天晚上要不要一起复习？\n乙：可以。\n甲：那我带上书。"},
    {"label": 0, "text": "甲：你这周有空帮我搬个东西吗？\n乙：可以，周六。\n甲：谢谢，到时候请你吃饭。"},
    {"label": 0, "text": "甲：我把合同打印好了。\n乙：我晚点签。\n甲：好的。"},
    {"label": 0, "text": "甲：你好，您预约的理发需要改时间吗？\n乙：改到明天下午。\n甲：没问题。"},
    {"label": 0, "text": "甲：你今天早点休息。\n乙：好。\n甲：明天还要早起。"},
    {"label": 0, "text": "甲：我刚把报告提交了。\n乙：辛苦。\n甲：等反馈出来我再同步。"},
    {"label": 0, "text": "甲：周末我可能回趟老家。\n乙：路上注意安全。\n甲：好，我到了给你消息。"},
    {"label": 0, "text": "甲：你要的资料我整理成表了。\n乙：太好了。\n甲：我发你邮箱。"},
    {"label": 0, "text": "甲：你好，今天下午能送货吗？\n乙：可以，三点左右到。\n甲：好的我等你。"},
    {"label": 0, "text": "甲：你晚点来我这拿一下东西。\n乙：行。\n甲：我放门口给你。"},
]

def predict_label(text: str):
    r = predict_single(text, model, tokenizer, device, CONFIG["max_len"])
    # 你原逻辑：score>50 判诈骗
    return 1 if r["phishing_detected"] else 0, r["llm_score"]

y_true, y_pred, scores = [], [], []
for ex in OOD_SAMPLES:
    pred, score = predict_label(ex["text"])
    y_true.append(ex["label"])
    y_pred.append(pred)
    scores.append(score)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print("\n========== OOD 测试结果 ==========")
print(f"OOD Acc={acc:.4f} | F1={f1:.4f}")
print("Confusion:\n", confusion_matrix(y_true, y_pred))
print("\nReport:\n", classification_report(y_true, y_pred, target_names=["正常", "诈骗"]))

# 打印错例（方便你改进样例/阈值/模型）
print("\n========== OOD 错例 ==========")
for i, ex in enumerate(OOD_SAMPLES):
    if y_true[i] != y_pred[i]:
        print("-" * 60)
        print(f"GT={y_true[i]} PRED={y_pred[i]} SCORE={scores[i]}")
        print(ex["text"][:300], "..." if len(ex["text"]) > 300 else "")


# ===================== Cell 14: 保存配置 & 下载指引 =====================
print("\n" + "=" * 70)
print("📦 Step 10: 保存配置 & 模型下载指引")
print("=" * 70)

# 保存训练配置 (方便本地加载时参考)
config_save = {
    "model_name": CONFIG["model_name"],
    "max_len": CONFIG["max_len"],
    "hidden_size": CONFIG["hidden_size"],
    "num_classes": CONFIG["num_classes"],
    "dr_rate": CONFIG["dr_rate"],
    "best_val_f1": best_val_f1,
    "best_epoch": best_epoch,
    "total_train_samples": len(train_texts),
    "total_val_samples": len(val_texts),
}

config_path = os.path.join(CONFIG["model_save_dir"], "training_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config_save, f, indent=2, ensure_ascii=False)

print(f"✅ 训练配置已保存: {config_path}")
print(f"\n📁 输出文件列表:")
for item in Path("/kaggle/working").rglob("*"):
    if item.is_file():
        size_mb = item.stat().st_size / (1024 * 1024)
        print(f"   {item.relative_to('/kaggle/working')} ({size_mb:.2f} MB)")

print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🎯 训练完成! 请下载以下文件到本地项目:                      ║
║                                                              ║
║  1. model/best_model.pt    → ChineseBERTModel/model/train.pt ║
║  2. model/train.pt         → 备用 (最后epoch)                ║
║  3. model/training_config.json → 配置参考                    ║
║  4. training_log.csv       → 训练日志                        ║
║                                                              ║
║  本地 predict.py 中需要将:                                   ║
║    monologg/kobert → bert-base-chinese                       ║
║    KoBERTClassifier → ChineseBERTClassifier                  ║
╚══════════════════════════════════════════════════════════════╝
""")

print("🏁 Notebook 运行完毕!")


