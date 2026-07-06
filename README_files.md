# README_files.md

本文档用于论文写作阶段快速定位项目关键文件。它不是完整开发文档，而是一个 thesis-oriented file map：只记录对方法、系统实现、实验设计、结果解释和复现有用的文件。

## 使用边界

- 论文主线优先围绕 `baseline` 动态风险追踪、Chinese BERT 文本风险、CNN-BiLSTM 声学风险、late fusion、smoothing、controlled evaluation 和 learned late fusion 变体展开。
- `gated_v1` / `gated_v2` / `gated_v3`、`progression_v1` 属于历史实验或 engineering ablation。它们可以作为内部探索背景，但不建议写入论文主方法，也不作为本索引的重点。
- `test_samples/metadata_final.csv` 是目前系统测试使用的样本索引；`test_samples/audio_final/` 是与之配套的当前样本音频目录。
- Comparison Across Different Windows and Steps 全量测试已完成，当前可用三组 baseline 结果：`final_baseline_w5_s2p5`、`final_baseline_w10_s5`、`final_baseline_w20_s10`。
- `metadata.csv`、`metadata_long.csv`、`test_samples/audio_long/`、`test_samples/audio/`、`test_samples/audio_fake/` 和 `long_baseline_w10_s5` 属于旧评估文件或旧结果，暂不纳入当前论文索引。
- `evaluation/predictions/` 和 `evaluation/reports/` 是本地生成产物目录，适合论文取数和查证，但通常不应直接提交到 Git。
- 运行项目脚本或服务前应先进入项目 conda 环境：`conda activate dissertation`。

## 一句话主线

当前论文可采用的系统主线是：上传完整音频或浏览器分片音频后，后端用滑动窗口模拟实时分析；每个窗口执行 ASR、窗口级文本风险、当前窗口声学风险、固定权重融合和平滑，前端展示随时间演化的 risk timeline。离线评估脚本再把 timeline 转换成 final classification、alert recall、lead time、detection delay 和 case-type breakdown。

## 核心关系图

```text
templates/main.html
  -> server.py
     -> /api/stream_audio_analysis
        -> streaming_analysis/window_pipeline.py
           -> speaker_analysis/asr_backend.py
           -> ChineseBERTModel/ensemble_utils.py
           -> audio_risk_detection/predict_audio_risk.py
           -> streaming_analysis/risk_scoring.py
        -> NDJSON timeline
     -> /api/live_audio_chunk / /api/live_audio_finish

test_samples/metadata_final.csv
  -> evaluation/generate_dynamic_predictions.py
     -> server.py real API path
     -> evaluation/predictions/<run_name>/dynamic_predictions.json
        -> evaluation/dynamic_metrics.py
           -> evaluation/reports/<run_name>/*.csv / *.json

evaluation/predictions/final_baseline_w5_s2p5/dynamic_predictions.json
evaluation/predictions/final_baseline_w10_s5/dynamic_predictions.json
evaluation/predictions/final_baseline_w20_s10/dynamic_predictions.json
  -> window/step sensitivity analysis
  -> evaluation/calibrated_late_fusion.py (currently use w10_s5 baseline if needed)
     -> evaluation/predictions/final_learned_late_fusion_w10_s5/dynamic_predictions.json
        -> evaluation/dynamic_metrics.py
```

## 系统入口与服务层

| 文件 | 位置 | 基本功能 | 论文用途 |
| --- | --- | --- | --- |
| `server.py` | 项目根目录 | Flask 服务入口。暴露 `/predict`、`/api/stream_audio_analysis`、`/api/live_audio_chunk`、`/api/live_audio_finish`、`/api/audio_result` 等接口。 | 写系统架构、API 入口、动态风险追踪主流程时引用。论文主线优先引用 `/api/stream_audio_analysis`。 |
| `templates/main.html` | `templates/` | 前端主界面。包含 Audio Stream、Live Mic、Text Check，展示 risk curve、timeline table、current window transcript 和 final/peak risk。 | 写 prototype UI、dynamic risk visualization、simulated streaming demo 时引用。用户可见 UI 文案应保持英文。 |
| `templates/test.html` | `templates/` | 旧的文本预测测试页。 | 通常不进入论文，只在追溯旧 `/predict` 测试入口时参考。 |
| `README.md` | 项目根目录 | 项目总览、启动命令、动态评估说明和旧结构描述。 | 可作为背景材料，但部分描述可能滞后；写论文时应以代码和本文件为准。 |
| `AGENTS.md` | 项目根目录 | 项目工作规则、运行环境、模型资产约束、当前核心流程、样本集组成。 | 写方法和复现设置前可查，尤其是环境、模型文件和样本集边界。 |

## 动态风险追踪主流程

| 文件 | 位置 | 基本功能 | 关系与论文用途 |
| --- | --- | --- | --- |
| `streaming_analysis/window_pipeline.py` | `streaming_analysis/` | 滑动窗口分析核心。负责切窗、临时窗口音频、ASR、窗口文本模型、滚动上下文文本模型、声学模型、融合和平滑，并输出 timeline。 | 论文方法章节的主实现依据。可对应 sliding-window simulated streaming、window-level inference、timeline generation。 |
| `streaming_analysis/risk_scoring.py` | `streaming_analysis/` | 风险分数计算。论文主线使用 `score_window(..., scoring_mode="baseline")`：文本风险结合当前窗口与近期上下文，声学风险按固定权重融合，再做 exponential smoothing。 | 写 multimodal fusion 和 smoothing 公式时引用。注意此文件也保留 gated 代码，论文主方法只取 baseline 部分。 |
| `streaming_analysis/__init__.py` | `streaming_analysis/` | 包初始化文件。 | 通常不需要写入论文。 |

### 当前 baseline 口径

- 文本模型直接输入不是完整 `cumulative_text`，而是当前窗口文本 `window_text.strip()` 以及由当前窗口优先、近期上下文补充得到的 `rolling_context_text`。
- `cumulative_text` 仍保留在 timeline 里，主要用于前端展示、人工回看和错误分析。
- 默认融合和平滑可写作 baseline setting：`fused_score = 0.8 * text_score + 0.2 * voice_score`，`smoothed_score = 0.65 * previous + 0.35 * current`。
- 风险阈值主口径：`Normal < 50`，`Suspicious 50-69`，`High Risk 70-89`，`Critical >= 90`；动态评估中常用 alert threshold 为 `70`。

## 文本风险模块

| 文件 | 位置 | 基本功能 | 论文用途 |
| --- | --- | --- | --- |
| `ChineseBERTModel/ensemble_utils.py` | `ChineseBERTModel/` | 运行时 Chinese BERT 推理入口。加载 `bert-base-chinese`、分类器权重和 tokenizer，提供 `ensemble_inference`、token counting、token budget trimming。 | 写文本风险模型部署、checkpoint、rolling-context token budget 时重点引用。当前 checkpoint 优先级为 `best_model.pt` -> `train.pt` -> legacy path。 |
| `ChineseBERTModel/predict.py` | `ChineseBERTModel/` | 单条文本预测相关逻辑。 | 可作为旧文本预测入口参考；论文主线更建议引用 `ensemble_utils.py`。 |
| `ChineseBERTModel/train.py` | `ChineseBERTModel/` | 早期训练脚本。 | 如需写模型训练来源可参考，但最终论文结果应以当前 checkpoint 和评估脚本为准。 |
| `ChineseBERTModel/train_chinese_bert_kaggle.py` | `ChineseBERTModel/` | Kaggle 环境下的 Chinese BERT 训练脚本。 | 写训练设置、Kaggle 复现实验时参考。 |
| `ChineseBERTModel/finetune_chinese_bert_hard_negatives_kaggle.py` | `ChineseBERTModel/` | hard negative / semantic contrast fine-tuning 脚本。 | 适合写 fine-tuning ablation 或 negative result；不要把 construction set 当最终测试集。 |
| `ChineseBERTModel/compare_surface_baseline_chinese_bert_kaggle.py` | `ChineseBERTModel/` | surface-rule baseline 与 Proposed Chinese-BERT 的 paper-ready 对比脚本，输出总体、case type、错误样例等表格。 | 写 baseline-vs-BERT 章节的核心脚本。 |
| `ChineseBERTModel/surface_baseline_expert.py` | `ChineseBERTModel/` | 手工 surface feature baseline。 | 作为 handcrafted baseline 的实现参考。 |
| `ChineseBERTModel/baseline_of_chinese_bert_kaggle.py` | `ChineseBERTModel/` | 原 notebook-style baseline / BERT 评估脚本。 | 用于保证 surface baseline 对比口径不被刻意削弱。 |
| `ChineseBERTModel/model/best_model.pt` | `ChineseBERTModel/model/` | 当前主文本分类器权重。 | 论文、Kaggle、demo/runtime 应尽量保持同一 checkpoint 叙事。权重不应纳入 Git。 |
| `ChineseBERTModel/model/train.pt` | `ChineseBERTModel/model/` | 文本模型 fallback 权重。 | 仅作为兼容 fallback 说明。 |
| `ChineseBERTModel/model/training_config.json` | `ChineseBERTModel/model/` | 文本模型训练配置。 | 写训练超参或模型配置时查。 |

## 声学风险模块

| 文件 | 位置 | 基本功能 | 论文用途 |
| --- | --- | --- | --- |
| `audio_risk_detection/predict_audio_risk.py` | `audio_risk_detection/` | 声学风险运行时入口。提取 MFCC，加载 CNN-BiLSTM，输出 deepfake probability、校准后的 `voice_score` 和 decision threshold。 | 写 audio branch、MFCC + CNN-BiLSTM、auxiliary acoustic risk signal 时重点引用。 |
| `audio_risk_detection/validate_audio_model_assets.py` | `audio_risk_detection/` | 检查 `best_f1_model.pt`、`audio_risk_config.json`、`training_meta.json` 是否 deployment-ready。 | 写实验前检查、部署资产校验时引用。 |
| `audio_risk_detection/train_audio_cnn_lstm_kaggle.py` | `audio_risk_detection/` | Kaggle 重训 CNN-BiLSTM 声学模型脚本。 | 写声学模型训练、threshold calibration、OOD validation 时参考。 |
| `audio_risk_detection/train_audio_cnn_lstm_kaggle.ipynb` | `audio_risk_detection/` | Kaggle notebook 版本训练记录。 | 可作为训练过程补充材料，不建议作为主要复现入口。 |
| `audio_risk_detection/train_audio_risk.py` | `audio_risk_detection/` | 较早的音频模型训练脚本。 | 历史参考。 |
| `audio_risk_detection/model/best_f1_model.pt` | `audio_risk_detection/model/` | 当前部署使用的声学模型权重。 | 权重资产，不应纳入 Git；论文中可描述为 deployed audio model checkpoint。 |
| `audio_risk_detection/model/audio_risk_config.json` | `audio_risk_detection/model/` | 声学模型特征参数、模型结构、输出路径和校准阈值。 | 写 MFCC 参数、模型结构和 decision threshold 时引用。 |
| `audio_risk_detection/model/training_meta.json` | `audio_risk_detection/model/` | 声学模型训练与验证元信息，包括 deployment readiness、threshold 和评估字段。 | 写音频模块能力边界、验证结果和局限时引用。 |
| `audio_risk_detection/model/training_log.csv` | `audio_risk_detection/model/` | 训练日志。 | 需要展示训练曲线或收敛过程时参考。 |

论文表述建议：声学模型应写成 auxiliary acoustic risk signal，而不是独立最终诈骗判定器。系统级最终判断主要依赖 text-dominant fusion 缓冲声学分支的 FP/FN。

## ASR 与旧整段音频分析

| 文件 | 位置 | 基本功能 | 论文用途 |
| --- | --- | --- | --- |
| `speaker_analysis/asr_backend.py` | `speaker_analysis/` | 统一 ASR 后端。默认 `funasr_paraformer`，也兼容 Whisper / SenseVoice。负责清洗中文 ASR 文本，并把模型缓存放在项目 `.cache/` 下。 | 写 speech-to-text preprocessing、FunASR Paraformer backend、ASR failure handling 时引用。 |
| `speaker_analysis/speaker_pipeline.py` | `speaker_analysis/` | 旧整段音频分析 pipeline：说话人分离、STT、文本风险、声学风险并按 speaker 聚合。 | 可作为 legacy full-audio analysis 对照；论文主贡献不要把它写成动态风险追踪主线。 |
| `speaker_analysis/diarization_utils.py` | `speaker_analysis/` | 说话人分离工具。 | 如果论文还保留 speaker diarization 背景可参考，但当前动态主流程不依赖它作为核心贡献。 |
| `speaker_analysis/whisper_stt.py` | `speaker_analysis/` | Whisper STT 旧路径。 | ASR fallback 或历史对比时参考。 |
| `speaker_analysis/test_speaker_analysis.py` | `speaker_analysis/` | speaker pipeline 测试脚本。 | 通常不进入论文正文。 |

## 评估与论文取数

| 文件或目录 | 位置 | 基本功能 | 论文用途 |
| --- | --- | --- | --- |
| `evaluation/generate_dynamic_predictions.py` | `evaluation/` | 从 `test_samples/metadata_final.csv` 或指定 metadata 读取当前测试样本，调用真实 `/api/stream_audio_analysis` 生成 dynamic timeline，并写入 `evaluation/predictions/<run_name>/dynamic_predictions.json`。 | 论文动态系统评估的预测生成入口。 |
| `evaluation/dynamic_metrics.py` | `evaluation/` | 从 prediction JSON/JSONL 计算 final metrics、alert metrics、lead time、detection delay、ablation summary、case-type summary、window summary。 | 论文表格和结果分析的主评估入口。 |
| `evaluation/calibrated_late_fusion.py` | `evaluation/` | 从缓存 baseline timeline 训练/应用 calibrated learned late-fusion 层，输出 `learned_late_fusion_score`。只使用当前和过去窗口的数值特征，避免 metadata/keyword leakage。 | 可写为 offline learned fusion evaluation variant，用来修正固定 `0.8/0.2` fusion 的 failure mode；不要写成已证明 cross-dataset generalization。 |
| `evaluation/test_calibrated_late_fusion.py` | `evaluation/` | learned late fusion 的 feature contract 测试。 | 写避免 leakage 的实现保障时可引用。 |
| `evaluation/external_frozen_v1.py` | `evaluation/` | 固化100条 external stress 样本、执行重叠审计并拆分 core 80 prediction。 | 复现 frozen external evaluation 时使用。 |
| `evaluation/external_frozen_v1_manifest.json` | `evaluation/` | external frozen v1 的样本、metadata/model 哈希和 freeze rules。 | 证明样本与模型在测试前已冻结。 |
| `evaluation/README_external_frozen_v1.md` | `evaluation/` | core 80 与 expanded 100 的结果摘要、边界和复现命令。 | Chapter 5 external stress evidence 主索引。 |
| `evaluation/generate_window20_text_predictions.py` | `evaluation/` | 生成 20 秒窗口级文本风险预测，用于测试 baseline ChineseBERT 在 simulated streaming 文本窗口上的行为。 | 文本分支动态/窗口级诊断。 |
| `evaluation/generate_full_text_predictions.py` | `evaluation/` | 生成 post-hoc full-audio text-risk predictions。 | 可作为 “post-hoc full transcript” 对照，不应混同为实时动态主线。 |
| `evaluation/build_cached_error_attribution.py` | `evaluation/` | 基于已有缓存结果做错误归因。 | 写 error analysis 时参考。 |
| `evaluation/generate_mimo_tts_samples.py` | `evaluation/` | 生成 mimo-v2.5 TTS 样本并登记 metadata。 | 数据构造与 synthetic/semantic fraud sample provenance。 |
| `evaluation/generate_google_tts_samples.py` | `evaluation/` | 生成 gTTS 样本。 | mixed_risk / synthetic_voice 样本构造时参考。 |
| `evaluation/extract_magicdata_normal_samples.py` | `evaluation/` | 从 MagicData 抽取 normal 样本并登记 metadata。 | normal_daily / normal_finance 数据来源说明。 |
| `evaluation/extract_magicdata_ramc_normal_samples.py` | `evaluation/` | RAMC / MagicData normal 样本抽取脚本。 | 若论文写 normal speech source，可辅助查证。 |
| `evaluation/prepare_risk_samples.py` | `evaluation/` | 风险样本准备脚本。 | 数据准备辅助，按需要引用。 |
| `evaluation/example_predictions.json` | `evaluation/` | `dynamic_metrics.py` 的示例输入。 | 只适合解释评估 JSON schema，不作为结果证据。 |

### 推荐论文结果目录

| 目录 | 用途 | 论文建议 |
| --- | --- | --- |
| `evaluation/predictions/final_baseline_w5_s2p5/` | final controlled set 的 5 s window / 2.5 s step prediction timeline。 | Chapter 5.5 短窗口 sensitivity 证据；响应更早，但误报更高。 |
| `evaluation/reports/final_baseline_w5_s2p5/` | 5 s / 2.5 s 的 dynamic metrics 报表。 | 可取 window/step comparison、case type、ablation、window summary。 |
| `evaluation/predictions/final_baseline_w10_s5/` | final controlled set 的 10 s window / 5 s step baseline prediction timeline。 | 当前 baseline 主证据；综合 performance / stability / timing 最平衡。 |
| `evaluation/reports/final_baseline_w10_s5/` | 10 s / 5 s baseline 的 dynamic metrics 报表。 | 可取 final classification、case type、ablation、window summary。 |
| `evaluation/predictions/final_baseline_w20_s10/` | final controlled set 的 20 s window / 10 s step prediction timeline。 | Chapter 5.5 长窗口 sensitivity 证据；更平滑但召回和 lead time 下降。 |
| `evaluation/reports/final_baseline_w20_s10/` | 20 s / 10 s 的 dynamic metrics 报表。 | 可取 window/step comparison、case type、ablation、window summary。 |
| `evaluation/predictions/final_learned_late_fusion_w10_s5/` | learned late fusion 变体 prediction。 | offline variant 证据。 |
| `evaluation/reports/final_learned_late_fusion_w10_s5/` | learned late fusion final-fit 报表。 | 可作为 controlled benchmark improvement，但需谨慎表述。 |
| `evaluation/reports/final_learned_late_fusion_w10_s5_sample_cv/` | learned late fusion sample-level CV 报表。 | 用于 OOF diagnostics 和过拟合风险讨论。 |
| `evaluation/reports/external_frozen_v1_core80/` | 四类各20条的 frozen matched stress 报表。 | 与 final benchmark 做 case-type 对照；结果显示 learned gain 未稳定迁移。 |
| `evaluation/reports/external_frozen_v1_expanded100/` | core 80 加20条 normal_finance 的 frozen stress 报表。 | 报告 benign financial speech 的 false-positive 压力。 |
| `evaluation/reports/baseline_route_equivalence_latency/` | baseline 双路由等价性与 browser 5 秒 chunk 延迟原始结果。 | 系统级实时可行性证据；结果目录被 Git 忽略。 |

### Window/Step 全量测试摘要

| run | window / step | 核心观察 | Chapter 5 用途 |
| --- | --- | --- | --- |
| `final_baseline_w5_s2p5` | 5 s / 2.5 s | alert 更早、lead time 更大，但 normal alert FPR 和 final FP 更高。 | 说明短窗口提高响应速度，同时放大瞬时误报。 |
| `final_baseline_w10_s5` | 10 s / 5 s | final F1、normal final FP、alert stability 和 timing 最平衡。 | 当前 baseline 主口径。 |
| `final_baseline_w20_s10` | 20 s / 10 s | 过程更平滑，但 fraud recall、alert recall 和 early-warning lead time 下降。 | 说明长窗口会牺牲动态预警能力。 |

### 不建议作为论文主结果的目录

- `evaluation/predictions/long_baseline_w10_s5/`
- `evaluation/reports/long_baseline_w10_s5/`
- `evaluation/reports/funasr_gated_*`
- `evaluation/predictions/funasr_gated_*`
- `evaluation/reports/long_gated_*`
- `evaluation/predictions/long_gated_*`
- `evaluation/reports/progression_*`
- `evaluation/predictions/progression_*`

这些目录可作为历史工程探索、旧评估或 ablation 背景保存，但暂不纳入当前论文主结果。

## Chinese BERT 评估材料

| 文件或目录 | 位置 | 基本功能 | 论文用途 |
| --- | --- | --- | --- |
| `evaluation/chinese-bert/README_finetune_stress_files.md` | `evaluation/chinese-bert/` | Chinese BERT fine-tuning、stress test 和 construction sets 的稳定索引。 | 写文本模型实验前应先看。 |
| `evaluation/chinese-bert/External Evaluation Sets/external_normal_finance_hard_negative_stress.jsonl` | `evaluation/chinese-bert/External Evaluation Sets/` | 120 条外部文本级 stress set，覆盖 normal_daily、normal_finance、semantic_fraud、mixed_risk。 | 文本模型主 stress test，用于 Proposed Chinese-BERT vs surface baseline。 |
| `evaluation/chinese-bert/External Evaluation Sets/final_text_label_diagnostic_metadata_final.jsonl` | 同上 | 80 条 final multimodal system samples 派生文本诊断集。 | 补充 text-only branch diagnostic，不作为主 normal-finance stress test。 |
| `evaluation/chinese-bert/Fine-tuning Construction Sets/benign_finance_hard_negatives_train_val.*` | `evaluation/chinese-bert/Fine-tuning Construction Sets/` | benign finance hard-negative train/validation construction set。 | fine-tuning ablation / negative result；不可当 final test。 |
| `evaluation/chinese-bert/Fine-tuning Construction Sets/finance_semantic_contrast_train_val.*` | 同上 | finance semantic contrast train/validation construction set。 | contrastive fine-tuning ablation；不可当 final test。 |
| `evaluation/chinese-bert/generate_benign_finance_hard_negatives.py` | `evaluation/chinese-bert/` | 生成上述 construction sets。 | 写数据构造方法和复现时引用。 |

## 样本与元数据

| 文件或目录 | 位置 | 基本功能 | 论文用途 |
| --- | --- | --- | --- |
| `test_samples/metadata_final.csv` | `test_samples/` | 当前系统测试样本索引。列包含 `sample_id`、`audio_path`、`label`、`case_type`、`event_time_sec`、`source`、`transcript_text` 等，音频路径指向 `test_samples/audio_final/`。 | 当前论文系统测试和样本 provenance 的 authoritative source。 |
| `test_samples/audio_final/` | `test_samples/` | 当前系统测试音频目录，配套 `metadata_final.csv`。包含 normal_daily、synthetic_voice、mixed_risk、semantic_fraud 等最终样本类别。 | 写当前测试集组成、样本来源和系统评估时优先查。 |

论文写作时建议优先以 `sample_id` 和 metadata 表达样本，而不是只靠文件名。`event_time_sec` 是动态指标中 lead time / detection delay 的关键字段。

## 论文与计划材料

| 文件 | 位置 | 基本功能 | 论文用途 |
| --- | --- | --- | --- |
| `后续计划.md` | 项目根目录 | 项目计划和论文推进记录。 | 可用于回看阶段性决策，但写正式论文前应重新核对代码和评估产物。 |
| `MinerU_markdown_InterimReport.md` | 项目根目录 | Interim report 的 markdown 转换稿。 | 可复用背景、早期架构和写作素材，但需更新到当前实现。 |
| `MinerU_markdown_Proposal.md` | 项目根目录 | Proposal 的 markdown 转换稿。 | 可复用 motivation / problem statement，方法和实验应以当前代码为准。 |

## 运行和依赖文件

| 文件或目录 | 位置 | 基本功能 | 论文用途 |
| --- | --- | --- | --- |
| `requirements.txt` | 项目根目录 | Python 依赖列表。 | 写复现环境时引用。 |
| `Dockerfile` | 项目根目录 | 容器化环境草案。 | 如果论文或附录需要部署说明可参考。 |
| `.cache/` | 项目根目录 | 项目内缓存目录，用于 Hugging Face、ModelScope、ASR、stream windows、live audio streams 等。 | 不作为论文证据，但复现时可解释缓存位置。 |
| `uploads/` | 项目根目录 | 上传音频临时目录。 | 运行时目录，不写入论文。 |
| `output/` | 项目根目录 | 生成产物目录。 | 不作为论文主索引，通常不提交。 |

## 排除或谨慎引用的文件

| 文件或目录 | 原因 |
| --- | --- |
| `streaming_analysis/progression_scoring.py` | progression 路线目前不采纳为论文方法。 |
| `evaluation/apply_gated_v1.py` | gated offline replay 历史脚本，不代表当前论文主方法。 |
| `evaluation/run_long_controlled_eval.py` | 依赖旧的 `metadata_long.csv` long controlled set，当前测试口径暂不纳入。 |
| `evaluation/reports/*gated*`、`evaluation/predictions/*gated*` | gated 系列结果只适合历史探索或 ablation 背景。 |
| `evaluation/reports/*progression*`、`evaluation/predictions/*progression*` | progression 系列结果不进入论文主线。 |
| `evaluation/reports/long_baseline_w10_s5/`、`evaluation/predictions/long_baseline_w10_s5/` | 旧 long baseline 评估结果，当前测试口径暂不纳入。 |
| `test_samples/metadata.csv`、`test_samples/metadata_long.csv` | 旧样本索引，当前系统测试以 `metadata_final.csv` 为准。 |
| `test_samples/audio_long/`、`test_samples/audio/`、`test_samples/audio_fake/` | 旧评估或历史样本目录，当前系统测试以 `audio_final/` 为准。 |
| `.DS_Store`、`__pycache__/` | 系统/缓存文件，无论文价值。 |
| `uploads/` | 运行时上传缓存，无论文价值。 |

## 写论文时的快速定位

- 写系统架构：先看 `server.py`、`streaming_analysis/window_pipeline.py`、`templates/main.html`。
- 写文本模型：先看 `ChineseBERTModel/ensemble_utils.py`、`ChineseBERTModel/compare_surface_baseline_chinese_bert_kaggle.py`、`evaluation/chinese-bert/README_finetune_stress_files.md`。
- 写声学模型：先看 `audio_risk_detection/predict_audio_risk.py`、`audio_risk_detection/model/audio_risk_config.json`、`audio_risk_detection/model/training_meta.json`。
- 写动态评估：先看 `evaluation/generate_dynamic_predictions.py`、`evaluation/dynamic_metrics.py`、`evaluation/reports/final_baseline_w10_s5/`。
- 写 window/step 对比：先看 `evaluation/reports/final_baseline_w5_s2p5/`、`evaluation/reports/final_baseline_w10_s5/`、`evaluation/reports/final_baseline_w20_s10/`，对应 prediction timeline 在 `evaluation/predictions/` 下同名目录。
- 写 learned fusion：先看 `evaluation/calibrated_late_fusion.py`、`evaluation/test_calibrated_late_fusion.py`、`evaluation/reports/final_learned_late_fusion_w10_s5_sample_cv/`。
- 写系统级实时验证：先看 `evaluation/README_baseline_route_equivalence_latency.md`、`evaluation/baseline_route_equivalence_latency.py` 和 `evaluation/figures/route_equivalence_browser_latency.pdf`。
- 写 ablation 与 window/step 答辩图：使用 `evaluation/generate_ablation_window_tradeoff_figure.py`，输出位于 `evaluation/figures/ablation_window_tradeoff.{pdf,png}`。
- 写数据集：先看 `test_samples/metadata_final.csv`、`test_samples/audio_final/`、`evaluation/chinese-bert/External Evaluation Sets/`。
- 写局限：重点讨论 audio branch 的辅助性质、fixed fusion/smoothing 的 failure mode、controlled benchmark 与 external generalization 的区别。
