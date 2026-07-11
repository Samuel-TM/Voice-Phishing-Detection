# Final Report 修改计划（corrected `defense_all180` 版）

**目标论文目录：** `/Users/sunjiashan/Material/HKU/Dissertation/FinalReport/HKU_DASE_Dissertation_FinalReport`
**本计划更新日期：** 2026-07-11

## 0. 论文主线与证据边界

论文统一采用 corrected `defense_all180` predictions/reports 的叙事：本研究将语音诈骗检测从事后整段二分类扩展到通话过程中的因果滑动窗口风险追踪。系统以文本语义风险为主、声学真实性异常为辅助证据，采用固定 `0.8*text + 0.2*voice` 融合和时间平滑，输出最终判定、过程报警和报警时序。

五类受控样本的标签固定为：ND 40、NF 20、SV 40 为 normal；SF 40、MR 40 为 fraud。因此，benchmark 共 180 条样本，其中 normal/fraud 为 100/80。它是 controlled benchmark evidence，不应写成独立外部泛化或真实世界患病率估计。

文中仅需保留一条术语修正：凡是旧文将声学支路说成“能够检测纯声学诈骗”，统一改为“检测声学真实性异常，作为辅助证据”。不要在各章反复展开同一概念对照。

## 1. 唯一可引用的实验产物

本次只改论文文本与表图引用，不运行脚本、不重算产物。所有 Chapter 5 数值、表格和 case-level 叙述均从下列已有 corrected `defense_all180` 产物提取：

| 论文用途 | 直接来源 |
|---|---|
| 样本定义、标签与计数 | `test_samples/metadata_all_corrected.csv` |
| timeline 与逐样本核对 | `evaluation/predictions/defense_all180_baseline_w10_s5/dynamic_predictions.json`；`dynamic_predictions_summary.csv` |
| 主系统总体指标与逐样本明细 | `evaluation/reports/defense_all180_baseline_w10_s5/dynamic_eval_report.json`；`dynamic_eval_detail.csv` |
| 五类 case breakdown | `evaluation/reports/defense_all180_baseline_w10_s5/dynamic_eval_case_type_summary.csv` |
| 消融与平滑取舍 | `evaluation/reports/defense_all180_baseline_w10_s5/dynamic_eval_ablation_summary.csv` |
| 固定权重敏感性 | `evaluation/reports/defense_all180_baseline_w10_s5/dynamic_eval_fusion_smoothing_sweep.csv`；`dynamic_eval_fusion_smoothing_case_type_sweep.csv` |
| 窗口配置结果与异常案例 | `evaluation/reports/defense_all180_baseline_w10_s5/dynamic_eval_window_summary.csv`；`dynamic_eval_high_raw_unalerted.csv` |

不在论文中引用或列入参考产物：corrected nested-CV 的 CSV、JSON、README，以及任何以这些文件为依据的策略排名或 admission 结论。相应地，learned fusion 不再构成论文实验主线或独立结果章节。

每一个写入论文的数字应可回溯至同一 `defense_all180` report：正常类只报告 final/alert FPR，SF/MR 才报告 fraud recall、alert recall、lead time 与 delay。

## 2. 全篇修改原则

1. 删除全部旧评估集的样本数、总体指标、case recall、lead time、TP/TN/FP/FN 和图表；不得保留旧结果作为背景性比较。
2. 固定 `0.8/0.2 + smoothing` 是论文的 runtime mainline。权重 sweep 只说明敏感性和取舍，不是事后选出“最优”方法的过程。
3. 主结果必须同时呈现最终判定、过程报警和报警时序；不得只用一个 aggregate F1 覆盖五类 case 的差异。
4. `synthetic_voice` 是 normal control。它的声学高分或报警只能用于误报/局限分析，不能记为 fraud detection 成功、FN 或 fraud recall 的分母。
5. learned fusion 的实现历史不再展开：Chapter 4 删除改进融合策略小节，Chapter 5 删除 offline learned-fusion / nested-CV 小节；如确有交代必要，只在局限性中用一句话说明“本文聚焦固定、可解释的运行时融合规则”，不报告其数值。

## 3. 分章修改计划

### 3.1 Abstract — `Files/titlepage.tex`

重写为四段：

1. 任务：从 post-hoc classification 转为 in-event causal sliding-window risk tracking。
2. 方法：文本风险与声学真实性异常证据并行，固定融合和平滑生成风险时间线。
3. 证据：corrected 180-sample five-way controlled benchmark；报告 final、alert 与 timing 三类结果。
4. 结论：固定方法在受控条件下提供可解释的动态风险追踪证据；结论限于 controlled benchmark，不声称生产部署或外部泛化。

摘要仅从 `dynamic_eval_report.json` 和对应 case summary 填写 `N=180`、normal/fraud `100/80`、主方法的 Accuracy/Precision/Recall/F1、normal final FPR、fraud alert recall 及 SF/MR 时序指标。不要写任何被删除的实验结果，也不讨论 learned fusion。

### 3.2 Chapter 1 — `Contents/chap1.tex`

- 保留 static classification 到 dynamic risk tracking 的动机、滑动窗口流程及三层评价目标。
- 将研究问题写为：如何在不同证据来源下分别评价最终诈骗判定、过程报警和报警时机。
- 将研究目标改为：在五类 controlled case types 中检验固定、text-dominant fusion 的边界；ND/NF/SV 是不同类型的正常干扰，SF/MR 是 fraud cases。
- 删除任何“声学支路能够检测纯声学诈骗”或 learned fusion 是后续主方法的表述。
- Chapter 5 的预告只保留主结果、五类分析、消融/平滑取舍、权重敏感性、窗口设置、runtime feasibility 和误差分析。

### 3.3 Chapter 2 — `Contents/chap2.tex`

文献综述主体无需重写。只在 Audio Deepfake 与 Streaming Multimodal Processing 的收束处补充：anti-spoofing 输出可作为声学辅助证据；多模态系统需按正常干扰与 fraud case 分开评价，而非只比较 aggregate F1。不要加入 learned/adaptive fusion 的解决方案叙事。

### 3.4 Chapter 3 — `Contents/chap3.tex`

- 保留 ChineseBERT 与音频模型的训练/验证内容，但明确其分别是文本模块与 anti-spoofing 模块的证据，不能替代 Chapter 5 的系统级结果。
- 在 Dataset Preparation 中给出 five-way corrected benchmark 的标签表及 normal/fraud 计数。
- 将 reproducibility 的数据层级写为：训练数据、模块验证数据、corrected `defense_all180` controlled benchmark、route validation 子集。不要保留已废止评估集作为论文实验层级。
- 对既有模块验证高分加上 model-validation boundary，避免外推成端到端诈骗检测结论。

### 3.5 Chapter 4 — `Contents/chap4.tex`

#### 4.4 Late Fusion and Temporal Smoothing Design

将本节压缩为运行时固定设计，不再讨论 learned fusion：

- 保留 Eq. (4.7) 与 Eq. (4.8)，将其定义为预先设定的 runtime baseline，而不是由本轮 benchmark 选出的最优权重。
- 明确 `text_score` 表示诈骗语义风险，`voice_score` 表示声学真实性异常证据；融合后的分数用于诈骗报警决策。
- 保留平滑的递推定义及其目的：降低单窗口波动，并在最终误报与报警时机之间形成可报告的取舍。
- 删除 4.4.3 “Position of Improved Fusion Strategies”及其中所有 learned/adaptive、grouped CV、admission constraint 和策略比较内容。

#### 运行时、指标与样本边界

- uploaded-audio sliding-window 是论文所称的 simulated-streaming experimental task。
- `/api/stream_audio_analysis` 与 `/api/live_audio_chunk` 的说明限于同窗 baseline 处理逻辑和本机处理可行性；不得写成运营商级或端到端网络实时部署。
- Fraud recall、TP/FN、lead time 与 delay 仅对 SF/MR 定义；ND/NF/SV 只报告 final FPR 与 alert FPR。
- 在 4.6.1 固定给出 corrected 180 composition，并注明 controlled evidence 边界。

### 3.6 Chapter 5 — `Contents/chap5.tex`

按下列结构重写，不保留 learned-fusion 或 nested-CV 小节：

1. **5.1 Experimental protocol and corrected benchmark**：五类样本、100 normal/80 fraud、10s/5s 预定义配置、threshold 70，以及按 case type 划分的 metric contract。
2. **5.2 Text-risk model evidence**：保留 Surface-vs-ChineseBERT，但限定为 text-module evidence；NF 用作 hard negative 的解释背景。
3. **5.3 Main dynamic tracking result: fixed fusion + smoothing**：从 `dynamic_eval_report.json` 引用固定主方法的总体表，包含 Acc/Prec/Rec/F1、TP/TN/FP/FN、fraud alert recall、normal final/alert FPR、TTA、SF/MR lead time 与 delay。
4. **5.4 Five-way case-type analysis**：从 `dynamic_eval_case_type_summary.csv` 生成一张表；ND/NF/SV 写 FPR，SF/MR 写 recall、alert 与 timing。SV 的讨论限于正常合成语音控制条件下的误报表现。
5. **5.5 Ablation and smoothing trade-off**：从 `dynamic_eval_ablation_summary.csv` 报告 text only、voice only、fusion without smoothing 和 fixed fusion with smoothing。voice-only 只说明声学辅助证据的局限；平滑写成最终误报、过程报警与报警时机之间的取舍。
6. **5.6 Fixed-weight sensitivity**：引用两个 fusion/smoothing sweep，比较预先列出的固定 text/voice 权重；0.8/0.2 保持运行时主线，其他权重仅作敏感性分析。
7. **5.7 Window/step setting**：引用 `dynamic_eval_window_summary.csv`；若产物只含当前配置，则只报告当前 10s/5s 配置，不添加不存在的窗口比较（后续补充 5s/2.5s 和 20s/10s）。
8. **5.8 Runtime feasibility and error analysis**：分开陈述 route/browser-chunk 的测量边界，并结合 `dynamic_eval_detail.csv`、`dynamic_eval_high_raw_unalerted.csv` 分析 NF false positives、SF misses、MR trade-offs 与 SV acoustic false alarms。

### 3.7 Chapter 6 — `Contents/chap6.tex`

结论压缩为四点：

1. 核心产出是因果滑动窗口风险时间线与 final/alert/timing 三层评估，而非单个整段分类器。
2. corrected five-way design 将不同正常干扰与诈骗 case 分开，避免把声学异常直接当作诈骗结论。
3. fixed `0.8/0.2 + smoothing` 是当前 controlled evidence 下的运行时主线；用 corrected `defense_all180` 指标说明其性能与取舍。
4. 受控 benchmark 和本机处理时延不等于跨数据集泛化或生产部署；未来工作是未触碰外部验证、提升音频可靠性，并评估网络、编码与隐私条件下的实时路径。

删除关于 learned/adaptive fusion 成功、失败、比较或未来必然升级的全部总结。

## 4. 表图来源与写作限制

| 论文表/图 | 数据来源 | 写作限制 |
|---|---|---|
| 五类 benchmark composition | `metadata_all_corrected.csv` | 写清标签、计数、用途与 controlled 边界 |
| 主系统总体指标 | `dynamic_eval_report.json`、`dynamic_eval_detail.csv` | 仅固定主方法；final 与 alert 分开 |
| 五类 case breakdown | `dynamic_eval_case_type_summary.csv` | 正常类写 FPR；fraud 类写 recall/timing |
| 消融与平滑取舍 | `dynamic_eval_ablation_summary.csv` | 不把声学报警解释成诈骗检出 |
| 固定权重敏感性 | `dynamic_eval_fusion_smoothing_sweep.csv`、case-type sweep | 不写成 test-set optimization |
| 窗口配置与异常案例 | `dynamic_eval_window_summary.csv`、`dynamic_eval_high_raw_unalerted.csv` | 只呈现已存在的 corrected 输出 |

所有 caption 均写明：corrected 180、决策层（fixed fusion）、口径（final 或 alert）与证据边界（controlled）。

## 5. 最终验收清单

- [ ] 全文只使用 corrected `defense_all180` predictions/reports 的实验数值与案例证据。
- [ ] 全文不存在任何已废止评估集的数值、表格、图或比较结论。
- [ ] `synthetic_voice` 不作为 phishing 正类、FN 或 fraud recall 的分母。
- [ ] 所有“声学支路检测诈骗”的旧表述均改为“检测声学真实性异常，作为辅助证据”。
- [ ] Chapter 4 和 Chapter 5 均无 learned fusion、nested-CV 或策略排名的小节、图表和数值。
- [ ] Abstract、Chapter 4、5、6 的样本构成均为 180 = ND40 + NF20 + SV40 + SF40 + MR40，normal/fraud=100/80。
- [ ] final FPR、alert FPR、fraud recall 与 timing 的适用 case type 清楚分开。
- [ ] runtime 结论限于 simulated streaming、同窗处理逻辑和本机测量范围。
- [ ] 编译后无未解析的表图、公式或交叉引用，且 Abstract/Conclusion 没有遗留旧叙事。
