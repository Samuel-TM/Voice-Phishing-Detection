# README_structure.md

本文档用于规划最终论文结构和写作索引，不是最终英文正文。正式报告题目沿用中期报告：

**Real-Time Dynamic Risk Tracking of Telecommunications Fraud Based on Parallel Multimodal Analysis**

核心写作原则：正文围绕 **dynamic risk tracking** 展开，重点放在研究方法、系统实现、结果、讨论和结论。第 1 章和第 2 章应基于 `MinerU_markdown_InterimReport.md` 优化压缩，不逐段沿用；正文证据口径以当前系统测试集为准。

## Guideline 约束摘要

### 一、 论文结构与前置页面（Front Matter）

前置非正文页（Front Matter）一律使用小写罗马数字（i, ii, iii...）编排页码（封面与摘要页除外）。

* **封面（Cover Page）**：须严格包含学校、学院、学系（Department of Data and Systems Engineering）、对应的硕士专业（M.Sc.(Eng.) in Robotics and Intelligent Systems）、完整的论文题目、作者姓名及大学 UID、导师姓名、提交日期（2026年8月2日）。


* **摘要（Abstract）**：
* **字数限制**：约 **400 至 500 字**（这是英文字数，中文字数自己估算）。
* **内容要求**：必须是对报告内容的精炼、准确概括，明确阐述研究内容、方法、主要结果、结论、建议以及必要的范围。
* **避坑指南**：该页**不设页码**；严禁堆砌泛泛的背景介绍，应直奔研究核心。


* **声明（Declaration）**：独立成页，正式声明该报告为个人独立作品，须包含个人手写或电子签名及日期，页码通常为 **i**。
* **致谢（Acknowledgements）**：属于**可选内容（Optional）**。若无需致谢则直接删除此页并重新对后续页面进行编号；若报告中使用了来自特定公司或组织的数据，可在此处撰写简短的感谢声明。
* **目录与图表清单（Table of Contents / List of Figures & Tables）**：依次排列，图表清单需精确对应正文中的图表编号与标题。

---

### 二、 篇幅与版面格式规范

从正文第一章（Chapter 1 Introduction）开始，页面必须重新计数并切换为**阿拉伯数字（1, 2, 3...）**。

* **正文页数限制**：
* **目标篇幅**：约为 **80 页**。
* **硬性边界**：最低不得少于 **70 页**，最高不得超过 **90 页**。
* **计算范围**：仅包含第 1 章至第 5/6 章的正文报告主体。参考文献（Bibliography）和附录（Appendix）**不计入**正文页数限制。


* **版面边距（Page Margins）**：上、下、左、右页边距一律严格设定为 **25 毫米（25mm）**。
* **行间距（Line Spacing）**：正文必须使用**双倍行距（Double line spacing）**。
* **字体大小与样式（Fonts & Titles）**：
* **正文主体（Text Body）**：统一使用 **12 号字体（Roman Times / Times New Roman）**。
* **小节标题（Section Titles）**：可设定为 **14 号字体**。
* **章节大标题（Chapter Titles）**：可设定为 **14 号 或 16 号字体**。


* **写作风格**：保持严谨的**学术报告风格（Academic Reporting Style）**，正文段落通常建议采用**两端对齐（Justified）**，每个新段落开头保持标准缩进。

---

### 三、 图表、公式与学术规范

#### 1. 图表标注规范

* **图（Figures）**：
* 图的标题和编号必须放置在**图的下方**（例如：`Figure 1.1 Model structure...`）。
* 图的编号必须反映其所在的章节（如第 1 章的第一张图为 `Figure 1.1`）。


* **表（Tables）**：
* 表的标题和编号必须放置在**表的上方**（例如：`Table 3.1 Summary of notations.`）。
* **核心细节**：当表题不是一个完整的句子时，末尾**绝对不能加句号（No full stop）**。如果表格使用了特殊符号，需在表下添加简要的 Legend 或 Note 进行说明。



#### 2. 数学公式（Equations）

* 所有公式需在文中独立成行，并在右侧进行对齐编号。
* **公式编号必须按章节进行对齐**（如第 3 章的公式依次编号为 `(3.14)`、`(3.15)`，其中 3 代表 Chapter 3）。

#### 3. 学术引用与专业术语

* **首次引入缩写**：在正文中第一次出现某个专业术语时，必须先提供英文全称，并在括号内注明缩写（例如：`information acquisition (IA)`）。
* **文中引用（In-text Citation）**：严格使用作者的姓氏/家族名（Last name/Family name）加年份（如 `Guo (2009)` 或 `(Fu & Zhu, 2010)`）。超过 3 名作者时使用 *et al.* 格式。

---

### 四、 核心内容侧重与附录定位

| 章节/板块 | 内容定位与核心要求 |
| --- | --- |
| **第 1 章 导论** | 根据中期报告内容进行优化与压缩。必须进一步细化和详尽阐述：**问题识别与界定**、**研究目标**、**研究议题与任务**。 |
| **第 2 章 文献综述** | 根据中期报告内容进行提炼、优化与压缩。明确划分研究流派（Research Streams），并在章节末尾准确归纳出研究切入点。 |
| **正文核心章节** | 最终报告必须聚焦并着重阐述：**研究方法（Methodology）**、**实施细节（Implementation Details）**、**实验/仿真结果（Results）**、**对结果的深入讨论（Discussion）** 以及 **得出的结论（Conclusions）**。 |
| **避坑要求** | 坚决避免撰写任何诸如“有待进一步开展的工作（Future Work）”等无实质意义、空洞泛泛的内容。 |
| **附录（Appendix）** | **严格定位**：只放置虽与课题相关、但若直接写入正文会破坏行文流畅性的**非关键支持信息**。例如：**原始数据**、**程序代码**、**补充表格**、**额外的数学公式推导或证明**、以及**被弱化处理的对比实验**。 |

---

## 当前论文证据口径

| 类别 | 当前采用 | 写作用途 |
| --- | --- | --- |
| 样本索引 | `test_samples/metadata_final.csv` | 当前系统测试的 authoritative sample index。 |
| 音频目录 | `test_samples/audio_final/` | 当前系统测试音频目录。 |
| 5s/2.5s 预测 | `evaluation/predictions/final_baseline_w5_s2p5/dynamic_predictions.json` | window/step sensitivity 的短窗口结果。 |
| 5s/2.5s 报表 | `evaluation/reports/final_baseline_w5_s2p5/` | Chapter 5.5 的短窗口对比证据。 |
| 10s/5s 预测 | `evaluation/predictions/final_baseline_w10_s5/dynamic_predictions.json` | 当前 baseline dynamic timeline 和逐样本结果。 |
| 10s/5s 报表 | `evaluation/reports/final_baseline_w10_s5/` | Chapter 5 的主要表格和默认 baseline 证据。 |
| 20s/10s 预测 | `evaluation/predictions/final_baseline_w20_s10/dynamic_predictions.json` | window/step sensitivity 的长窗口结果。 |
| 20s/10s 报表 | `evaluation/reports/final_baseline_w20_s10/` | Chapter 5.5 的长窗口对比证据。 |
| 主评估变体 | `fusion_with_smoothing` baseline | 论文主线结果，重点报告 final classification 与 dynamic alert metrics。 |
| learned late fusion | Appendix 或 Chapter 5 caveat 中弱化 | 不作为主方法，不替代 baseline。 |

暂不纳入当前主大纲的旧材料：

- `test_samples/metadata.csv`
- `test_samples/metadata_long.csv`
- `test_samples/audio_long/`
- `test_samples/audio/`
- `test_samples/audio_fake/`
- `evaluation/predictions/long_baseline_w10_s5/`
- `evaluation/reports/long_baseline_w10_s5/`
- `gated_*` 和 `progression_*` 相关结果

## 正文页数预算

| 章节 | 建议页数 | 写作重点 |
| --- | ---: | --- |
| Chapter 1 Introduction | 8-10 | 压缩背景，强化问题识别、研究目标、任务和范围。 |
| Chapter 2 Literature Review | 10-12 | 按三条研究流派组织，在 2.5 集中总结 research gap。 |
| Chapter 3 Research Methodology | 14-16 | 讲训练数据准备、ChineseBERT 文本模型和 synthetic-speech 声学模型。 |
| Chapter 4 System Design and Implementation | 15-17 | 讲动态多模态风险追踪框架的方法设计，不写代码文件、工程接口或界面实现细节。 |
| Chapter 5 Experiments and Analysis of Results | 25-28 | 重点章节，报告模型对比、动态结果、消融、window/step 对比和 case-type 错误分析。 |
| Chapter 6 Conclusions | 2-3 | 简洁回应 objectives，总结 main achievements、findings、contributions 和 innovations。 |

合计弹性范围约 **74-86 页**，实际写作目标仍控制在 **80-82 页** 左右，并满足 70-90 页要求。若正文过长，优先压缩 Chapter 1/2；不要压缩 Chapter 5 的结果讨论。

## Chapter 1: Introduction

建议页数：8-10 页。

写作目标：压缩中期报告里的背景材料，重点细化 problem identification、research motivation、research issues 和 objectives。导论要清楚说明：本研究不是只追求整段录音分类 accuracy，而是解决通话过程中的 warning timing gap 和 single-modality insufficiency。

### 1.1 Research Background

- 从 telecom fraud 的现实危害切入，但避免堆叠行业统计。
- 保留 Generative AI、deepfake voice 和 synthetic speech 使诈骗更自动化、拟人化、低成本的背景。
- 说明诈骗风险已经从单纯文本话术扩展到 text semantics + voice authenticity 的多模态问题。
- 引出 post-hoc classification 对事后取证有价值，但对通话中预警和干预不足。

### 1.2 Motivation of this research

- 动机一：现有 full recording / full transcript classification 多数是 post-hoc judgment，无法回答风险应在何时被提醒。
- 动机二：text-only 方法能理解诈骗语义，但不能感知 synthetic voice；audio-only 方法能感知声学异常，但不能理解转账、验证码、屏幕共享等诈骗意图。
- 动机三：需要将文本语义风险和声学真实性风险放在同一时间轴上，观察风险如何随通话进程演化。
- 研究边界：本研究是 application-layer content risk control，采用 uploaded-audio simulated streaming / incremental analysis，不声称 telecom operator network deployment。

### 1.3 Research issues and objectives

#### 1.3.1 Research issues

- 如何在通话过程中识别逐步显现的诈骗语义风险，而不是只做整段文本分类。
- 如何利用 synthetic-speech detection 捕捉 text-only 方法无法覆盖的声学风险。
- 如何将 text risk、voice risk 和 temporal smoothing 组织为连续 risk timeline。
- 如何用 alert recall、time to alert、early-warning lead time 和 detection delay 评估动态预警价值。

#### 1.3.2 Research objectives

- 构建 simulated streaming multimodal risk tracking prototype。
- 实现 window-level text risk、voice risk、fusion risk 和 smoothed risk timeline。
- 使用 `metadata_final.csv` 和 `audio_final/` 构建当前 final controlled benchmark。
- 对比 ChineseBERT 与 surface baseline，验证文本语义模型的贡献。
- 通过 dynamic metrics、ablation、window/step comparison 和 fraud type error analysis 评估系统行为。

### 1.4 Organization of the thesis

- Chapter 1 介绍研究背景、研究动机、research issues 和 research objectives，并界定本研究的 scope。
- Chapter 2 综述 fraud semantics、audio deepfake detection 和 streaming multimodal processing，并在 2.5 总结 research gap。
- Chapter 3 介绍数据准备、ChineseBERT 文本诈骗检测模型和 synthetic speech detection 模型。
- Chapter 4 说明 dynamic multimodal risk tracking framework 的方法设计。
- Chapter 5 报告实验设置、模型对比、动态结果、消融、窗口步长对比和错误分析。
- Chapter 6 总结主要发现、贡献和创新点。

## Chapter 2: Literature Review

建议页数：10-12 页。

写作目标：按研究流派组织文献，不做模型百科。2.2-2.4 分别梳理 fraud semantics、audio deepfake detection 和 streaming multimodal processing 的演进与现状；research gap 统一在 2.5 总结，作为 Chapter 3-5 的切入理由。

### 2.1 Introduction

#### 2.1.1 Review Scope

- 明确本章只综述与本研究问题直接相关的三条文献线：文本诈骗语义理解、音频 deepfake / synthetic speech detection、以及 streaming multimodal processing。
- 不展开与通话内容动态预警关系较弱的泛化网络安全检测、传统 CDR 风控或百科式深度学习模型清单。

#### 2.1.2 Organizing Logic

- 综述顺序从单模态内容理解到多模态动态处理。
- 每一条研究线先讲代表性思路和能力边界，最后在 2.5 综合出本研究的知识空白。

### 2.2 Text Semantic Understanding for Fraud Detection

#### 2.2.1 From Keyword and Surface Features to Semantic Modeling

- 介绍 keyword matching、hand-crafted lexical features、surface-rule baselines 在诈骗文本识别中的早期作用。
- 强调这类方法可解释但容易受话术变体、上下文省略和对抗性表达影响。

#### 2.2.2 Transformer-Based Fraud Semantics

- 综述 BERT / Chinese-BERT 类模型如何利用上下文表示提升中文诈骗语义识别能力。
- 说明 text semantic understanding 对识别转账诱导、验证码索取、冒充身份、投资骗局等内容风险的价值。

#### 2.2.3 Limitations of Text-Only Fraud Detection

- 说明 text-only 方法无法感知说话声音是否合成，也无法单独处理声学伪造风险。
- 为后续多模态设计埋下问题背景，但不在此处单独写 research gap。

### 2.3 Audio Deepfake and Synthetic Speech Detection

#### 2.3.1 Development of Synthetic Speech Detection

- 综述 ASVspoof、synthetic speech detection、voice anti-spoofing 等研究任务。
- 说明 deepfake voice 使 telecom fraud 从文本话术风险扩展到声学身份与真实性风险。

#### 2.3.2 Acoustic Features and Neural Architectures

- 概述 MFCC、spectrogram、CNN、LSTM / BiLSTM 等声学建模思路。
- 重点服务于本论文的 CNN-BiLSTM synthetic-speech detector，而不是穷举所有声学模型。

#### 2.3.3 Limitations of Voice-Only Detection

- 说明 voice-only detection 可以发现合成语音风险，但不能理解通话内容中的诈骗意图。
- 特别指出正常文本由合成语音朗读、诈骗文本由真人或高质量生成音色承载时，单靠声学信号会出现边界。

### 2.4 Streaming Multimodal Processing

#### 2.4.1 Multimodal Fusion in Classification Tasks

- 介绍 text + audio fusion 在情感识别、欺诈检测、deepfake detection 或对话理解中的常见作用。
- 区分 early fusion、late fusion、score-level fusion 等基本思路。

#### 2.4.2 Streaming and Incremental Processing

- 讨论 streaming 场景下的 partial transcript、window segmentation、latency、silent windows 和 incremental evidence accumulation。
- 强调动态任务中，模型不仅要判断“是不是风险”，还要回答“什么时候风险足够明显”。

#### 2.4.3 Timeline-Oriented Evaluation

- 介绍 final accuracy / F1 之外的 timing-oriented evaluation 思路。
- 为 Chapter 4 的 risk timeline formulation 和 Chapter 5 的 alert / delay metrics 做铺垫。

### 2.5 Summary of Research Gap

- 集中总结：现有研究分别覆盖 fraud semantics、deepfake detection 和 multimodal classification，但多停留在 static / post-hoc classification 或离线融合评估。
- 核心空白一：缺少面向通话过程的 dynamic risk tracking prototype，难以解释风险如何随通话推进而上升、波动或稳定。
- 核心空白二：缺少同时考虑 warning timing、alert false positives、lead time 和 detection delay 的 timeline metrics。
- 核心空白三：缺少把文本诈骗语义和声学 synthetic-speech risk 放在同一 simulated streaming 框架中分析的实验设计。
- 因此，本研究转向 parallel multimodal analysis + simulated streaming + timeline-based evaluation，而不是只追求整段录音的最终分类准确率。

## Chapter 3: Research Methodology

建议页数：14-16 页。

写作目标：讲基础模型与数据方法，而不是讲系统运行细节。Chapter 3 负责说明训练数据如何准备、文本模型如何建模中文诈骗语义、声学模型如何识别 synthetic speech。动态 fusion、smoothing 和 timeline metrics 放在 Chapter 4；实验数值和 final test set 放在 Chapter 5。

### 3.1 Dataset Preparation and Preprocessing

#### 3.1.1 Text Fraud Dataset and Label Definition

- 说明 ChineseBERT 文本模型使用的中文诈骗文本训练数据来源、样本标签和 fraud / non-fraud 定义。
- 重点解释标签与论文任务的关系：识别诈骗语义风险，而不是识别说话人身份或声学真实性。
- 若使用 TeleAntiFraud 或其他中文诈骗语料，在正文中写清数据来源、样本规模、类别分布和清洗标准。

#### 3.1.2 Synthetic Speech Dataset and Label Definition

- 说明声学模型使用的 synthetic / bona fide speech 数据来源与标签定义。
- 区分 synthetic speech detection 与 fraud semantic detection：声学标签只表示语音真实性风险，不直接等同于诈骗标签。
- 若训练数据来自 ASVspoof、Kaggle 或本地整理版本，正文按最终实验记录填写准确来源。

#### 3.1.3 Preprocessing, Split Strategy, and Reproducibility

- 文本侧说明中文清洗、tokenization、长度截断、训练/验证/测试划分原则。
- 音频侧说明采样率统一、片段长度处理、MFCC 或谱特征提取前的标准化流程。
- 写清随机种子、划分策略和避免数据泄漏的原则；不要把 `metadata_final.csv` 写成训练数据。

### 3.2 ChineseBERT-Based Text Phishing Detection Model

#### 3.2.1 Model Architecture

- 说明 ChineseBERT / BERT-style encoder 如何生成句子级或片段级语义表示。
- 说明 classification head 如何输出 phishing / non-phishing probability，并进一步映射为 text risk score。
- 解释选择 ChineseBERT 的理由：中文语义建模、上下文理解、相比 surface baseline 对话术变体更稳健。

#### 3.2.2 Training Procedure

- 写训练超参数、loss function、optimizer、batch size、epoch、validation strategy 和 early stopping / checkpoint selection。
- 说明 baseline comparison 的位置：surface baseline 与 ChineseBERT 的定量对比放在 Chapter 5.2。
- 本节只写训练过程和模型选择依据，不提前展开最终实验分析。

### 3.3 Synthetic Speech Detection Model

#### 3.3.1 Feature Extraction and Architecture

- 说明声学分支从音频片段提取 MFCC 或最终采用的声学特征。
- 说明 CNN-BiLSTM 的角色：CNN 捕获局部声学模式，BiLSTM 捕获时间序列变化。
- 输出应写成 synthetic-speech probability / voice risk score，用于后续 Chapter 4 的 multimodal risk tracking。

#### 3.3.2 Training Procedure and Results

- 写训练/验证过程、loss、optimizer、checkpoint selection 和主要 validation / held-out 指标。
- 结果只报告模型训练层面的表现，用于证明声学分支具备 synthetic-speech discrimination 能力。
- 不把 audio branch 写成独立诈骗检测器；其系统价值在 Chapter 4/5 中通过 multimodal tracking 与 case-type analysis 说明。

## Chapter 4: System Design and Implementation

建议页数：15-17 页。

写作目标：讲研究方法如何被组织成可运行的 dynamic multimodal risk tracking framework。这里的 implementation 指方法设计层面的系统方案，而不是逐行解释代码。正文不写工程接口、界面组件、文件路径或底层调用关系；这些实现索引放入 Appendix 或 `README_files.md`。

### 4.1 Design Goals and Conceptual Architecture

#### 4.1.1 Design Goals

- 目标不是只输出整段通话的 final label，而是在通话过程中形成可更新的 risk timeline。
- 设计重点包括 dynamic tracking、incremental analysis、multimodal complementarity 和 interpretable risk evolution。
- 强调系统原型服务于研究验证：证明风险可以随通话进程被追踪，而不是声称已经达到运营商级部署。

#### 4.1.2 Conceptual Architecture

- 框架由 audio stream abstraction、text semantic risk branch、voice synthetic-speech risk branch、late fusion module、temporal smoothing module 和 timeline evaluator 组成。
- 各模块的关系用框架图说明，不展开具体代码文件或接口名称。
- 输出对象是按时间排列的 risk points，用于支持后续 Chapter 5 的动态评估。

#### 4.1.3 Research-Level Implementation Meaning

- 本章的 implementation 只说明设计如何落地为可运行原型：输入如何被抽象为连续窗口，模型输出如何被转化为时间序列风险，异常窗口如何被纳入流程。
- 不讨论具体页面组件、数据传输字段、临时文件路径或底层控制流。

### 4.2 Simulated Streaming Formulation

#### 4.2.1 Definition of Simulated Streaming

- 本论文中的 real-time 定义为 uploaded-audio simulated streaming / incremental analysis。
- 完整音频只作为输入载体；分析过程按窗口递增展开，以模拟通话进行中逐步更新风险判断。
- 该设定保留动态追踪问题的核心，同时避免把研究范围扩大到真实电信网络部署。

#### 4.2.2 Window and Step Design

- 以 sliding window 将音频划分为一组有重叠的分析单元。
- 每个窗口只使用当前可见的音频片段和相应转写内容，体现 incremental observation。
- 默认主结果可采用 10 s window / 5 s step；不同 window/step 的敏感性放在 Chapter 5.5 报告。

#### 4.2.3 Incremental Observation Assumption

- 早期窗口信息有限，风险判断允许不确定；随着更多语义和声学证据进入，风险曲线逐步更新。
- trailing short windows、silent windows 和 partial transcript 被视为动态过程中的自然不完美观测，而不是单独的代码异常。
- 该设计使评价重点从 post-hoc classification 转向 in-event warning timing。

### 4.3 Multimodal Risk Tracking Workflow

#### 4.3.1 Text Semantic Risk Branch

- 文本分支把窗口级转写内容转化为 semantic fraud risk。
- 写作重点是语义证据如何随通话推进累积，例如转账、验证码、屏幕共享、冒充身份等意图信号。
- 具体模型结构和训练细节已经在 Chapter 3 说明，本节只讲其在动态框架中的角色。

#### 4.3.2 Voice Synthetic-Speech Risk Branch

- 声学分支把当前窗口音频转化为 synthetic-speech risk。
- 它用于补充文本语义无法感知的 voice-level attack，例如正常文本内容被合成语音承载。
- 不把 voice branch 写成 standalone fraud detector，而是写成辅助声学风险信号。

#### 4.3.3 Parallel Evidence Alignment

- 文本风险与声学风险按同一窗口时间轴对齐。
- 每个窗口形成一个多模态证据单元，再进入 fusion 和 smoothing。
- 对齐逻辑服务于 timeline-level evaluation，而不是只服务于最终分类。

### 4.4 Late Fusion and Temporal Smoothing Design

#### 4.4.1 Fixed Late Fusion Baseline

- baseline 使用 late fusion 将 text risk 和 voice risk 合成为当前窗口风险。
- fixed text-dominant fusion 的动机是多数诈骗行为主要通过语义内容暴露。
- 该 fusion 只是当前 baseline，不应被写成最终最优策略；synthetic_voice 应作为 non-fraud synthetic-speech stress case 讨论，用于检验声学风险是否会造成误报或短暂预警。

#### 4.4.2 Temporal Smoothing

- smoothing 将当前窗口风险与前序状态结合，降低窗口级预测抖动。
- 设计目的不是隐藏错误，而是在动态曲线中提高连续性和可解释性。
- 需要区分 alert-level false positives 与 final-level false positives，避免把平滑后的最终稳定性误写成全程无误报。

#### 4.4.3 Position of Improved Fusion Strategy

- 后续 improved/adaptive fusion strategy 应放在此设计框架下解释：它针对 fixed fusion 在纯声学风险场景中的不足。
- 在最终写作前，根据实验结果决定具体 fusion 方案名称和数学定义。
- 若 learned late fusion 只作为诊断实验，则不要在本章写成主方法。

### 4.5 Dynamic Timeline Metrics

#### 4.5.1 Timeline as Evaluation Object

- 动态评估对象是一条风险时间线，而不是单个最终标签。
- 每个 risk point 包含窗口时间、单模态风险、融合风险、平滑风险和风险等级等概念字段。
- 具体字段 schema 可放 Appendix，不进入正文主叙事。

#### 4.5.2 Alert and Timing Metrics

- alert recall 衡量 fraud case 是否在过程中触发有效预警。
- time to alert 衡量系统从通话开始到首次报警所需时间。
- early-warning lead time 衡量报警相对关键诈骗行为的提前量。
- detection delay 衡量关键行为发生后系统延迟多久才报警。

### 4.6 Design Boundaries and Practical Assumptions

#### 4.6.1 Controlled Benchmark Boundary

- 当前结论来自 controlled benchmark，而不是大规模自然电话流。
- `metadata_final.csv` 和 `audio_final/` 是当前测试口径，旧评估集不纳入主结论。

#### 4.6.2 Simulated Real-Time Boundary

- uploaded-audio simulated streaming 能研究 incremental risk tracking，但不等同于电信运营商网络中的实时部署。
- 正文应避免把 prototype 写成 production system。

#### 4.6.3 Imperfect Observation Handling

- 短窗口、静音窗口、空转写和局部推理失败应被视为动态追踪中的 imperfect observations。
- 设计原则是保持 timeline 连续，并在 Chapter 5 讨论它们对 alert 和 final decision 的影响。

## Chapter 5: Experiments and Analysis of Results

建议页数：25-28 页。

写作目标：全篇结果主章。必须同时报告 ChineseBERT 相对 baseline 的提升、dynamic risk tracking 的整体表现、消融实验、window/step 敏感性和不同 fraud type 下的错误模式。不要只写“模型效果不错”；要解释系统为什么成功、哪里失败、失败说明什么。

### 5.1 Experimental Setup

#### 5.1.1 Current Final Test Set

- `metadata_final.csv`：80 条样本。
- 40 normal，40 fraud。
- 四个 case_type 各 20。
- `synthetic_voice` 属于 normal fraud label，但属于 synthetic-speech / voice-authenticity positive condition；不要把“合成语音”直接写成“诈骗”。
- 当前音频目录为 `test_samples/audio_final/`。

#### 5.1.2 Dynamic Evaluation Configuration

- 主结果目录：`evaluation/reports/final_baseline_w10_s5/`。
- prediction timeline：`evaluation/predictions/final_baseline_w10_s5/dynamic_predictions.json`。
- 当前 baseline window = 10 s，step = 5 s。
- 报告 final classification metrics、alert metrics 和 timing metrics。

#### 5.1.3 Window and Step Experiment Matrix

- Comparison Across Different Windows and Steps 全量测试已完成。
- 测试矩阵：5 s / 2.5 s、10 s / 5 s、20 s / 10 s。
- 结果目录分别为：
  - `evaluation/predictions/final_baseline_w5_s2p5/` 与 `evaluation/reports/final_baseline_w5_s2p5/`。
  - `evaluation/predictions/final_baseline_w10_s5/` 与 `evaluation/reports/final_baseline_w10_s5/`。
  - `evaluation/predictions/final_baseline_w20_s10/` 与 `evaluation/reports/final_baseline_w20_s10/`。
- 写作重点是响应速度、稳定性、误报和延迟之间的 trade-off。

### 5.2 Quantitative Comparison Between Baseline and Proposed Models (ChineseBERT)

#### 5.2.1 Baseline Definition

- Surface baseline 作为文本侧传统方法对照，代表 keyword / surface-feature style detection。
- 该 baseline 用于证明 ChineseBERT 的语义建模价值，不作为最终系统主方法。

#### 5.2.2 Overall Text Classification Comparison

- 报告 Surface Baseline 与 Proposed Chinese-BERT 的整体指标：
  - Surface Baseline：accuracy = 0.7667，F1 = 0.7255，fraud recall = 0.9250。
  - Proposed Chinese-BERT：accuracy = 0.9167，F1 = 0.8889，fraud recall = 1.0000。
- 重点讨论 ChineseBERT 如何降低正常样本误报并保持诈骗召回。

#### 5.2.3 Case-Level Text Model Behavior

- 对比 normal_daily、normal_finance、semantic_fraud、mixed_risk 等文本侧 case behavior。
- 不把该节写成最终系统结果；它只是说明 text branch 为什么适合作为动态风险追踪中的语义风险来源。

### 5.3 Overall Dynamic Evaluation Results

#### 5.3.1 Final Classification Performance

- 重点报告 `fusion_with_smoothing` baseline：
  - accuracy = 1.0。
  - precision = 1.0。
  - recall = 1.0。
  - F1 = 1.0。
  - TP = 40，TN = 40，FP = 0，FN = 0。
- 强调 final FP = 0 不等于过程完全无误报。

#### 5.3.2 Alert Performance

- `fusion_with_smoothing`：
  - fraud_alert_recall = 1.0。
  - normal_alert_false_positive_rate = 0.1。
  - normal_final_false_positive_rate = 0.0。
  - normal_alert_false_positives = 4。
  - normal_final_false_positives = 0。
- 必须区分 alert-level false positives 与 final-level false positives。

#### 5.3.3 Timing Performance

- `fusion_with_smoothing`：
  - mean_time_to_alert_sec = 33.38。
  - mean_early_warning_lead_time_sec = 17.7605。
  - mean_detection_delay_sec = 0.5325。
- 解释 timing metrics 比单一 final label 更贴近 in-event intervention。

### 5.4 Ablation Experiments and Improved Fusion Strategy

#### 5.4.1 Text-Only

- text_only accuracy = 0.925，F1 = 0.9302。
- semantic_fraud 和 mixed_risk 表现强，但 normal false positives 需要讨论。

#### 5.4.2 Voice-Only

- voice_only accuracy = 0.4125，F1 = 0.2295。
- 说明声学分支不能独立承担 fraud detection；对 synthetic_voice 的高响应应解释为 voice-authenticity signal，而不是诈骗正类证据。

#### 5.4.3 Fixed Weighted Fusion and Smoothing

- 对比 text_only、voice_only、fusion_without_smoothing 和 fusion_with_smoothing 四个 baseline variants。
- `fusion_without_smoothing` accuracy = 0.925，F1 = 0.9302；`fusion_with_smoothing` accuracy = 1.0，F1 = 1.0。
- `fusion_with_smoothing` 改善 final normal FP，同时保留 semantic_fraud 和 mixed_risk 的 fraud recall。

#### 5.4.4 Improved / Adaptive Fusion Strategy

- 旧的 final 80 + frozen-v2 100 grouped nested-CV 结果是在 synthetic_voice 被标成 fraud 的口径下生成；采用 corrected semantic-fraud label 后，相关 learned/adaptive fusion 表格必须重新生成后再写入正文。
- 当前主线先报告 fixed 0.8/0.2 + smoothing baseline；不把它写成最终最优，只把它作为已验证、可复现的 conservative baseline。

### 5.5 Comparison Across Different Windows and Steps

#### 5.5.1 Purpose of Window/Step Comparison

- 比较更短窗口带来的更快响应与更高抖动风险。
- 比较更长窗口带来的更稳定风险估计与更大 detection delay。
- 讨论 window/step 设置对 alert recall、normal alert FPR、lead time 和 final F1 的影响。

#### 5.5.2 Completed Full Evaluation Matrix

- 已完成三组 full evaluation：
  - 5 s window / 2.5 s step。
  - 10 s window / 5 s step。
  - 20 s window / 10 s step。
- 10 s / 5 s 是当前 baseline，对应现有 `final_baseline_w10_s5` 结果。

#### 5.5.3 Overall Window/Step Results

- 5 s / 2.5 s：accuracy = 0.725，F1 = 0.7800，fraud alert recall = 0.7000，normal alert FPR = 0.3000，normal final FPR = 0.0500，mean time to alert = 15.4762 s，mean early-warning lead time = 33.2290 s。
- 10 s / 5 s：accuracy = 0.750，F1 = 0.8000，fraud alert recall = 0.6833，normal alert FPR = 0.1500，normal final FPR = 0.0000，mean time to alert = 33.2976 s，mean early-warning lead time = 16.5956 s。
- 20 s / 10 s：accuracy = 0.6625，F1 = 0.7097，fraud alert recall = 0.5667，normal alert FPR = 0.1500，normal final FPR = 0.0000，mean time to alert = 39.8624 s，mean early-warning lead time = 10.7674 s。

#### 5.5.4 Interpretation

- 短窗口 5 s / 2.5 s 提供最快报警和最大 lead time，但 normal alert / final false positives 更高。
- 10 s / 5 s 在 final F1、normal final FP、alert stability 和 timing 之间最平衡，适合作为当前 baseline 主口径。
- 长窗口 20 s / 10 s 降低了过程波动，但 fraud recall、fraud alert recall 和 early-warning lead time 明显下降。
- 三组结果共同支持：window/step 是 dynamic risk tracking 中的关键设计参数，而不是纯工程超参。

#### 5.5.5 Case-Type Effect of Window Length

- mixed_risk 在三组设置下 final recall 均保持 1.0，说明强语义 + 声学风险最稳定。
- semantic_fraud 在 5 s / 2.5 s 下 final recall = 0.95，在 10 s / 5 s 下 final recall = 1.0，但在 20 s / 10 s 下下降到 0.65，说明过长窗口会削弱部分语义风险的及时累积。
- synthetic_voice 应按 normal fraud label + synthetic-speech acoustic condition 重新解释；旧的 synthetic_voice recall 口径不再适用。
- normal_daily 在 10 s / 5 s 和 20 s / 10 s 下 final FP = 0；5 s / 2.5 s 下 final FP = 1，说明短窗口更易产生瞬时误报。

### 5.6 Error Analysis by Fraud Type

#### 5.6.1 Normal Daily

- `fusion_with_smoothing` final FP = 0。
- 但 normal alert false positives = 3，需要讨论 transient risk spikes。

#### 5.6.2 Semantic Fraud

- `fusion_with_smoothing` final recall = 1.0。
- 说明文本语义追踪对 fraud semantics 有效。

#### 5.6.3 Mixed Risk

- `fusion_with_smoothing` final recall = 1.0。
- text + voice signals 都能支持风险上升。

#### 5.6.4 Synthetic Voice as Non-Fraud Acoustic Stress

- 不是 fraud false negative；它是正常文本由合成语音朗读的 acoustic stress case。
- 分析重点改为：voice_only 是否会把正常合成语音推成诈骗误报，以及 fusion/smoothing 是否能抑制这种声学误报。
- 讨论：声学分支适合作为 authenticity cue，不应直接等同于 fraud decision。

## Chapter 6: Conclusions

建议页数：2-3 页。

写作目标：简洁回应 Chapter 1 的 research objectives，不设置独立 Future Work 章节。结论应总结已经完成的工作、主要发现、贡献和创新点，同时用 evidence boundary 的方式说明 controlled benchmark 与 simulated streaming 的范围。

### 6.1 Main achievements and findings

- 回应 objective 1：完成 simulated streaming multimodal risk tracking prototype，并将完整音频转化为 window-level risk timeline。
- 回应 objective 2：实现 text risk、voice risk、fusion risk 和 smoothed risk 的动态输出，使系统不只给出 final label。
- 回应 objective 3：通过 dynamic metrics 评估 alert recall、time to alert、early-warning lead time 和 detection delay。
- 回应 objective 4：实验显示 semantic_fraud 和 mixed_risk 能被有效追踪，normal_daily final false positives 可被 smoothing 控制。
- 主要边界：synthetic-speech risk 与 fraud semantic risk 必须分开解释；当前结论来自 controlled final benchmark 和 uploaded-audio simulated streaming。

### 6.2 Contributions and innovations

- 方法贡献：将 telecom fraud detection 从 post-hoc classification 扩展到 in-event dynamic risk tracking。
- 框架贡献：构建 parallel multimodal analysis framework，把中文诈骗语义风险和 synthetic-speech 声学风险放在同一时间轴上分析。
- 评估贡献：引入 timeline-based metrics，区分 final decision performance、alert-level behavior 和 warning timing。
- 实证贡献：基于 normal_daily、semantic_fraud、mixed_risk、synthetic_voice 四类 controlled cases，揭示不同攻击类型下的能力边界。
- 设计启示：fixed fusion baseline 能解释当前系统行为，但最终论文应根据后续实验进一步确定 improved/adaptive fusion strategy。

## Suggested Figures and Tables

### Figures

- Figure 1.1：Post-hoc classification vs in-event dynamic risk tracking。
- Figure 3.1：ChineseBERT-based text phishing detection model。
- Figure 3.2：CNN-BiLSTM synthetic speech detection model。
- Figure 4.1：Conceptual architecture of dynamic multimodal risk tracking。
- Figure 4.2：Simulated streaming formulation and risk timeline generation。
- Figure 5.1：Risk curves for representative samples。
- Figure 5.2：Window/step comparison of dynamic risk curves。
- Figure 5.3：Case-type comparison of final and alert behavior。

### Tables

- Table 3.1：Training datasets and preprocessing summary。
- Table 3.2：Text and audio model training settings。
- Table 4.1：Design components and assumptions for dynamic risk tracking。
- Table 5.1：Final test set composition from `metadata_final.csv`。
- Table 5.2：ChineseBERT vs surface baseline comparison。
- Table 5.3：Overall dynamic evaluation results。
- Table 5.4：Ablation and fusion strategy comparison。
- Table 5.5：Window/step comparison summary。
- Table 5.6：Case-type error analysis summary。

## Appendix Plan

Appendix 只放相关但非关键内容，不进入正文页数。

- Appendix A：Detailed file map，可引用 `README_files.md`。
- Appendix B：Full dynamic metric tables。
- Appendix C：Representative timeline JSON schema。
- Appendix D：Additional text-model stress test details。
- Appendix E：Learned late fusion diagnostic note。
- Appendix F：Excluded historical experiments, including gated / progression / old long baseline outputs。

## 写作检查清单

- 第 1 章不要写成行业背景堆叠；必须清楚回答“为什么 post-hoc 不够”。
- 第 2 章不在每节末尾分散写 gap；统一在 2.5 总结 research gap。
- 第 3 章讲训练数据、ChineseBERT 和 synthetic-speech model，不写 final test set 和动态结果。
- 第 4 章讲方法设计层面的系统方案，不写工程接口、界面组件、文件路径或底层调用关系。
- 第 5 章必须包含 ChineseBERT comparison、dynamic results、ablation、window/step full test 和 synthetic_voice failure mode。
- 第 6 章不设独立 Future Work；只写 main achievements、findings、contributions 和 innovations。
- 全文避免把 controlled benchmark 写成真实外部泛化。
- 全文避免把 simulated streaming 写成 telecom carrier deployed real-time system。
- 全文避免把 learned late fusion 写成主方法。
