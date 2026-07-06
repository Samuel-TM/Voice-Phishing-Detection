# Oral Defence PPT Structure and Slide Text

题目：Real-Time Dynamic Risk Tracking of Telecommunications Fraud Based on Parallel Multimodal Analysis

使用方式：
- PPT 页面文字建议使用英文，符合 HKU oral defence 场景。
- 每页下面的“讲解重点”是你练习时用的中文提示，不一定放进 PPT。
- 15 分钟答辩建议控制在 14 页左右，平均每页 45-65 秒。背景页不要展开。

---

## Slide 1. Title

**Slide text**

Real-Time Dynamic Risk Tracking of Telecommunications Fraud Based on Parallel Multimodal Analysis

Oral Defence

M.Sc.(Eng.) in Robotics and Intelligent Systems  
Department of Data and Systems Engineering, The University of Hong Kong

**讲解重点**

开场只用 15-20 秒。直接说本研究关注的不是“整段录音最后是不是诈骗”，而是“通话过程中风险如何逐步出现，以及系统能不能在时间线上给出可解释的预警”。

---

## Slide 2. Defence Message

**Slide text**

This project is about dynamic risk tracking, not only post-hoc classification.

What I built:
- A simulated streaming prototype for uploaded audio and browser audio chunks
- Parallel text-risk and audio-risk inference on sliding windows
- Score-level fusion, temporal smoothing, and a risk timeline
- A final controlled benchmark with timing-oriented evaluation

Main claim:
The system can track in-event risk evolution, but the results also reveal a clear failure mode in fixed fusion for acoustic-only attacks.

**讲解重点**

这一页要定调：你不是来讲泛泛背景，而是讲你做出来的系统、实验和发现。最后一句很关键：不要把结果包装成全部成功，要主动指出 fixed fusion 对 synthetic_voice 的问题，这会显得你真正理解系统。

---

## Slide 3. Research Objective

**Slide text**

Objective:
Build and evaluate a prototype that detects telecommunications fraud risk as a call unfolds.

Research tasks:
1. Convert full audio into incremental sliding-window analysis
2. Estimate semantic fraud risk from transcribed speech
3. Estimate acoustic synthetic-speech risk from the audio window
4. Fuse both risks into a smoothed timeline
5. Evaluate not only final classification, but also alert timing and case-type behaviour

**讲解重点**

不要讲太多 telecom fraud 背景。强调任务定义：动态追踪、双模态、时间线评估。老师已经看过报告，这里只需要证明你清楚自己的 objective。

---

## Slide 4. System I Implemented

**Slide text**

Core runtime path:

`/api/stream_audio_analysis`

Uploaded audio  
-> sliding windows  
-> ASR  
-> text risk + audio risk  
-> weighted fusion  
-> exponential smoothing  
-> timeline output

Also implemented:
- `/api/live_audio_chunk` and `/api/live_audio_finish` for browser audio chunks
- Front-end risk curve, timeline table, replay, and current transcript display
- Structured prediction outputs for offline evaluation

**讲解重点**

这里要说清楚 current deliverable 是 uploaded-audio simulated streaming，而不是声称已经做成 telecom operator network deployment。可以补一句：live chunk endpoint 有实现，但最终实验主线用 `/api/stream_audio_analysis`，因为它稳定、可复现、适合答辩演示。

---

## Slide 5. Method: Parallel Multimodal Timeline

**Slide text**

Per-window processing:

1. Window segmentation  
   10 s window / 5 s step as the main setting

2. ASR and text branch  
   Chinese-BERT risk from current-window text and rolling context

3. Audio branch  
   MFCC-20 + CNN-BiLSTM synthetic-speech detector

4. Baseline fusion  
   `fused = 0.8 * text_score + 0.2 * voice_score`

5. Temporal smoothing  
   `smoothed = 0.65 * previous + 0.35 * current`

6. Risk level and alert timeline  
   Normal, Suspicious, High Risk, Critical

**讲解重点**

这里要讲“为什么这样设计”：文本分支负责诈骗语义，音频分支负责合成语音风险；fusion 是 text-dominant，因为诈骗意图主要在话术里；smoothing 是为了减少单窗口尖峰误报。也要说明这是 baseline，不是唯一可能的最终规则。

---

## Slide 6. Final Benchmark

**Slide text**

Final controlled benchmark:

80 audio samples from `metadata_final.csv` and `audio_final/`

Case types:
- 20 normal_daily: real normal speech
- 20 synthetic_voice: synthetic voice reading normal scripts
- 20 mixed_risk: synthetic voice reading fraud scripts
- 20 semantic_fraud: fraud scripts generated with high-quality voice

Labels:
- 20 normal
- 60 fraud

Why this split matters:
It separates semantic risk, acoustic risk, and multimodal risk.

**讲解重点**

这一页要说明你的测试集不是随便拼的。四类样本分别测试不同能力：正常真人控制误报；synthetic_voice 测纯声学攻击；mixed_risk 测文本和声学同时有风险；semantic_fraud 测声学可能失效时文本语义是否能撑住。

---

## Slide 7. Text-Risk Branch

**Slide text**

Text model role:
Detect fraud semantics in partial transcripts.

External text-level stress test:

| Model | Accuracy | Precision | Recall | F1 | Normal FP Rate | Fraud Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Surface baseline | 0.7667 | 0.5968 | 0.9250 | 0.7255 | 0.3125 | 0.9250 |
| Proposed Chinese-BERT | 0.9167 | 0.8000 | 1.0000 | 0.8889 | 0.1250 | 1.0000 |

Finding:
Chinese-BERT reduces surface-rule false alarms while preserving fraud recall.

**讲解重点**

不要把 text model 讲成孤立贡献。它服务于动态系统中的窗口级语义风险。重点解释 surface baseline 的局限：关键词/规则对金融正常语句容易过敏，Chinese-BERT 更能看上下文语义。

---

## Slide 8. Audio-Risk Branch

**Slide text**

Audio model role:
Provide an auxiliary synthetic-speech risk signal.

Model:
- MFCC-20 features
- CNN-BiLSTM classifier
- Trained with ASVspoof 2019 LA + In-the-Wild Audio Deepfake

Deployment evidence:
- Decision threshold: 0.7495
- In-the-Wild OOD macro-F1: 0.9345
- Real recall: 0.9420
- Fake recall: 0.9270

Important boundary:
The audio branch detects voice authenticity risk, not fraud intention.

**讲解重点**

要主动说清楚边界。audio branch 不能理解转账、验证码、冒充身份这些语义，它只能辅助发现 synthetic speech。所以系统最终不能只靠 voice-only。

---

## Slide 9. Main Dynamic Result: 10 s / 5 s Baseline

**Slide text**

Main baseline: `final_baseline_w10_s5`, `fusion_with_smoothing`

| Metric | Result |
| --- | ---: |
| Samples | 80 |
| Accuracy | 0.7500 |
| Precision | 1.0000 |
| Recall | 0.6667 |
| F1 | 0.8000 |
| TP / TN / FP / FN | 40 / 20 / 0 / 20 |
| Fraud alert recall | 0.6833 |
| Normal alert false positives | 3 / 20 |
| Mean early-warning lead time | 16.60 s |

Interpretation:
The baseline is conservative and avoids final normal false positives, but misses one important fraud subtype.

**讲解重点**

这页不能只说 accuracy。要讲 precision=1 和 FP=0 说明最终判断很保守；recall=0.6667 和 FN=20 暗示有整类风险没被最终识别。马上用下一页解释是哪一类。

---

## Slide 10. Case-Type Behaviour

**Slide text**

Baseline case-type results (`fusion_with_smoothing`, 10 s / 5 s):

| Case type | Final recall | Alert recall | Mean time to alert | Key behaviour |
| --- | ---: | ---: | ---: | --- |
| mixed_risk | 1.00 | 1.00 | 24.0 s | Strong semantic + acoustic evidence |
| semantic_fraud | 1.00 | 1.00 | 42.8 s | Text semantics work, but smoothing delays alert |
| normal_daily | FP = 0 | alert FP = 3 | - | Final false positives controlled |
| synthetic_voice | 0.00 | 0.05 | 30.0 s | Acoustic-only risk is suppressed by fixed fusion |

Finding:
The system succeeds when fraud semantics are present, but fixed text-dominant fusion is weak for pure synthetic-voice attacks.

**讲解重点**

这是答辩最重要的一页之一。老师很可能问：为什么 synthetic_voice 全挂？你要回答：这些样本是正常文本 + 合成语音，文本分数低，而 baseline 融合 80% 依赖文本，再加 smoothing，会把 audio-only evidence 压低。这不是 bug，而是设计权衡暴露出的 failure mode。

---

## Slide 11. Ablation: What Each Component Contributes

**Slide text**

10 s / 5 s ablation summary:

| Variant | Accuracy | Precision | Recall | F1 | Final normal FP | Fraud alert recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| text_only | 0.7000 | 0.8913 | 0.6833 | 0.7736 | 5 | 0.7667 |
| voice_only | 0.5125 | 1.0000 | 0.3500 | 0.5185 | 0 | 0.8833 |
| fusion_without_smoothing | 0.7000 | 0.8913 | 0.6833 | 0.7736 | 5 | 0.7500 |
| fusion_with_smoothing | 0.7500 | 1.0000 | 0.6667 | 0.8000 | 0 | 0.6833 |

Finding:
Smoothing improves final stability and removes final normal false positives, but it can delay or suppress alerts.

**讲解重点**

解释每一列背后的意义：text_only 语义能力强但正常误报较多；voice_only 能早报警但最终召回低；smoothing 带来稳定性和 FP 控制，但牺牲部分 recall/alert speed。这正是动态系统需要 trade-off 的地方。

---

## Slide 12. Window / Step Sensitivity

**Slide text**

Window-step comparison under the same baseline:

| Window / step | Accuracy | F1 | Final FP | Alert recall | Normal alert FPR | Mean lead time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 s / 2.5 s | 0.7250 | 0.7800 | 1 | 0.7000 | 0.3000 | 33.23 s |
| 10 s / 5 s | 0.7500 | 0.8000 | 0 | 0.6833 | 0.1500 | 16.60 s |
| 20 s / 10 s | 0.6625 | 0.7097 | 0 | 0.5667 | 0.1500 | 10.77 s |

Selected setting:
10 s / 5 s is the most balanced setting for the current prototype.

**讲解重点**

不要说 10s/5s 每个指标都最好。正确说法：5s 更早但更容易 alert FP；20s 更平滑但召回和动态预警弱；10s/5s 是最终 precision、F1、alert stability 和 timing 的平衡点。

---

## Slide 13. Learned Late Fusion as a Diagnostic Variant

**Slide text**

Why I tested learned late fusion:
The fixed `0.8 text + 0.2 voice` rule exposed a failure mode in synthetic_voice.

Leakage control:
- Sample-level 5-fold nested CV
- Held-out folds contain complete samples, not random windows
- Threshold selected only inside training folds
- Prefix-only numerical timeline features
- No `case_type`, `sample_id`, keywords, metadata, or future windows

Sample-level CV result:
- Accuracy: 0.8625
- Precision: 0.9804
- Recall: 0.8333
- F1: 0.9009
- TP / TN / FP / FN: 50 / 19 / 1 / 10

Position:
Useful diagnostic evidence, but not the deployed runtime baseline.

**讲解重点**

老师可能会追问：既然 learned fusion 更好，为什么不用它当主方法？回答：因为 final set 小，full-fit 容易过拟合；sample-level CV 更可信，但它仍是离线 decision-layer variant，还没有接入 service path 做端到端 runtime validation。因此论文主线保持 baseline，learned fusion 作为诊断和改进方向的证据。

---

## Slide 14. What the Results Show

**Slide text**

Main findings:

1. Timeline evaluation is necessary  
Final F1 alone cannot show warning timing, alert false positives, or delayed detection.

2. The two modalities cover different risks  
Text captures fraud intention; audio captures synthetic-speech risk.

3. Conservative smoothing is useful but not free  
It reduces final normal false positives, but may delay or suppress acoustic-only alerts.

4. The current baseline is interpretable  
Its failure mode is clear and measurable, which supports further evidence-based refinement.

**讲解重点**

这一页是理解程度总结。要把“结果好不好”转成“我从结果里学到了什么”。尤其强调 dynamic evaluation：first crossing, time to alert, lead time, detection delay，这些都是整段分类看不到的。

---

## Slide 15. Contributions and Conclusion

**Slide text**

Contributions:

1. Built a working simulated-streaming multimodal fraud-risk tracking prototype
2. Integrated ASR, Chinese-BERT text risk, CNN-BiLSTM audio risk, late fusion, and smoothing
3. Constructed a final controlled benchmark separating normal, semantic, acoustic, and multimodal risks
4. Evaluated the system with final classification and timeline-oriented metrics
5. Identified a concrete failure mode in fixed fusion and validated learned fusion as a cautious offline diagnostic

Conclusion:
The project demonstrates dynamic in-event risk tracking and, equally important, explains where the current baseline succeeds and where it fails.

**讲解重点**

结尾不要讲空泛 future work。重点说：我的工作已经形成一个可运行系统、一个可复现实验口径、以及对结果边界的清楚理解。最后一句可以说：This is why I treat the project as a dynamic risk tracking prototype, not a static classifier.

