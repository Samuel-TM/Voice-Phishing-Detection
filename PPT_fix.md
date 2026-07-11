# PPT_fix.md

实际 Beamer 源文件为：

`/Users/sunjiashan/Material/HKU/Dissertation/Oral Defence/HKU_Beamer_Slides-master/defense.tex`

---

## 1. 本轮修改原则

根据新的论文叙述和 corrected 180 实验结果，PPT 需要从旧的“80-sample + learned acoustic alert”叙事，改为：

> This project extends voice-phishing detection from post-hoc binary classification to in-event dynamic risk tracking. The system generates a sliding-window risk timeline and evaluates warning ability using alert recall, alert time, lead time, and detection delay. The corrected 180-sample benchmark contains five controlled multimodal case types, so the evaluation exposes how fixed and learned fusion behave under semantic fraud, acoustic authenticity risk, benign financial interference, benign synthetic speech, and mixed attacks.

核心改动方向：

- 保留旧 PPT 中稳定的系统背景、架构、模块介绍和大部分实验展示页面。
- 删除旧的 Learned Late Fusion 正向主线，不再把 learned fusion 描述成“解决 acoustic-only synthetic voice alert”的成功方案。
- 明确 `voice_score` 不是 phishing 的充分证据。它只能作为辅助 acoustic authenticity risk evidence。
- 重点突出两个贡献：
  1. 五类 multimodal case-type evaluation。
  2. alert timing / lead time / detection delay 等时间指标。
- 主方法保持 Fixed fusion + smoothing。
- Learned late fusion 只作为 diagnostic comparison。

---

## 2. 旧幻灯片处理决策

用户指定的保留/删除策略如下，应作为修改 `defense.tex` 的优先约束。

| 旧页码 | 当前主题 | 处理方式 | 说明 |
|---:|---|---|---|
| Slide 1 | Title | 保留 | 只检查日期和标题 |
| Slide 2 | Static classification -> dynamic tracking | 保留并强化 | 放到开头，承接总贡献 |
| Slide 3 | Research objective and scope | 保留并更新 | 加入 corrected 180 和 timing metrics |
| Slide 4 | Runtime pipeline | 保留 | 强调 sliding-window risk timeline |
| Slide 5 | Parallel risk modelling | 保留并修正 | 强调 voice score 是辅助真实性风险，不是诈骗充分证据 |
| Slide 6 | Baseline fusion rationale | 保留并强化 | 改成 final main method: fixed 8:2 + smoothing |
| Slide 7 | Runtime validation | 保留，但必须按新样本重跑 | 需要写明脚本调用 |
| Slide 8 | Enhanced decision layer: learned fusion | 删除/替换 | 换成五类 case-type evaluation 贡献页 |
| Slide 9 | Benchmark design | 保留但替换内容 | 从 80 samples 改成 corrected 180 samples |
| Slide 10 | Text branch evaluation | 保留 | 作为模块证据，不抢主线 |
| Slide 11 | Audio branch evaluation | 保留并修正 | 声学真实性检测，不等于诈骗判断 |
| Slide 12 | Main performance | 保留但替换数字 | 使用 corrected 180 main result |
| Slide 13 | Risk-source coverage | 保留但替换表述 | 用五类 case-type breakdown |
| Slide 14 | Synthetic voice timeline: fixed vs learned | 删除/替换 | 旧叙事错误，SV 是 normal，不应当作 learned 成功案例 |
| Slide 15 | Learned late fusion result | 删除/替换 | 换成 fixed vs learned trade-off，不作为主方法 |
| Slide 16 | Ablation/window-step | 保留并更新 | 使用 corrected 180 消融和 window/step 比较 |
| Slide 17 | Weight sensitivity / fusion strategy | 保留/新增 | 放 fixed weight sweep 和策略选择 |
| Slide 18 | Contributions/conclusion | 保留并重写 | 明确两项核心贡献和最终结论 |

注意：如果当前 `defense.tex` 中实际页数少于用户上传旧 PPT 的页数，仍按“旧 PPT 页码语义”理解；最终应按下面的新逻辑顺序重排。

---

## 3. 推荐新页面顺序

建议最终控制在 15-18 页。保留页面不一定保持原来的位置，按逻辑重排如下。

| 新顺序 | 来源 | 标题建议 | 目的 |
|---:|---|---|---|
| 1 | 旧 Slide 1 | Title | 标题页 |
| 2 | 旧 Slide 2 | From Static Classification to Dynamic Risk Tracking | 总研究范式转变 |
| 3 | 旧 Slide 3 | Research Objective and Contributions | 提前点出两项核心贡献 |
| 4 | 旧 Slide 4 | Implemented Runtime Pipeline | 系统如何生成风险时间线 |
| 5 | 旧 Slide 5 | Parallel Risk Modelling | 文本语义风险 + 声学真实性辅助风险 |
| 6 | 旧 Slide 6 | Fixed Fusion and Temporal Smoothing | 主方法公式与设计理由 |
| 7 | 替换旧 Slide 8 | Contribution I: Five Multimodal Case Types | 五类 case-type evaluation |
| 8 | 保留旧 Slide 9 | Corrected 180-Sample Benchmark Design | 数据集与标签原则 |
| 9 | 保留旧 Slide 7 | Runtime Validation on Corrected Samples | 用新样本重跑 route equivalence / latency |
| 10 | 保留旧 Slide 10 | Text-Risk Branch Evaluation | 模块证据，可压缩 |
| 11 | 保留旧 Slide 11 | Audio-Risk Branch Evaluation | 模块证据，并声明不是诈骗充分证据 |
| 12 | 保留旧 Slide 12 | Main Dynamic Tracking Performance | corrected 180 主结果 |
| 13 | 保留旧 Slide 13 | Case-Type Breakdown | 五类样本下的边界与 trade-off |
| 14 | 替换旧 Slide 14 | Why Voice Score Alone Is Not Phishing Evidence | 删除 synthetic voice learned-success 旧页 |
| 15 | 替换旧 Slide 15 | Fixed Fusion vs Learned Late Fusion | learned 作为诊断对比 |
| 16 | 保留旧 Slide 16 | Ablation and Timing Metrics | 消融 + alert timing |
| 17 | 保留/新增旧 Slide 17 | Fusion Weight Sensitivity | w=0.7/0.8/0.9/1.0 对比 |
| 18 | 保留旧 Slide 18 | Contributions and Conclusion | 汇总贡献与结论 |

如果答辩时间很紧：

- Text branch / audio branch 可以合并成 1 页。
- Runtime validation 可以放到 backup，但用户希望 Slide 7 保留，因此主版本建议保留。

---

## 4. 需要突出的贡献表述

### 4.1 贡献一：五类 multimodal case-type evaluation

建议在 Slide 3 和新 Slide 7 同时出现。

中文解释给自己讲：

> 我不是只做 normal/phishing 二分类验证，而是设计了五类受控样本，分别测试正常日常语音、正常金融干扰、正常内容的合成语音、真人语音诈骗内容、以及合成语音诈骗内容。这样可以系统观察文本语义风险、声学真实性风险和混合风险下，固定融合与学习式融合分别会出现什么边界。

PPT 英文表述：

> I designed a five-way multimodal case-type benchmark, rather than a simple normal/phishing split. The benchmark separates benign daily speech, benign financial hard negatives, benign synthetic speech, semantic fraud, and mixed semantic-acoustic attacks, exposing how fusion behaves under different evidence sources.

### 4.2 贡献二：时间指标 evaluation

建议在 Slide 3、Slide 12、Slide 16 出现。

中文解释给自己讲：

> 我的系统不是只输出最终分类，而是在通话过程中不断输出风险曲线。因此评估也不能只看 Accuracy、Precision、Recall、F1，还需要看什么时候报警，是否提前于诈骗关键事件，漏报时延是多少。

PPT 英文表述：

> The evaluation introduces alert-timing metrics, including alert recall, alert time, early-warning lead time, and detection delay. These metrics evaluate whether the system can warn during the event, not only whether the final label is correct.

### 4.3 总贡献一句话

PPT/答辩口径：

> This project extends voice-phishing detection from post-hoc binary classification to in-event dynamic risk tracking by generating sliding-window risk timelines and evaluating both final decisions and warning timing across five controlled multimodal case types.

---

## 5. 新实验口径与结果

### 5.1 Corrected 180 sample composition

| Case Type | Count | Label | Voice Source | Purpose |
|---|---:|---|---|---|
| normal_daily (ND) | 40 | normal | real recording | benign daily-speech false-positive control |
| normal_finance (NF) | 20 | normal | real recording | hard negative with financial vocabulary |
| synthetic_voice (SV) | 40 | normal | Google TTS | benign synthetic speech; tests acoustic false alarms |
| semantic_fraud (SF) | 40 | fraud | real recording | fraud semantics with real speech |
| mixed_risk (MR) | 40 | fraud | Google TTS | fraud semantics plus synthetic speech |

核心标签原则：

> Fraud is defined by semantic/behavioral content, not by synthetic speech. Synthetic voice reading benign content is normal for fraud detection.

### 5.2 Main method result: Fixed 8:2 fusion + smoothing

| Metric | Result |
|---|---:|
| Samples | 180 |
| Normal / Fraud | 100 / 80 |
| Accuracy | 0.9278 |
| Precision | 0.9718 |
| Recall | 0.8625 |
| F1 | 0.9139 |
| TP / TN / FP / FN | 69 / 98 / 2 / 11 |
| Fraud alert recall | 0.9250 |
| Normal final FPR | 0.0200 |
| Mean alert time | 23.55s |
| Mean early-warning lead time | 24.56s |
| Mean detection delay | 0.02s |

Interpretation:

> Fixed fusion with smoothing is selected because it provides a stable, interpretable, low-FP dynamic warning rule with strong early-warning lead time.

### 5.3 Corrected 180 ablation table

| Variant | Recall | Prec | F1 | FPR | NF FP | SV FP | SF Recall | MR Recall | AlertT | LeadT | Role |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| text_only | 0.900 | 0.857 | 0.878 | 0.120 | 3/20 | 1/40 | 33/40 | 39/40 | 16.8s | 31.2s | semantic baseline |
| voice_only | 0.175 | 0.368 | 0.237 | 0.240 | 1/20 | 22/40 | 2/40 | 12/40 | 15.9s | 33.1s | acoustic diagnostic |
| fusion_without_smoothing | 0.900 | 0.900 | 0.900 | 0.080 | 2/20 | 0/40 | 33/40 | 39/40 | 17.2s | 30.8s | fusion ablation |
| **fusion_with_smoothing** | **0.863** | **0.972** | **0.914** | **0.020** | **2/20** | **0/40** | **32/40** | **37/40** | **23.5s** | **24.6s** | **Main method** |
| learned_late_fusion | 0.913 | 0.936 | 0.924 | 0.050 | 3/20 | 2/40 | 33/40 | 40/40 | 45.5s | 3.1s | diagnostic comparison |

Key message:

> Learned late fusion has slightly higher final F1, but fixed fusion has lower false positives and much stronger early-warning lead time.

### 5.4 Case-type breakdown for main method

| Case Type | Label | Main Result | Interpretation |
|---|---|---|---|
| normal_daily | normal | 0/40 final FP | clean benign control |
| normal_finance | normal | 2/20 final FP | hardest normal class |
| synthetic_voice | normal | 0/40 final FP | benign synthetic speech is not misclassified as fraud |
| semantic_fraud | fraud | 32/40 final TP | main remaining false-negative source |
| mixed_risk | fraud | 37/40 final TP | multimodal risk is mostly detected |

Key message:

> The case-type design reveals where the system is reliable and where it struggles. This is more informative than reporting only overall Accuracy, Precision, Recall, and F1.

### 5.5 Fusion weight sweep

| Strategy | Text Weight | Precision | Recall | F1 | FP | FN | LeadT | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fixed_7_3_with_smoothing | 0.70 | 1.000 | 0.788 | 0.881 | 0 | 17 | 25.0s | very conservative, loses fraud recall |
| **fixed_8_2_with_smoothing** | **0.80** | **0.972** | **0.863** | **0.914** | **2** | **11** | **24.6s** | selected mainline |
| fixed_9_1_with_smoothing | 0.90 | 0.959 | 0.875 | 0.915 | 3 | 10 | 25.5s | similar but more text-dominant |
| text_only_with_smoothing | 1.00 | 0.960 | 0.900 | 0.929 | 3 | 8 | 25.5s | strong reference, but removes multimodal fusion |

Important warning:

> This sweep is sensitivity analysis. It must not be presented as post-hoc test-set tuning. The selected mainline remains the pre-defined 8:2 multimodal fusion rule.

---

## 6. Slide-specific modification plan

### Slide 1: Title

Status: 保留。

Update:

- 保持题目不变。
- 检查日期是否为实际答辩日期。
- 不放实验数字。

### Slide 2: From Static Classification to Dynamic Risk Tracking

Status: 保留并强化。

Keep:

- `Is this call fraud?` -> `When does risk emerge during the call?`

Add one sentence:

> The project evaluates not only the final label, but also when the warning appears during the call.

### Slide 3: Research Objective and Contributions

Status: 保留但建议改标题，从 `Research Objective and Scope` 改为 `Research Objective and Contributions`。

Replace/insert content:

1. Objective:

> Build a prototype that transforms evolving speech into a causal, window-level fraud-risk timeline.

2. Contribution 1:

> Five-way multimodal case-type evaluation: ND, NF, SV, SF, and MR separate semantic risk, acoustic authenticity risk, benign financial interference, and mixed attacks.

3. Contribution 2:

> Timing-aware evaluation: alert recall, alert time, early-warning lead time, and detection delay evaluate in-event warning capability.

4. Boundary:

> The corrected 180-sample benchmark supports controlled robustness evidence, not broad real-world deployment generalization.

### Slide 4: Implemented Runtime Pipeline

Status: 保留。

Keep:

- Audio window -> ASR / MFCC -> text risk / voice risk -> fixed fusion -> smoothing -> risk timeline.

Update keyline:

> The runtime output is a dynamic risk timeline produced by sliding-window causal analysis, not only a final binary label.

### Slide 5: Parallel Risk Modelling

Status: 保留并修正。

Required correction:

The voice branch must not be described as phishing evidence.

Use:

| Branch | Output Meaning |
|---|---|
| Text branch | semantic fraud-intention risk |
| Audio branch | acoustic authenticity risk; auxiliary evidence only |

Add clear note:

> A high `voice_score` means the speech may be synthetic. It is not sufficient evidence of phishing unless semantic fraud evidence is also present.

### Slide 6: Fixed Fusion and Temporal Smoothing

Status: 保留并强化为主方法页。

Keep formulas:

```text
fused_score = 0.8 * text_score + 0.2 * voice_score
smoothed_score_t = 0.65 * smoothed_score_{t-1} + 0.35 * fused_score_t
```

Update rationale:

> Text risk is the primary evidence for fraud intention. Voice risk is an auxiliary authenticity cue. The fixed 8:2 design prevents benign synthetic speech from being treated as fraud solely because of its acoustic source.

Add small result box:

| Metric | Fixed 8:2 + smoothing |
|---|---:|
| F1 | 0.9139 |
| Precision | 0.9718 |
| Normal final FPR | 0.0200 |
| Mean lead time | 24.56s |

### Slide 7: Runtime Validation

Status: 保留，但必须根据新样本重新生成。

Purpose:

- 验证 `/api/stream_audio_analysis` 与 `/api/live_audio_chunk` 路径的一致性。
- 验证 browser-style chunk processing latency 是否快于真实音频时长。
- 不用于证明模型泛化，只用于 runtime feasibility。

Use scripts:

```bash
conda activate dissertation
cd /Users/sunjiashan/Material/HKU/Dissertation/Code/Voice-Phishing-Detection
python evaluation/baseline_route_equivalence_latency.py \
  --metadata test_samples/metadata_all_corrected.csv \
  --audio-dir test_samples/audio_all \
  --output-dir evaluation/reports/baseline_route_equivalence_latency_corrected180
python evaluation/generate_route_equivalence_latency_figure.py \
  --report-dir evaluation/reports/baseline_route_equivalence_latency_corrected180 \
  --output-dir evaluation/figures \
  --basename route_equivalence_browser_latency_corrected180
```

If the current script does not yet support `--metadata` / `--audio-dir`, modify `evaluation/baseline_route_equivalence_latency.py` before rerunning so the selected samples come from `metadata_all_corrected.csv` and `audio_all/`.

Expected output assets:

- `evaluation/reports/baseline_route_equivalence_latency_corrected180/baseline_route_equivalence_latency_report.json`
- `evaluation/reports/baseline_route_equivalence_latency_corrected180/baseline_browser_chunk_latency.csv`
- `evaluation/figures/route_equivalence_browser_latency_corrected180.pdf`
- `evaluation/figures/route_equivalence_browser_latency_corrected180.png`

Slide wording:

> Runtime validation uses corrected benchmark audio to test route equivalence and in-process chunk-processing latency. It supports feasibility of simulated streaming and browser chunk processing, not internet-scale production latency.

### Replacement for old Slide 8: Contribution I - Five Multimodal Case Types

Status: 删除旧 learned-fusion motivation，换成新的贡献页。

Title:

> Contribution I: Five Multimodal Case-Type Evaluation

Suggested table:

| Case Type | Semantic Risk | Acoustic Risk | Label | What It Tests |
|---|---|---|---|---|
| ND | low | real | normal | ordinary false positives |
| NF | benign finance | real | normal | finance hard negatives |
| SV | benign | synthetic | normal | acoustic false alarms |
| SF | fraud | real | fraud | semantic fraud without synthetic voice |
| MR | fraud | synthetic | fraud | mixed semantic-acoustic attack |

Keyline:

> This benchmark evaluates fusion behavior across evidence sources, not just normal/phishing classification.

### Slide 9: Corrected 180-Sample Benchmark Design

Status: 保留原位置或移到 Slide 8 后；内容必须替换。

Delete old 80-sample content.

Use table from Section 5.1.

Add label principle:

> Fraud labels are semantic/behavioral. Synthetic voice is not automatically fraud.

### Slide 10: Text-Risk Branch Evaluation

Status: 保留，但降级为 supporting evidence。

Update note:

> This module-level result supports semantic-risk modelling. The final defence claim is based on the corrected 180-sample dynamic evaluation.

Do not let this slide dominate the main story.

### Slide 11: Audio-Risk Branch Evaluation

Status: 保留并增加边界说明。

Add a highlighted note:

> The audio model estimates synthetic-speech authenticity risk. A high voice score alone is not sufficient evidence of phishing.

If possible, change any label like `audio fraud risk` to:

> voice authenticity risk

### Slide 12: Main Dynamic Tracking Performance

Status: 保留但替换所有数字。

Use table from Section 5.2.

Interpretation:

> The main method maintains high precision and low normal final FPR while producing measurable early-warning lead time.

Add contribution tie-in:

> These timing metrics are necessary because the system is designed for in-event risk tracking, not post-hoc classification only.

### Slide 13: Case-Type Breakdown

Status: 保留但替换旧内容。

Use table from Section 5.4.

Keyline:

> Case-type evaluation exposes boundary behavior: NF remains the hardest normal condition, SF is the main false-negative source, and SV verifies that benign synthetic speech is not treated as fraud.

### Replacement for old Slide 14: Why Voice Score Alone Is Not Phishing Evidence

Status: 删除旧 `Synthetic-Voice Timeline: Fixed vs Learned Fusion`。

Reason:

- Under corrected labels, SV is normal.
- A learned alert on SV is a false positive, not a success.

New title:

> Voice Score Is Auxiliary Evidence, Not Sufficient Phishing Evidence

Suggested table:

| Evidence | Meaning | Fraud-decision implication |
|---|---|---|
| High text score | semantic fraud intent | primary evidence |
| High voice score | synthetic or suspicious acoustic source | auxiliary evidence |
| Benign text + high voice | possible synthetic normal speech | should not automatically alert as fraud |
| Fraud text + high voice | mixed-risk attack | stronger alert evidence |

Keyline:

> The corrected label design prevents the system from equating synthetic speech with fraud.

### Replacement for old Slide 15: Fixed Fusion vs Learned Late Fusion

Status: 删除旧 80-sample learned result，换成 fixed vs learned trade-off。

Title:

> Why Fixed Fusion Is Selected over Learned Late Fusion

Use comparison table:

| Criterion | Fixed 8:2 + smoothing | Learned OOF |
|---|---:|---:|
| F1 | 0.9139 | 0.9241 |
| FP | 2 | 5 |
| NF FP | 2/20 | 3/20 |
| SV FP | 0/40 | 2/40 |
| Mean alert time | 23.55s | 45.55s |
| Mean lead time | 24.56s | 3.13s |
| Main method? | Yes | No |

Interpretation:

> Learned fusion slightly improves final F1, but it triggers much later and introduces more false positives. Fixed fusion is more aligned with early-warning-oriented dynamic tracking.

Leakage note if there is space:

> Learned result uses `causal_late_fusion_v2_score` with 5-fold outer / 4-fold inner grouped OOF; it remains a diagnostic comparison.

### Slide 16: Ablation and Window-Step Trade-off

Status: 保留但更新。

Recommended content:

- If using a table, use Section 5.3.
- If using the old figure, regenerate it using corrected 180 results first.

Key bullets:

- Voice-only performs poorly as a fraud detector because voice authenticity is not fraud intent.
- Smoothing reduces final false positives from 8 to 2.
- Learned late fusion improves final F1 but loses early-warning advantage.
- Window/step comparison should be reported as runtime/evaluation sensitivity, not a different main method.

### Slide 17: Fusion Weight Sensitivity

Status: 保留/新增。

Use Section 5.5.

Key message:

> The fixed 8:2 rule is selected as a pre-defined interpretable multimodal operating point. The sweep demonstrates sensitivity and robustness; it is not a post-hoc search for the best F1.

### Slide 18: Contributions and Conclusion

Status: 保留并重写。

Use three contribution blocks:

1. Dynamic tracking system:

> A working uploaded-audio simulated-streaming prototype that converts speech into a causal risk timeline.

2. Case-type evaluation:

> A five-way multimodal benchmark separating semantic fraud, acoustic authenticity risk, benign synthetic speech, benign financial interference, and mixed attacks.

3. Timing-aware evaluation:

> Alert recall, alert time, early-warning lead time, and detection delay evaluate in-event warning ability beyond final classification.

Final conclusion:

> The project extends voice-phishing detection from post-hoc binary classification to in-event dynamic risk tracking. On the corrected 180-sample controlled benchmark, fixed 8:2 fusion with smoothing offers the best practical trade-off between interpretability, low false positives, and early warning.

---

## 7. Claims to remove or avoid

Remove these claims from the PPT:

1. `synthetic_voice` is fraud.
2. Synthetic voice alone should trigger a phishing alert.
3. Learned fusion “restores” an acoustic-only alert as a success.
4. Voice score is phishing evidence by itself.
5. The final benchmark is 80 samples.
6. Learned late fusion is the main method.
7. The corrected 180 benchmark proves broad real-world generalization.
8. Weight sweep results justify post-hoc selecting the highest-F1 weight.

Use these replacements:

1. `synthetic_voice` is normal if content is benign.
2. Voice score is auxiliary acoustic authenticity evidence.
3. Fraud label is semantic/behavioral.
4. Learned fusion is a diagnostic comparison.
5. Fixed 8:2 + smoothing is the selected main method.
6. The benchmark provides controlled robustness evidence.

---

## 8. Figure and artifact update checklist

| Asset | Action |
|---|---|
| `materials/figures/route_equivalence_browser_latency.pdf` | keep concept, but regenerate from corrected 180 samples if used in main deck |
| `materials/figures/synthetic_voice_timeline_sv_long_05.pdf` | remove or relabel as a false-positive risk example, not a learned success |
| `materials/figures/ablation_window_tradeoff.pdf` | regenerate from corrected 180 or replace with corrected table |
| text branch figures | keep as supporting module evidence |
| audio branch figures | keep, but label as authenticity-risk evidence |

Suggested copied figure for new runtime validation:

`materials/figures/route_equivalence_browser_latency_corrected180.pdf`

Do not overwrite the old figure until the new script run is verified.

---

## 9. Script calls that should be documented in speaker notes / backup

### 9.1 Regenerate corrected 180 defence metrics

```bash
conda activate dissertation
cd /Users/sunjiashan/Material/HKU/Dissertation/Code/Voice-Phishing-Detection
python evaluation/rerun_corrected_label_ablation.py
```

Outputs:

- `evaluation/reports/defense_all180_baseline_w10_s5/dynamic_eval_ablation_summary.csv`
- `evaluation/reports/defense_all180_baseline_w10_s5/dynamic_eval_case_type_summary.csv`
- `evaluation/reports/defense_all180_baseline_w10_s5/dynamic_eval_fusion_smoothing_sweep.csv`
- `evaluation/reports/defense_all180_baseline_w10_s5/dynamic_eval_fusion_strategy_comparison.csv`
- `evaluation/predictions/defense_all180_causal_learned_oof_w10_s5/nested_oof_report.json`

### 9.2 Regenerate runtime validation on corrected samples

Preferred command:

```bash
conda activate dissertation
cd /Users/sunjiashan/Material/HKU/Dissertation/Code/Voice-Phishing-Detection
python evaluation/baseline_route_equivalence_latency.py \
  --metadata test_samples/metadata_all_corrected.csv \
  --audio-dir test_samples/audio_all \
  --output-dir evaluation/reports/baseline_route_equivalence_latency_corrected180
python evaluation/generate_route_equivalence_latency_figure.py \
  --report-dir evaluation/reports/baseline_route_equivalence_latency_corrected180 \
  --output-dir evaluation/figures \
  --basename route_equivalence_browser_latency_corrected180
```

If unsupported:

- Update `evaluation/baseline_route_equivalence_latency.py` to accept `--metadata`, `--audio-dir`, and `--output-dir`.
- Use a small balanced subset if full 180 route validation is too slow, but disclose the subset composition.
- The subset should include all five case types.

Minimum balanced subset recommendation:

| Case Type | Samples |
|---|---:|
| ND | 2-4 |
| NF | 2-4 |
| SV | 2-4 |
| SF | 2-4 |
| MR | 2-4 |

Slide wording if subset is used:

> Runtime validation was rerun on a balanced corrected-label subset covering all five case types. It validates route equivalence and processing latency, not benchmark accuracy.

---

## 10. Answer to expected defence questions

### Q1. Why is synthetic_voice normal?

> The fraud label is defined by semantic or behavioral intent, not by whether the voice is synthetic. Many legitimate systems use synthetic speech to read benign content. Therefore, synthetic voice with normal content is normal for fraud detection, although it may have high acoustic authenticity risk.

### Q2. Why is voice_score not enough?

> Voice score measures acoustic authenticity risk. It can indicate that speech may be synthetic, but it does not show fraudulent intent. In this system, voice evidence is auxiliary; semantic risk remains the primary phishing evidence.

### Q3. Why not use learned late fusion as the main method?

> Learned late fusion achieved slightly higher final F1 under grouped OOF evaluation, but it produced more false positives and much weaker early-warning lead time. The project is about in-event dynamic risk tracking, so alert timing and false-positive control matter. Fixed 8:2 fusion with smoothing is more interpretable, stable, and better aligned with early warning.

### Q4. What is the main contribution beyond normal/phishing classification?

> The project evaluates dynamic risk tracking over time and separates five multimodal evidence conditions. This reveals fusion trade-offs across semantic fraud, acoustic authenticity risk, benign finance, benign synthetic speech, and mixed attacks, rather than hiding them inside one overall F1 score.

---

## 11. Final one-sentence narrative

> My contribution is to move voice-phishing detection from post-hoc binary classification to in-event dynamic risk tracking, using sliding-window risk timelines, timing-aware alert metrics, and a five-way multimodal case-type benchmark to expose how different fusion strategies behave under semantic, acoustic, benign, and mixed-risk conditions.
