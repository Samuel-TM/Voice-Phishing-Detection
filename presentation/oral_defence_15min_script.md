# Oral Defence 15-Minute Script

说明：
- 这份稿子按 15 页 PPT 写，每页约 45-65 秒。
- 建议正式答辩时用英文 PPT，但你练习时先用中文把逻辑讲顺。
- 如果需要英文口播，可以在这版定稿后再逐段翻译。

---

## Slide 1. Title

各位老师好，我的毕业设计题目是 **Real-Time Dynamic Risk Tracking of Telecommunications Fraud Based on Parallel Multimodal Analysis**。

我今天的陈述会非常聚焦在我自己完成的工作：系统目标、实现过程、实验结果，以及这些结果说明了什么。我的重点不是泛泛介绍电信诈骗背景，而是说明我如何把一个通话音频转化成随时间变化的风险追踪问题。

---

## Slide 2. Defence Message

这个项目的核心不是做一个整段录音的 post-hoc classifier。整段分类只能告诉我们“最后是不是诈骗”，但它不能回答更关键的问题：风险是在什么时候出现的？系统能不能在通话过程中逐步发现风险？

所以我实现的是一个 simulated streaming prototype。它把上传的完整音频按滑动窗口切分，每个窗口分别做文本风险和音频风险，然后融合、平滑，并输出一条 risk timeline。

我想先把结论说清楚：这个系统在包含诈骗语义的样本上能追踪风险变化，但实验也暴露出一个明确的 failure mode，也就是固定 text-dominant fusion 对纯 synthetic voice attack 不够敏感。

---

## Slide 3. Research Objective

我的研究目标可以概括为一句话：构建并评估一个能在通话展开过程中追踪电信诈骗风险的原型系统。

为了实现这个目标，我完成了五个任务。第一，把完整音频转换成滑动窗口下的增量分析。第二，从窗口转写文本中估计诈骗语义风险。第三，从音频窗口中估计合成语音或 deepfake voice 的声学风险。第四，把文本风险和音频风险融合成连续 timeline。第五，不只看最终 accuracy 或 F1，也评估 alert timing、lead time、detection delay 和不同 case type 的表现。

---

## Slide 4. System I Implemented

系统的主运行路径是 `/api/stream_audio_analysis`。用户上传一段完整音频，后端按窗口切分，逐个窗口执行 ASR、文本风险推理、音频风险推理、融合和平滑，最后返回 timeline。

我还实现了 browser audio chunks 对应的 `/api/live_audio_chunk` 和 `/api/live_audio_finish`，以及前端的 risk curve、timeline table、replay 和当前窗口 transcript 展示。

不过在论文和答辩主线里，我把 uploaded-audio simulated streaming 作为主要评估路径，因为它更稳定、可重复，也更适合用同一批 benchmark 样本做严格比较。

---

## Slide 5. Method: Parallel Multimodal Timeline

在每个窗口里，系统先做 ASR，然后并行产生两个风险信号。

文本分支使用 Chinese-BERT，输入不是完整的 cumulative transcript，而是当前窗口文本和近期上下文构成的 rolling context。这样做是为了保持动态性，避免把未来文本提前泄漏给当前窗口。

音频分支使用 MFCC-20 特征和 CNN-BiLSTM 模型，输出 synthetic-speech risk。它的作用不是理解诈骗意图，而是捕捉声音真实性风险。

baseline 融合公式是 `0.8 * text_score + 0.2 * voice_score`，然后使用 `0.65 * previous + 0.35 * current` 做平滑。这个设置是可解释的：文本语义是主要依据，声学风险是辅助依据，平滑用于减少单个窗口的尖峰误报。

---

## Slide 6. Final Benchmark

最终评估使用 `metadata_final.csv` 和 `audio_final/`，一共 80 条样本。

这 80 条被分成四类，每类 20 条。`normal_daily` 是真人正常语音，用来检查误报；`synthetic_voice` 是合成语音朗读正常文本，用来测试纯声学攻击；`mixed_risk` 是合成语音朗读诈骗文本，用来测试多模态联合风险；`semantic_fraud` 是高质量语音承载诈骗话术，用来测试当声学证据不明显时，文本语义能不能发挥作用。

这个划分对理解结果很重要，因为它能把“语义风险”和“声学风险”分开看，而不是只给出一个总体 accuracy。

---

## Slide 7. Text-Risk Branch

文本风险分支的目标是从 partial transcript 中识别诈骗语义。

我用外部 text-level stress test 比较了 surface baseline 和 Proposed Chinese-BERT。Surface baseline 的 accuracy 是 0.7667，F1 是 0.7255，normal false positive rate 是 0.3125。Proposed Chinese-BERT 的 accuracy 提升到 0.9167，F1 提升到 0.8889，normal false positive rate 降到 0.125，同时 fraud recall 达到 1.0。

这个结果说明，简单关键词或 surface rules 容易把正常金融语句误判为诈骗；Chinese-BERT 更能利用上下文语义，适合在动态系统中作为文本风险分支。

---

## Slide 8. Audio-Risk Branch

音频风险分支使用 MFCC-20 和 CNN-BiLSTM，用来提供 synthetic-speech risk。

训练和验证使用了 ASVspoof 2019 LA 以及 In-the-Wild Audio Deepfake。当前部署阈值是 0.7495。在 In-the-Wild OOD test 上，macro-F1 是 0.9345，real recall 是 0.942，fake recall 是 0.927。

但这里我必须强调边界：audio branch 检测的是 voice authenticity risk，不是 fraud intention。也就是说，如果一段合成语音朗读的是完全正常内容，音频分支可能提示声学风险；但它不能判断是否有转账、验证码、冒充身份等诈骗语义。

---

## Slide 9. Main Dynamic Result

主 baseline 使用 10 秒窗口和 5 秒步长，也就是 `final_baseline_w10_s5`。在 `fusion_with_smoothing` 下，80 条样本的 accuracy 是 0.75，precision 是 1.0，recall 是 0.6667，F1 是 0.8。

从混淆矩阵看，TP 是 40，TN 是 20，FP 是 0，FN 是 20。也就是说，最终判断没有把 normal_daily 误判成 fraud，但漏掉了 20 条 fraud。

动态指标方面，fraud alert recall 是 0.6833，normal alert false positives 是 3 条，mean early-warning lead time 是 16.60 秒。

我的解读是：这个 baseline 是保守的，最终误报控制很好，但它牺牲了一部分召回。下一页可以看到，漏掉的并不是随机样本，而是一个很明确的 case type。

---

## Slide 10. Case-Type Behaviour

按 case type 分析后，结果非常清楚。

在 `mixed_risk` 上，final recall 和 alert recall 都是 1.0，因为文本诈骗语义和合成语音声学风险同时存在。

在 `semantic_fraud` 上，final recall 和 alert recall 也是 1.0，说明当声学证据不强时，Chinese-BERT 文本语义仍然能识别诈骗。但 mean time to alert 是 42.76 秒，说明 smoothing 会让报警更保守、更晚。

在 `normal_daily` 上，final false positive 是 0，但过程中的 alert false positive 有 3 条，说明 timeline 仍存在局部尖峰。

最大问题是 `synthetic_voice`：final recall 是 0，alert recall 只有 0.05。原因是这些样本是合成语音朗读正常文本，文本风险很低，而 baseline 融合 80% 依赖文本，再加 smoothing，导致纯声学风险被压低。

---

## Slide 11. Ablation

消融实验帮助我理解每个组件的作用。

`text_only` 的 F1 是 0.7736，但 final normal FP 有 5 条，说明语义模型能抓诈骗，但也会误伤部分正常语音。

`voice_only` 的 precision 是 1.0，final normal FP 是 0，但 recall 只有 0.35，说明它不是可靠的最终诈骗分类器。

`fusion_without_smoothing` 和 text-only 很接近，说明固定 0.8/0.2 融合仍然主要由文本分支主导。

`fusion_with_smoothing` 把 final normal FP 降到 0，F1 到 0.8，但 fraud alert recall 降到 0.6833。这说明 smoothing 提高稳定性，但代价是报警可能更晚或者被压住。

---

## Slide 12. Window / Step Sensitivity

我还比较了 5/2.5、10/5 和 20/10 三组 window-step。

5 秒窗口的 mean lead time 是 33.23 秒，alert 更早，但 normal alert FPR 是 0.30，并且 final FP 有 1 条。20 秒窗口更保守，final FP 是 0，但 alert recall 只有 0.5667，F1 也降到 0.7097。

10 秒窗口和 5 秒步长的 F1 是 0.8，final FP 是 0，normal alert FPR 是 0.15，整体更平衡。

所以我选择 10/5 不是因为它每个指标都是最高，而是因为它在最终稳定性、动态报警和召回之间取得了最合理的 trade-off。

---

## Slide 13. Learned Late Fusion

因为 baseline 暴露了 fixed fusion 对 synthetic_voice 的问题，我又测试了 learned late fusion。但我把它定位为 offline diagnostic variant，而不是当前 runtime baseline。

为了避免数据泄漏，我使用 sample-level 5-fold nested CV。每个 outer fold hold out 完整样本，而不是随机窗口；threshold 只在 training fold 内选择；特征只允许当前和过去窗口的 numerical timeline aggregates，不允许使用 `case_type`、`sample_id`、关键词、metadata 或未来窗口。

结果是 accuracy 0.8625，precision 0.9804，recall 0.8333，F1 0.9009，TP/TN/FP/FN 是 50/19/1/10。

这说明 learned fusion 能缓解固定融合的 failure mode，但由于 final set 较小，而且它还没有作为 service path 做端到端验证，所以我没有把它写成主方法，只把它作为谨慎的诊断证据。

---

## Slide 14. What the Results Show

从这些结果里，我认为有四点发现。

第一，timeline evaluation 是必要的。只看最终 F1 不能说明报警时间、过程误报、延迟检测和风险演化。

第二，两个模态覆盖的是不同风险。文本分支负责诈骗意图，音频分支负责声音真实性。

第三，smoothing 有价值，但不是免费的。它能减少最终误报，但也可能延迟或压制报警。

第四，当前 baseline 的优点是可解释。它的成功和失败都可以通过 case-type analysis 解释出来，而不是只给一个黑箱分数。

---

## Slide 15. Contributions and Conclusion

总结来说，我完成了五方面工作。

第一，我实现了一个可以运行的 simulated-streaming multimodal fraud-risk tracking prototype。第二，我集成了 ASR、Chinese-BERT 文本风险、CNN-BiLSTM 声学风险、late fusion 和 smoothing。第三，我构建了一个 80 条样本的 final controlled benchmark，用来区分 normal、semantic、acoustic 和 multimodal risk。第四，我用 final classification 和 timeline-oriented metrics 评估系统。第五，我识别出 fixed fusion 的具体 failure mode，并用 learned late fusion 做了谨慎的离线诊断。

所以我的结论是：这个项目已经展示了 dynamic in-event risk tracking 的可行性；同时，它也清楚说明了当前 baseline 在什么场景下有效、在什么场景下失败。这正是我认为本项目最有价值的地方。

