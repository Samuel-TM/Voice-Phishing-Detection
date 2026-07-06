# Oral Defence Q&A Preparation

目标：准备 15 分钟问答。回答风格要短、准、诚实。不要把所有实验都说成成功；老师更看重你是否理解边界。

---

## 1. 你的系统为什么叫 real-time？是不是严格实时？

**建议回答**

我在论文中把它定义为 simulated streaming / incremental analysis。主评估路径是上传完整音频后，用滑动窗口模拟通话过程中的增量分析。它不是 telecom operator network-level deployment，也不是必须依赖真实电话流的生产系统。

我也实现了 browser audio chunk 相关接口，但最终 benchmark 使用 `/api/stream_audio_analysis`，因为它更稳定、可复现，适合系统性评估。

---

## 2. 你为什么不用整段 transcript 做文本风险？

**建议回答**

因为研究目标是动态预警。如果当前窗口使用完整 transcript，就等于把未来信息提前给了模型，会破坏 timing evaluation。

当前设计使用 current-window text 和 rolling context。这样既保留局部上下文，又避免把后续诈骗话术提前泄漏给当前时间点。

---

## 3. 为什么 baseline 融合权重是 0.8 text 和 0.2 voice？

**建议回答**

这是一个可解释的 baseline setting。电信诈骗的关键意图通常在话术语义里，比如转账、验证码、冒充身份；音频分支主要捕捉 synthetic-speech risk，所以我让文本占主导、音频作为辅助。

实验也证明这个设计有边界：它对 semantic_fraud 和 mixed_risk 表现好，但对 synthetic_voice 这种 acoustic-only attack 不够敏感。这正是后续 learned fusion 诊断的动机。

---

## 4. 既然 learned late fusion 的 F1 更高，为什么不把它作为主方法？

**建议回答**

因为我需要区分 runtime baseline 和 offline diagnostic variant。learned fusion 在 sample-level nested CV 下 F1 达到 0.9009，确实比 fixed fusion 好，但 final controlled set 只有 80 条，full-fit 容易过拟合。

我已经做了 leakage control：sample-level split、inner fold threshold selection、prefix-only numerical features，不使用 `case_type`、`sample_id`、keywords、metadata 或 future windows。但它还没有作为 service endpoint 做端到端 runtime validation，所以我把它定位为诊断证据，而不是当前部署主路径。

---

## 5. synthetic_voice 为什么失败？这是否说明系统没用？

**建议回答**

不是系统没用，而是 baseline 的设计权衡暴露出一个明确 failure mode。

synthetic_voice 是合成语音朗读正常文本，所以 text score 很低。baseline 使用 `0.8 * text_score + 0.2 * voice_score`，再经过 smoothing，纯声学风险容易被压低。因此 final recall 是 0，alert recall 也只有 0.05。

这个结果反而说明 case-type benchmark 有价值：它不是只给总体分数，而是揭示哪种风险没有被当前融合策略覆盖好。

---

## 6. 你的 normal false positive 控制得好吗？

**建议回答**

从 final classification 看，10s/5s baseline 的 normal final false positives 是 0/20，这是一个保守且稳定的结果。

但从 timeline 看，normal alert false positives 仍有 3/20。也就是说最终结果稳定，但过程中还有局部尖峰。这就是为什么我同时报告 final FP 和 alert FP，而不是只看最终分类。

---

## 7. 为什么需要 smoothing？

**建议回答**

窗口级模型会产生局部尖峰，尤其是 ASR 片段不完整、文本上下文短或音频窗口质量波动时。Smoothing 可以让风险曲线更稳定，更符合“逐步积累风险证据”的展示目标。

但 smoothing 不是纯收益。它会降低最终误报，但也可能推迟报警或压住短时强证据，所以我在 ablation 和 window/step comparison 中专门报告了这个 trade-off。

---

## 8. 你的 early-warning lead time 怎么解释？为什么 synthetic_voice 里有负数？

**建议回答**

对 mixed_risk 和 semantic_fraud，`event_time_sec` 表示诈骗动作或关键风险话术开始出现的时间，所以 early-warning lead time 可以解释为系统相对 fraud action start 提前多少报警。

对 synthetic_voice，风险从音频开始就存在，因为它测试的是合成语音本身，所以 event time 更接近 0。此时更适合讲 time-to-alert 或 detection delay，而不是把 lead time 当成传统意义上的“提前预警”。

---

## 9. 为什么 10 秒窗口 / 5 秒步长是主设置？

**建议回答**

因为它是最平衡的，不是因为每个指标都最高。

5s/2.5s alert 更早，mean lead time 33.23s，但 normal alert FPR 更高，final FP 也有 1 条。20s/10s 更保守，但 recall 和 alert recall 明显下降。10s/5s 的 F1 是 0.8，final FP 是 0，normal alert FPR 是 0.15，所以更适合作为当前 prototype baseline。

---

## 10. 你的 benchmark 是否能证明 generalization？

**建议回答**

不能过度声称。`metadata_final.csv` 是 controlled benchmark，用来检验系统在四类明确设计的风险场景中的行为。它适合做机制验证和 case-type analysis，但不是大规模真实世界 generalization 证明。

我会把结论限定为：在当前 controlled benchmark 下，系统展示了 dynamic risk tracking 能力，并揭示了 fixed fusion 的边界。

---

## 11. 你如何避免 window-level leakage？

**建议回答**

在动态 baseline 中，当前窗口只使用当前和过去上下文，不使用未来 transcript。

在 learned fusion 中，我没有做 random window split，而是 sample-level nested CV。每个 held-out fold 包含完整 sample 的所有窗口，threshold selection 只在 train fold 内完成。此外，特征契约禁止 metadata、case type、sample id、关键词和 future windows。

---

## 12. 文本模型和音频模型分别负责什么？

**建议回答**

文本模型负责 fraud semantics，比如冒充身份、转账诱导、验证码索取、屏幕共享和资金核验等语义风险。

音频模型负责 voice authenticity risk，比如 synthetic speech 或 deepfake voice 的声学迹象。

这两个风险不是同一个问题，所以系统需要 multimodal timeline，而不是单一模型替代一切。

---

## 13. 你自己主要做了哪些工作？

**建议回答**

我完成了系统集成和动态评估主线，包括：

- 将音频处理改成 sliding-window simulated streaming；
- 接入 ASR、Chinese-BERT 文本风险和 CNN-BiLSTM 声学风险；
- 实现 baseline fusion、smoothing 和 timeline 输出；
- 构建 final benchmark 的样本索引和四类 case-type 评估；
- 生成 dynamic predictions 和 metrics；
- 做 ablation、window/step comparison、case-type error analysis；
- 设计 learned late fusion 的离线诊断并控制 leakage。

---

## 14. 如果 ASR 某个窗口失败怎么办？

**建议回答**

系统设计上不应该因为单个静音窗口、短窗口或 ASR 失败让整段分析返回 500。窗口需要保留 start/end time，并尽可能返回可用分数。如果文本为空，文本风险可以按低风险或不可用处理，后续窗口继续分析。

这也符合 dynamic tracking 的设计，因为真实通话中窗口质量会波动，系统需要有容错能力。

---

## 15. 你最重要的发现是什么？

**建议回答**

最重要的发现不是某一个 accuracy 数字，而是 case-type behaviour。

系统在 mixed_risk 和 semantic_fraud 上表现强，说明文本语义分支对诈骗意图有效；normal final FP 为 0，说明 smoothing 后最终判断较稳；但 synthetic_voice 失败说明 fixed text-dominant fusion 不能充分处理 acoustic-only risk。

这说明 dynamic multimodal risk tracking 有价值，但融合策略必须根据风险类型更细致地设计和评估。

---

## 16. 如果老师质疑你的系统只是 demo，怎么办？

**建议回答**

我会承认它是 prototype，而不是 production system。但它不是只有界面的 demo；它包含完整的 runtime path、模型推理、timeline output、final controlled benchmark、dynamic metrics、ablation 和 error analysis。

毕业设计的贡献在于提出并验证 dynamic risk tracking 的系统框架，而不是完成 telecom-grade deployment。

