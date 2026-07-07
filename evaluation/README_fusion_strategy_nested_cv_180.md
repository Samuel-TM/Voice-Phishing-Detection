# Fusion Strategy Comparison on the 180-Sample Controlled Pool

## Decision

No preregistered fusion strategy passed all five admission constraints. Following
the preregistered fallback rule, the thesis mainline remains **fixed 0.8/0.2
fusion with smoothing**. This is a conservative fallback, not evidence that the
fixed strategy solves synthetic-voice detection.

The unconstrained learned strategy has the highest OOF macro F1, but it is not
eligible: it loses 0.325 semantic-fraud recall relative to the text-only
reference and exceeds the normal-finance FPR ceiling by 0.20.

## Protocol

- Pool: `audio_final` (80) + `audio_external_frozen_v2` (100).
- Evidence status: controlled model-selection evidence, not an independent
  external-generalization test.
- Evaluation: 5-fold outer / 4-fold inner nested grouped cross-validation.
- Leakage control: all windows from one sample remain in one fold; paired v2
  samples sharing the same script remain in one fold.
- Threshold selection: label-only inner OOF objective. `case_type` is used only
  for the final preregistered admission checks.
- Text-only semantic-fraud recall reference: 0.95.

## OOF Results

| Strategy | Macro F1 | Precision | SV recall | SF recall | MR recall | ND FPR | NF FPR | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Fixed 0.8/0.2 + smoothing | 0.7317 | 0.9865 | 0.000 | 0.900 | 0.925 | 0.000 | 0.050 | No |
| Calibrated max | 0.5328 | 0.8333 | 0.575 | 0.150 | 0.400 | 0.100 | 0.250 | No |
| Calibrated noisy-OR | 0.4888 | 0.7414 | 0.575 | 0.150 | 0.350 | 0.175 | 0.400 | No |
| Monotonic evidence-preserving | 0.7143 | 0.8889 | 0.050 | 0.950 | 1.000 | 0.125 | 0.250 | No |
| Unconstrained learned | **0.8303** | 0.9107 | **0.925** | 0.625 | 1.000 | 0.100 | 0.300 | No |

Admission requires: SV recall >= 0.80; SF recall >= 0.90; MR recall >= 0.90;
ND FPR <= 0.10; and NF FPR <= 0.10.

The post-hoc threshold sensitivity check is diagnostic only and was not used for
selection. Across 257 candidate thresholds, none of the four non-fixed
strategies has a threshold that passes all five constraints. The failure is
therefore not explained by the displayed decision threshold alone.

## Interpretation for the Thesis

The current evidence does not support replacing fixed fusion with learned late
fusion as the main method. The learned counterexample repairs synthetic-voice
recall but sacrifices the text branch's semantic-fraud advantage and produces
too many normal-finance false positives. Monotonic fusion preserves semantic
evidence, but the voice evidence needed to recover synthetic voice is not
separable from normal speech at the required FPR.

The defensible thesis claim is therefore: the two branches provide complementary
signals, while robust adaptive fusion remains unresolved on the present
controlled pool. Fixed fusion remains the deployed/mainline decision layer, and
learned fusion should be reported as an ablation or negative result rather than
as the proposed method.

## Reproduction

```bash
conda activate dissertation
cd /Users/sunjiashan/Material/HKU/Dissertation/Code/Voice-Phishing-Detection
python -m unittest evaluation.test_compare_fusion_strategies_nested_cv
python evaluation/compare_fusion_strategies_nested_cv.py
```

Machine-readable outputs:

- `evaluation/reports/fusion_strategy_nested_cv_180/fusion_strategy_nested_cv_summary.csv`
- `evaluation/reports/fusion_strategy_nested_cv_180/fusion_strategy_nested_cv_report.json`
- `evaluation/predictions/fusion_strategy_nested_cv_180/oof_predictions.json`

