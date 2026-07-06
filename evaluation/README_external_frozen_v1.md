# External Frozen V1 Stress Evaluation

## Evaluation contract

`external_frozen_v1` is a frozen, non-overlapping legacy-pool stress evaluation for the learned late-fusion decision layer.

- Source registry: `test_samples/metadata.csv`.
- Excluded fitting registry: `test_samples/metadata_final.csv` and its 80 `audio_final` samples.
- Core matched set: 80 samples, with 20 each from `normal_daily`, `semantic_fraud`, `mixed_risk`, and `synthetic_voice`.
- Hard-negative extension: 20 `normal_finance` samples.
- Expanded set: 100 samples total.
- Window / step: 10 s / 5 s.
- Frozen model: `evaluation/models/calibrated_late_fusion_w10_s5.joblib`.
- Frozen model SHA-256: `697af9a59a2c8e79878b0d6924f8952dbf545a38e6c2041281a8059120ea073f`.
- Frozen probability threshold: `0.61004068`.
- No model fitting, calibration, threshold scan, or membership change was performed after evaluation began.

The leakage audit found zero overlap with `metadata_final.csv` by sample ID, non-empty transcript text, and audio SHA-256. These samples are external to the learned-fusion fitting set, but they come from a previously maintained legacy pool. The result therefore supports a frozen stress-test claim, not a cross-dataset generalization claim.

## Core matched results (80 samples)

| Variant | Accuracy | Precision | Recall | F1 | Final FP | Final FN | Fraud alert recall | Normal alert FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed fusion + smoothing | 0.7500 | 0.9762 | 0.6833 | **0.8039** | **1** | 19 | 0.7333 | **10** |
| Frozen learned late fusion | 0.6500 | 0.8333 | 0.6667 | 0.7407 | 8 | 20 | **0.7667** | 13 |

### Case-type comparison

| Case type | Fixed final recall | Learned final recall | Fixed alert recall / FP | Learned alert recall / FP |
| --- | ---: | ---: | ---: | ---: |
| `normal_daily` | N/A | N/A | 10 FP | 13 FP |
| `semantic_fraud` | 0.95 | 0.95 | 1.00 | 1.00 |
| `mixed_risk` | 0.90 | 0.90 | 1.00 | 1.00 |
| `synthetic_voice` | **0.20** | 0.15 | 0.20 | **0.30** |

The frozen learned fusion did not reproduce the controlled final-benchmark improvement. It slightly increased overall fraud alert recall, but reduced final F1, increased normal final false positives, and did not improve synthetic-voice final recall.

## Expanded hard-negative results (100 samples)

| Variant | Accuracy | Precision | Recall | F1 | Final FP | Normal final FPR | Normal alert FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed fusion + smoothing | **0.7000** | **0.7885** | **0.6833** | **0.7321** | **11** | **0.275** | **0.650** |
| Frozen learned late fusion | 0.5500 | 0.6154 | 0.6667 | 0.6400 | 25 | 0.625 | 0.800 |

On `normal_finance`, fixed fusion produced 10/20 final false positives and learned fusion produced 17/20. The hard-negative extension therefore exposes a major unresolved false-positive problem in both decision layers, especially the learned variant.

## Thesis interpretation

The result should be reported as negative external stress evidence:

> Learned late fusion improved the controlled final benchmark under sample-level nested cross-validation, but the frozen model did not preserve that advantage on a non-overlapping legacy-pool stress set. Its gains are therefore benchmark-specific rather than evidence of cross-dataset generalization. The principal external weaknesses are synthetic-voice domain shift and false alarms on benign financial speech.

Do not tune the saved model or threshold on this set. Any subsequent model revision must define a new model version and preserve `external_frozen_v1` as an already-observed evaluation set.

## Artifacts

- Selection and hashes: `evaluation/external_frozen_v1_manifest.json`.
- Generated metadata: `test_samples/metadata_external_frozen_v1.csv`.
- Baseline timelines: `evaluation/predictions/external_frozen_v1_baseline_w10_s5/`.
- Frozen learned predictions and inference audit: `evaluation/predictions/external_frozen_v1_learned_late_fusion_w10_s5/`.
- Core reports: `evaluation/reports/external_frozen_v1_core80/`.
- Expanded reports: `evaluation/reports/external_frozen_v1_expanded100/`.

## Reproduction

Activate the project environment before every command:

```bash
conda activate dissertation
python evaluation/external_frozen_v1.py prepare
python evaluation/generate_dynamic_predictions.py \
  --metadata test_samples/metadata_external_frozen_v1.csv \
  --run-name external_frozen_v1_baseline_w10_s5 \
  --window-seconds 10 --step-seconds 5 \
  --text-weight 0.8 --smoothing-previous-weight 0.65 \
  --scoring-mode baseline --resume
python evaluation/calibrated_late_fusion.py \
  --apply-frozen-model \
  --predictions evaluation/predictions/external_frozen_v1_baseline_w10_s5/dynamic_predictions.json \
  --output-predictions evaluation/predictions/external_frozen_v1_learned_late_fusion_w10_s5/dynamic_predictions.json \
  --model-path evaluation/models/calibrated_late_fusion_w10_s5.joblib \
  --report-path evaluation/predictions/external_frozen_v1_learned_late_fusion_w10_s5/frozen_application_report.json
```
