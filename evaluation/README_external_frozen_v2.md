# External Frozen V2: Pre-Evaluation Runbook

## Frozen contract

- Status: `frozen_pre_evaluation`.
- Metadata: `test_samples/metadata_external_frozen_v2.csv`.
- Audio: `test_samples/audio_external_frozen_v2/`.
- Expanded set: 100 samples (`ND`, `NF`, `SV`, `SF`, `MR`, 20 each).
- Paired core: 80 samples (`ND`, `SV`, `SF`, `MR`).
- Hard-negative extension: 20 `normal_finance` (`NF`) samples.
- Window / step: 10 s / 5 s.
- Fixed fusion: `0.8 * text + 0.2 * voice` with previous smoothing weight `0.65`.
- Frozen learned model: `evaluation/models/causal_late_fusion_v2_w10_s5.joblib`.
- Frozen learned threshold: `0.6446452282499999` probability, mapped to display score 70.
- Manifest: `evaluation/external_frozen_v2_manifest.json`.

Do not change sample membership, labels, audio, model, threshold, or scoring configuration after this point. Do not run threshold, weight, or smoothing sweeps on v2.

## About the earlier metadata error

`Missing final sample specs in TTS markdown` came from the legacy Mimo generator trying to rebuild `metadata_final.csv` after audio generation. It did not mean the generated `EV2_ND_*` audio was missing or invalid. External generation must use `--skip-metadata`; v2 metadata is built independently by `evaluation/external_frozen_v2.py`.

## Commands to run once

```bash
conda activate dissertation
cd /Users/sunjiashan/Material/HKU/Dissertation/Code/Voice-Phishing-Detection
```

Verify that all frozen hashes still match:

```bash
python evaluation/external_frozen_v2.py verify
```

Run the fixed-fusion baseline and upstream model inference:

```bash
python evaluation/generate_dynamic_predictions.py \
  --metadata test_samples/metadata_external_frozen_v2.csv \
  --run-name external_frozen_v2_baseline_w10_s5 \
  --window-seconds 10 \
  --step-seconds 5 \
  --text-weight 0.8 \
  --smoothing-previous-weight 0.65 \
  --scoring-mode baseline \
  --resume
```

Apply the already-frozen causal learned late-fusion model and threshold without fitting or selection:

```bash
python evaluation/causal_late_fusion_v2.py apply-frozen \
  --predictions evaluation/predictions/external_frozen_v2_baseline_w10_s5/dynamic_predictions.json \
  --model-path evaluation/models/causal_late_fusion_v2_w10_s5.joblib \
  --output-predictions evaluation/predictions/external_frozen_v2_causal_learned_w10_s5/dynamic_predictions.json \
  --report-path evaluation/predictions/external_frozen_v2_causal_learned_w10_s5/frozen_application_report.json
```

Validate the completed 100-record run and create the predefined paired core-80 view:

```bash
python evaluation/external_frozen_v2.py prepare-result-views \
  --learned-predictions evaluation/predictions/external_frozen_v2_causal_learned_w10_s5/dynamic_predictions.json \
  --core-output evaluation/predictions/external_frozen_v2_causal_learned_w10_s5/core80_dynamic_predictions.json \
  --audit-output evaluation/predictions/external_frozen_v2_causal_learned_w10_s5/results_audit.json
```

Generate the expanded-100 report. The prediction file contains both fixed-fusion scores and frozen causal learned-fusion scores, so both variants are evaluated from the same timelines:

```bash
python evaluation/dynamic_metrics.py \
  --predictions evaluation/predictions/external_frozen_v2_causal_learned_w10_s5/dynamic_predictions.json \
  --run-name external_frozen_v2_expanded100
```

Generate the paired core-80 report:

```bash
python evaluation/dynamic_metrics.py \
  --predictions evaluation/predictions/external_frozen_v2_causal_learned_w10_s5/core80_dynamic_predictions.json \
  --run-name external_frozen_v2_core80
```

Re-run the immutable-hash check after evaluation:

```bash
python evaluation/external_frozen_v2.py verify
```

After results are observed, do not replace samples or adjust the learned threshold. Any later method revision must use a new model version and a new untouched test set.
