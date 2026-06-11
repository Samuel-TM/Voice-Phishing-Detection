# Fine-tune and Stress-test Files

This note records the recent text-model fine-tuning and stress-test artifacts in
`evaluation/`. It is an index only; the referenced files keep their original
paths for reproducibility.

## External / Holdout Evaluation Sets

### `external_normal_finance_hard_negative_stress.jsonl`

- Rows: 120
- Label field: `text_label`
- Label distribution: 80 normal, 40 fraud
- Case types:
  - `normal_daily`: 50
  - `normal_finance`: 30
  - `semantic_fraud`: 20
  - `mixed_risk`: 20
- Intended use: main external text-level stress test.
- Thesis role: compare Proposed Chinese-BERT against a surface-rule baseline on
  normal finance hard negatives and semantic fraud cases.
- Do not use for training or threshold tuning after reporting final results.

### `final_text_label_diagnostic_metadata_final.jsonl`

- Rows: 80
- Label field: `text_label`
- Label distribution: 40 normal, 40 fraud
- Case types:
  - `normal_daily`: 20
  - `synthetic_voice`: 20
  - `semantic_fraud`: 20
  - `mixed_risk`: 20
- Intended use: text-branch diagnostic over the final multimodal system samples.
- Thesis role: supplementary diagnostic, not the main normal-finance stress test.
- Note: the final 80 samples are primarily for multimodal system evaluation, but
  this derived JSONL can be used to report text-only branch behavior.

## Fine-tuning Construction Sets

### `benign_finance_hard_negatives_train_val.jsonl`

- Rows: 240
- Label field: `text_label`
- Label distribution: 240 normal
- Case type: `normal_finance_hard_negative`
- Split: 192 train, 48 validation
- Intended use: benign finance hard-negative fine-tuning data.
- Thesis role: training/validation construction set for the hard-negative
  fine-tuning ablation.
- Do not use as an independent final test set.

### `benign_finance_hard_negatives_train_val.csv`

- CSV mirror of `benign_finance_hard_negatives_train_val.jsonl`.
- Intended use: inspection, spreadsheet review, and paper appendix checks.

### `benign_finance_hard_negatives_summary.csv`

- Summary of the benign finance hard-negative construction set.
- Intended use: quick reporting of sample counts and construction metadata.

### `finance_semantic_contrast_train_val.jsonl`

- Rows: 384
- Label field: `text_label`
- Label distribution: 240 normal, 144 fraud
- Case types:
  - `normal_finance_hard_negative`: 240
  - `finance_semantic_contrast_fraud`: 144
- Split: 306 train, 78 validation
- Intended use: semantic contrast fine-tuning data.
- Thesis role: training/validation construction set for the contrastive
  fine-tuning ablation.
- Do not use as an independent final test set.

### `finance_semantic_contrast_train_val.csv`

- CSV mirror of `finance_semantic_contrast_train_val.jsonl`.
- Intended use: inspection, spreadsheet review, and paper appendix checks.

### `finance_semantic_contrast_summary.csv`

- Summary of the finance semantic contrast construction set.
- Intended use: quick reporting of sample counts and construction metadata.

## Generation Script

### `generate_benign_finance_hard_negatives.py`

- Generates the benign finance hard-negative and finance semantic contrast
  train/validation files listed above.
- Keep this script with the generated files so the fine-tuning data construction
  process remains reproducible.

## Recommended Reporting Structure

1. Main baseline comparison:
   `external_normal_finance_hard_negative_stress.jsonl`
2. Supplementary final-system text diagnostic:
   `final_text_label_diagnostic_metadata_final.jsonl`
3. Fine-tuning ablation / negative result:
   `benign_finance_hard_negatives_train_val.jsonl` and
   `finance_semantic_contrast_train_val.jsonl`

