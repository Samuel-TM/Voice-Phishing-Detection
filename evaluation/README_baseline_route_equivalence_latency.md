# Baseline Route Equivalence and Browser-Chunk Latency

## Purpose

This experiment validates the deployed baseline at the system level. It does not re-estimate model accuracy. It asks whether the uploaded simulated-streaming route and the live browser-chunk route execute the same causal baseline logic, and whether 5-second browser chunks can be processed before the next chunk arrives.

## Frozen protocol

- Samples: 12 recordings from `test_samples/metadata_final.csv`, with three samples from each of `normal_daily`, `semantic_fraud`, `mixed_risk`, and `synthetic_voice`.
- Route-equivalence input: the same 10-second windows at 5-second steps are sent to `/api/stream_audio_analysis` and sequentially to `/api/live_audio_chunk`.
- Browser-style input: each recording is replayed as sequential, non-overlapping 5-second WAV chunks through `/api/live_audio_chunk`.
- Baseline: `0.8 * text_score + 0.2 * voice_score`, followed by `0.65 * previous + 0.35 * current` smoothing.
- Alert threshold: smoothed score at least 70.

## Results

### Identical-window route equivalence

- 12/12 samples and 124/124 windows completed successfully.
- Text, voice, fused, and smoothed score MAE: **0.000**; maximum absolute differences: **0.000**.
- Window timing, transcript, final label, alert state, and first-alert time agreement: **100%**.
- Live-chunk processing latency for these 10-second windows: median **1.54 s**, p95 **2.03 s**.

This is direct evidence that the uploaded and live endpoints execute an equivalent baseline when their window inputs are held constant.

### Browser-style 5-second chunks

- 127/127 chunks returned successfully.
- Processing latency: median **1.16 s**, p95 **1.58 s**, maximum **1.63 s**.
- 127/127 chunks completed within their audio duration; median real-time factor was **0.234**.
- Alert-state agreement with the 10-second/5-second uploaded route: **12/12**.
- Final-label agreement: **11/12**. `SF_long_01` alerted on both paths, but its browser-style final score fell below the threshold after the earlier alert, while its uploaded-route final score remained above it.
- For the seven samples that alerted on both paths, the mean absolute first-alert difference was **16.15 s** and the maximum was **53.08 s**. Non-overlapping 5-second chunks and overlapping 10-second windows have different evidence boundaries, so this is not a pointwise-equivalence test.

## Interpretation and limits

The defensible claim is: the two server routes are equivalent under identical window input, and browser-style chunks are processed faster than real time on the evaluation machine. The browser replay was accelerated and used Flask's in-process test client. Its latency excludes network transport and MediaRecorder encoding. Therefore, this experiment supports runtime feasibility, not an internet-scale production latency claim.

## Reproduction

```bash
conda activate dissertation
python evaluation/baseline_route_equivalence_latency.py
python evaluation/generate_route_equivalence_latency_figure.py
```

Ignored raw artifacts are under `evaluation/reports/baseline_route_equivalence_latency/`. The slide-ready vector and PNG figures are under `evaluation/figures/`.
