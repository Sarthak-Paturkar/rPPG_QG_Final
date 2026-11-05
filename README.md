# UBFC-rPPG project

## Phase B: signal preprocessing

From the project root, run the preprocessing stage for a subject with the
project virtual environment:

```powershell
.\venv\Scripts\python.exe -m src.preprocessing.preprocess_signals --subject subject3
```

The stage reads `datasets/processed/<subject>/rgb_signals.csv` and leaves that
source file unchanged. It derives the sampling frequency from the CSV's
`video_time_seconds` values and rejects missing, non-increasing, or materially
irregular timestamps rather than silently resampling them.

Preprocessing invalidates samples where a face was not detected, the skin-mask
ratio is below the extractor's 0.20 acceptance threshold, or any RGB channel is
non-finite. Only internal gaps no longer than 0.10 seconds are linearly
interpolated; longer gaps and gaps at either end remain missing. Detrending,
second-order Butterworth bandpass filtering (0.7–4.0 Hz), and z-score
normalization are applied independently to each contiguous finite segment.
Zero-phase filtering is skipped for segments too short to support
`sosfiltfilt`; those samples remain missing in the processed columns.

The stage saves the original columns plus explicit filled, detrended, filtered,
and normalized RGB columns under `datasets/cleaned/<subject>/`. A JSON report is
saved under `outputs/reports/<subject>/`, with before/after plots under
`outputs/plots/<subject>/preprocessing/`. Thresholds and filter settings are
defined in `config.py`.

## Batch processing: Phases A and B

Run the validated 28-subject batch from the project root:

```powershell
.\venv\Scripts\python.exe .\scripts\run_phase_a_b_batch.py
```

An optional subset can be supplied for a controlled run:

```powershell
.\venv\Scripts\python.exe .\scripts\run_phase_a_b_batch.py --subjects subject3 subject8
```

The runner reuses the YuNet RGB extractor, Phase A inspector, and Phase B
preprocessor. It downloads missing inputs through the configured `gdrive:`
rclone remote, skips only validated stage outputs, and saves an incrementally
updated summary to `outputs/reports/batch_phase_a_b_summary.json`. A local
`vid.avi` is deleted only after RGB extraction, Phase A, and Phase B outputs
have all passed verification; ground truth and generated results are retained.