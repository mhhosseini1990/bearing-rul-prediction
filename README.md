# Wavelet-Based Bearing RUL Prediction

Canonical direct Remaining Useful Life regression for the 15 XJTU-SY bearing
histories. LSTM and GRU use the same fixed 48 features, splits, optimization
budget, and validation-only checkpoint selection. Historical V1–V8 evidence is
archived outside the active repository; it is not executable here.

## Final protocol

- Feature set: `fixed48_v1` — 8 DWT, 16 WPT, 16 time-domain, and 8
  frequency-domain features from both vibration channels.
- Five predeclared bearing-disjoint folds, each with 9 train, 3 validation, and
  3 test bearings and all three operating conditions represented.
- Shared LSTM/GRU sequence lengths: fold 1=10, fold 2=30, fold 3=30, fold 4=10,
  fold 5=10. Primary test scoring uses the common range `step >= 29`.
- Seeds 42, 123, 2026; hidden size 32; one recurrent layer; Softplus output;
  batch 128; AdamW; Smooth L1; 40 maximum and 8 minimum epochs; patience 7.
- Primary metrics are equal-bearing macro MAE and RMSE in minutes.

The final 30-model experiment has **not** been run. Earlier work inspected some
outer-test results, so the future final table remains exploratory rather than a
fresh untouched confirmation.

## Setup and data

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[all]"
.\.venv\Scripts\python.exe scripts\prepare_data.py --validate
```

The canonical data files are
`data/processed/xjtu_fixed_features_48.csv` and its JSON manifest. If absent,
obtain the original XJTU-SY data as described in `data/README.md`, then run
`scripts/prepare_data.py --extract`. Extraction never performs feature
selection.

## Final experiment

Read-only readiness check:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_experiment.py --check-only
```

Complete resumable 30-checkpoint run:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_experiment.py
```

Display and verify the completed LSTM/GRU summary:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_experiment.py --summary-only
```

The runner verifies dataset and registry hashes before training or resuming,
refuses incompatible checkpoints, preserves raw predictions, and writes primary
common-range predictions and per-bearing/fold/condition/model summaries.

## Active entry points

```text
scripts/prepare_data.py
scripts/train.py
scripts/run_final_experiment.py
scripts/evaluate.py
scripts/reproduce_results.py
scripts/run_maintenance_replay.py
dashboard/app.py
```

The dashboard and maintenance replay use only final fixed48 direct-RUL
artifacts. Until the 30-model run completes they report missing evidence rather
than falling back to historical HI/Yao results. Replay costs are hypothetical
and do not establish industrial savings.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
git diff --check
```

See `docs/TEST_PLAN.md`, `docs/FEATURE_DEFINITIONS.md`,
`docs/EXPERIMENT_REGISTRY.md`, and `docs/FINAL_TECHNICAL_REPORT.md`.
