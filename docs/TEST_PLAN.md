# Test Plan — fixed48 direct-RUL final protocol

## Release gates

1. Validate the fixed48 registry, deterministic extraction, two-channel use,
   manifest, 9,216 rows, 15 bearings, and three conditions.
2. Verify every fold is disjoint 9/3/3, represents all conditions, and each
   bearing is test exactly once.
3. Verify scaling and target scaling use training bearings only; windows remain
   within a bearing and contain only past/current observations.
4. Exercise LSTM and GRU forward/backward paths with `(batch, sequence, 48)`,
   `(batch, 1)` Softplus output, AdamW/Smooth-L1 training, and deterministic seed.
5. Validate checkpoint provenance/save/load, validation-only selection,
   minimum-epoch/early-stopping behavior, and resume refusal on mismatch.
6. Score primary test predictions only at `step >= 29`; compute per-bearing
   MAE/RMSE before equal-bearing aggregation.
7. Confirm maintenance actions use only time and current/past predictions;
   ground truth is retrospective scoring evidence only.
8. Run all six dashboard pages, full pytest, readiness check, smoke training,
   checkpoint inference, and `git diff --check`.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --validate
.\.venv\Scripts\python.exe scripts\run_final_experiment.py --check-only
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
git diff --check
```

The full 30-model training is explicitly outside readiness verification and is
started only by the repository owner. Passing unit tests does not constitute a
completed scientific experiment.
