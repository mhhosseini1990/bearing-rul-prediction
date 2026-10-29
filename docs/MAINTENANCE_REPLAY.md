# Historical Maintenance Replay (offline baseline simulation)

## What is completed

- Python replay implementation with an explicit, chronological warning rule.
- Tests for consecutive-alert confirmation, no future-information leakage, event categories, schema failures, bearing/model alignment, and command-line file outputs.
- An executable CLI that reads the *existing* LSTM and GRU held-out prediction CSVs; no retraining required.

**Important:** Passing tests based on synthetic predictions does not prove that the replay has been executed on the real XJTU-SY test predictions. Execute the command below on the project machine and preserve the actual output as evidence.

## Run on the project's saved evaluation predictions

From the repository root on Windows PowerShell:

```powershell
python -m pytest tests/test_maintenance_replay.py -q
python scripts/run_maintenance_replay.py --run-name main --threshold-minutes 120 --consecutive-snapshots 3 --minimum-lead-minutes 30
Get-Content outputs/main/maintenance_replay/replay_events.csv
Get-Content outputs/main/maintenance_replay/replay_summary.csv
```

Expected existing inputs:

- `outputs/main/evaluation/lstm_predictions.csv`
- `outputs/main/evaluation/gru_predictions.csv`

Default output files:

- `outputs/main/maintenance_replay/replay_events.csv` — 1 event per model and test bearing
- `outputs/main/maintenance_replay/replay_summary.csv` — event counts and mean warning lead time per model
- `outputs/main/maintenance_replay/replay_timeline.csv` — each time step, prediction and alert flag
- `outputs/main/maintenance_replay/replay_config.json` — settings for reproducibility

## Decision policy (illustrative, not optimized)

1. Read each test bearing's predictions in strict chronological order (by `target_step` or accepted step alias).
2. For each model, trigger *one first alert* when predicted RUL is <= 120 min for 3 consecutive snapshots. This requires no future predictions, labels or failure time.
3. **Only after the decision**, compare the recorded actual RUL at the alert step to determine the alert lead time.
4. Label each bearing/model event as:
   - `early`: lead > 120 min (earlier than the intended warning horizon)
   - `timely`: 30 <= lead <= 120 min
   - `late`: lead < 30 min
   - `missed`: no alert by the recorded failure endpoint

Thresholds are demonstration design choices, **not validated industrial limits**. With a 120-minute threshold, a short-lived bearing may already be inside the warning horizon at the first eligible snapshot; state this limitation if it occurs. The user may choose other values through CLI flags, but report the chosen values and do not tune them using the test data to claim generalization.

## Input CSV compatibility

The program auto-recognizes these common column names (case-insensitive):

- Bearing: `bearing_id`, `bearing`, `bearing_name`, `test_bearing`, `unit_id`
- Chronology: `target_step`, `step`, `snapshot_index`, `snapshot_step`, `time_step`, `sequence_end_step`, `file_index`, `index_in_bearing`
- Ground-truth RUL: `actual_rul_minutes`, `true_rul_minutes`, `rul_true_minutes`, `rul_actual_minutes`, `y_true_minutes`, `actual_rul`, `true_rul`, `rul_true`, `rul_minutes`, `y_true`
- Predicted RUL: `predicted_rul_minutes`, `prediction_minutes`, `pred_rul_minutes`, `rul_pred_minutes`, `y_pred_minutes`, `predicted_rul`, `prediction`, `pred_rul`, `rul_pred`, `y_pred`

If your CSV schema differs, the CLI prints a meaningful error listing the actual columns. Update the alias mapping only after verifying the meaning and **units** of the fields. Do not silently derive chronological order from `actual_rul_minutes`.

## Validity and limitations

- A given step must have the same true RUL for both models; the script rejects misaligned test predictions.
- Assumes actual RUL is stored in **minutes**, not normalized training units. Confirm your evaluation CSV before running.
- Expects the test prediction history to include the failure endpoint (last true RUL = 0).
- The three test bearings are not a large independent sample; the 448 overlapping windows are correlated.
- All run-to-failure histories eventually fail, so this experiment cannot measure false-alarm rates on genuinely healthy systems, maintenance cost savings, or deployed industrial reliability.
- Do not claim the replay has been experimentally completed on XJTU-SY until its output CSVs have been generated and inspected on the project computer.
