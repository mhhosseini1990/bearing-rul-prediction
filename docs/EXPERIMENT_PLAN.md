# Experiment plan

## Phase 1 – Repository and literature (20 Oct)

- [x] Define project scope: XJTU-SY, RUL, RF versus LSTM.
- [x] Create folders, initial code, tests and GitHub setup.
- [ ] Confirm dataset source, vibration channels, time intervals and end-of-life definition.
- [ ] Add papers and create literature map (15–20 references).

## Phase 2 – First baseline (by 3 Nov)

- [ ] Download XJTU-SY locally and inspect its folders and files.
- [ ] Validate the input feature and RUL-label assumptions.
- [ ] Run RF and record the actual number of train/test bearings.
- [ ] Evaluate MAE and RMSE, overall and per bearing.
- [ ] Plot predicted vs. true RUL for held-out bearings.
- [ ] Explain simple chronological historical replay.

## Phase 3 – Model comparison

- [ ] Define a fixed, leakage-safe train/validation/test protocol.
- [ ] Build the LSTM with chronological windows.
- [ ] Fit preprocessing only with training bearings.
- [ ] Use the **same held-out test bearings, target units, and metrics** for RF and LSTM.
- [ ] Compare generalization across operating conditions.

## Phase 4 – Reporting

- [ ] Record methods, packages, data origin and reproducibility steps.
- [ ] Discuss uncertainty, sample size, data leakage risks and deployment limits.
- [ ] Complete report and presentation/handout.

## Evaluation table (fill ONLY with real experiment results)

| Model | MAE (remaining recordings) | RMSE (remaining recordings) | Test bearings |
|---|---:|---:|---|
| Random Forest | TBD | TBD | TBD |
| LSTM | TBD | TBD | TBD |

**Do not invent benchmark scores.**
