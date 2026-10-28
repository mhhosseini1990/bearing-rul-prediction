# Test Plan — Rolling Bearing RUL Prediction (LSTM vs. GRU)

**Project:** Predictive Maintenance of Rolling Bearings  
**Course:** Case Study: Intelligent Systems in Production  
**Dataset:** XJTU-SY  
**Models:** LSTM and GRU  
**Document type:** Living verification and evaluation specification  
**Version:** 2.2 | 28 October 2026

## 1. Purpose and Scope

This document defines the planned verification and evaluation strategy for predicting rolling-bearing Remaining Useful Life (RUL) from historical vibration measurements. The project compares LSTM and GRU under a common, past-only, wavelet-based feature pipeline. This is a living specification applicable throughout development and evaluation. Test execution evidence is maintained separately; the plan itself does not claim passing tests or measured results.

**In scope:** data ingestion, chronology, wavelet feature extraction, RUL target generation, bearing-level dataset splits, sequence construction, normalization, model interfaces, training behavior, regression metrics, fair comparison, reproducibility, and historical replay.

**Out of scope:** live industrial deployment, sensor hardware, verified maintenance cost savings, and claims of real-world operational reliability.

## 2. Project Test Architecture

```text
rolling-bearing-rul-prediction/
├── configs/
│   └── bearing_split_v1.yaml    # Fixed 9/3/3 bearing assignments
├── data/raw/                   # Local XJTU-SY files (not committed)
├── docs/
│   ├── TEST_PLAN.md            # This verification contract
│   └── DATA_SPLIT.md           # Split rationale and bearing IDs
├── scripts/
│   └── validate_dataset.py     # Local structural CSV validation
├── src/bearing_rul/
│   ├── data/
│   │   ├── xjtu_loader.py      # CSV loading and chronological manifest
│   │   ├── rul_labels.py       # RUL labels
│   │   └── splits.py           # Bearing partition utilities
│   ├── features/               # Wavelet feature extraction
│   ├── models/                 # LSTM and GRU (planned)
│   └── evaluation/             # Metrics and replay
├── tests/
│   ├── test_loader.py          # Loader tests
│   ├── test_targets.py         # RUL target tests
│   ├── test_splits.py          # Split tests
│   ├── test_features.py        # Feature tests
│   └── test_metrics.py         # Metric tests
└── results/                    # Experiment evidence (planned)
```

Additional tests for sequences, scaling, model behavior and end-to-end integration are planned. The tree identifies intended responsibilities; the exact presence of every file must be checked against the repository. The repository may still contain legacy model files; these are not part of the approved LSTM-versus-GRU comparison.

## 3. Scientific Evaluation Contract

1. **Unit of independence:** a complete bearing run. No windows or samples from one bearing may appear in more than one train/validation/test partition.
2. **Temporal causality:** a prediction at time *t* may use only measurements available at or before *t*. Centered windows or future-informed preprocessing are prohibited.
3. **RUL definition (confirmed sampling protocol):** XJTU-SY records one 32,768-sample, 1.28-second vibration snapshot per minute, at 25.6 kHz. For an ordered, complete run with `N` snapshots and zero-based index `i`, define `rul_minutes(i) = (N - 1 - i) × 1 minute`. This is the remaining nominal elapsed time **between the current recorded snapshot and the final recorded snapshot**. The final recorded snapshot receives zero minutes as an **experimental end-of-life proxy**, not a claim of total mechanical breakdown. The official dataset documentation reports the experiments as complete run-to-failure and describes a stop criterion of vibration amplitude exceeding `10 × Ah`, where `Ah` is a normal-stage maximum. The model must not receive `N`, the future endpoint, the remaining-count label, or any future-derived feature at inference time. Distinguish the reported full experimental lifetime (which may count the first recording interval) from the last-to-current snapshot difference.
4. **Feature pipeline (initial contract):** apply a level-3 discrete wavelet transform (DWT), `db4` wavelet, `symmetric` boundary mode, independently to each available vibration channel in each single measurement snapshot. Compute mean squared coefficient energy for approximation `A3` and details `D3`, `D2`, `D1` (four features per channel). Reject snapshots too short for a valid level-3 decomposition rather than silently changing levels. The snapshot itself must be available at prediction time; never include later snapshots. Any additional features require a documented version change.
5. **Normalization:** fit scalers on training bearings only; apply the same fitted transformation to validation and test bearings.
6. **Model fairness:** use identical bearing partitions, prediction targets, available input history, feature definitions and evaluation code for LSTM and GRU. Report model-specific hyperparameters and tuning budgets.
7. **Model selection:** use validation data for tuning; reserve the test bearings for final evaluation. Do not tune based on test scores.
8. **Reported performance:** include MAE and RMSE, units, sample aggregation method, and preferably per-bearing results. A single pooled score must not hide uneven bearing performance.

**Fixed split policy:** 9 training, 3 validation and 3 test bearings from 15 XJTU-SY bearing histories. Each of the three operating conditions contributes 3 training, 1 validation and 1 test bearing. Complete bearing runs are indivisible; assignments are in `configs/bearing_split_v1.yaml` and `docs/DATA_SPLIT.md`. Local checks confirmed all 15 bearing directories and 9,216 CSV files; filename sequences were consecutive and all 9,216 files passed the project structural validation (two numeric finite vibration columns and 32,768 rows). These checks do not independently establish physical signal quality or exact timestamp integrity. This is a project-defined split, not an official benchmark split. Keep the test bearings untouched during tuning.


### 3.1 Configurable Baseline Experiment Contract

| Parameter | Initial setting | Rationale / limitation |
|---|---|---|
| RUL unit | Minutes, derived from the documented 1-minute sampling interval | Nominal time to the last recorded snapshot, not independently timestamp-verified time |
| Failure endpoint | Last snapshot of a documented complete run-to-failure experiment | Experimental stop criterion: maximum vibration amplitude exceeds `10 × Ah`; endpoint proxy, not necessarily total breakdown |
| Vibration channels | Horizontal and vertical accelerometer signals | Two columns confirmed in local CSV validation |
| Wavelet transform | DWT `db4`, level `3`, mode `symmetric` | Reproducible starting configuration, not a proven optimum |
| Wavelet features | Mean squared energy of `A3`, `D3`, `D2`, `D1` per channel | Fixed eight-dimensional feature vector if two channels are available |
| Sequence length | 20 consecutive measurement snapshots | Initial hyperparameter; short runs produce no full windows |
| Sequence target | RUL at the final snapshot of each input window | Strict causal alignment |
| Window stride | 1 snapshot | Fixed initial evaluation cadence |
| Scaling | StandardScaler fit on training snapshot features only | Avoids validation/test leakage; train-only statistics |
| Output | One scalar predicted RUL in minutes | Identical output contract for LSTM and GRU |
| Dataset split | 9 train / 3 validation / 3 test bearings | Fixed IDs; local structural validation of 9,216 CSVs passed |
| Primary metrics | MAE and RMSE in minutes | Report pooled and per-bearing results; note unequal bearing lengths and weighting |

**Important:** These are explicit **initial engineering choices**, not parameters asserted to be specified by the three literature sources or optimal for XJTU-SY. If the final approved proposal mandates a different feature representation, change the configuration and corresponding tests together under version control.

**Causality detail:** DWT is computed **within a completed measurement snapshot**, not across the full future bearing history. A prediction at snapshot `i` may use feature vectors only from snapshots `max(0, i-19)` through `i`; the initial version requires 20 complete snapshots and therefore starts at `i=19`.

**RUL label detail:** `rul_remaining_recordings = N - 1 - i` is an auxiliary integer count; `rul_minutes = rul_remaining_recordings × 1` is the primary regression target. The equality of numerical values is specific to the XJTU-SY one-minute sampling protocol. These supervised labels use the known future endpoint **offline**; neither endpoint nor label may be supplied as an inference feature. For `Bearing1_1` (123 files), targets run from 122 minutes at `1.csv` to 0 minutes at `123.csv`.

## 4. Test Cases and Acceptance Criteria

| ID | Area / proposed file | Test case | Expected result / acceptance criterion | Priority |
|---|---|---|---|---|
| D-01 | Loading / `test_loader.py` | Load a valid small vibration fixture | Expected channels, rows and numeric values are preserved | P0 |
| D-02 | Loading | Missing file, empty file or malformed schema | Explicit, informative error; no silent fallback | P0 |
| D-03 | Loading | Read successive measurements | Chronological ordering and bearing identity are preserved | P0 |
| F-01 | Features / `test_features.py` | Known synthetic signal through `db4` level-3 DWT energy function | Deterministic finite values, four energies per channel and reference values computed independently | P0 |
| F-02 | Features | Empty, nonnumeric or invalid signal | Explicit validation error | P1 |
| F-03 | Features | Change samples after prediction time *t* | Feature vector for time *t* remains unchanged | P0 |
| T-01 | Targets / `test_targets.py` | Five ordered snapshots with one-minute spacing | `rul_minutes` equals `[4, 3, 2, 1, 0]` minutes and `rul_remaining_recordings` equals `[4, 3, 2, 1, 0]` | P0 |
| T-02 | Targets | Missing bearing ID, invalid/duplicate/nonconsecutive steps, empty data | Explicit validation error; no fabricated labels | P0 |
| T-03 | Targets | Label alignment and bearing independence | Each sequence predicts minute-RUL of its last observed step; labels are computed independently for each bearing | P0 |
| S-01 | Splits / `test_splits.py` | Partition bearing IDs | Train, validation and test ID sets are pairwise disjoint | P0 |
| S-02 | Splits | Fixed seed and fixed bearing list | Repeated splitting yields the same assignments | P0 |
| S-03 | Splits | Check completeness and configured proportions | Each intended bearing assigned once; expected 9/3/3 counts (3/1/1 per condition) | P0 |
| Q-01 | Sequences / `test_sequences.py` | Construct 20-step temporal windows with stride 1 | For `N >= 20`, exactly `N-19` windows, each shaped `(20, feature_count)`, with endpoint-aligned labels | P0 |
| Q-02 | Sequences | First available prediction and short histories | Boundary behavior matches the documented policy | P1 |
| Q-03 | Sequences | Mutate future observations | Earlier windows and their inputs do not change | P0 |
| N-01 | Scaling / `test_scaling.py` | Extreme values only in test data | StandardScaler means/variances equal training-only fit and remain unchanged | P0 |
| M-01 | Metrics / `test_metrics.py` | `y_true=[100,80,60]`, `y_pred=[90,70,50]` in minutes | MAE = 10 minutes and RMSE = 10 minutes | P0 |
| M-02 | Metrics | Perfect predictions | MAE = 0 and RMSE = 0 | P0 |
| M-03 | Metrics | Shape mismatch, empty arrays or NaN/Inf | Explicit error according to the documented metric API | P0 |
| A-01 | Models / `test_models.py` | LSTM and GRU forward pass on tiny tensors | One finite RUL prediction per input sequence; correct shape | P0 |
| A-02 | Models | One minimal training step | Loss is finite; gradients/weights update where expected | P1 |
| A-03 | Models | Compare shared input and output contracts | Both models accept equivalent features and return comparable RUL units | P0 |
| I-01 | Pipeline / `test_pipeline.py` | Tiny synthetic end-to-end fixture | Loading → features → targets → split → sequences → inference → metrics completes | P1 |
| R-01 | Reproducibility | Repeat a controlled experiment | Same splits/configuration; deterministic components reproduce results within stated tolerance | P1 |
| H-01 | Historical replay | Simulate decision at past time *t* | Only information available at *t* affects that decision | P1 |

**Priority:** P0 = required correctness gate; P1 = important additional coverage. This table specifies required behavior; execution outcomes belong in separate test reports.

## 5. Test Data and Fixtures

- **Unit tests:** tiny synthetic arrays, short signals and fabricated bearing IDs; no full dataset download required.
- **Integration tests:** small, clearly identified subsets of actual XJTU-SY files after access and format validation.
- **Evaluation:** held-out complete bearing runs, never random rows/windows from the same bearing across partitions.
- **Fixture policy:** commit only small synthetic fixtures unless the dataset license explicitly permits redistribution. Do not commit large raw vibration data.

## 6. Model Evaluation Procedure

1. Verify data schema and chronological order.
2. Freeze bearing-level train/validation/test IDs and record operating conditions.
3. Use the documented 1-minute measurement cadence and the experimental run-to-failure endpoint; calculate minute-RUL as `(N - 1 - i) × 1 minute`.
4. Extract causal wavelet-based features using the same settings for both models.
5. Fit any learned preprocessing on training bearings only.
6. Construct aligned past-only sequences and targets.
7. Train LSTM and GRU with documented configurations and seeds.
8. Select hyperparameters using validation bearings only.
9. Evaluate both models on the same untouched test bearings.
10. Report MAE and RMSE **in minutes**, both pooled and per-bearing, with aggregation/weighting, configurations and limitations.
11. Optionally replay historical maintenance decisions as a **simulation**, not verified industrial cost savings.

### Required result record (once experiments exist)

| Field | Description |
|---|---|
| Model | LSTM or GRU |
| Dataset and bearings | Dataset version and exact train/val/test IDs |
| RUL unit | Minutes to the final recorded snapshot: `(N - 1 - i) × 1 minute` |
| Features | Wavelet type/level/statistics and channel selection |
| Sequence length | Number of past observations used |
| Training settings | Seed, epochs, batch size, optimizer, learning rate |
| MAE / RMSE | Held-out test values in minutes |
| Per-bearing scores | Metrics for each held-out bearing |
| Runtime / hardware | Environment for reproducibility |
| Notes | Failures, limitations, and deviations from plan |

No performance targets are asserted before a baseline experiment. Acceptance for the scientific comparison requires a leakage-free, reproducible protocol and correctly computed metrics, **not** a predetermined accuracy claim.

### Dataset provenance and verification evidence

The project uses the supplied *Introduction to XJTU-SY Bearing Dataset* (sections 1–3) as its primary sampling and failure-protocol source. It states 15 complete run-to-failure bearings, three operating conditions (35 Hz/12 kN, 37.5 Hz/11 kN, 40 Hz/10 kN), 25.6 kHz acquisition, 32,768 samples per 1.28-second recording, one recording per minute, and a vibration-amplitude stopping criterion of `10 × Ah`. It also tabulates bearing lifetimes and CSV counts. The project's local VS Code evidence (28 October 2026) reports 9/9 synthetic loader tests passing and structural validation of 9,216 CSVs from 15 bearings with zero reported errors. This does **not** constitute independent verification of sensor calibration, physically plausible signal behavior, or an external benchmark result.

## 7. Test Execution and Evidence

Use `pytest` for automated tests once test modules exist:

```bash
python -m pytest tests/ -v
```

Record execution date, commit hash, environment, passed/failed/skipped counts, and any failure explanation. A missing production module during test-first development is an expected initial **RED** state; it is not evidence that the implemented system failed validation.

A test can be marked **PASS** only after execution confirms the documented assertion. Mark unimplemented tests **NOT RUN**, not PASS.

## 8. Quality Gates

**Gate 1 — Test design:** contracts, target definition, data split policy, and P0 assertions agreed.

**Gate 2 — Core implementation:** all applicable P0 unit tests pass; failures and exclusions documented.

**Gate 3 — Integration:** small end-to-end pipeline works with correctly aligned sequences and labels.

**Gate 4 — Model comparison:** LSTM and GRU are evaluated on identical held-out bearings with recorded MAE/RMSE and experiment settings.

**Gate 5 — Final reporting:** results are reproducible to the extent supported by the environment, limitations are disclosed, and any historical replay is labeled as simulation.

## 9. Test Maintenance and Change Control

This document is the standing reference for tests across the project lifecycle. Any change to RUL units, wavelet decomposition, sequence construction, split policy, preprocessing, model interfaces or evaluation metrics must be reflected in the relevant configuration, test assertions and documentation in the same change set.

- Keep test fixtures small, deterministic and independent of the full dataset wherever possible.
- Add regression tests for each corrected defect.
- Review causal leakage and bearing-level independence whenever preprocessing or windowing changes.
- Keep baseline parameters configurable; do not hard-code them as universal scientific requirements.
- Store test execution results (commit, environment, outcomes and failures) separately from this plan.
- Do not silently relax acceptance criteria to make a failing implementation pass; document and justify any specification change.

## 10. Traceability and Ownership

Each implemented test should retain its test-case ID in its name, documentation or test metadata. Map the test case to the relevant source module and requirement. Code reviewers should verify that new functionality has appropriate test coverage and that any changes preserve the agreed LSTM-versus-GRU comparison protocol.

**Document maintenance rule:** Update this plan when the system contract changes, not merely when a task is completed. Operational status, deadlines and next-action checklists belong in project tracking documents rather than this test plan.
# Test Plan — Rolling Bearing RUL Prediction (LSTM vs. GRU)

**Project:** Predictive Maintenance of Rolling Bearings  
**Course:** Case Study: Intelligent Systems in Production  
**Dataset:** XJTU-SY  
**Models:** LSTM and GRU  
**Document type:** Living verification and evaluation specification  
**Version:** 2.2 | 28 October 2026

## 1. Purpose and Scope

This document defines the planned verification and evaluation strategy for predicting rolling-bearing Remaining Useful Life (RUL) from historical vibration measurements. The project compares LSTM and GRU under a common, past-only, wavelet-based feature pipeline. This is a living specification applicable throughout development and evaluation. Test execution evidence is maintained separately; the plan itself does not claim passing tests or measured results.

**In scope:** data ingestion, chronology, wavelet feature extraction, RUL target generation, bearing-level dataset splits, sequence construction, normalization, model interfaces, training behavior, regression metrics, fair comparison, reproducibility, and historical replay.

**Out of scope:** live industrial deployment, sensor hardware, verified maintenance cost savings, and claims of real-world operational reliability.

## 2. Project Test Architecture

```text
rolling-bearing-rul-prediction/
├── configs/
│   └── bearing_split_v1.yaml    # Fixed 9/3/3 bearing assignments
├── data/raw/                   # Local XJTU-SY files (not committed)
├── docs/
│   ├── TEST_PLAN.md            # This verification contract
│   └── DATA_SPLIT.md           # Split rationale and bearing IDs
├── scripts/
│   └── validate_dataset.py     # Local structural CSV validation
├── src/bearing_rul/
│   ├── data/
│   │   ├── xjtu_loader.py      # CSV loading and chronological manifest
│   │   ├── rul_labels.py       # RUL labels
│   │   └── splits.py           # Bearing partition utilities
│   ├── features/               # Wavelet feature extraction
│   ├── models/                 # LSTM and GRU (planned)
│   └── evaluation/             # Metrics and replay
├── tests/
│   ├── test_loader.py          # Loader tests
│   ├── test_targets.py         # RUL target tests
│   ├── test_splits.py          # Split tests
│   ├── test_features.py        # Feature tests
│   └── test_metrics.py         # Metric tests
└── results/                    # Experiment evidence (planned)
```

Additional tests for sequences, scaling, model behavior and end-to-end integration are planned. The tree identifies intended responsibilities; the exact presence of every file must be checked against the repository. The repository may still contain legacy model files; these are not part of the approved LSTM-versus-GRU comparison.

## 3. Scientific Evaluation Contract

1. **Unit of independence:** a complete bearing run. No windows or samples from one bearing may appear in more than one train/validation/test partition.
2. **Temporal causality:** a prediction at time *t* may use only measurements available at or before *t*. Centered windows or future-informed preprocessing are prohibited.
3. **RUL definition (confirmed sampling protocol):** XJTU-SY records one 32,768-sample, 1.28-second vibration snapshot per minute, at 25.6 kHz. For an ordered, complete run with `N` snapshots and zero-based index `i`, define `rul_minutes(i) = (N - 1 - i) × 1 minute`. This is the remaining nominal elapsed time **between the current recorded snapshot and the final recorded snapshot**. The final recorded snapshot receives zero minutes as an **experimental end-of-life proxy**, not a claim of total mechanical breakdown. The official dataset documentation reports the experiments as complete run-to-failure and describes a stop criterion of vibration amplitude exceeding `10 × Ah`, where `Ah` is a normal-stage maximum. The model must not receive `N`, the future endpoint, the remaining-count label, or any future-derived feature at inference time. Distinguish the reported full experimental lifetime (which may count the first recording interval) from the last-to-current snapshot difference.
4. **Feature pipeline (initial contract):** apply a level-3 discrete wavelet transform (DWT), `db4` wavelet, `symmetric` boundary mode, independently to each available vibration channel in each single measurement snapshot. Compute mean squared coefficient energy for approximation `A3` and details `D3`, `D2`, `D1` (four features per channel). Reject snapshots too short for a valid level-3 decomposition rather than silently changing levels. The snapshot itself must be available at prediction time; never include later snapshots. Any additional features require a documented version change.
5. **Normalization:** fit scalers on training bearings only; apply the same fitted transformation to validation and test bearings.
6. **Model fairness:** use identical bearing partitions, prediction targets, available input history, feature definitions and evaluation code for LSTM and GRU. Report model-specific hyperparameters and tuning budgets.
7. **Model selection:** use validation data for tuning; reserve the test bearings for final evaluation. Do not tune based on test scores.
8. **Reported performance:** include MAE and RMSE, units, sample aggregation method, and preferably per-bearing results. A single pooled score must not hide uneven bearing performance.

**Fixed split policy:** 9 training, 3 validation and 3 test bearings from 15 XJTU-SY bearing histories. Each of the three operating conditions contributes 3 training, 1 validation and 1 test bearing. Complete bearing runs are indivisible; assignments are in `configs/bearing_split_v1.yaml` and `docs/DATA_SPLIT.md`. Local checks confirmed all 15 bearing directories and 9,216 CSV files; filename sequences were consecutive and all 9,216 files passed the project structural validation (two numeric finite vibration columns and 32,768 rows). These checks do not independently establish physical signal quality or exact timestamp integrity. This is a project-defined split, not an official benchmark split. Keep the test bearings untouched during tuning.


### 3.1 Configurable Baseline Experiment Contract

| Parameter | Initial setting | Rationale / limitation |
|---|---|---|
| RUL unit | Minutes, derived from the documented 1-minute sampling interval | Nominal time to the last recorded snapshot, not independently timestamp-verified time |
| Failure endpoint | Last snapshot of a documented complete run-to-failure experiment | Experimental stop criterion: maximum vibration amplitude exceeds `10 × Ah`; endpoint proxy, not necessarily total breakdown |
| Vibration channels | Horizontal and vertical accelerometer signals | Two columns confirmed in local CSV validation |
| Wavelet transform | DWT `db4`, level `3`, mode `symmetric` | Reproducible starting configuration, not a proven optimum |
| Wavelet features | Mean squared energy of `A3`, `D3`, `D2`, `D1` per channel | Fixed eight-dimensional feature vector if two channels are available |
| Sequence length | 20 consecutive measurement snapshots | Initial hyperparameter; short runs produce no full windows |
| Sequence target | RUL at the final snapshot of each input window | Strict causal alignment |
| Window stride | 1 snapshot | Fixed initial evaluation cadence |
| Scaling | StandardScaler fit on training snapshot features only | Avoids validation/test leakage; train-only statistics |
| Output | One scalar predicted RUL in minutes | Identical output contract for LSTM and GRU |
| Dataset split | 9 train / 3 validation / 3 test bearings | Fixed IDs; local structural validation of 9,216 CSVs passed |
| Primary metrics | MAE and RMSE in minutes | Report pooled and per-bearing results; note unequal bearing lengths and weighting |

**Important:** These are explicit **initial engineering choices**, not parameters asserted to be specified by the three literature sources or optimal for XJTU-SY. If the final approved proposal mandates a different feature representation, change the configuration and corresponding tests together under version control.

**Causality detail:** DWT is computed **within a completed measurement snapshot**, not across the full future bearing history. A prediction at snapshot `i` may use feature vectors only from snapshots `max(0, i-19)` through `i`; the initial version requires 20 complete snapshots and therefore starts at `i=19`.

**RUL label detail:** `rul_remaining_recordings = N - 1 - i` is an auxiliary integer count; `rul_minutes = rul_remaining_recordings × 1` is the primary regression target. The equality of numerical values is specific to the XJTU-SY one-minute sampling protocol. These supervised labels use the known future endpoint **offline**; neither endpoint nor label may be supplied as an inference feature. For `Bearing1_1` (123 files), targets run from 122 minutes at `1.csv` to 0 minutes at `123.csv`.

## 4. Test Cases and Acceptance Criteria

| ID | Area / proposed file | Test case | Expected result / acceptance criterion | Priority |
|---|---|---|---|---|
| D-01 | Loading / `test_loader.py` | Load a valid small vibration fixture | Expected channels, rows and numeric values are preserved | P0 |
| D-02 | Loading | Missing file, empty file or malformed schema | Explicit, informative error; no silent fallback | P0 |
| D-03 | Loading | Read successive measurements | Chronological ordering and bearing identity are preserved | P0 |
| F-01 | Features / `test_features.py` | Known synthetic signal through `db4` level-3 DWT energy function | Deterministic finite values, four energies per channel and reference values computed independently | P0 |
| F-02 | Features | Empty, nonnumeric or invalid signal | Explicit validation error | P1 |
| F-03 | Features | Change samples after prediction time *t* | Feature vector for time *t* remains unchanged | P0 |
| T-01 | Targets / `test_targets.py` | Five ordered snapshots with one-minute spacing | `rul_minutes` equals `[4, 3, 2, 1, 0]` minutes and `rul_remaining_recordings` equals `[4, 3, 2, 1, 0]` | P0 |
| T-02 | Targets | Missing bearing ID, invalid/duplicate/nonconsecutive steps, empty data | Explicit validation error; no fabricated labels | P0 |
| T-03 | Targets | Label alignment and bearing independence | Each sequence predicts minute-RUL of its last observed step; labels are computed independently for each bearing | P0 |
| S-01 | Splits / `test_splits.py` | Partition bearing IDs | Train, validation and test ID sets are pairwise disjoint | P0 |
| S-02 | Splits | Fixed seed and fixed bearing list | Repeated splitting yields the same assignments | P0 |
| S-03 | Splits | Check completeness and configured proportions | Each intended bearing assigned once; expected 9/3/3 counts (3/1/1 per condition) | P0 |
| Q-01 | Sequences / `test_sequences.py` | Construct 20-step temporal windows with stride 1 | For `N >= 20`, exactly `N-19` windows, each shaped `(20, feature_count)`, with endpoint-aligned labels | P0 |
| Q-02 | Sequences | First available prediction and short histories | Boundary behavior matches the documented policy | P1 |
| Q-03 | Sequences | Mutate future observations | Earlier windows and their inputs do not change | P0 |
| N-01 | Scaling / `test_scaling.py` | Extreme values only in test data | StandardScaler means/variances equal training-only fit and remain unchanged | P0 |
| M-01 | Metrics / `test_metrics.py` | `y_true=[100,80,60]`, `y_pred=[90,70,50]` in minutes | MAE = 10 minutes and RMSE = 10 minutes | P0 |
| M-02 | Metrics | Perfect predictions | MAE = 0 and RMSE = 0 | P0 |
| M-03 | Metrics | Shape mismatch, empty arrays or NaN/Inf | Explicit error according to the documented metric API | P0 |
| A-01 | Models / `test_models.py` | LSTM and GRU forward pass on tiny tensors | One finite RUL prediction per input sequence; correct shape | P0 |
| A-02 | Models | One minimal training step | Loss is finite; gradients/weights update where expected | P1 |
| A-03 | Models | Compare shared input and output contracts | Both models accept equivalent features and return comparable RUL units | P0 |
| I-01 | Pipeline / `test_pipeline.py` | Tiny synthetic end-to-end fixture | Loading → features → targets → split → sequences → inference → metrics completes | P1 |
| R-01 | Reproducibility | Repeat a controlled experiment | Same splits/configuration; deterministic components reproduce results within stated tolerance | P1 |
| H-01 | Historical replay | Simulate decision at past time *t* | Only information available at *t* affects that decision | P1 |

**Priority:** P0 = required correctness gate; P1 = important additional coverage. This table specifies required behavior; execution outcomes belong in separate test reports.

## 5. Test Data and Fixtures

- **Unit tests:** tiny synthetic arrays, short signals and fabricated bearing IDs; no full dataset download required.
- **Integration tests:** small, clearly identified subsets of actual XJTU-SY files after access and format validation.
- **Evaluation:** held-out complete bearing runs, never random rows/windows from the same bearing across partitions.
- **Fixture policy:** commit only small synthetic fixtures unless the dataset license explicitly permits redistribution. Do not commit large raw vibration data.

## 6. Model Evaluation Procedure

1. Verify data schema and chronological order.
2. Freeze bearing-level train/validation/test IDs and record operating conditions.
3. Use the documented 1-minute measurement cadence and the experimental run-to-failure endpoint; calculate minute-RUL as `(N - 1 - i) × 1 minute`.
4. Extract causal wavelet-based features using the same settings for both models.
5. Fit any learned preprocessing on training bearings only.
6. Construct aligned past-only sequences and targets.
7. Train LSTM and GRU with documented configurations and seeds.
8. Select hyperparameters using validation bearings only.
9. Evaluate both models on the same untouched test bearings.
10. Report MAE and RMSE **in minutes**, both pooled and per-bearing, with aggregation/weighting, configurations and limitations.
11. Optionally replay historical maintenance decisions as a **simulation**, not verified industrial cost savings.

### Required result record (once experiments exist)

| Field | Description |
|---|---|
| Model | LSTM or GRU |
| Dataset and bearings | Dataset version and exact train/val/test IDs |
| RUL unit | Minutes to the final recorded snapshot: `(N - 1 - i) × 1 minute` |
| Features | Wavelet type/level/statistics and channel selection |
| Sequence length | Number of past observations used |
| Training settings | Seed, epochs, batch size, optimizer, learning rate |
| MAE / RMSE | Held-out test values in minutes |
| Per-bearing scores | Metrics for each held-out bearing |
| Runtime / hardware | Environment for reproducibility |
| Notes | Failures, limitations, and deviations from plan |

No performance targets are asserted before a baseline experiment. Acceptance for the scientific comparison requires a leakage-free, reproducible protocol and correctly computed metrics, **not** a predetermined accuracy claim.

### Dataset provenance and verification evidence

The project uses the supplied *Introduction to XJTU-SY Bearing Dataset* (sections 1–3) as its primary sampling and failure-protocol source. It states 15 complete run-to-failure bearings, three operating conditions (35 Hz/12 kN, 37.5 Hz/11 kN, 40 Hz/10 kN), 25.6 kHz acquisition, 32,768 samples per 1.28-second recording, one recording per minute, and a vibration-amplitude stopping criterion of `10 × Ah`. It also tabulates bearing lifetimes and CSV counts. The project's local VS Code evidence (28 October 2026) reports 9/9 synthetic loader tests passing and structural validation of 9,216 CSVs from 15 bearings with zero reported errors. This does **not** constitute independent verification of sensor calibration, physically plausible signal behavior, or an external benchmark result.

## 7. Test Execution and Evidence

Use `pytest` for automated tests once test modules exist:

```bash
python -m pytest tests/ -v
```

Record execution date, commit hash, environment, passed/failed/skipped counts, and any failure explanation. A missing production module during test-first development is an expected initial **RED** state; it is not evidence that the implemented system failed validation.

A test can be marked **PASS** only after execution confirms the documented assertion. Mark unimplemented tests **NOT RUN**, not PASS.

## 8. Quality Gates

**Gate 1 — Test design:** contracts, target definition, data split policy, and P0 assertions agreed.

**Gate 2 — Core implementation:** all applicable P0 unit tests pass; failures and exclusions documented.

**Gate 3 — Integration:** small end-to-end pipeline works with correctly aligned sequences and labels.

**Gate 4 — Model comparison:** LSTM and GRU are evaluated on identical held-out bearings with recorded MAE/RMSE and experiment settings.

**Gate 5 — Final reporting:** results are reproducible to the extent supported by the environment, limitations are disclosed, and any historical replay is labeled as simulation.

## 9. Test Maintenance and Change Control

This document is the standing reference for tests across the project lifecycle. Any change to RUL units, wavelet decomposition, sequence construction, split policy, preprocessing, model interfaces or evaluation metrics must be reflected in the relevant configuration, test assertions and documentation in the same change set.

- Keep test fixtures small, deterministic and independent of the full dataset wherever possible.
- Add regression tests for each corrected defect.
- Review causal leakage and bearing-level independence whenever preprocessing or windowing changes.
- Keep baseline parameters configurable; do not hard-code them as universal scientific requirements.
- Store test execution results (commit, environment, outcomes and failures) separately from this plan.
- Do not silently relax acceptance criteria to make a failing implementation pass; document and justify any specification change.

## 10. Traceability and Ownership

Each implemented test should retain its test-case ID in its name, documentation or test metadata. Map the test case to the relevant source module and requirement. Code reviewers should verify that new functionality has appropriate test coverage and that any changes preserve the agreed LSTM-versus-GRU comparison protocol.

**Document maintenance rule:** Update this plan when the system contract changes, not merely when a task is completed. Operational status, deadlines and next-action checklists belong in project tracking documents rather than this test plan.
