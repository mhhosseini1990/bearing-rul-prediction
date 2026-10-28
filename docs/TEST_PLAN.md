# Test Plan — Rolling Bearing RUL Prediction (LSTM vs. GRU)

**Project:** Predictive Maintenance of Rolling Bearings  
**Course:** Case Study: Intelligent Systems in Production  
**Dataset:** XJTU-SY  
**Models:** LSTM and GRU  
**Document type:** Living verification and evaluation specification  
**Version:** 2.0 | 28 October 2026

## 1. Purpose and Scope

This document defines the planned verification and evaluation strategy for predicting rolling-bearing Remaining Useful Life (RUL) from historical vibration measurements. The project compares LSTM and GRU under a common, past-only, wavelet-based feature pipeline. This is a living specification applicable throughout development and evaluation. Test execution evidence is maintained separately; the plan itself does not claim passing tests or measured results.

**In scope:** data ingestion, chronology, wavelet feature extraction, RUL target generation, bearing-level dataset splits, sequence construction, normalization, model interfaces, training behavior, regression metrics, fair comparison, reproducibility, and historical replay.

**Out of scope:** live industrial deployment, sensor hardware, verified maintenance cost savings, and claims of real-world operational reliability.

## 2. Project Test Architecture

```text
bearing-rul-prediction/
├── configs/                 # Experiment settings, model hyperparameters and seeds
├── data/                    # Dataset instructions and local data organization
├── docs/
│   └── TEST_PLAN.md         # This document: verification and evaluation contract
├── src/                     # Production implementation (to be developed)
│   ├── data/                # File loading, bearing splits and RUL labels
│   ├── features/            # Wavelet feature extraction and scaling
│   ├── sequences/           # Past-only temporal window construction
│   ├── models/              # LSTM and GRU architectures and training
│   └── evaluation/          # Metrics, comparisons and historical replay
├── tests/                   # Automated tests (to be implemented)
│   ├── test_loader.py       # Data reading, schema, ordering and invalid inputs
│   ├── test_features.py     # Wavelet features and past-only behavior
│   ├── test_targets.py      # RUL target definition and boundary cases
│   ├── test_splits.py       # Bearing-level isolation and split reproducibility
│   ├── test_metrics.py      # MAE/RMSE correctness and invalid inputs
│   ├── test_sequences.py    # Temporal windows and target alignment (planned)
│   ├── test_scaling.py      # Train-only fitting and transformation (planned)
│   ├── test_models.py       # LSTM/GRU input-output and basic training (planned)
│   └── test_pipeline.py     # Small end-to-end synthetic pipeline (planned)
└── results/                 # Metrics, plots and experiment records (planned)
```

The `src/` subdirectories and additional test files above are **proposed responsibilities**, not claims that these paths already exist. Adapt names to the repository before implementing imports.

## 3. Scientific Evaluation Contract

1. **Unit of independence:** a complete bearing run. No windows or samples from one bearing may appear in more than one train/validation/test partition.
2. **Temporal causality:** a prediction at time *t* may use only measurements available at or before *t*. Centered windows or future-informed preprocessing are prohibited.
3. **RUL definition (initial contract):** predict the number of remaining measurement snapshots: `RUL(i) = N - 1 - i` for zero-based snapshot index `i` in a complete, ordered run of `N` snapshots. The last recorded snapshot is assigned zero **only as an experimental endpoint proxy**; verify that it represents the end of the run-to-failure experiment, not a censored recording. Do not describe step-based RUL as hours or minutes. If timestamps are later verified, time-based RUL is a separate explicitly versioned experiment.
4. **Feature pipeline (initial contract):** apply a level-3 discrete wavelet transform (DWT), `db4` wavelet, `symmetric` boundary mode, independently to each available vibration channel in each single measurement snapshot. Compute mean squared coefficient energy for approximation `A3` and details `D3`, `D2`, `D1` (four features per channel). Reject snapshots too short for a valid level-3 decomposition rather than silently changing levels. The snapshot itself must be available at prediction time; never include later snapshots. Any additional features require a documented version change.
5. **Normalization:** fit scalers on training bearings only; apply the same fitted transformation to validation and test bearings.
6. **Model fairness:** use identical bearing partitions, prediction targets, available input history, feature definitions and evaluation code for LSTM and GRU. Report model-specific hyperparameters and tuning budgets.
7. **Model selection:** use validation data for tuning; reserve the test bearings for final evaluation. Do not tune based on test scores.
8. **Reported performance:** include MAE and RMSE, units, sample aggregation method, and preferably per-bearing results. A single pooled score must not hide uneven bearing performance.

**Initial split policy:** 10 training, 2 validation and 3 test bearings from 15 XJTU-SY bearing histories. Use complete bearing IDs as indivisible units. Freeze and commit the explicit ID lists in configuration before any training. Aim for operating-condition coverage and report unavoidable imbalance; do not claim a random 10/2/3 split guarantees representative conditions. The exact IDs remain pending verification of the downloaded dataset inventory.


### 3.1 Configurable Baseline Experiment Contract

| Parameter | Initial setting | Rationale / limitation |
|---|---|---|
| RUL unit | Remaining measurement snapshots (integer steps) | Avoids assuming unverified wall-clock spacing; not physical time |
| Failure endpoint | Last snapshot of a confirmed complete run-to-failure bearing | Verify dataset metadata before generating labels |
| Vibration channels | Use both recorded vibration axes when present | Validate actual file schema; no invented channels |
| Wavelet transform | DWT `db4`, level `3`, mode `symmetric` | Reproducible starting configuration, not a proven optimum |
| Wavelet features | Mean squared energy of `A3`, `D3`, `D2`, `D1` per channel | Fixed eight-dimensional feature vector if two channels are available |
| Sequence length | 20 consecutive measurement snapshots | Initial hyperparameter; short runs produce no full windows |
| Sequence target | RUL at the final snapshot of each input window | Strict causal alignment |
| Window stride | 1 snapshot | Fixed initial evaluation cadence |
| Scaling | StandardScaler fit on training snapshot features only | Avoids validation/test leakage; train-only statistics |
| Output | One scalar predicted RUL in snapshot steps | Identical output contract for LSTM and GRU |
| Dataset split | 10 train / 2 validation / 3 test bearings | IDs frozen after inspecting dataset inventory |
| Primary metrics | MAE and RMSE in snapshot steps | Report pooled and per-bearing results; note weighting |

**Important:** These are explicit **initial engineering choices**, not parameters asserted to be specified by the three literature sources or optimal for XJTU-SY. If the final approved proposal mandates a different feature representation, change the configuration and corresponding tests together under version control.

**Causality detail:** DWT is computed **within a completed measurement snapshot**, not across the full future bearing history. A prediction at snapshot `i` may use feature vectors only from snapshots `max(0, i-19)` through `i`; the initial version requires 20 complete snapshots and therefore starts at `i=19`.

**RUL label detail:** `RUL(i)` is a supervised training label derived from the known experimental endpoint. The model never receives the endpoint or the label as an input feature during inference.

## 4. Test Cases and Acceptance Criteria

| ID | Area / proposed file | Test case | Expected result / acceptance criterion | Priority |
|---|---|---|---|---|
| D-01 | Loading / `test_loader.py` | Load a valid small vibration fixture | Expected channels, rows and numeric values are preserved | P0 |
| D-02 | Loading | Missing file, empty file or malformed schema | Explicit, informative error; no silent fallback | P0 |
| D-03 | Loading | Read successive measurements | Chronological ordering and bearing identity are preserved | P0 |
| F-01 | Features / `test_features.py` | Known synthetic signal through `db4` level-3 DWT energy function | Deterministic finite values, four energies per channel and reference values computed independently | P0 |
| F-02 | Features | Empty, nonnumeric or invalid signal | Explicit validation error | P1 |
| F-03 | Features | Change samples after prediction time *t* | Feature vector for time *t* remains unchanged | P0 |
| T-01 | Targets / `test_targets.py` | Five ordered observations with step-based RUL | Targets equal `[4, 3, 2, 1, 0]` under the step convention | P0 |
| T-02 | Targets | Invalid length, missing endpoint or inconsistent timestamps | Explicit validation error; no fabricated labels | P0 |
| T-03 | Targets | Label alignment with sequence endpoint | Each sequence predicts the RUL of its last observed time step | P0 |
| S-01 | Splits / `test_splits.py` | Partition bearing IDs | Train, validation and test ID sets are pairwise disjoint | P0 |
| S-02 | Splits | Fixed seed and fixed bearing list | Repeated splitting yields the same assignments | P0 |
| S-03 | Splits | Check completeness and configured proportions | Each intended bearing assigned once; expected 10/2/3 counts if selected | P0 |
| Q-01 | Sequences / `test_sequences.py` | Construct 20-step temporal windows with stride 1 | For `N >= 20`, exactly `N-19` windows, each shaped `(20, feature_count)`, with endpoint-aligned labels | P0 |
| Q-02 | Sequences | First available prediction and short histories | Boundary behavior matches the documented policy | P1 |
| Q-03 | Sequences | Mutate future observations | Earlier windows and their inputs do not change | P0 |
| N-01 | Scaling / `test_scaling.py` | Extreme values only in test data | StandardScaler means/variances equal training-only fit and remain unchanged | P0 |
| M-01 | Metrics / `test_metrics.py` | `y_true=[100,80,60]`, `y_pred=[90,70,50]` | MAE = 10 and RMSE = 10 | P0 |
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
3. Define RUL units, failure endpoints and prediction cadence.
4. Extract causal wavelet-based features using the same settings for both models.
5. Fit any learned preprocessing on training bearings only.
6. Construct aligned past-only sequences and targets.
7. Train LSTM and GRU with documented configurations and seeds.
8. Select hyperparameters using validation bearings only.
9. Evaluate both models on the same untouched test bearings.
10. Report MAE, RMSE, units, per-bearing results, aggregation, configurations and limitations.
11. Optionally replay historical maintenance decisions as a **simulation**, not verified industrial cost savings.

### Required result record (once experiments exist)

| Field | Description |
|---|---|
| Model | LSTM or GRU |
| Dataset and bearings | Dataset version and exact train/val/test IDs |
| RUL unit | Steps or time, with definition |
| Features | Wavelet type/level/statistics and channel selection |
| Sequence length | Number of past observations used |
| Training settings | Seed, epochs, batch size, optimizer, learning rate |
| MAE / RMSE | Held-out test values with units |
| Per-bearing scores | Metrics for each held-out bearing |
| Runtime / hardware | Environment for reproducibility |
| Notes | Failures, limitations, and deviations from plan |

No performance targets are asserted before a baseline experiment. Acceptance for the scientific comparison requires a leakage-free, reproducible protocol and correctly computed metrics, **not** a predetermined accuracy claim.

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
