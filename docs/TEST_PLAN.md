# Test Plan — Rolling Bearing RUL Prediction (LSTM vs. GRU)

**Project:** Predictive Maintenance of Rolling Bearings  
**Course:** Case Study: Intelligent Systems in Production  
**Dataset:** XJTU-SY  
**Models:** LSTM and GRU  
**Document:** Verification, validation, regression and release test plan  
**Version:** 3.0 — 29 October 2026  
**Status:** Living specification; baseline implementation completed, additional verification pending  
**Repository target:** `docs/TEST_PLAN.md`

> **Evidence convention.** **Observed** means an execution result previously reported by the team, not independently rerun while preparing this document. **Planned / NOT VERIFIED** means a requirement or test whose passing result has not been supplied. A test is **PASS** only when a reproducible run log demonstrates it. The test plan is not a substitute for a test execution report.

## 1. Purpose, scope and claims

The system estimates **Remaining Useful Life (RUL), in minutes**, for rolling bearings from historical vibration recordings. The implementation extracts past-only wavelet features, builds temporal sequences, trains LSTM and GRU regressors, evaluates on held-out complete bearing histories, and replays illustrative maintenance decisions. A local Streamlit dashboard visualizes stored results.

**In scope:** dataset ingestion and validation; chronology; target construction; wavelet extraction; leakage prevention; split integrity; normalization; sequence alignment; LSTM/GRU training and inference; metrics; comparison fairness; historical replay; dashboard; reproducibility; release readiness; negative/error-path tests.

**Out of scope:** live industrial sensing; calibrated defect probabilities; actual factory maintenance interventions; independently verified savings; production safety certification; guaranteed cross-machine generalization. The dashboard is a local **historical-results viewer**, not a live predictive monitoring service.

## 2. Baseline system contract

| Component | Baseline specification | Evidence / verification requirement |
|---|---|---|
| Dataset | XJTU-SY, 15 complete bearing runs, 3 operating conditions | Dataset provenance + validation log |
| Recordings | 9,216 CSV snapshots; each 32,768 vibration measurements per channel; horizontal and vertical channels | Structural validation observed; verify counts on rerun |
| Cadence | One 1.28-second recording each nominal minute, 25.6 kHz | Dataset protocol; not an independent timestamp audit |
| Endpoint | Final recorded snapshot of the experimental run-to-failure history | End-of-recording proxy; not necessarily mechanical disintegration |
| Target | `rul_minutes(i) = (N - 1 - i) × 1 minute` for zero-based snapshot index `i` | Offline label generation; endpoint must never enter model inputs |
| Features | `db4` DWT, level 3, `symmetric`; mean squared coefficient energies of A3/D3/D2/D1 in both channels | Exactly 8 finite features per valid snapshot |
| Feature table | 9,216 rows × 8 wavelet features, plus identifiers/targets as applicable | Check unique `(bearing_id, step)` |
| Split | 9 train / 3 validation / 3 test **bearings**, 3/1/1 per operating condition | Fixed `configs/bearing_split_v1.yaml` |
| Bearing IDs | Train: `Bearing1_1..1_3`, `Bearing2_1..2_3`, `Bearing3_1..3_3`; validation: `Bearing1_4`, `Bearing2_4`, `Bearing3_4`; test: `Bearing1_5`, `Bearing2_5`, `Bearing3_5` | No shared bearing IDs across partitions |
| Snapshots by split | 7,032 / 1,679 / 505 (train/validation/test) | Reported baseline counts |
| Scaling | `StandardScaler` fit **only on training** bearing features; RUL scaling fitted on training labels only | Persist/reuse fitted transformations |
| Sequences | 20 past-and-current snapshots, stride 1, `(20, 8)` input | 6,861 / 1,622 / 448 windows reported |
| Models | LSTM and GRU; hidden size 64, 1 layer, scalar RUL output | 19,009 / 14,273 parameters reported |
| Training | Adam, LR 0.001, MSE, batch 64, seed 42, early stopping patience 7 | Record full configuration, best checkpoint and history |
| Evaluation | MAE/RMSE **in minutes**, pooled over test windows and per-bearing | Same targets, partitions and evaluation code |
| Replay | Threshold 120 min, 3 consecutive confirmations, minimum useful lead 30 min | Historical offline decision simulation only |
| Dashboard | Streamlit + Plotly; reads persisted `outputs/main/` artifacts | Must label source/run and avoid fabricated data |

**Causal prediction rule:** At snapshot `i`, the model may use wavelet features computed from snapshots `i-19` through `i` only. A valid sequence first exists at `i=19`. Wavelet decomposition may use all samples **inside the current completed snapshot**, but never a later snapshot. Ground-truth RUL is derived using the known final snapshot **for supervised training/evaluation only**, never supplied as a model feature.

**Scientific caveat:** The split is project-defined, not an official benchmark. The pooled score is **window-weighted**; unequal bearing lengths may dominate. Hyperparameter selection must use training/validation only. If the existing test results have already influenced tuning choices, disclose that limitation and use a new untouched evaluation protocol for strong post-optimization claims.

## 3. Test strategy, levels and priorities

| Level | Purpose | Typical evidence |
|---|---|---|
| Unit | Validate pure functions and failure handling with tiny deterministic fixtures | `pytest` output |
| Contract | Check schema, shapes, units, model signatures, artifact names | Assertions + fixture files |
| Integration | Validate preprocessing → sequences → inference → evaluation | Small end-to-end test |
| Scientific | Prevent leakage and misleading evaluation | Split/causality/weighting checks |
| Regression | Preserve previously fixed behavior after changes | CI/local test history |
| Dashboard | Verify correct artifact loading, filters, empty/error states and honest labels | Streamlit smoke test + screenshots |
| Acceptance | Verify baseline run and evidence package can be reproduced | Run manifest, logs, commit SHA |

**P0 — blocking:** wrong targets, leakage, corrupt data silently accepted, incorrect metrics, non-causal replay, misleading fabricated dashboard outputs.  
**P1 — required for final submission:** robustness, error handling, reproducibility, dashboard acceptance, meaningful per-bearing reporting.  
**P2 — enhancements:** accessibility, performance profiling, deployment, optional model optimization.

## 4. Traceable test cases

The IDs below are stable. Record each outcome as `PASS`, `FAIL`, `NOT RUN`, `BLOCKED` or `N/A` with evidence and date. **Do not infer passing status from implemented code.**

### 4.1 Data ingestion and dataset integrity

| ID | Priority | Test / input | Expected acceptance criterion |
|---|---|---|---|
| D-01 | P0 | Valid two-channel synthetic CSV | Numeric channels and exact row count preserved |
| D-02 | P0 | Missing, empty, malformed or wrong-column CSV | Informative error; no silent fallback |
| D-03 | P0 | Filenames `1.csv`, `2.csv`, `10.csv` | Numeric chronology, not lexical sorting |
| D-04 | P0 | Duplicate/missing steps, duplicate bearing IDs | Detected or explicitly documented; no silent reordering |
| D-05 | P0 | NaN, Inf, nonnumeric and unexpected column count | Reject invalid data; report file and reason |
| D-06 | P1 | Full local dataset scan | Exactly 15 bearing directories and 9,216 valid snapshots; store log |
| D-07 | P1 | Run metadata/provenance | Dataset version, source and operating conditions documented |

### 4.2 Targets, features and chronology

| ID | Priority | Test / input | Expected acceptance criterion |
|---|---|---|---|
| T-01 | P0 | Five snapshots | RUL `[4,3,2,1,0]` minutes |
| T-02 | P0 | Two bearings with different lengths | Labels calculated independently; last snapshot RUL = 0 for each |
| T-03 | P0 | Missing/duplicate/nonconsecutive steps | Explicit error or documented strict handling; never fabricated time labels |
| T-04 | P0 | Input feature matrix inspection | Future endpoint, lifetime `N`, RUL label and test partition flags absent from model inputs |
| F-01 | P0 | Known signal with independent reference calculation | DWT `db4`, level 3, symmetric; 4 finite energies per channel |
| F-02 | P0 | Two channels | Exactly 8 features in documented A3/D3/D2/D1 order |
| F-03 | P0 | Modify future snapshots after time `t` | Features at `t` unchanged |
| F-04 | P1 | Short/empty/nonnumeric signal | Explicit validation error, no silent wavelet-level downgrade |
| F-05 | P1 | Repeat same signal | Identical deterministic feature values within numerical tolerance |
| F-06 | P1 | Extreme but finite vibration amplitudes | No silent overflow or invalid energies; explicit failure if nonfinite |

### 4.3 Split, scaling and sequences

| ID | Priority | Test / input | Expected acceptance criterion |
|---|---|---|---|
| S-01 | P0 | Load split YAML | Train/validation/test bearing sets pairwise disjoint |
| S-02 | P0 | Compare split with dataset manifest | All 15 bearings assigned exactly once, counts 9/3/3 |
| S-03 | P0 | Count per operating condition | 3/1/1 bearings in train/validation/test per condition |
| S-04 | P1 | Repeat fixed configuration | Same split and documented ordering |
| N-01 | P0 | Inject extreme values into validation/test | Fitted scaler statistics unchanged from training-only fit |
| N-02 | P0 | RUL target scaling | Fit on train labels only; inverse transform returns minutes |
| N-03 | P1 | Missing fitted scaler or wrong feature order | Clear error; no refit on test data |
| Q-01 | P0 | `N=25`, length 20, stride 1 | Exactly 6 windows, each `(20,8)` |
| Q-02 | P0 | Sequence endpoint | Target equals RUL at the final observed step of that window |
| Q-03 | P0 | Change observations after time `t` | Earlier windows and predictions' inputs unaffected |
| Q-04 | P0 | Multiple bearing histories | No window crosses a bearing boundary |
| Q-05 | P1 | `N<20`, `N=20` | 0 and 1 windows respectively; explicit documented behavior |
| Q-06 | P1 | Full baseline matrix | Expected 6,861 / 1,622 / 448 windows, each `(20,8)` |

### 4.4 Models, training and reproducibility

| ID | Priority | Test / input | Expected acceptance criterion |
|---|---|---|---|
| A-01 | P0 | LSTM/GRU forward pass with `(batch,20,8)` | One finite scalar output per sequence |
| A-02 | P1 | One optimizer step | Finite loss, nonzero gradients and intended parameter update |
| A-03 | P0 | Same synthetic batch through both models | Same input/output contract and RUL units |
| A-04 | P0 | Checkpoint selection | Saved best validation checkpoint, not automatically final epoch |
| A-05 | P1 | Early stopping with controlled loss sequence | Stops according to documented patience and improvement rule |
| A-06 | P1 | Invalid model input dimensions or missing checkpoint | Informative failure, no fabricated predictions |
| A-07 | P1 | Fixed-seed repeat in pinned environment | Splits and deterministic outputs reproducible within stated tolerance |
| A-08 | P1 | Baseline/optimization experiment naming | Separate run directories; baseline outputs never overwritten |
| A-09 | P1 | Model config audit | Model sizes, hyperparameters and training history logged |

### 4.5 Evaluation and scientific comparison

| ID | Priority | Test / input | Expected acceptance criterion |
|---|---|---|---|
| M-01 | P0 | `y=[100,80,60]`, `ŷ=[90,70,50]` | MAE = RMSE = 10 minutes |
| M-02 | P0 | Perfect predictions | MAE = RMSE = 0 |
| M-03 | P0 | Empty/mismatched/NaN/Inf arrays | Explicit error |
| M-04 | P0 | Test predictions vs target CSV | Same bearing/step keys, no duplicate or missing matched rows |
| M-05 | P0 | Compare LSTM and GRU | Identical 448 held-out sequence targets and evaluation units |
| M-06 | P0 | Pooled metrics | Window-weighted aggregate computed from **all** matching predictions |
| M-07 | P1 | Per-bearing metrics | Report 3 test bearings separately with window counts |
| M-08 | P1 | Run comparison | State model selection procedure and any test-informed tuning limitation |
| M-09 | P1 | Predicted RUL postprocessing | Negative/clipped values handled by an explicit, consistent policy |
| M-10 | P1 | Artifact reproducibility | CSV/JSON summaries agree with independently recalculated metrics |

### 4.6 Historical maintenance replay

| ID | Priority | Test / input | Expected acceptance criterion |
|---|---|---|---|
| H-01 | P0 | Future prediction changed after time `t` | Decision at `t` unchanged |
| H-02 | P0 | Three consecutive predictions below 120 min | Alert triggers only on third qualifying observation |
| H-03 | P0 | One/two isolated low predictions | No premature alert |
| H-04 | P0 | Alert lead < 30 min | Classified `late` under baseline policy |
| H-05 | P0 | Alert with sufficient lead but earlier than configured timely band | Classified `early` consistently with implementation contract |
| H-06 | P0 | Alert in configured timely band | Classified `timely` |
| H-07 | P0 | No qualifying alert before endpoint | Classified `missed` |
| H-08 | P1 | Empty/unsorted/duplicate steps or invalid predictions | Explicit rejection or documented handling |
| H-09 | P1 | Compare replay summary with detailed events | Counts and mean lead times reconcile |
| H-10 | P1 | Labeling and limitations | Results always called **offline historical simulation**, not verified maintenance savings |
| H-11 | P1 | Change replay threshold/confirmation count | New run tagged by configuration; old baseline preserved |

**Replay evidence already reported:** `python -m pytest tests/test_maintenance_replay.py -q` returned **14 passed**; `python scripts/run_maintenance_replay.py --run-name main` completed. This is evidence for that executed test module as a whole, **not proof that every new H-01…H-11 case above exists or passes**.

### 4.7 Streamlit dashboard and visualization

| ID | Priority | Test / input | Expected acceptance criterion |
|---|---|---|---|
| UI-01 | P0 | Start `python -m streamlit run app.py` | App starts without traceback in supported environment |
| UI-02 | P0 | Load `outputs/main/evaluation/comparison.csv` | Model comparison matches saved MAE/RMSE; units minutes |
| UI-03 | P0 | Select each test bearing/model | Actual/predicted RUL traces use correct bearing, order and time steps |
| UI-04 | P0 | Replay tab | Early/timely/late/missed and lead-time values match replay artifacts |
| UI-05 | P0 | No result files | Explicit helpful message, **no synthetic fallback presented as measured results** |
| UI-06 | P0 | Change selected model/bearing | Other model/bearing data never leak into current view |
| UI-07 | P1 | Training tab | Training/validation curves and best epoch match saved histories |
| UI-08 | P1 | Run app from another working directory | Paths resolved relative to repository/app location, or clear documented requirement |
| UI-09 | P1 | Malformed CSV/missing columns/NaN values | Clear validation error, no crash or misleading plot |
| UI-10 | P1 | Chart legends and axes | Clear units, model names, source run and test-only labeling |
| UI-11 | P1 | Local interactive smoke test | Sidebar navigation, controls and plots work in browser |
| UI-12 | P1 | Reproducible deployment bundle | Only permitted, appropriately sized non-sensitive derived artifacts included |
| UI-13 | P2 | Accessibility/responsive layout | Readable on laptop; distinguish lines without color alone |
| UI-14 | P2 | Optional online deployment | Public link opens without access to local filesystem or raw dataset |

**Dashboard integrity rule:** A synthetic S-curve defect probability is **not** an output of the RUL regression models. It must not be presented as actual LSTM/GRU predictions. If a synthetic demo is retained, label it prominently as simulated and separate from the project results.

### 4.8 Visualization, integration and operational checks

| ID | Priority | Test / input | Expected acceptance criterion |
|---|---|---|---|
| V-01 | P1 | Regenerate baseline figures | Expected figure files exist and are readable |
| V-02 | P1 | Compare chart points to source CSVs | Chart values equal source values; no fabricated interpolation |
| V-03 | P1 | Plot training histories | Best validation epoch and overfitting patterns correctly described |
| I-01 | P1 | Tiny synthetic pipeline | Load → wavelet → target → split → scale → sequence → model → metrics |
| I-02 | P1 | Real subset smoke test | Valid finite features and aligned predictions with recorded IDs |
| R-01 | P1 | Clean environment install | `pip install -r requirements.txt` succeeds on documented supported Python |
| R-02 | P1 | Full test suite | Exit code 0; record passed/failed/skipped and environment |
| R-03 | P1 | Fresh clone without raw data | Unit tests and dashboard error states work; no undocumented data requirement |
| R-04 | P1 | Repository hygiene | No raw XJTU-SY, secrets, oversized checkpoints or accidental generated data committed |
| R-05 | P1 | Experiment provenance | Commit SHA, seed, configs, versions, source files and output paths recorded |
| R-06 | P1 | CI | CI runs portable tests; full dataset/training tests explicitly separate |

## 5. Test fixtures and methodology

1. **Unit fixtures:** synthetic finite two-channel arrays; tiny bearing histories of lengths 0, 1, 5, 19, 20 and 25; intentionally malformed CSVs; known metric vectors. Keep small and commit-safe.
2. **Integration fixtures:** small explicitly identified XJTU-SY subsets if redistribution is permitted; otherwise local-only fixture paths and documented reproduction commands.
3. **Evaluation fixtures:** three complete held-out test bearings (`Bearing1_5`, `Bearing2_5`, `Bearing3_5`). Never randomly mix windows from a bearing across splits.
4. **Dashboard fixtures:** minimal valid comparison, predictions, histories and replay CSV/JSON plus missing/corrupt variants. Test dashboard behavior without full model training.
5. **Negative controls:** modify future data; inject extreme test-only values; reorder snapshots; duplicate bearing IDs; delete artifacts; swap model labels. Each should either leave earlier predictions invariant or produce an explicit failure.
6. **Numerical tolerance:** use exact equality for IDs/counts and deterministic labels; use explicit `pytest.approx`/`numpy.testing.assert_allclose` tolerances for floating-point transforms. Record device-dependent tolerance for PyTorch repeats.

## 6. Recorded baseline evidence (not a new execution)

The following values were reported from the completed local baseline workflow. They should be re-verified from saved artifacts and a timestamped execution report before final submission.

### 6.1 Training and held-out evaluation

| Model | Best epoch | Best validation MSE (scaled) | Test MAE (min) | Test RMSE (min) |
|---|---:|---:|---:|---:|
| LSTM | 1 | 0.082069 | 541.104 | 741.903 |
| GRU | 5 | 0.030729 | 148.382 | 331.900 |

**Test set:** 448 overlapping windows across 3 unseen bearing runs. GRU has lower pooled MAE/RMSE in this run, but this does **not** establish universal model superiority. LSTM validation loss rises after its first epoch; GRU validation is lowest at epoch 5 and later worsens. This is consistent with overfitting and/or bearing distribution shift; investigate using validation data rather than selecting settings by repeatedly checking the test set.

| Test bearing | Windows | LSTM MAE (min) | GRU MAE (min) |
|---|---:|---:|---:|
| Bearing1_5 | 33 | 814.867 | 928.545 |
| Bearing2_5 | 320 | 619.461 | 77.789 |
| Bearing3_5 | 95 | 182.069 | 115.166 |

**Interpretation:** `Bearing2_5` contributes 320/448 windows (71.4%) and therefore heavily influences the pooled metrics. GRU is worse than LSTM on `Bearing1_5`. Report both pooled and per-bearing results; consider equal-bearing aggregation as an additional, clearly labeled sensitivity analysis.

### 6.2 Historical maintenance replay

**Policy:** RUL alert threshold 120 minutes; 3 consecutive confirmations; minimum useful lead 30 minutes.

| Model | Bearing1_5 | Bearing2_5 | Bearing3_5 | Timely rate | Mean alert lead |
|---|---|---|---|---:|---:|
| LSTM | Late (7 min) | Early (144 min) | Timely (88 min) | 1/3 = 33.33% | 79.67 min |
| GRU | Late (8 min) | Early (171 min) | Timely (84 min) | 1/3 = 33.33% | 87.67 min |

**Limit:** This is a counterfactual replay on historical predictions, not a live deployment, an economic impact estimate, or a validated maintenance policy. An apparent lead time is relative to the experimental final snapshot.

### 6.3 Evidence status ledger

| Evidence | Current status | Next verification |
|---|---|---|
| Dataset structural scan (9,216 files) | **Observed earlier**: zero errors | Save scan output with date + commit |
| Loader synthetic tests | **Observed earlier**: 9 passed | Run full suite and archive log |
| Replay test module | **Observed earlier**: 14 passed | Save command, date, SHA, full log |
| Baseline training/evaluation | **Observed earlier**: metrics above | Recompute from source prediction CSVs |
| Baseline replay execution | **Observed earlier**: completed | Reconcile detailed and summary artifacts |
| Eleven static charts | **Reported generated**, not all manually reviewed | Verify existence, labels, source consistency |
| Streamlit dashboard | **Code prepared**, not independently smoke-tested here | UI-01…UI-11 local test |
| Entire pytest suite | **NOT VERIFIED for current commit** | `python -m pytest tests/ -v` |
| Fresh clone / clean install | **NOT VERIFIED** | R-01 and R-03 |
| Online deployment | **NOT DONE** | Optional UI-14 |

## 7. Execution procedure and evidence capture

**Environment:** Windows PowerShell in the project root; use the project Python environment. Replace any paths only if the repository actually differs. Do not run full training during every smoke test.

```powershell
# 1. Identify exact source revision and Python version
git rev-parse HEAD
python --version
python -m pip freeze > test-environment-freeze.txt

# 2. Run all portable automated tests
$env:PYTHONPATH = "src"
python -m pytest tests/ -v 2>&1 | Tee-Object -FilePath test-results-pytest.txt

# 3. Recheck replay tests specifically
python -m pytest tests/test_maintenance_replay.py -q

# 4. Reproduce historical replay using existing prediction artifacts
python scripts/run_maintenance_replay.py --run-name main

# 5. Smoke-test local dashboard (then inspect in browser)
python -m streamlit run app.py
```

**Note:** `test-environment-freeze.txt` and `test-results-pytest.txt` are local evidence files; inspect before committing. If `scripts/validate_dataset.py` or any other path is different in the current repository, use its actual CLI rather than assuming the command. The `--run-name main` replay may replace existing derived outputs: back up or use a distinct run name when comparing experiments.

**Required execution record:**

```text
Execution ID:
Date/time (timezone):
Operator:
Git commit SHA / branch:
OS / Python / PyTorch / Streamlit versions:
Dataset provenance / local file count:
Configuration and seed:
Command:
Test IDs covered:
PASS / FAIL / SKIP / NOT RUN counts:
Artifact paths / hashes:
Observed deviations and issue links:
Reviewer and approval date:
```

## 8. Quality gates and release acceptance

| Gate | Blocking condition / acceptance |
|---|---|
| G1 — Scientific contract | Target units, causal feature policy and fixed bearing split documented and approved |
| G2 — Core correctness | All applicable P0 loader, label, feature, split, scaling, sequence and metric tests pass |
| G3 — Training/evaluation | Models produce finite outputs; best checkpoint used; same held-out targets and metrics; limitations stated |
| G4 — Replay | Replay P0 tests pass; summary reconciles with events; offline simulation labeled clearly |
| G5 — Dashboard | UI P0 tests pass; displayed values match artifacts; missing files cannot yield fabricated results |
| G6 — Reproducibility | Full automated suite run logged, dependencies documented, baseline artifacts/configs preserved |
| G7 — Final submission | Report contains pooled/per-bearing metrics, replay interpretation, unresolved issues and no unsupported deployment claims |

**Release decision:** `READY` only if all applicable P0 gates pass with evidence, required P1 exceptions are documented and approved, and the exact Git revision is identified. Otherwise `CONDITIONAL` (known nonblocking gaps) or `NOT READY` (P0 failure/unknown).

## 9. Open issues and planned improvement work

These are **proposed GitHub issues**, not assertions that each is a confirmed software bug.

| Issue | Priority | Description / acceptance |
|---|---|---|
| ISS-01 — Diagnose validation divergence | P1 | Plot and log per-bearing validation error, check scaler/label alignment, evaluate train/val distribution shift; explain LSTM best epoch 1 and GRU best epoch 5 without claiming a single cause |
| ISS-02 — Controlled hyperparameter experiments | P2 | Tune on train/validation only; one change per run; preserve `main` baseline; log configs/seeds and avoid test-based model selection |
| ISS-03 — Per-bearing and equal-bearing evaluation | P1 | Publish per-bearing MAE/RMSE, sample counts and optional equal-bearing mean alongside window-weighted pooled score |
| ISS-04 — Dashboard integration and smoke tests | P1 | Validate actual `outputs/main/` schemas; add UI-01…UI-11 tests and browser screenshots |
| ISS-05 — Missing/corrupt artifact handling | P0 | Friendly errors, no fabricated values, clear run/source labels |
| ISS-06 — CI and reproducibility | P1 | Run portable tests on clean install; record pinned compatible environment; exclude full dataset/training from routine CI |
| ISS-07 — Baseline evidence manifest | P1 | Archive run config, SHA, metrics, replay CSV/JSON and test log; checksum derived artifacts where useful |
| ISS-08 — Deployment packaging | P2 | Optional online deployment with permitted small derived data only; ensure paths and dependency compatibility |
| ISS-09 — Replay sensitivity analysis | P2 | Compare threshold and consecutive-confirmation settings on validation/historical examples; preserve baseline policy and do not claim industrial optimization |

## 10. Change control, ownership and traceability

- **Owner:** Team 3. Each change should name an implementer and reviewer in the issue/PR.
- Each test should retain its ID in the test function name, parameter ID, docstring or test mapping.
- Changes to the RUL definition, DWT configuration, split IDs, feature order, scaler policy, sequence length, replay policy or metric aggregation require corresponding updates to tests, configs and documentation in the **same PR**.
- Keep raw dataset and large generated outputs out of Git unless license and repository policy explicitly permit them. For a dashboard release, commit only an approved minimal derived-results bundle or use documented artifact storage.
- Preserve baseline outputs; optimization must use a distinct run name and record differences.
- Add a regression test for every fixed P0/P1 defect. Never weaken assertions merely to turn failures green.
- Maintain a separate `docs/TEST_EXECUTION_REPORT.md` (or equivalent) containing actual dated executions; do not overwrite this plan with transient status notes.
- Before final presentation, reconcile the project proposal, README, code and slides so they consistently describe **LSTM vs GRU**, **RUL regression in minutes**, and **offline historical replay**.

## 11. Final reviewer checklist

- [ ] Exact commit SHA and Python environment recorded.
- [ ] Data integrity scan log archived; dataset provenance documented.
- [ ] All P0 tests executed and passing, or release blocked.
- [ ] Leakage checks cover split, normalization, window boundaries and future features.
- [ ] Evaluation metrics independently recomputed from saved predictions.
- [ ] Pooled and per-bearing results include window counts and units.
- [ ] Replay policy and classifications checked against detailed event CSVs.
- [ ] Dashboard started locally and every page inspected with actual artifacts.
- [ ] Dashboard fails safely when outputs are missing or malformed.
- [ ] Baseline run preserved separately from optimization experiments.
- [ ] Limitations and outstanding issues included in final report.
- [ ] No live-industrial, defect-probability, maintenance-savings or generalization claim unsupported by evidence.

---

**Revision note (v3.0):** Consolidates the duplicated v2.2 source, updates outdated “models planned / results planned” language, adds verified-vs-planned evidence discipline, documents reported baseline results and replay, extends tests to dashboard, integration, release and error paths, and introduces traceable proposed GitHub issues. No new tests were executed while authoring this revision.
