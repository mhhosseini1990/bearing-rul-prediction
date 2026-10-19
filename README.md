# Predictive Maintenance of Rolling Bearings

**Random Forest vs. LSTM for Remaining Useful Life (RUL) Prediction**

**Course:** MDM2 – Case Study: Intelligent Systems in Production  
**Team:** 3  
**Status:** In progress — research pipeline and prototype design

## Project Overview

Unexpected bearing failures can interrupt industrial production and increase maintenance costs. This project investigates prediction of the **remaining useful life (RUL)** of rolling bearings using historical run-to-failure vibration measurements.

We plan to compare a **Random Forest regressor** with a **Long Short-Term Memory (LSTM)** sequence model under a consistent evaluation protocol. The main contribution is a reproducible comparison on previously unseen bearings, not a new prediction algorithm or a production-ready monitoring system.

A **Streamlit user-interface prototype** has also been designed to illustrate time-series monitoring and threshold-based maintenance alerts. The displayed defect probability is **not** the same as RUL in minutes, and the prototype must not be interpreted as validated model performance.

## Research Question

> How accurately can Random Forest regression and an LSTM sequence model predict the remaining useful life of rolling bearings from vibration measurements on previously unseen bearings in the XJTU-SY dataset?

## Datasets

### Primary — XJTU-SY

- **Dataset:** XJTU-SY Bearing Dataset
- **Scope:** 15 run-to-failure bearing histories under three operating conditions
- **Measurements:** Laboratory vibration signals
- **Source:** [XJTU-SY Bearing Datasets](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)

We will verify file order, sampling intervals, and recorded run endpoints before deriving RUL targets. These laboratory results do not establish industrial deployment readiness.

### Optional — PRONOSTIA / PHM 2012

The PRONOSTIA / PHM 2012 bearing dataset may be used for **additional external validation if time permits**. It is not required for the baseline or core model comparison. Differences in operating conditions, recording conventions, and label availability must be assessed before any cross-dataset claims.

## Planned Methodology

1. **Data verification and labeling:** Check bearing histories, observation order, time intervals, and run endpoints; derive RUL labels in minutes where supported by timestamps.
2. **Feature engineering:** Extract RMS, standard deviation, kurtosis, and peak-to-peak amplitude from vibration measurements. Frequency-domain features are optional.
3. **Leakage-safe splitting:** Keep each complete bearing history within one of train, validation, or test. Fit learned preprocessing on training data only.
4. **Random Forest baseline:** Train `RandomForestRegressor` on vibration feature vectors using scikit-learn.
5. **LSTM sequence model:** Train a causal PyTorch LSTM on ordered feature windows using past and current observations only.
6. **Fair evaluation:** Compare both models using the same RUL definition, held-out test bearings, and metrics; report results by bearing and operating condition.
7. **Historical replay:** Demonstrate predictions over archived observations without claiming live sensor connectivity.

## Evaluation

- **MAE (Mean Absolute Error):** Average absolute difference between predicted and true RUL, in minutes.
- **RMSE (Root Mean Squared Error):** Square-error-based metric that gives greater weight to large errors, in minutes.
- **Breakdowns:** Per-bearing and per-operating-condition errors.

**Lower MAE and RMSE indicate better predictive accuracy.** We do not assume that LSTM will outperform Random Forest. No experimental scores or model winner are claimed until the experiments have been run.

## Implementation Plan and Deliverables

### Phase 1 — Baseline MVP

- Verified XJTU-SY histories and RUL labels
- Reproducible feature-extraction workflow
- Trained Random Forest baseline
- Saved held-out predictions, MAE/RMSE tables, and plots
- Simple historical RUL prediction replay

### Phase 2 — Core Research Comparison

- Causal LSTM implementation
- Matching evaluation protocol for Random Forest and LSTM
- Per-bearing and per-condition comparative analysis
- Reproducible experiment documentation and final report

### Optional Extensions

- Further development/integration of the Streamlit dashboard
- PRONOSTIA / PHM 2012 evaluation
- Additional frequency-domain features

## Streamlit Prototype — UI Design

The current dashboard design illustrates **threshold-based alert behavior** over a sequence of time steps. Two example configurations show the same displayed defect score (**0.98**) with different thresholds:

| Scenario | Alert threshold | Displayed defect score | Interpretation |
| --- | ---: | ---: | --- |
| A | 0.49 | 0.98 | Alert triggered: maintenance needed |
| B | 0.99 | 0.98 | No alert triggered: score below threshold |

**Important:** “No alert triggered” does not mean the bearing is healthy. The prototype's **defect probability over time** is distinct from the research target, **remaining useful life in minutes**. The screenshots demonstrate the planned interface behavior, not verified XJTU-SY predictions or measured model accuracy. The UI is not a live factory deployment.

## Timeline

| Date | Planned milestone |
| --- | --- |
| 20 October 2026 | Proposal, scope, tech stack, and MVP definition |
| 3 November 2026 | Dataset checks, literature map, features, and Random Forest baseline |
| 24 November – 8 December 2026 | LSTM development and comparative experiments |
| 22 December 2026 | Demo, testing, and reproducibility |
| 12 January 2027 | Final report and presentation |

## Tech Stack

| Purpose | Technology |
| --- | --- |
| Programming | Python |
| Data analysis | NumPy, pandas |
| Classical machine learning | scikit-learn |
| Deep learning | PyTorch |
| UI prototype | Streamlit |
| Plots | Matplotlib / Plotly (as needed) |

## Repository Status

This repository documents an **in-progress** course project. The Streamlit interface design is available as a prototype concept; this README does not claim that the full RUL pipeline, trained models, or evaluated predictions are already complete. Training scripts, setup commands, data-loading instructions, and results should be documented here **when the corresponding files are committed and verified**.

The original datasets should be obtained from their respective sources; avoid committing large raw datasets without checking their licenses and repository size limits.

## Limitations and Responsible Research

- XJTU-SY contains only **15 independent bearing histories**; performance on unseen bearings may vary substantially.
- Randomly splitting windows from the same bearing across train/test can cause data leakage.
- LSTM input windows must not include future observations.
- Lab-based results cannot be assumed to generalize to factory machines.
- Dataset and paper attribution, transparent methods, and honest reporting are required; no personal data is involved.

## Team

- Akram Serrar
- Seyed Hosseini Rivandi
- Yeganeh Seyyed Abady

## Selected References

1. Dhungana et al. (2026). *The International Journal of Advanced Manufacturing Technology*. [DOI: 10.1007/s00170-025-17139-6](https://doi.org/10.1007/s00170-025-17139-6)
2. Dai et al. (2026). *Frontiers in Computer Science*. [DOI: 10.3389/fcomp.2026.1833609](https://doi.org/10.3389/fcomp.2026.1833609)
3. Ahmad et al. (2019). *Reliability Engineering & System Safety*. [DOI: 10.1016/j.ress.2018.02.003](https://doi.org/10.1016/j.ress.2018.02.003)

Additional supporting papers are documented in the project literature review.

---

*Course project — Case Study: Intelligent Systems in Production (Team 3).*
