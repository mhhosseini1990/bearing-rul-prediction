# Predictive Maintenance of Rolling Bearings

## Wavelet-Based Remaining Useful Life Prediction Using LSTM and GRU

**Course:** MDM2 – Case Study: Intelligent Systems in Production  
**University:** Technische Hochschule Deggendorf  
**Project Track:** Software & Simulation  
**Team:** 3  
**Status:** In Progress – Research and Prototype Development

---

## 1. Project Overview

Unexpected rolling bearing failures can cause production downtime, equipment damage, and increased maintenance costs.

This project investigates the prediction of **Remaining Useful Life (RUL)** of rolling bearings using historical vibration measurements and machine learning.

The main objective is to compare two recurrent neural network architectures:

- **Long Short-Term Memory (LSTM)**
- **Gated Recurrent Unit (GRU)**

Both models will use equivalent wavelet-based vibration feature sequences to predict RUL directly.

The project focuses on a reproducible comparison using previously unseen bearings from the **XJTU-SY dataset**.

A simple Random Forest regression model may also be implemented as an initial baseline to verify the data-processing and prediction pipeline.

In addition, a historical maintenance simulation will investigate how predicted RUL could support replacement decisions.

The project is based on historical laboratory measurements and does not involve real-time industrial deployment.

---

## 2. Research Question

> How accurately can standard LSTM and GRU models predict the Remaining Useful Life of rolling bearings using vibration-based wavelet features from the XJTU-SY dataset?

### Research Objectives

1. Extract meaningful statistical and wavelet-based features from vibration signals.
2. Develop standard LSTM and GRU regression models for direct RUL prediction.
3. Compare both models under equivalent experimental conditions.
4. Evaluate prediction accuracy using MAE and RMSE.
5. Investigate model generalization on previously unseen bearings.
6. Explore the potential use of RUL predictions in a hypothetical maintenance decision simulation.

---

## 3. Dataset

### Primary Dataset: XJTU-SY

The project uses the publicly available **XJTU-SY Rolling Bearing Dataset**.

**Dataset source:**  
https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets

| Property | Description |
|---|---|
| Dataset | XJTU-SY |
| Equipment | Rolling bearings |
| Bearing histories | 15 |
| Operating conditions | 3 |
| Bearings per condition | 5 |
| Data type | Vibration measurements |
| Experiment type | Run-to-failure |
| Prediction target | Remaining Useful Life (RUL) |

Each bearing contains a sequence of vibration measurements recorded during its experimental lifetime.

The final recorded observation is treated as an experimental end-of-life proxy.

### RUL Definition

For bearing `i`, RUL at observation time `t` is defined as:

```text
RUL_i(t) = T_end,i - t
```

Where:

- `RUL_i(t)` = Remaining Useful Life at time `t`
- `T_end,i` = Final recorded observation time of bearing `i`
- `t` = Current observation time

RUL will be expressed in minutes after verifying the dataset acquisition intervals.

The observed end time is used to generate retrospective labels and evaluation scores. It is never provided as a model input.

### Optional Dataset

**PRONOSTIA / FEMTO-ST (PHM 2012)** may be considered for additional external validation if time permits.

It is not required for the main experiment.

---

## 4. Proposed Methodology

The project follows a software-based experimental workflow.

```text
XJTU-SY Vibration Dataset
          |
          v
Data Loading and Verification
          |
          v
Statistical + Wavelet Feature Extraction
          |
          v
RUL Label Generation
          |
          v
Bearing-Disjoint Data Splitting
          |
          v
Sequence Construction
          |
          +-------------------+
          |                   |
          v                   v
     LSTM Model           GRU Model
          |                   |
          v                   v
     RUL Prediction      RUL Prediction
          |                   |
          +---------+---------+
                    |
                    v
             MAE / RMSE
                    |
                    v
          Model Comparison
                    |
                    v
      Historical Maintenance Replay
```

### 4.1 Data Preprocessing

The preprocessing stage will include:

- Loading vibration measurements
- Verifying chronological order
- Checking missing or invalid records
- Confirming acquisition intervals
- Generating RUL labels
- Preparing numerical model inputs

Any preprocessing parameters learned from data will be fitted using training bearings only.

### 4.2 Feature Extraction

The initial feature representation will combine statistical and wavelet-based descriptors.

**Statistical features may include:**

- Root Mean Square (RMS)
- Standard deviation
- Kurtosis
- Skewness
- Peak-to-peak amplitude

**Wavelet-based features:**

- Daubechies 4 (db4) wavelet
- Wavelet-packet decomposition
- Wavelet RMS or energy descriptors
- Selected wavelet-coefficient statistics

The final decomposition level and feature dimensions will be determined using training and validation data.

Both LSTM and GRU will receive the same selected features.

### 4.3 Sequence Construction

A sliding-window approach will be used to construct time-series inputs.

Each input window will contain observations at or before the current prediction time.

Future observations will not be included.

Window length and stride will be selected during model development.

---

## 5. Model Development

### 5.1 Random Forest Baseline

A Random Forest regressor is proposed as an initial baseline for testing the end-to-end pipeline.

Its purpose is to verify:

- Feature extraction
- RUL label generation
- Model training
- Prediction generation
- MAE and RMSE calculation

Random Forest is not one of the two main neural models in the final research comparison.

### 5.2 LSTM Model

LSTM is a recurrent neural network architecture designed to learn temporal dependencies.

The proposed LSTM model will:

1. Receive sequences of vibration features.
2. Learn temporal patterns associated with bearing degradation.
3. Predict a continuous RUL value through a regression output layer.

### 5.3 GRU Model

GRU is another gated recurrent architecture with a simpler internal gating structure than LSTM.

The proposed GRU model will:

1. Receive the same feature sequences as LSTM.
2. Learn temporal degradation patterns.
3. Predict the same continuous RUL target.

The project uses a standard GRU regression model rather than reproducing the probabilistic GRU-DeepAR architecture described in the reviewed literature.

---

## 6. Test Plan and Evaluation

### 6.1 Dataset Splitting

The dataset will be divided by complete bearing histories.

A proposed initial split is:

| Dataset Partition | Bearings |
|---|---:|
| Training | 9 |
| Validation | 3 |
| Testing | 3 |
| **Total** | **15** |

The proposed allocation is three training bearings, one validation bearing, and one test bearing per operating condition.

The exact bearing identifiers will be fixed and documented before model training.

**Important:** No bearing may appear in more than one partition.

### 6.2 Controlled Model Comparison

LSTM and GRU will be compared using:

- Identical training, validation, and test bearings
- Identical feature definitions
- Identical sequence lengths
- Identical RUL targets
- Identical prediction times
- Equivalent evaluation procedures
- Comparable hyperparameter-tuning strategies

### 6.3 Evaluation Metrics

Two regression metrics will be used.

**Mean Absolute Error (MAE)**

```text
MAE = mean(|Actual RUL - Predicted RUL|)
```

**Root Mean Squared Error (RMSE)**

```text
RMSE = sqrt(mean((Actual RUL - Predicted RUL)^2))
```

Both metrics will be reported in minutes.

Lower MAE and RMSE indicate smaller prediction errors.

### 6.4 Evaluation Results

The final comparison will report:

- MAE per test bearing
- RMSE per test bearing
- Performance by operating condition
- Overall test MAE and RMSE
- Actual versus predicted RUL plots
- Comparative LSTM and GRU results

No experimental results or model winner are claimed before completing the experiments.

---

## 7. Historical Maintenance Simulation

The project will include a retrospective maintenance decision simulation.

The purpose is to investigate how RUL predictions could influence replacement timing.

### Strategy A: Fixed-Interval Replacement

Bearings are replaced according to a predefined operating-time interval.

### Strategy B: RUL-Based Replacement

Replacement is triggered when predicted RUL falls below a predefined lead-time threshold.

```text
If Predicted RUL <= Replacement Threshold:
    Recommend Replacement
Else:
    Continue Operation
```

### Simulation Evaluation

The simulation will compare:

- Replacement timing
- Hypothetical preventive replacement costs
- Hypothetical failure penalties
- Differences between maintenance policies

Policy thresholds and cost assumptions will be specified without using future outcomes from test bearings.

**Important:** The simulation uses historical experimental data and hypothetical replacement decisions. It does not demonstrate validated industrial cost savings.

---

## 8. Streamlit Prototype

A Streamlit dashboard concept has been designed to illustrate time-series monitoring and threshold-based maintenance alerts.

The existing prototype illustrates interface behavior rather than validated RUL prediction performance.

The displayed defect probability is not equivalent to RUL in minutes.

Further integration with the trained RUL models may be considered after the core experiments are completed.

The prototype is not connected to live factory sensors.

---

## 9. Technology Stack

| Purpose | Technology |
|---|---|
| Programming language | Python |
| Data processing | NumPy, pandas |
| Signal processing | SciPy, PyWavelets |
| Baseline machine learning | scikit-learn |
| Deep learning | PyTorch |
| Evaluation | scikit-learn metrics |
| Visualization | Matplotlib, Plotly |
| Optional dashboard | Streamlit |
| Version control | Git and GitHub |

---

## 10. Project Timeline

| Date | Planned Milestone |
|---|---|
| 20 October 2026 | Project Proposal, Tech Stack, and MVP Definition |
| 3 November 2026 | Literature Review, Core Code, Baseline Simulation, and Test Plan |
| 24 November 2026 | Feature Development, Testing Progress, and Issue Tracking |
| 8 December 2026 | Feature Development, Testing Progress, and Issue Tracking |
| 22 December 2026 | Demonstration, Run Instructions, and Packaging |
| 12 January 2027 | Final Report and Presentation (subject to course confirmation) |

---

## 11. Planned Deliverables

### November 3 Milestone

- [x] Project proposal completed
- [x] Three research papers selected
- [x] Literature review drafted
- [x] Test plan prepared
- [ ] XJTU-SY data-loading pipeline
- [ ] Statistical and wavelet feature extraction
- [ ] RUL label generation
- [ ] Random Forest baseline implementation
- [ ] Initial MAE/RMSE evaluation
- [ ] Reproducible baseline simulation

### Subsequent Development

- [ ] Standard LSTM implementation
- [ ] Standard GRU implementation
- [ ] Hyperparameter tuning
- [ ] Bearing-disjoint model comparison
- [ ] Per-bearing evaluation
- [ ] Historical maintenance simulation
- [ ] Final report and demonstration

---

## 12. Reproducibility

The project aims to provide a reproducible experimental workflow.

The repository will progressively include:

- Source code
- Dataset-loading instructions
- Preprocessing scripts
- Feature-extraction scripts
- Model implementations
- Evaluation scripts
- Configuration files
- Test results
- Documentation

The original datasets should be obtained from their respective sources.

Large raw datasets should not be committed without checking licensing conditions and repository size limits.

---

## 13. Limitations

The project has several important limitations:

1. XJTU-SY contains only 15 independent bearing histories.
2. Model performance may vary across operating conditions.
3. Randomly splitting windows from the same bearing can cause data leakage.
4. Future observations must not be included in model inputs.
5. Laboratory results do not establish real-world industrial performance.
6. Maintenance simulation costs are hypothetical.
7. The project does not claim a novel prediction algorithm.
8. The project does not claim that either LSTM or GRU is superior before experimental evaluation.

---

## 14. Team Members

**Team 3**

- Akram Serrar
- Seyed Hosseini Rivandi
- Yeganeh Seyyed Abady

---

## 15. Selected References

**[1]** H. Dhungana, A.-L. Kampen, A. S. Lundervold, and K. Øvsthus,  
"Remaining Useful Life Prediction of Rolling Bearings via Wavelet Feature Fusion and LSTM Networks,"  
*The International Journal of Advanced Manufacturing Technology*, vol. 142, pp. 2913–2930, 2026.  
DOI: https://doi.org/10.1007/s00170-025-17139-6

**[2]** J. Li, Z. Wang, X. Liu, and Z. Feng,  
"Remaining Useful Life Prediction of Rolling Bearings Using GRU-DeepAR with Adaptive Failure Threshold,"  
*Sensors*, vol. 23, no. 3, article 1144, 2023.  
DOI: https://doi.org/10.3390/s23031144

**[3]** H. Wang, H. Liao, X. Ma, and R. Bao,  
"Remaining Useful Life Prediction and Optimal Maintenance Time Determination for a Single Unit Using Isotonic Regression and Gamma Process Model,"  
*Reliability Engineering & System Safety*, vol. 210, article 107504, 2021.  
DOI: https://doi.org/10.1016/j.ress.2021.107504

---

## Project Status

**This repository documents an ongoing Master's-level course project.**

The proposed methodology, literature review, and evaluation strategy have been defined.

Model training, comparative experiments, and maintenance simulation will be documented as implementation progresses.

No unverified model accuracy, completed experiment, or industrial cost saving is claimed.

---

*Case Study: Intelligent Systems in Production — Team 3*  
*Technische Hochschule Deggendorf*
