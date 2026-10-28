# XJTU-SY Dataset Split — Baseline v1

## Purpose
A fixed **bearing-level** split for a fair comparison of LSTM and GRU RUL prediction. A complete bearing run, including all its measurement files and generated windows, belongs to exactly one partition.

## Operating conditions
| Folder | Condition | Bearing IDs |
|---|---|---|
| `35Hz12kN` | 35 Hz, 12 kN | `Bearing1_1`–`Bearing1_5` |
| `37.5Hz11kN` | 37.5 Hz, 11 kN | `Bearing2_1`–`Bearing2_5` |
| `40Hz10kN` | 40 Hz, 10 kN | `Bearing3_1`–`Bearing3_5` |

## Fixed assignments
| Condition | Train | Validation | Test |
|---|---|---|---|
| 35 Hz, 12 kN | Bearing1_1, Bearing1_2, Bearing1_3 | Bearing1_4 | Bearing1_5 |
| 37.5 Hz, 11 kN | Bearing2_1, Bearing2_2, Bearing2_3 | Bearing2_4 | Bearing2_5 |
| 40 Hz, 10 kN | Bearing3_1, Bearing3_2, Bearing3_3 | Bearing3_4 | Bearing3_5 |
| **Total** | **9** | **3** | **3** |

Machine-readable source: `configs/bearing_split_v1.yaml`.

## Rules
1. Never split files, snapshots or windows from the same bearing across partitions.
2. Use the exact same assignments for LSTM and GRU.
3. Fit data-dependent preprocessing, including scalers, on training bearings only.
4. Tune model hyperparameters using validation bearings; reserve test bearings for final evaluation.
5. Record per-bearing MAE/RMSE and operating conditions; report pooled metrics with their aggregation method.
6. Do not change this split after inspecting test performance. Any revised evaluation protocol must be versioned and justified.

## Limitations and verification
- Training, validation and test each cover all three operating conditions. However, each validation and test condition contains only one bearing, so performance estimates may be sensitive to the chosen bearing.
- Local PowerShell directory inventory on 28 October 2026 confirmed all 15 expected bearing folder names across the three operating conditions. **CSV file contents, chronology, and measurement integrity have not yet been verified.**
- This split evaluates unseen bearings under **known** operating conditions, not generalization to unseen operating conditions.
- This split is a project design choice, not a published official split or evidence of model performance.
