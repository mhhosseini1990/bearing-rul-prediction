# Dataset and RUL target policy

## Primary dataset

**XJTU-SY** – vibration recordings from run-to-failure bearing experiments under different operating conditions. The dataset is NOT part of this repository. Download and cite the original dataset/paper in the final report.

## Expected checks before model training

- Verify the actual download source, license/citation information and extraction layout.
- Verify that each CSV corresponds to one measurement and that files are sorted chronologically.
- Verify which vibration column is horizontal and which is vertical.
- Verify recording interval and units if converting remaining recordings to time.
- Verify how the experiment defines the bearing end-of-life event.

## Baseline labels

For bearing `b`, let the recorded measurement steps be `0, ..., N_b - 1` in chronological order.

`RUL_steps(b, t) = (N_b - 1) - t`

This **does not** automatically represent real operating minutes or hours. It assumes the last measurement is the end of the run-to-failure history. Document limitations and modify the target policy if dataset metadata requires it.

## Prevent data leakage

**Use bearing-level train/validation/test splits.** No snapshots of a test bearing should appear during model fitting. Future labels cannot be used to compute inference features. Time-dependent feature normalization must be fitted on training bearings only.

The first example implementation uses one bearing-level train/test split. A dedicated bearing-level validation procedure should be added before tuning models and reporting a fair RF-vs-LSTM comparison.
