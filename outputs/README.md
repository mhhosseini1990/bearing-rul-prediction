# Generated outputs

The final runner writes only to:

```text
outputs/final/model/fixed48_direct_final/
  checkpoints/                         # exactly 30 when complete
  predictions_raw.csv
  predictions_common_range.csv         # primary step >= 29 evidence
  learning_curves.csv
  per_bearing_metrics.csv
  per_fold_metrics.csv
  per_condition_metrics.csv
  summary_by_seed.csv
  model_summary.csv
  paired_bearing_comparison.csv
  per_run_summary.csv
  split_manifest.json
  run_config.json
  LIMITATIONS.txt
```

Maintenance replay outputs are generated afterward under
`outputs/final/maintenance_replay/`. All outputs are Git-ignored. Historical
experiments were moved to the verified external research archive.
