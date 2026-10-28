# Team checklist (assign owners together)

| Workstream | Suggested task | Owner | Status |
|---|---|---|---|
| Research | Literature map and related work | TBD | Planned |
| Data | Download/inspect XJTU-SY; validate labels | TBD | Planned |
| Baseline | Feature extraction and Random Forest | TBD | Initial scaffold |
| Deep learning | LSTM sequence pipeline | TBD | Planned |
| Evaluation | MAE/RMSE, per-bearing replay plots | TBD | Initial scaffold |
| Documentation | GitHub README, report and slides | TBD | Initial scaffold |

## Team Git workflow

1. Keep `main` as the stable branch.
2. Create topic branches such as `feature/data-loading` or `feature/lstm`.
3. Commit small logical changes with a clear message.
4. Use a pull request to merge changes into `main`.
5. Never commit downloaded datasets or large model files.
