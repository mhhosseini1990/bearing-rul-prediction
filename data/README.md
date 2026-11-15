# XJTU-SY data setup

Place the original 15 run-to-failure histories under:

```text
data/raw/XJTU-SY_Bearing_Datasets/
  35Hz12kN/Bearing1_1..Bearing1_5/
  37.5Hz11kN/Bearing2_1..Bearing2_5/
  40Hz10kN/Bearing3_1..Bearing3_5/
```

Extract or validate the single active feature matrix:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py --extract
.\.venv\Scripts\python.exe scripts\prepare_data.py --validate
```

Expected fixed contract: 9,216 rows, 15 bearings, 3 conditions, 48 features,
registry SHA-256 `18692d899aafd55be58a5a01fd66892c2b7f8cb8e1bdad75d2f09a34deaf2b0f`,
and CSV SHA-256
`d72fb0c3d05badf455ccf9cc0aceac5a898d7d30b7c13e424feacc669420451d`.

Raw and processed datasets are Git-ignored. Raw PRONOSTIA material, if present
locally for historical research, is not read by the final pipeline.
