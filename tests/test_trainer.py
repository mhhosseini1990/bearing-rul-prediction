"""Model fitting and final-runner provenance tests."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_final_experiment as final  # noqa: E402
import train  # noqa: E402


@pytest.mark.parametrize("model_name", ["lstm", "gru"])
def test_fit_one_returns_nonnegative_softplus_predictions(model_name):
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(12, 10, 48)).astype(np.float32)
    y_train = np.linspace(0, 100, 12, dtype=np.float32)
    owners = np.array(["A"] * 6 + ["B"] * 6)
    x_val = rng.normal(size=(6, 10, 48)).astype(np.float32)
    y_val = np.linspace(0, 50, 6, dtype=np.float32)
    meta = pd.DataFrame({"bearing_id": ["V1"] * 3 + ["V2"] * 3,
                         "condition": [1] * 3 + [2] * 3,
                         "step": range(29, 35), "rul_minutes": y_val})
    model, scale, epoch, score, curve = train.fit_one(
        x_train, y_train, owners, x_val, y_val, meta, model_name, 32, 42,
        epochs=1, patience=1, min_epochs=1, batch_size=128, output_activation="softplus")
    prediction = train.infer(model, x_val, scale, 128)
    assert prediction.shape == (6,) and (prediction >= 0).all()
    assert scale == pytest.approx(float(np.quantile(y_train, 0.95)))
    assert epoch == 1 and np.isfinite(score) and len(curve) == 1


def test_final_protocol_has_thirty_tasks_and_approved_lengths():
    assert len(final.tasks()) == 30
    assert final.SEQUENCE_BY_FOLD == {"fold_1": 10, "fold_2": 30, "fold_3": 30,
                                      "fold_4": 10, "fold_5": 10}
    assert final.COMMON_MIN_STEP == 29 and final.OUTPUT_ACTIVATION == "softplus"


def test_checkpoint_round_trip_and_provenance(tmp_path):
    expected = final.expected_metadata("fold_1", "gru", 42, "data-hash")
    model = train.MODELS["gru"](48, 32, 1, "softplus")
    checkpoint = {**expected, "model_state_dict": model.state_dict(),
                  "feature_mean": [0.] * 48, "feature_std": [1.] * 48,
                  "target_scale": 100., "best_epoch": 8,
                  "validation_macro_mae_minutes": 10., "learning_curve": []}
    path = tmp_path / "model.pt"; torch.save(checkpoint, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    final.verify_checkpoint(restored, expected)
    loaded = final.load_model(restored)
    with torch.inference_mode():
        torch.testing.assert_close(model(torch.zeros(2, 10, 48)), loaded(torch.zeros(2, 10, 48)))


def test_checkpoint_provenance_mismatch_is_rejected():
    expected = final.expected_metadata("fold_1", "gru", 42, "correct")
    with pytest.raises(ValueError, match="provenance mismatch"):
        final.verify_checkpoint({**expected, "csv_sha256": "wrong"}, expected)


def test_validation_only_run_never_constructs_test_sequences(monkeypatch, tmp_path):
    ids = {f"Bearing{c}_{i}" for c in (1, 2, 3) for i in range(1, 6)}
    records = []
    for bearing in ids:
        for step in range(3):
            row = {"bearing_id": bearing, "condition": int(bearing[7]), "step": step,
                   "elapsed_minutes": step, "rul_minutes": 2 - step}
            row.update({name: float(step) for name in train.FEATURE_NAMES})
            records.append(row)
    frame = pd.DataFrame(records)
    monkeypatch.setattr(train, "validate_inputs", lambda path: (frame, "hash"))
    seen = []

    def fake_sequences(part, sequence, mean, std):
        seen.append(set(part.bearing_id))
        n = len(part.bearing_id.unique())
        metadata = part.groupby("bearing_id", as_index=False).tail(1)[
            ["bearing_id", "condition", "step", "rul_minutes"]].reset_index(drop=True)
        return np.zeros((n, sequence, 48), np.float32), np.ones(n, np.float32), metadata.bearing_id.to_numpy(), metadata

    monkeypatch.setattr(train, "sequences", fake_sequences)
    monkeypatch.setattr(train, "fit_one", lambda *a, **k: (
        train.MODELS["gru"](48, 2, 1, "softplus"), 1., 1, 1.,
        pd.DataFrame([{"epoch": 1, "train_loss": 1., "val_macro_mae_minutes": 1.}])) )
    monkeypatch.setattr(train, "infer", lambda model, x, scale, batch: np.ones(len(x)))
    args = type("Args", (), {"check_only": False, "validation_only": True,
        "sequence": 2, "batch_size": 2, "hidden_size": 2, "epochs": 1,
        "patience": 1, "min_epochs": 1, "features": tmp_path / "fake.csv",
        "fold": "fold_1", "model": "gru", "seed": "42", "output": tmp_path / "out",
        "overwrite": False, "output_activation": "softplus"})()
    train.run(args)
    assert seen == [set(train.FOLDS["fold_1"]["train"]), set(train.FOLDS["fold_1"]["val"])]
    assert not (args.output / "predictions.csv").exists()
    assert (args.output / "validation_predictions.csv").is_file()

