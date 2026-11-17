import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.features import feature_names
from src.models.classical import MODELS
from src.training import train as T
from src.training import tune as TU

CLASSES = ["buena", "regular", "mala"]


def test_class_weights():
    w = T.class_weights(np.array([0, 0, 1, 2, 2, 2]), 3)
    assert torch.allclose(w, torch.tensor([1.0, 2.0, 2 / 3]))


def test_class_weights_missing_class_no_div0():
    w = T.class_weights(np.array([0, 0, 1]), 3)
    assert torch.isfinite(w).all()


def test_macro_f1_zero_division():
    assert T.macro_f1(["buena"] * 3, ["buena"] * 3, CLASSES) == pytest.approx(1 / 3)
    assert T.macro_f1(["buena"], ["mala"], CLASSES) == 0.0


def test_log_experiment(tmp_path):
    cfg = {"paths": {"experiments_log": str(tmp_path / "a/b/log.csv")}}
    T.log_experiment(cfg, {"exp_id": "x", "model": "m"})
    T.log_experiment(cfg, {"exp_id": "y", "model": "m", "extra": 1})
    df = pd.read_csv(cfg["paths"]["experiments_log"])
    assert list(df.columns) == T.LOG_COLUMNS and list(df["exp_id"]) == ["x", "y"]


def test_train_unknown_model():
    with pytest.raises(ValueError):
        T.train("nope", {"seed": 1})


# ---------------------------------------------------------------- tune
def synth(n_per=12, seed=0):
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for k, c in enumerate(CLASSES):
        for i in range(n_per):
            X.append(rng.normal(k * 6, 1, 4))
            y.append(c)
            g.append(f"{c}{i // 2}")
    return np.array(X), np.array(y), np.array(g)


def test_tune_logreg():
    grid = {"clf__C": [0.1, 1, 10]}
    cfg = {"seed": 0, "classical": {"cv_folds": 3, "grids": {"logreg": grid}}}
    X, y, g = synth()
    model, params, f1 = TU.tune("logreg", cfg, X, y, g)
    assert params["clf__C"] in grid["clf__C"]
    assert f1 > 0.8
    assert (model.predict(X) == y).mean() > 0.9


def test_tune_passes_groups(monkeypatch):
    seen = {}
    orig = TU.GridSearchCV.fit

    def spy(self, X, y=None, **kw):
        seen.update(kw)
        return orig(self, X, y, **kw)
    monkeypatch.setattr(TU.GridSearchCV, "fit", spy)
    cfg = {"seed": 0, "classical": {"cv_folds": 3, "grids": {"logreg": {"clf__C": [1]}}}}
    X, y, g = synth()
    TU.tune("logreg", cfg, X, y, g)
    assert list(seen["groups"]) == list(g)


# ---------------------------------------------------------------- humo
@pytest.fixture
def smoke(tmp_path, monkeypatch):
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    t = tmp_path
    cfg["split"]["output"] = str(t / "splits.csv")
    cfg["paths"].update(features=str(t / "features.csv"), experiments_log=str(t / "res/log.csv"),
                        checkpoints=str(t / "ckpt"), logs=str(t / "logs"))
    cfg["cnn"].update(epochs=2, batch_size=8, patience=5, grid={"lr": [0.001], "weight_decay": [0.0], "dropout": [0.3]})
    cfg["classical"]["cv_folds"] = 2
    cfg["classical"]["grids"] = {"logreg": {"clf__C": [1]}, "svm": {"clf__estimator__C": [1]},
                                 "rf": {"clf__n_estimators": [5]}}
    raw = t / "raw"
    raw.mkdir()
    monkeypatch.setattr(T, "RAW_DIR", raw)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(0)
    rows = []
    n = 0
    for split, per in (("train", 8), ("val", 4)):
        for k, c in enumerate(CLASSES):
            for i in range(per):
                name = f"im{n}.png"
                img = np.full((40, 40, 3), 40 + 80 * k, np.uint8) + rng.integers(0, 10, (40, 40, 3), dtype=np.uint8)
                cv2.imwrite(str(raw / name), img)
                rows.append(dict(image_id=f"i{n}", filename=name, source="kaggle", fruit_id=f"{split}{c}{i // 2}",
                                 quality_class=c, split=split))
                n += 1
    splits = pd.DataFrame(rows)
    splits.to_csv(cfg["split"]["output"], index=False)
    feat = splits.drop(columns="filename").copy()
    feat["seg_ok"] = True
    q = feat["quality_class"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    for j, nm in enumerate(feature_names()):
        feat[nm] = q * 5.0 + rng.normal(0, 0.5, len(feat)) if j < 30 else rng.normal(0, 1, len(feat))
    feat.to_csv(cfg["paths"]["features"], index=False)
    return cfg


def test_train_cnn_smoke(smoke):
    exp_id, f1 = T.train("cnn", smoke)
    assert exp_id.startswith("cnn_") and 0 <= f1 <= 1
    from pathlib import Path
    assert (Path(smoke["paths"]["checkpoints"]) / f"{exp_id}.pt").exists()
    assert (Path(smoke["paths"]["logs"]) / f"{exp_id}.csv").exists()
    log = pd.read_csv(smoke["paths"]["experiments_log"])
    assert len(log) == 1 and log.loc[0, "model"] == "cnn"


@pytest.mark.parametrize("name", ["baseline", "logreg", "svm", "rf"])
def test_train_classical_smoke(smoke, name):
    from pathlib import Path
    assert name in MODELS
    if name == "baseline":
        smoke["classical"]["grids"]["baseline"] = {}
    exp_id, val_f1 = T.train(name, smoke)
    assert (Path(smoke["paths"]["checkpoints"]) / f"{exp_id}.joblib").exists()
    log = pd.read_csv(smoke["paths"]["experiments_log"])
    assert log.loc[0, "model"] == f"{name}-masked"
    if name != "baseline":
        assert val_f1 > 0.8


def test_train_classical_invalid_features(smoke):
    with pytest.raises(ValueError):
        T.train("logreg", smoke, features="otro")
