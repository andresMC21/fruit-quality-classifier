import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.models.classical import MODELS, build_model


@pytest.mark.parametrize("name", MODELS)
def test_build_model_pipeline(name):
    assert isinstance(build_model(name, 0), Pipeline)


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("xgb")


def test_baseline_majority():
    X = np.random.default_rng(0).normal(size=(10, 3))
    y = np.array(["buena"] * 6 + ["mala"] * 3 + ["regular"])
    m = build_model("baseline").fit(X, y)
    assert set(m.predict(X)) == {"buena"}
