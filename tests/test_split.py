import sys

import pandas as pd
import pytest

from src.data import split as split_mod
from src.data.split import group_labels, make_splits
from src.utils.config import load_config

CFG = load_config()
CLASSES = CFG["classes"]


def _annotations():
    rows = []

    def add(source, fid, cls):
        n = len(rows) + 1
        rows.append({"image_id": f"i{n}", "filename": f"f{n}.jpg", "source": source,
                     "fruit_id": fid, "quality_class": cls})

    for ci, cls in enumerate(CLASSES):
        for k in range(20):                      # propias: 20 frutas por clase, 3 vistas
            for _ in range(3):
                add("own", f"own_{ci}_{k}", cls)
        for k in range(40):                      # publicas: 40 grupos por clase
            add("kaggle", f"kg_{ci}_{k}", cls)
    add("own", "own_blank", "")
    add("own", "own_ex", "excluir")
    add("kaggle", "kg_blank", None)
    add("kaggle", "kg_ex", "excluir")
    return pd.DataFrame(rows)


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "annotations.csv"
    _annotations().to_csv(p, index=False)
    return str(p)


def test_no_group_in_two_splits(csv_path):
    out = make_splits(csv_path, CFG)
    assert (out.groupby("fruit_id")["split"].nunique() == 1).all()
    assert out["split"].notna().all()


def test_invalid_rows_dropped(csv_path):
    out = make_splits(csv_path, CFG)
    assert set(out["quality_class"]) <= set(CLASSES)
    assert not out["fruit_id"].isin(["own_blank", "own_ex", "kg_blank", "kg_ex"]).any()
    assert len(out) == 3 * 20 * 3 + 3 * 40


def test_proportions_per_source_and_class(csv_path):
    out = make_splits(csv_path, CFG)
    s = CFG["split"]
    for source, frac, n in (("own", s["test_own"], 20), ("kaggle", s["test_public"], 40)):
        part = out[out["source"] == source].drop_duplicates("fruit_id")
        for cls in CLASSES:
            g = part[part["quality_class"] == cls]["split"].value_counts()
            assert abs(g.get("test", 0) - frac * n) <= 1
            assert abs(g.get("val", 0) - s["val"] * n) <= 1
            assert g.get("train", 0) > 0


def test_group_label_is_worst():
    df = pd.DataFrame({"fruit_id": ["a", "a", "b", "b", "c"],
                       "quality_class": ["buena", "mala", "buena", "regular", "buena"]})
    assert group_labels(df, CLASSES) == {"a": "mala", "b": "regular", "c": "buena"}


def test_mixed_views_stratified_as_worst(tmp_path):
    rows = [{"image_id": f"i{i}", "filename": f"f{i}.jpg", "source": "own", "fruit_id": "m",
             "quality_class": c} for i, c in enumerate(["buena", "mala"])]
    p = tmp_path / "a.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    out = make_splits(str(p), CFG)
    assert out["split"].nunique() == 1


def test_deterministic(csv_path):
    pd.testing.assert_frame_equal(make_splits(csv_path, CFG), make_splits(csv_path, CFG))


def test_seed_changes_result(csv_path):
    other = {**CFG, "seed": CFG["seed"] + 1}
    assert not make_splits(csv_path, CFG)["split"].equals(make_splits(csv_path, other)["split"])


def _patched(monkeypatch, tmp_path, csv_path):
    out = tmp_path / "out" / "splits.csv"
    cfg = {**CFG, "paths": {**CFG["paths"], "annotations": csv_path},
           "split": {**CFG["split"], "output": str(out)}}
    monkeypatch.setattr(split_mod, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(sys, "argv", ["split"])
    return out


def test_main_refuses_overwrite_unless_force(monkeypatch, tmp_path, csv_path):
    out = _patched(monkeypatch, tmp_path, csv_path)
    split_mod.main()
    assert out.exists()
    before = out.read_text()
    with pytest.raises(SystemExit):
        split_mod.main()
    assert out.read_text() == before
    out.write_text("viejo")
    monkeypatch.setattr(sys, "argv", ["split", "--force"])
    split_mod.main()
    assert out.read_text() == before


def test_shared_fruit_id_between_sources_raises(tmp_path):
    """Propias y publicas se reparten por separado: un fruit_id en ambas se pisaria sin aviso."""
    rows = [
        dict(image_id="a", filename="a", source="own", fruit_id="F1", quality_class="buena"),
        dict(image_id="b", filename="b", source="kaggle", fruit_id="F1", quality_class="mala"),
    ]
    p = tmp_path / "ann.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    with pytest.raises(ValueError):
        make_splits(str(p), load_config())


@pytest.mark.parametrize("n, expected", [(1, (0, 0)), (2, (1, 0)), (3, (1, 1)), (4, (2, 1)), (20, (8, 3))])
def test_split_sizes_small_classes(n, expected):
    """test_own=0.40, val=0.15: con n >= 3 siempre hay al menos 1 en test, val y train."""
    n_test, n_val = split_mod.split_sizes(n, 0.40, 0.15)
    assert (n_test, n_val) == expected
    assert n - n_test - n_val >= 1


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_split_sizes_public_fraction_never_empty_test(n):
    n_test, n_val = split_mod.split_sizes(n, 0.15, 0.15)
    assert n_test >= 1 and n_val >= 1 and n - n_test - n_val >= 1
