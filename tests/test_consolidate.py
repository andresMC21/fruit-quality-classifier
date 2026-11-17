import pandas as pd
import pytest

from src.data import consolidate as C

LCOLS = ["filename", "annotator", "batch", "quality_class", "hand", "background", "doubt",
         "exclude", "exclude_reason", "timestamp"]


def lab(filename, annotator, batch, q="buena", exclude="0", doubt="0"):
    return dict(filename=filename, annotator=annotator, batch=batch, quality_class=q, hand="0",
                background="liso", doubt=doubt, exclude=exclude, exclude_reason="", timestamp="t")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/annotations/labels").mkdir(parents=True)
    cfg = {"paths": {"kaggle_labels": "data/annotations/labels",
                     "annotations": "data/annotations/annotations.csv"}}
    names = ["own1.jpg", "own2.jpg", "k_agree.jpg", "k_dis.jpg", "k_both_ex.jpg", "k_one.jpg", "k_ex_vs_q.jpg"]
    q = pd.DataFrame({"filename": names, "folder": "f", "fruit_id": [f"F{i}" for i in range(len(names))],
                      "group_size": "1", "assigned_to": "a"})
    q.to_csv(tmp_path / "data/annotations/label_queue.csv", index=False)

    def write(a, b):
        pd.DataFrame(a, columns=LCOLS).to_csv(tmp_path / "data/annotations/labels/ana.csv", index=False)
        pd.DataFrame(b, columns=LCOLS).to_csv(tmp_path / "data/annotations/labels/beto.csv", index=False)
    return tmp_path, cfg, write


def base_rows():
    a = [lab("own1.jpg", "ana", "own", "regular", doubt="1"),
         lab("own2.jpg", "ana", "own", exclude="1"),
         lab("k_agree.jpg", "ana", "kappa", "buena"),
         lab("k_dis.jpg", "ana", "kappa", "buena"),
         lab("k_both_ex.jpg", "ana", "kappa", exclude="1"),
         lab("k_one.jpg", "ana", "kappa", "mala")]
    b = [lab("k_agree.jpg", "beto", "kappa", "buena"),
         lab("k_dis.jpg", "beto", "kappa", "mala"),
         lab("k_both_ex.jpg", "beto", "kappa", exclude="1")]
    return a, b


def test_own_exclusions_and_columns(env):
    _, cfg, write = env
    write(*base_rows())
    merged, _ = C.consolidate(cfg)
    assert list(merged.columns) == C.COLUMNS
    assert "kaggle/own2.jpg" not in set(merged["filename"])
    r = merged[merged["filename"] == "kaggle/own1.jpg"].iloc[0]
    assert r["quality_class"] == "regular" and r["notes"] == "duda"


def test_kappa_agreement_and_prefix_and_ids(env):
    _, cfg, write = env
    write(*base_rows())
    merged, _ = C.consolidate(cfg)
    r = merged[merged["filename"] == "kaggle/k_agree.jpg"].iloc[0]
    assert r["quality_class"] == r["annotator2_quality"] == "buena"
    assert r["image_id"] == r["fruit_id"] + "_k_agree" and r["fruit_id"] == "F2"
    assert r["source"] == "kaggle" and r["notes"] == ""
    assert merged["filename"].str.startswith("kaggle/").all()


def test_disagreement_unresolved(env):
    _, cfg, write = env
    write(*base_rows())
    merged, _ = C.consolidate(cfg)
    r = merged[merged["filename"] == "kaggle/k_dis.jpg"].iloc[0]
    assert r["quality_class"] == "" and r["notes"] == "desacuerdo kappa"


def test_disagreement_resolved(env):
    tmp, cfg, write = env
    write(*base_rows())
    pd.DataFrame({"filename": ["k_dis.jpg"], "quality_class": ["regular"]}).to_csv(
        tmp / "data/annotations/kappa_resolutions.csv", index=False)
    merged, _ = C.consolidate(cfg)
    r = merged[merged["filename"] == "kaggle/k_dis.jpg"].iloc[0]
    assert r["quality_class"] == "regular" and r["notes"] == "desacuerdo kappa"
    assert r["annotator2_quality"] == "mala"


def test_both_exclude_dropped_and_single_annotator_ignored(env):
    _, cfg, write = env
    write(*base_rows())
    merged, _ = C.consolidate(cfg)
    names = set(merged["filename"])
    assert "kaggle/k_both_ex.jpg" not in names
    assert "kaggle/k_one.jpg" not in names   # solo un anotador: no es par kappa


def test_resolution_to_excluir_drops_row(env):
    tmp, cfg, write = env
    a = [lab("k_ex_vs_q.jpg", "ana", "kappa", "buena"), lab("k_agree.jpg", "ana", "kappa", "buena")]
    b = [lab("k_ex_vs_q.jpg", "beto", "kappa", exclude="1"), lab("k_agree.jpg", "beto", "kappa", "buena")]
    write(a, b)
    pd.DataFrame({"filename": ["k_ex_vs_q.jpg"], "quality_class": ["excluir"]}).to_csv(
        tmp / "data/annotations/kappa_resolutions.csv", index=False)
    merged, _ = C.consolidate(cfg)
    assert list(merged["filename"]) == ["kaggle/k_agree.jpg"]


def test_all_rows_excluded_does_not_crash(env):
    """Regresion: si todas las filas quedan excluidas, antes fallaba con KeyError 'filename'."""
    _, cfg, write = env
    write([lab("k_both_ex.jpg", "ana", "kappa", exclude="1")], [lab("k_both_ex.jpg", "beto", "kappa", exclude="1")])
    merged, _ = C.consolidate(cfg)
    assert len(merged) == 0


def test_kappa_perfect(env):
    _, cfg, write = env
    a = [lab("k_agree.jpg", "ana", "kappa", "buena"), lab("k_dis.jpg", "ana", "kappa", "mala")]
    b = [lab("k_agree.jpg", "beto", "kappa", "buena"), lab("k_dis.jpg", "beto", "kappa", "mala")]
    write(a, b)
    _, kappa = C.consolidate(cfg)
    assert kappa == pytest.approx(1.0)


def test_keeps_own_replaces_kaggle(env):
    tmp, cfg, write = env
    write(*base_rows())
    prev = pd.DataFrame([
        dict(image_id="OWN_1", filename="own_images/x.jpg", source="own", quality_class="buena"),
        dict(image_id="OLD", filename="kaggle/old.jpg", source="kaggle", quality_class="mala")])
    prev.to_csv(tmp / "data/annotations/annotations.csv", index=False)
    merged, _ = C.consolidate(cfg)
    assert "OWN_1" in set(merged["image_id"])
    assert "OLD" not in set(merged["image_id"])
    on_disk = pd.read_csv(tmp / "data/annotations/annotations.csv", dtype=str, keep_default_na=False)
    assert list(on_disk.columns) == C.COLUMNS and len(on_disk) == len(merged)
