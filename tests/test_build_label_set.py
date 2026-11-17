import csv
from collections import Counter

import pytest

from src.data.build_label_set import ANNOTATORS, build, write_csv
from tests.helpers import noise_array, save_img

N_GOOD_GROUPS, N_BAD_GROUPS = 12, 6


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "extracted"
    seed = 0
    for folder, n in (("Apple_Good", N_GOOD_GROUPS), ("Apple_Bad", N_BAD_GROUPS)):
        for g in range(n):
            seed += 1
            arr = noise_array(seed)
            # se crea primero z_ para comprobar que el representante es a_ (alfabetico)
            save_img(r / "train" / folder / f"z_{g}.jpg", arr)
            save_img(r / "train" / folder / f"a_{g}.jpg", arr.copy())
    return r


def run(root, **kw):
    args = dict(n_good=5, n_kappa=4, threshold=6, seed=42)
    args.update(kw)
    return build(root, **args)


def test_representante_primero_alfabetico(root):
    _, queue = run(root, n_good=N_GOOD_GROUPS)
    assert len(queue) == N_GOOD_GROUPS + N_BAD_GROUPS
    for row in queue:
        assert row["filename"].rsplit("/", 1)[1].startswith("a_")
        assert row["group_size"] == 2


def test_columnas_y_formato_fruit_id(root):
    group_rows, queue = run(root, n_good=N_GOOD_GROUPS)
    assert set(group_rows[0]) == {"filename", "folder", "fruit_id"}
    assert set(queue[0]) == {"filename", "folder", "fruit_id", "group_size", "assigned_to"}
    for r in group_rows:
        kind = "good" if r["folder"] == "Apple_Good" else "bad"
        assert r["fruit_id"].startswith(f"kaggle_{kind}_")
        assert len(r["fruit_id"].rsplit("_", 1)[1]) == 4


def test_todos_los_bad_incluidos(root):
    _, queue = run(root, n_good=1)
    assert sum(r["folder"] == "Apple_Bad" for r in queue) == N_BAD_GROUPS


def test_n_good_respetado(root):
    _, queue = run(root, n_good=5)
    assert sum(r["folder"] == "Apple_Good" for r in queue) == 5
    _, queue = run(root, n_good=1000)
    assert sum(r["folder"] == "Apple_Good" for r in queue) == N_GOOD_GROUPS


def test_group_rows_cubren_todas_las_imagenes(root):
    group_rows, _ = run(root)
    assert len(group_rows) == 2 * (N_GOOD_GROUPS + N_BAD_GROUPS)
    assert len({r["fruit_id"] for r in group_rows}) == N_GOOD_GROUPS + N_BAD_GROUPS


def test_kappa(root):
    _, queue = run(root, n_kappa=4)
    assert sum(r["assigned_to"] == "kappa" for r in queue) == 4
    assert all(r["assigned_to"] in ANNOTATORS + ["kappa"] for r in queue)


def test_reparto_equilibrado(root):
    _, queue = run(root, n_good=N_GOOD_GROUPS, n_kappa=2)
    counts = Counter(r["assigned_to"] for r in queue if r["assigned_to"] != "kappa")
    assert set(counts) == set(ANNOTATORS)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_misma_semilla_mismo_resultado(root):
    assert run(root, seed=7) == run(root, seed=7)


def test_semilla_distinta_cambia_resultado(root):
    assert run(root, seed=1)[1] != run(root, seed=2)[1]


def test_kappa_proporcion_good_bad(root):
    _, queue = run(root, n_good=N_GOOD_GROUPS, n_kappa=6)
    kappa = [r for r in queue if r["assigned_to"] == "kappa"]
    bad = sum(r["folder"] == "Apple_Bad" for r in kappa)
    assert bad in (1, 2, 3)  # proporcion 1/3 +- 1


def test_lotes_estratificados(root):
    _, queue = run(root, n_good=N_GOOD_GROUPS, n_kappa=2)
    for a in ANNOTATORS:
        rows = [r for r in queue if r["assigned_to"] == a]
        bad = sum(r["folder"] == "Apple_Bad" for r in rows)
        assert 1 <= bad <= 2  # 6 bad entre 4 anotadores


def test_carpeta_vacia_aborta(tmp_path):
    (tmp_path / "Apple_Good").mkdir()
    with pytest.raises(SystemExit):
        build(tmp_path, 1, 1, 6, 0)


def test_acepta_png(tmp_path):
    for folder in ("Apple_Good", "Apple_Bad"):
        save_img(tmp_path / folder / "x.png", noise_array(3 if folder == "Apple_Good" else 4))
    _, queue = build(tmp_path, 1, 0, 6, 0)
    assert len(queue) == 2


def test_write_csv_roundtrip(tmp_path, root):
    _, queue = run(root)
    out = tmp_path / "sub" / "q.csv"
    write_csv(out, queue)
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(queue)
    assert list(rows[0]) == list(queue[0])
