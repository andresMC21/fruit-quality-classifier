import csv
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.data import annotate_app
from src.data.build_label_set import ANNOTATORS
from tests.helpers import noise_array

ROOT = Path(__file__).resolve().parents[1]
APP_SRC = (ROOT / "src/data/annotate_app.py").read_text(encoding="utf-8")
ME = ANNOTATORS[0]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def env(tmp_path):
    images, out = tmp_path / "imgs", tmp_path / "out"
    (images / "Apple_Bad").mkdir(parents=True)
    rows = []
    for i in range(4):
        rel = f"Apple_Bad/secreto_{i}.jpg"
        Image.fromarray(noise_array(i)).save(images / rel)
        rows.append({"filename": rel, "folder": "Apple_Bad", "fruit_id": f"kaggle_bad_000{i}",
                     "group_size": 1, "assigned_to": ME if i < 3 else "kappa"})
    queue = tmp_path / "queue.csv"
    write_csv(queue, rows)
    prelabels = tmp_path / "prelabels.csv"
    write_csv(prelabels, [{"filename": "Apple_Bad/secreto_0.jpg", "quality_class": "regular"},
                          {"filename": "Apple_Bad/secreto_3.jpg", "quality_class": "mala"}])
    return images, queue, out, prelabels


def make_app(env, with_prelabels=False):
    images, queue, out, prelabels = env
    # Las rutas son constantes del propio script: se reescriben en una copia de prueba
    # (el codigo de produccion no se modifica).
    src = APP_SRC.replace("ROOT = Path(__file__).resolve().parents[2]", f"ROOT = Path({str(ROOT)!r})")
    nowhere = prelabels.parent / "no_existe.csv"
    vals = (("IMAGES_DIR", images), ("QUEUE_CSV", queue), ("OUT_DIR", out),
            ("PRELABELS_CSV", prelabels if with_prelabels else nowhere))
    for name, val in vals:
        lines = [l for l in src.splitlines() if l.startswith(f"{name} = ")]
        assert len(lines) == 1
        src = src.replace(lines[0], f"{name} = Path({str(val)!r})")
    return AppTest.from_string(src, default_timeout=30).run()


def read_out(out):
    p = out / f"{ME}.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def button(at, label):
    return next(b for b in at.button if b.label == label or b.label.startswith(label))


def test_app_carga(env):
    at = make_app(env)
    assert not at.exception
    assert at.sidebar.selectbox[0].value == ME
    assert "Imagen 1 de 3" in at.caption[0].value


def test_guardar_escribe_fila_y_avanza(env):
    _, _, out, _ = env
    at = make_app(env)
    button(at, "Buena").click().run()
    assert not at.exception
    rows = read_out(out)
    assert len(rows) == 1
    assert list(rows[0]) == annotate_app.FIELDS
    assert rows[0]["quality_class"] == "buena"
    assert rows[0]["filename"] == "Apple_Bad/secreto_0.jpg"
    assert rows[0]["annotator"] == ME and rows[0]["batch"] == "own"
    assert rows[0]["background"] == "complejo"
    assert "Imagen 2 de 3" in at.caption[0].value


def test_botones_de_clase(env):
    at = make_app(env)
    labels = [b.label for b in at.button]
    for name in ("Buena", "Regular", "Mala"):
        assert name in labels


def test_no_revela_carpeta_ni_fruit_id(env):
    for with_pre in (False, True):
        at = make_app(env, with_prelabels=with_pre)
        visible = [c.value for c in at.caption] + [m.value for m in at.markdown]
        visible += [m.value for m in at.sidebar.markdown] + [i.value for i in at.info]
        visible += [str(o) for o in at.radio[0].options] + [str(o) for s in at.selectbox for o in s.options]
        for text in visible:
            for secret in ("Apple_Bad", "Apple_Good", "kaggle_bad", "kaggle_good", "secreto", "fruit_id"):
                assert secret not in text


def test_propuesta_previa_en_lote_own(env):
    at = make_app(env, with_prelabels=True)
    assert len(at.info) == 1
    assert "regular" in at.info[0].value


def test_sin_archivo_de_prelabels_no_hay_propuesta(env):
    assert not make_app(env, with_prelabels=False).info


def test_propuesta_previa_no_aparece_en_kappa(env):
    at = make_app(env, with_prelabels=True)
    at.sidebar.radio[0].set_value("kappa").run()
    assert not at.exception
    assert "Imagen 1 de 1" in at.caption[0].value  # es la imagen que si tiene prelabel (secreto_3)
    assert not at.info


def test_deshacer_elimina_ultima_fila(env):
    _, _, out, _ = env
    at = make_app(env)
    button(at, "Buena").click().run()
    button(at, "Mala").click().run()
    assert [r["quality_class"] for r in read_out(out)] == ["buena", "mala"]
    button(at, "Deshacer").click().run()
    assert [r["quality_class"] for r in read_out(out)] == ["buena"]
    assert "Imagen 2 de 3" in at.caption[0].value


def test_deshacer_deshabilitado_sin_etiquetas(env):
    assert button(make_app(env), "Deshacer").disabled


def test_excluir_sin_motivo_avisa(env):
    _, _, out, _ = env
    at = make_app(env)
    button(at, "Excluir").click().run()
    assert at.warning and read_out(out) == []


def test_excluir_con_motivo_guarda(env):
    _, _, out, _ = env
    at = make_app(env)
    next(s for s in at.selectbox if s.label == "Motivo").select("no es manzana")
    button(at, "Excluir").click().run()
    rows = read_out(out)
    assert len(rows) == 1
    assert rows[0]["exclude"] == "1" and rows[0]["exclude_reason"] == "no es manzana"
    assert rows[0]["quality_class"] == ""


def mkrow(filename, batch, quality="buena"):
    r = {k: "" for k in annotate_app.FIELDS}
    r.update(filename=filename, batch=batch, quality_class=quality, annotator="x")
    return r


def test_save_label_no_duplica(tmp_path):
    p = tmp_path / "x.csv"
    annotate_app.save_label(p, mkrow("i1", "own", "buena"))
    annotate_app.save_label(p, mkrow("i1", "own", "mala"))
    annotate_app.save_label(p, mkrow("i1", "kappa", "regular"))  # otro lote: se conserva
    rows = annotate_app.read_labels(p)
    assert [(r["filename"], r["batch"], r["quality_class"]) for r in rows] == [
        ("i1", "own", "mala"), ("i1", "kappa", "regular")]
    assert not p.with_suffix(".tmp").exists()


def test_remove_last_label_respeta_lote(tmp_path):
    p = tmp_path / "x.csv"
    for img, b in (("i1", "own"), ("i2", "kappa"), ("i3", "kappa")):
        annotate_app.save_label(p, mkrow(img, b))
    assert annotate_app.remove_last_label(p, "own") == "i1"
    assert [r["filename"] for r in annotate_app.read_labels(p)] == ["i2", "i3"]
    assert annotate_app.remove_last_label(p, "own") is None
    assert annotate_app.remove_last_label(p, "kappa") == "i3"
    assert [r["filename"] for r in annotate_app.read_labels(p)] == ["i2"]


def test_load_prelabels(tmp_path):
    assert annotate_app.load_prelabels(tmp_path / "nope.csv") == {}
    write_csv(tmp_path / "p.csv", [{"filename": "a.jpg", "quality_class": "mala"}])
    assert annotate_app.load_prelabels(tmp_path / "p.csv") == {"a.jpg": "mala"}


def test_app_deshacer_solo_afecta_al_lote_activo(env):
    _, _, out, _ = env
    annotate_app.save_label(out / f"{ME}.csv", mkrow("otro.jpg", "kappa"))
    at = make_app(env)
    button(at, "Buena").click().run()
    button(at, "Deshacer").click().run()
    rows = read_out(out)
    assert [(r["filename"], r["batch"]) for r in rows] == [("otro.jpg", "kappa")]


def test_constantes_del_modulo():
    assert annotate_app.CLASSES == ["buena", "regular", "mala"]
    assert annotate_app.OUT_DIR == ROOT / "data/annotations/kaggle_labels"
    assert annotate_app.IMAGES_DIR == ROOT / "data/raw/kaggle"
    assert annotate_app.PRELABELS_CSV == ROOT / "data/interim/prelabels.csv"
