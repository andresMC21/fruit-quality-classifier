"""App de re-etiquetado de las manzanas de Kaggle (criterios en docs/guia_anotacion.md).

Muestra una imagen representante por fruit_id (cola de src/data/build_label_set.py). No muestra la
carpeta de origen ni el fruit_id, porque dicen si la foto venia de "buena" o "mala" y eso sesga.
Cada anotador guarda en su propio CSV (data/annotations/kaggle_labels/<usuario>.csv) para que no
haya conflictos al hacer merge.

Si existe data/interim/prelabels.csv (filename, quality_class), en "Mi lote" se muestra esa
propuesta previa para confirmarla o corregirla. En la muestra kappa nunca se muestra.

Uso:
    streamlit run src/data/annotate_app.py
"""
import csv
import datetime
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.build_label_set import ANNOTATORS  # noqa: E402

IMAGES_DIR = ROOT / "data/raw/kaggle"
QUEUE_CSV = ROOT / "data/annotations/label_queue.csv"
OUT_DIR = ROOT / "data/annotations/kaggle_labels"
PRELABELS_CSV = ROOT / "data/interim/prelabels.csv"

FIELDS = ["filename", "annotator", "batch", "quality_class", "hand", "background",
          "doubt", "exclude", "exclude_reason", "timestamp"]
CLASSES = ["buena", "regular", "mala"]
EXCLUDE_REASONS = ["varias frutas", "no es manzana", "borrosa o ilegible", "fruta cortada", "otro"]
BATCHES = {"own": "Mi lote", "kappa": "Muestra comun (kappa)"}
CRITERIA = (
    "**Criterios** (`docs/guia_anotacion.md`)\n"
    "- **Buena:** piel integra, sin golpes, cortes, podredumbre ni moho.\n"
    "- **Regular:** golpe sin piel rota, raspon, mancha o arrugamiento leve; < 10 % del area visible.\n"
    "- **Mala:** podredumbre, moho, herida abierta, arrugamiento severo o >= 10 % del area.\n"
    "- El color de la variedad **no** es defecto."
)


def read_labels(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_labels(path, rows):
    """Escribe a un temporal y lo renombra: si el proceso muere, el CSV anterior queda intacto."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def save_label(path, row):
    """Guarda la etiqueta; si la imagen ya estaba etiquetada en ese lote, la reemplaza."""
    rows = [r for r in read_labels(path)
            if (r["filename"], r["batch"]) != (row["filename"], row["batch"])]
    write_labels(path, rows + [row])


def remove_last_label(path, batch):
    """Borra la ultima etiqueta del lote indicado y devuelve su filename (o None)."""
    rows = read_labels(path)
    for i in range(len(rows) - 1, -1, -1):
        if rows[i]["batch"] == batch:
            removed = rows.pop(i)["filename"]
            write_labels(path, rows)
            return removed
    return None


def load_prelabels(path):
    if not path.exists():
        return {}
    return dict(pd.read_csv(path)[["filename", "quality_class"]].itertuples(index=False))


def main():
    st.set_page_config(page_title="Re-etiquetado de manzanas", layout="wide")
    annotator = st.sidebar.selectbox("Anotador", ANNOTATORS)
    batch = st.sidebar.radio("Lote", list(BATCHES), format_func=BATCHES.get)
    st.sidebar.markdown(CRITERIA)

    queue = pd.read_csv(QUEUE_CSV)
    items = queue[queue["assigned_to"] == (annotator if batch == "own" else "kappa")]
    out_path = OUT_DIR / f"{annotator}.csv"
    done = {r["filename"] for r in read_labels(out_path) if r["batch"] == batch}
    pending = items[~items["filename"].isin(done)]

    n_done = len(items) - len(pending)
    st.sidebar.progress(n_done / max(len(items), 1), text=f"{n_done} / {len(items)}")
    if st.sidebar.button("Deshacer la ultima etiqueta de este lote", disabled=n_done == 0):
        remove_last_label(out_path, batch)
        st.session_state["k"] = st.session_state.get("k", 0) + 1
        st.rerun()

    if pending.empty:
        st.success("Lote terminado.")
        return

    item = pending.iloc[0]
    # Claves con anotador, lote y contador: asi los widgets se reinician en cada imagen.
    k = f"{annotator}_{batch}_{st.session_state.setdefault('k', 0)}"
    col_img, col_form = st.columns([3, 2])
    image_path = IMAGES_DIR / item["filename"]
    if image_path.exists():
        col_img.image(str(image_path), width=512)
    else:
        # Sin la ruta en el mensaje: incluye la carpeta Apple_Good/Apple_Bad y sesgaria al anotador.
        col_img.error("No se encuentra la imagen. Revisa la extraccion en docs/instalacion.md.")
    col_img.caption(f"Imagen {n_done + 1} de {len(items)}")

    with col_form:
        proposal = load_prelabels(PRELABELS_CSV).get(item["filename"]) if batch == "own" else None
        if proposal:
            st.info(f"Propuesta previa: **{proposal}**. Confirmala o corrigela.")
        hand = st.checkbox("Aparece una mano o parte de una persona", key=f"hand{k}")
        bg_simple = st.checkbox("Fondo simple (uniforme, sin objetos)", key=f"bg{k}")
        doubt = st.checkbox("Tengo duda", key=f"doubt{k}")

        def save(quality, exclude=False, reason=""):
            save_label(out_path, {
                "filename": item["filename"], "annotator": annotator, "batch": batch,
                "quality_class": quality, "hand": int(hand),
                "background": "simple" if bg_simple else "complejo",
                "doubt": int(doubt), "exclude": int(exclude), "exclude_reason": reason,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            })
            st.session_state["k"] += 1
            st.rerun()

        st.markdown("**Calidad**")
        for col, name in zip(st.columns(3), CLASSES):
            if col.button(name.capitalize(), key=f"{name}{k}", width="stretch"):
                save(name)

        st.markdown("**O excluir la imagen**")
        reason = st.selectbox("Motivo", EXCLUDE_REASONS, index=None, key=f"reason{k}")
        if st.button("Excluir", key=f"excl{k}"):
            if reason:
                save("", exclude=True, reason=reason)
            else:
                st.warning("Elige un motivo para excluir.")


if __name__ == "__main__":
    main()
