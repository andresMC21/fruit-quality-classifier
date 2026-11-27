"""Fase 2/3: construye data/annotations/annotations.csv para las bases públicas de la especie elegida.

Fuentes:
- kaggle   : Fruit Quality Classification (ryandpark). Good -> buena, Bad -> mala.
- afruitdb : Fruit grade (AFruitDB / Asian Fruits Dataset). 1st -> buena, 2nd -> regular, 3rd -> mala.
- fruitvision: FruitVision (Bangladesh, fondo gris). Fresh -> buena, Rotten -> mala; Formalin-mixed se excluye
  (fruta tratada con formol: criterio químico, no visible en la foto).
- fruits360: Fruits-360 (100x100, fondo blanco). SIN calidad: solo entrena el clasificador de especie e incluye la
  clase `other` (frutas fuera del alcance) para que la app pueda rechazarlas.
- wgisd: WGISD (Embrapa, racimos de uva en viñedo). Cada racimo se recorta con su máscara de instancia sobre fondo
  blanco (src/data/wgisd_crops.py). SIN calidad: solo especie (uva en racimo, que las demás fuentes no tienen).
- mixed_quality (Kaggle): varias frutas por foto; se segmentan y se anotan a mano (no entra aquí).
- own / other_group: se agregan con `add_own_images` (fotos propias y de otros grupos).

La etiqueta de calidad de las bases públicas es una etiqueta DÉBIL: viene de la carpeta, no de los criterios
de docs/guia_anotacion.md. Por eso se conserva `orig_label` y se revisa una muestra (ver notebooks/02_eda.ipynb).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COLUMNS = ["image_id", "filename", "source", "group_id", "fruit_species", "fruit_id", "parent_image",
           "quality_class", "size_class", "diameter_mm", "annotator", "annotator2_quality",
           "capture_device", "lighting", "background", "date", "notes", "orig_label", "timestamp"]

IMG_EXT = {".jpg", ".jpeg", ".png"}
_TS = re.compile(r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})")

# (carpeta -> clase de calidad) por fuente; 'regular' solo existe en AFruitDB, mixed_quality y fotos propias.
KAGGLE_MAP = {"Good": "buena", "Bad": "mala"}
AFRUITDB_MAP = {"1st grade": "buena", "2nd grade": "regular", "3rd grade": "mala"}
# especie -> (carpeta en Kaggle, carpeta en AFruitDB); None = la base no la incluye.
# Pomegranate/Papayas/Tomatoes/Burmese_grape existen en las bases pero no están en la lista de frutas del curso.
SPECIES_DIR = {"apple": ("Apple", "Apple"), "banana": ("Banana", "Banana"), "mango": (None, "Mango"),
               "guava": ("Guava", None), "orange": ("Orange", None), "lime": ("Lime", None)}


def parse_timestamp(name: str):
    """Marca de tiempo embebida en el nombre del archivo (IMG20200728175856 / IMG_20230930_172415)."""
    m = _TS.search(name)
    if not m:
        return pd.NaT
    y, mo, d, h, mi, s = map(int, m.groups())
    try:
        return pd.Timestamp(y, mo, d, h, mi, s)
    except ValueError:
        return pd.NaT


def _files(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


# Sultana et al. (Mendeley, 2022): carpetas FreshX / RottenX; solo fresca/podrida (sin "regular").
SULTANA_SPECIES = {"Apple": "apple", "Banana": "banana", "Orange": "orange", "Guava": "guava",
                   "Grape": "grape", "Strawberry": "strawberry"}   # Jujube y Pomegranate fuera de la lista del curso
SULTANA_MAP = {"Fresh": "buena", "Rotten": "mala"}


# FruitVision: carpeta de especie -> especie del proyecto; calidad por subcarpeta
FRUITVISION_SPECIES = {"Apple": "apple", "Banana": "banana", "Grape": "grape", "Mango": "mango", "Orange": "orange"}
FRUITVISION_MAP = {"Fresh": "buena", "Rotten": "mala"}
# Prefijo de image_id por fuente (las tres primeras letras chocan entre fruitvision y fruits360)
ID_PREFIX = {"fruitvision": "FVI", "fruits360": "F36", "wgisd": "WGI"}


def id_prefix(source: str) -> str:
    return ID_PREFIX.get(source, source[:3].upper())


def _fruitvision_rows(cfg: dict, root: Path):
    rows = []
    froot = root / cfg["paths"]["raw_fruitvision"]
    for name, sp in FRUITVISION_SPECIES.items():
        if sp not in cfg["fruit_species"]:
            continue
        for sub, qc in FRUITVISION_MAP.items():
            for p in _files(froot / name / sub):
                rows.append(dict(source="fruitvision", fruit_species=sp, quality_class=qc, orig_label=sub, path=p))
    return rows


def _fruits360_rows(cfg: dict, root: Path):
    """Selección fija (data/raw/Fruits360/_seleccion.json: especie, carpeta, archivo), hecha con fotogramas
    equiespaciados de cada carpeta de Training. La carpeta (variedad) se guarda en orig_label."""
    import json
    froot = root / cfg["paths"]["raw_fruits360"]
    sel = froot / "_seleccion.json"
    if not sel.exists():
        return []
    rows = []
    for sp, folder, name in json.loads(sel.read_text(encoding="utf-8")):
        p = froot / folder / name
        if p.exists() and (sp == "other" or sp in cfg["fruit_species"]):
            rows.append(dict(source="fruits360", fruit_species=sp, quality_class="", orig_label=folder, path=p))
    return rows


def _wgisd_rows(cfg: dict, root: Path):
    """Recortes de racimos (data/raw/WGISD/_recortes.json: foto de origen, archivo); la foto va en orig_label."""
    import json
    wroot = root / cfg["paths"].get("raw_wgisd", "data/raw/WGISD")
    sel = wroot / "_recortes.json"
    if not sel.exists() or "grape" not in cfg["fruit_species"]:
        return []
    return [dict(source="wgisd", fruit_species="grape", quality_class="", orig_label=src, path=wroot / fn)
            for src, fn in json.loads(sel.read_text(encoding="utf-8")) if (wroot / fn).exists()]


def build_public_manifest(cfg: dict, root: str = ".") -> pd.DataFrame:
    root = Path(root)
    rows = []
    sroot = root / cfg["paths"].get("raw_sultana", "Sultana")
    for name, sp in SULTANA_SPECIES.items():
        if sp not in cfg["fruit_species"]:
            continue
        for prefix, qc in SULTANA_MAP.items():
            for p in _files(sroot / f"{prefix}{name}"):
                rows.append(dict(source="sultana", fruit_species=sp, quality_class=qc, orig_label=prefix, path=p))
    for sp in cfg["fruit_species"]:
        if sp not in SPECIES_DIR:
            continue
        kname, aname = SPECIES_DIR[sp]
        kroot = root / cfg["paths"]["raw_kaggle"]
        for qdir, label in (("Good Quality_Fruits", "Good"), ("Bad Quality_Fruits", "Bad")):
            for p in (_files(kroot / qdir / f"{kname}_{label}") if kname else []):
                rows.append(dict(source="kaggle", fruit_species=sp, quality_class=KAGGLE_MAP[label],
                                 orig_label=f"{label}", path=p))
        aroot = root / cfg["paths"]["raw_afruitdb"] / (aname or "_")
        for grade, qc in AFRUITDB_MAP.items():
            for p in (_files(aroot / grade) if aname else []):
                rows.append(dict(source="afruitdb", fruit_species=sp, quality_class=qc,
                                 orig_label=grade, path=p))
    # Fuentes añadidas en la versión 2: van AL FINAL para no cambiar ids ni grupos de las anteriores
    rows += _fruitvision_rows(cfg, root) + _fruits360_rows(cfg, root) + _wgisd_rows(cfg, root)   # v2.2: WGISD
    df = pd.DataFrame(rows)
    df["filename"] = [p.relative_to(root).as_posix() for p in df.pop("path")]
    df["timestamp"] = [parse_timestamp(Path(f).name) for f in df["filename"]]
    # ID estable: no cambia al agregar especies (se numera dentro de cada fuente+especie)
    df["image_id"] = [f"{id_prefix(s)}_{sp[:3]}_{i:05d}" for s, sp, i in
                      zip(df["source"], df["fruit_species"], df.groupby(["source", "fruit_species"]).cumcount())]
    df["group_id"] = "public"
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    df["fruit_id"] = ""            # se rellena con el grupo de duplicados (dedup.py)
    df["annotator"] = "folder_label"
    return df[COLUMNS]


def write_manifest(cfg: dict, root: str = ".") -> pd.DataFrame:
    df = build_public_manifest(cfg, root)
    out = Path(root) / cfg["paths"]["annotations"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
