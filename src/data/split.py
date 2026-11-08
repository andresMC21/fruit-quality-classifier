"""Division train/val/test FIJA, AGRUPADA y estratificada.

- Grupo = fruit_id (fotos propias) o grupo de duplicados de dedup.py (Kaggle). Un grupo nunca aparece en dos splits.
- Estratificar por quality_class (StratifiedGroupKFold o GroupShuffleSplit repetido hasta balancear).
- El test incluye una fraccion de frutas PROPIAS (config: split.test_own), para medir la generalizacion a datos locales.
- Guardar el resultado en data/processed/splits.csv (se versiona) y no regenerarlo despues de mirar el test.
"""


def make_splits(annotations_csv: str, cfg: dict):
    raise NotImplementedError
