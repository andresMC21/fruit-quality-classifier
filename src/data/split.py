"""División train/val/test FIJA y sin fuga de datos (misma fruta no puede estar en dos splits).
El test debe contener imágenes propias para medir generalización real."""

def make_splits(annotations_csv: str, seed: int = 42):
    raise NotImplementedError
