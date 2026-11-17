"""Búsqueda de hiperparámetros (GridSearchCV / validación cruzada estratificada).

La validacion cruzada es estratificada por clase y AGRUPADA por fruit_id (StratifiedGroupKFold): las
vistas de una misma fruta nunca quedan a la vez en entrenamiento y validacion de un pliegue.
"""
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.models.classical import build_model


def tune(model_name: str, cfg: dict, X, y, groups):
    """Devuelve (mejor modelo reentrenado con todo X, mejores hiperparametros, F1 macro medio en CV)."""
    model = build_model(model_name, cfg["seed"])
    grid = cfg["classical"]["grids"].get(model_name, {})
    cv = StratifiedGroupKFold(n_splits=cfg["classical"]["cv_folds"], shuffle=True, random_state=cfg["seed"])
    search = GridSearchCV(model, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_)
