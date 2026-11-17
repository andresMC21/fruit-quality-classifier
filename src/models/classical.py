"""ML tradicional (mín. 2): p. ej. SVM y Random Forest/XGBoost + línea base (mayoritaria, regresión logística).
Hiperparámetros por validación cruzada / GridSearchCV.

Cada modelo es un Pipeline de sklearn: la estandarizacion z = (x - mu) / sigma vive dentro del pipeline, asi
que en la validacion cruzada mu y sigma se calculan solo con los pliegues de entrenamiento (sin fuga).
class_weight="balanced" usa w_k = N / (K * N_k), la misma ponderacion que la perdida de la CNN.
"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS = ("baseline", "logreg", "svm", "rf")


def build_model(name: str, seed: int = 42):
    if name == "baseline":
        # Linea base trivial: siempre la clase mas frecuente de train. Su F1 macro es el piso a superar.
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)
    elif name == "svm":
        # La app muestra probabilidades por clase y el SVM solo da distancias al margen: se calibran con
        # CalibratedClassifierCV (sigmoide de Platt). cv=3 porque "regular" tiene pocos ejemplos.
        # Reemplaza a SVC(probability=True), obsoleto desde scikit-learn 1.9.
        clf = CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced", random_state=seed),
                                     method="sigmoid", cv=3, ensemble=False)
    elif name == "rf":
        # Los arboles no necesitan estandarizar, pero se deja el mismo pipeline para tratar igual a todos.
        # n_jobs=1: GridSearchCV ya reparte en paralelo los pliegues.
        clf = RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=seed)
    else:
        raise ValueError(f"Modelo desconocido: {name}. Opciones: {MODELS}")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])
