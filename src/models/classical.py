"""ML tradicional: línea base (mayoritaria, regresión logística), SVM y Random Forest (+ XGBoost opcional)."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Rejillas de hiperparámetros (validación cruzada agrupada en src/training/tune.py)
GRIDS = {
    "svm": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.003, 0.03]},
    "rf": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 20], "clf__min_samples_leaf": [1, 3]},
    "logreg": {"clf__C": [0.1, 1, 10]},
}


def build_model(name: str, seed: int = 42, probability: bool = True, **params):
    if name == "majority":
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    if name == "logreg":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    elif name == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced", probability=probability, random_state=seed)
    elif name == "rf":
        clf = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=2, random_state=seed)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=2, random_state=seed)
    else:
        raise ValueError(name)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return pipe.set_params(**params) if params else pipe
