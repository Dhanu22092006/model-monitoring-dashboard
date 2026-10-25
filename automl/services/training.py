"""Genuine scikit-learn training. Preprocessing lives inside the Pipeline and is fitted on the
training split only (no leakage); validation and test splits are only ever transformed."""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import datasets, evaluation
from .prediction import build_feature_schema, predict_row
from .utils import to_native

# key -> (label, estimator factory, needs feature scaling)
CLASSIFIERS = {
    "logistic_regression": ("Logistic Regression", lambda rs: LogisticRegression(max_iter=2000, random_state=rs), True),
    "decision_tree": ("Decision Tree Classifier", lambda rs: DecisionTreeClassifier(random_state=rs), False),
    "random_forest": ("Random Forest Classifier", lambda rs: RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=1), False),
    "knn": ("K-Nearest Neighbors", lambda rs: KNeighborsClassifier(), True),
    "svc": ("Support Vector Classifier", lambda rs: SVC(probability=True, random_state=rs), True),
}
REGRESSORS = {
    "linear_regression": ("Linear Regression", lambda rs: LinearRegression(), True),
    "decision_tree": ("Decision Tree Regressor", lambda rs: DecisionTreeRegressor(random_state=rs), False),
    "random_forest": ("Random Forest Regressor", lambda rs: RandomForestRegressor(n_estimators=100, random_state=rs, n_jobs=1), False),
    "ridge": ("Ridge Regression", lambda rs: Ridge(alpha=1.0, random_state=rs), True),
}
REFERENCE_SAMPLE = 500


def registry(problem_type):
    return CLASSIFIERS if problem_type == "classification" else REGRESSORS


def algorithm_choices():
    """[(value, label, problem_type)] where value = '<problem_type>:<key>'."""
    out = []
    for pt in ("classification", "regression"):
        for key, (label, _f, _s) in registry(pt).items():
            out.append((f"{pt}:{key}", label, pt))
    return out


def build_pipeline(numeric_cols, categorical_cols, estimator, scale):
    transformers = []
    if numeric_cols:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), list(numeric_cols)))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical_cols)))
    return Pipeline([("prep", ColumnTransformer(transformers, remainder="drop")), ("model", estimator)])


def _normalise(X, categorical_cols, numeric_cols):
    X = X.copy()
    for c in categorical_cols:
        X[c] = X[c].astype(object).map(lambda v: np.nan if pd.isna(v) else str(v))
    for c in numeric_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def prepare_data(df, target, problem_type, random_state=42, test_size=0.2, val_size=0.2):
    """Validate, clean and split once, so every algorithm sees identical train/val/test rows."""
    datasets.validate_target(df, target, problem_type)
    df = df.loc[df[target].notna()].reset_index(drop=True)
    y = df[target].astype(str) if problem_type == "classification" else df[target].astype(float)
    X = df.drop(columns=[target])
    dropped = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]  # constant / empty columns carry no signal
    X = X.drop(columns=dropped)
    if X.shape[1] == 0:
        raise datasets.DatasetError("No usable feature columns remain after removing constant columns.")
    kinds = {c: datasets.column_kind(X[c]) for c in X.columns}
    numeric = [c for c, k in kinds.items() if k == "numeric"]
    categorical = [c for c, k in kinds.items() if k == "categorical"]
    X = _normalise(X, categorical, numeric)

    strat = y if problem_type == "classification" else None
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)
    strat2 = y_tmp if problem_type == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size / (1 - test_size), random_state=random_state, stratify=strat2)
    return {
        "problem_type": problem_type, "target": target, "random_state": random_state,
        "numeric": numeric, "categorical": categorical, "dropped_columns": dropped,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "classes": sorted(y.unique().tolist()) if problem_type == "classification" else [],
        "schema": build_feature_schema(X_train),
        "split_info": {"train": len(X_train), "validation": len(X_val), "test": len(X_test),
                       "random_state": random_state, "stratified": problem_type == "classification"},
    }


def _score(pipe, X, y, problem_type, classes):
    pred = pipe.predict(X)
    if problem_type == "classification":
        proba = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") else None
        # predict_proba columns follow pipe.classes_, which equals sorted(classes)
        return evaluation.classification_metrics(y, pred, proba, classes)
    return evaluation.regression_metrics(y, pred), {}


def measure_latency_ms(pipe, schema, X_test, problem_type, classes, repeats=15):
    row = {k: (None if pd.isna(v) else v) for k, v in X_test.iloc[0].to_dict().items()}
    times = [predict_row(pipe, schema, row, problem_type, classes)["latency_ms"] for _ in range(repeats)]
    return float(np.median(times))


def fit_and_evaluate(prep, algorithm):
    problem_type = prep["problem_type"]
    label, factory, scale = registry(problem_type)[algorithm]
    rs = prep["random_state"]
    pipe = build_pipeline(prep["numeric"], prep["categorical"], factory(rs), scale)
    t0 = time.perf_counter()
    pipe.fit(prep["X_train"], prep["y_train"])
    train_seconds = time.perf_counter() - t0
    val_metrics, _ = _score(pipe, prep["X_val"], prep["y_val"], problem_type, prep["classes"])
    test_metrics, details = _score(pipe, prep["X_test"], prep["y_test"], problem_type, prep["classes"])
    latency = measure_latency_ms(pipe, prep["schema"], prep["X_test"], problem_type, prep["classes"])
    ref = prep["X_train"].sample(n=min(REFERENCE_SAMPLE, len(prep["X_train"])), random_state=rs)
    reference = {"rows": to_native(ref.to_dict("records")),
                 "predictions": to_native(pipe.predict(ref).tolist())}
    return {
        "pipeline": pipe, "algorithm": algorithm, "algorithm_label": label,
        "metrics": test_metrics, "validation_metrics": val_metrics, "details": details,
        "train_seconds": train_seconds, "latency_ms": latency, "reference": reference,
    }
