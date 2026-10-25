"""Feature schema, input validation and real single-row inference."""
import math
import time

import numpy as np
import pandas as pd

MAX_CHOICES = 50


def build_feature_schema(X: pd.DataFrame):
    schema = []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            schema.append({"name": col, "kind": "numeric", "min": float(s.min()), "max": float(s.max()),
                           "median": float(s.median())})
        else:
            vals = sorted({str(v) for v in s.dropna().unique()})
            schema.append({"name": col, "kind": "categorical",
                           "choices": vals if len(vals) <= MAX_CHOICES else None})
    return schema


def validate_input(schema, raw: dict):
    """Return (clean_row, errors). Blank values are allowed (the fitted imputer handles them)."""
    clean, errors = {}, {}
    for f in schema:
        name, val = f["name"], raw.get(f["name"])
        if val is None or (isinstance(val, str) and val.strip() == ""):
            clean[name] = None
            continue
        if f["kind"] == "numeric":
            try:
                num = float(val)
            except (TypeError, ValueError):
                errors[name] = "Enter a number."
                continue
            if not math.isfinite(num):
                errors[name] = "Enter a finite number."
                continue
            clean[name] = num
        else:
            sval = str(val).strip()
            if f.get("choices") is not None and sval not in f["choices"]:
                errors[name] = "Choose one of: " + ", ".join(f["choices"][:8]) + ("…" if len(f["choices"]) > 8 else "")
                continue
            if len(sval) > 200:
                errors[name] = "Value too long."
                continue
            clean[name] = sval
    return clean, errors


def _frame(schema, row):
    data = {}
    for f in schema:
        v = row.get(f["name"])
        if f["kind"] == "numeric":
            data[f["name"]] = [np.nan if v is None else float(v)]
        else:
            data[f["name"]] = pd.Series([np.nan if v is None else str(v)], dtype=object)
    return pd.DataFrame(data)


def predict_row(pipeline, schema, row, problem_type, classes):
    X = _frame(schema, row)
    t0 = time.perf_counter()
    pred = pipeline.predict(X)[0]
    proba = None
    if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
        p = pipeline.predict_proba(X)[0]
        proba = {str(c): float(v) for c, v in zip(pipeline.classes_, p)}
    latency_ms = (time.perf_counter() - t0) * 1000.0
    output = str(pred) if problem_type == "classification" else round(float(pred), 6)
    return {"prediction": output, "probabilities": proba, "latency_ms": latency_ms}
