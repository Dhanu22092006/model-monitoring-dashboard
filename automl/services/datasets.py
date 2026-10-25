"""CSV validation, profiling and target helpers. Pure pandas, no Django."""
import io

import numpy as np
import pandas as pd

from .utils import to_native

MIN_ROWS = 20
MAX_COLUMNS = 200
MIN_CLASS_COUNT = 5


class DatasetError(ValueError):
    """Raised for any user-correctable dataset problem."""


def read_csv_bytes(data: bytes, max_bytes: int) -> pd.DataFrame:
    if not data:
        raise DatasetError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise DatasetError(f"File is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DatasetError("File is not UTF-8 text. Please upload a plain CSV file.")
    if "\x00" in text:
        raise DatasetError("File looks binary, not a CSV.")
    try:
        df = pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DatasetError(f"Malformed CSV: {exc}")
    df.columns = [str(c).strip() for c in df.columns]
    validate_frame(df)
    return df


def validate_frame(df: pd.DataFrame) -> None:
    if df.shape[1] < 2:
        raise DatasetError("A dataset needs at least two columns (features and a target).")
    if df.shape[1] > MAX_COLUMNS:
        raise DatasetError(f"Too many columns (maximum {MAX_COLUMNS}).")
    if df.shape[0] < MIN_ROWS:
        raise DatasetError(f"Too few rows: {df.shape[0]} (minimum {MIN_ROWS}).")
    bad = [c for c in df.columns if not c or c.startswith("Unnamed:")]
    if bad:
        raise DatasetError("Every column needs a header name (found blank/unnamed headers).")
    if df.columns.duplicated().any():
        raise DatasetError("Duplicate column names are not allowed.")


def column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


def profile_dataframe(df: pd.DataFrame) -> dict:
    features = []
    for col in df.columns:
        s = df[col]
        features.append({
            "name": col, "dtype": str(s.dtype), "kind": column_kind(s),
            "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True)),
        })
    numeric = [f["name"] for f in features if f["kind"] == "numeric"]
    categorical = [f["name"] for f in features if f["kind"] == "categorical"]
    describe = {}
    if numeric:
        describe = df[numeric].describe().T.round(4).to_dict("index")
    return to_native({
        "rows": int(df.shape[0]), "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "features": features, "numeric": numeric, "categorical": categorical,
        "describe": describe,
    })


def infer_problem_type(series: pd.Series) -> str:
    if column_kind(series) == "categorical":
        return "classification"
    if pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= 20:
        return "classification"
    return "regression"


def validate_target(df: pd.DataFrame, target: str, problem_type: str) -> None:
    if problem_type not in ("classification", "regression"):
        raise DatasetError("Problem type must be 'classification' or 'regression'.")
    if target not in df.columns:
        raise DatasetError(f"Target column '{target}' does not exist in the dataset.")
    y = df[target].dropna()
    if y.empty:
        raise DatasetError("The target column has no values.")
    if len(y) < MIN_ROWS:
        raise DatasetError(f"Only {len(y)} rows have a target value (minimum {MIN_ROWS}).")
    if problem_type == "regression":
        if column_kind(df[target]) != "numeric":
            raise DatasetError("Regression needs a numeric target column.")
        if y.nunique() < 3:
            raise DatasetError("Regression target has fewer than 3 distinct values.")
    else:
        counts = y.astype(str).value_counts()
        if len(counts) < 2:
            raise DatasetError("Classification target needs at least two classes.")
        if len(counts) > 50:
            raise DatasetError("Classification target has more than 50 classes; choose another target or regression.")
        small = counts[counts < MIN_CLASS_COUNT]
        if len(small):
            raise DatasetError(
                f"Class '{small.index[0]}' has only {int(small.iloc[0])} rows (minimum {MIN_CLASS_COUNT} per class)."
            )


def target_distribution(df: pd.DataFrame, target: str, problem_type: str) -> dict:
    y = df[target].dropna()
    if problem_type == "classification":
        counts = y.astype(str).value_counts().sort_index()
        return {"labels": list(counts.index), "values": [int(v) for v in counts.values]}
    counts, edges = np.histogram(y.astype(float), bins=10)
    labels = [f"{edges[i]:.3g} – {edges[i + 1]:.3g}" for i in range(len(counts))]
    return {"labels": labels, "values": [int(v) for v in counts]}
