"""Helpers shared by the service modules (no Django imports)."""
import math
from datetime import date, datetime

import numpy as np
import pandas as pd


def to_native(o):
    """Recursively convert numpy/pandas values into JSON-safe Python values (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_native(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_native(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if o is pd.NaT or o is pd.NA:
        return None
    if isinstance(o, (datetime, date, pd.Timestamp)):
        return o.isoformat()
    return o
