"""Statistical drift detection: PSI, two-sample Kolmogorov-Smirnov, and chi-square for categoricals.

Drift = the *inputs/outputs changed*; it is NOT proof that accuracy fell. Performance is judged
separately (monitoring.performance) and only when ground-truth labels exist.
"""
from datetime import datetime, timezone

import numpy as np
from scipy import stats

PSI_WARN, PSI_DRIFT, ALPHA = 0.10, 0.25, 0.05
MIN_CURRENT = 30
EPS = 1e-4
THRESHOLDS = {"psi_warning": PSI_WARN, "psi_drift": PSI_DRIFT, "p_value_alpha": ALPHA, "min_current_samples": MIN_CURRENT}


def _clean_num(values):
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    return arr[np.isfinite(arr)]


def _psi_from_counts(ref_counts, cur_counts):
    r = np.maximum(ref_counts / max(ref_counts.sum(), 1), EPS)
    c = np.maximum(cur_counts / max(cur_counts.sum(), 1), EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref, cur, bins=10):
    ref, cur = _clean_num(ref), _clean_num(cur)
    if len(ref) == 0 or len(cur) == 0:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    inner = edges[1:-1] if len(edges) > 2 else edges[:1]
    nb = len(inner) + 1
    rc = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=nb).astype(float)
    cc = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=nb).astype(float)
    return _psi_from_counts(rc, cc)


def _cat_counts(ref, cur):
    ref = ["__missing__" if v is None else str(v) for v in ref]
    cur = ["__missing__" if v is None else str(v) for v in cur]
    cats = sorted(set(ref) | set(cur))
    rc = np.array([ref.count(c) for c in cats], dtype=float)
    cc = np.array([cur.count(c) for c in cats], dtype=float)
    return cats, rc, cc


def psi_categorical(ref, cur):
    if not len(ref) or not len(cur):
        return None
    _c, rc, cc = _cat_counts(ref, cur)
    return _psi_from_counts(rc, cc)


def ks_test(ref, cur):
    ref, cur = _clean_num(ref), _clean_num(cur)
    if len(ref) < 2 or len(cur) < 2:
        return None, None
    res = stats.ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def chi_square_test(ref, cur):
    cats, rc, cc = _cat_counts(ref, cur)
    keep = (rc + cc) > 0
    table = np.vstack([rc[keep], cc[keep]])
    if table.shape[1] < 2:
        return 0.0, 1.0
    chi2, p, _dof, _exp = stats.chi2_contingency(table)
    return float(chi2), float(p)


def _status(psi, p):
    """PSI measures effect size, the KS/chi-square p-value measures statistical evidence.
    Requiring both avoids flagging PSI sampling noise on small samples (and tiny shifts on huge ones)."""
    if psi is None:
        return "insufficient_data"
    significant = p is not None and p < ALPHA
    if psi >= PSI_DRIFT and significant:
        return "drift"
    if (psi >= PSI_WARN and significant) or psi >= PSI_DRIFT:
        return "warning"
    return "stable"


def numeric_report(name, ref, cur):
    psi = psi_numeric(ref, cur)
    stat, p = ks_test(ref, cur)
    return {"name": name, "kind": "numeric", "psi": psi, "test": "Kolmogorov-Smirnov",
            "statistic": stat, "p_value": p, "status": _status(psi, p)}


def categorical_report(name, ref, cur):
    psi = psi_categorical(ref, cur)
    stat, p = chi_square_test(ref, cur) if len(ref) and len(cur) else (None, None)
    return {"name": name, "kind": "categorical", "psi": psi, "test": "Chi-square",
            "statistic": stat, "p_value": p, "status": _status(psi, p)}


def detect_drift(reference_rows, current_rows, schema, ref_predictions, cur_predictions, problem_type,
                 min_current=MIN_CURRENT):
    now = datetime.now(timezone.utc).isoformat()
    base = {"reference_size": len(reference_rows), "current_size": len(current_rows),
            "thresholds": {**THRESHOLDS, "min_current_samples": min_current}, "generated_at": now,
            "features": [], "prediction_drift": None, "affected_features": []}
    if len(current_rows) < min_current or len(reference_rows) < min_current:
        base.update(status="insufficient_data",
                    message=f"Need at least {min_current} logged predictions to test for drift "
                            f"(have {len(current_rows)} current, {len(reference_rows)} reference). No drift scores computed.")
        return base
    feats = []
    for f in schema:
        ref = [r.get(f["name"]) for r in reference_rows]
        cur = [r.get(f["name"]) for r in current_rows]
        feats.append(numeric_report(f["name"], ref, cur) if f["kind"] == "numeric"
                     else categorical_report(f["name"], ref, cur))
    if problem_type == "classification":
        pred = categorical_report("(prediction)", ref_predictions, cur_predictions)
    else:
        pred = numeric_report("(prediction)", ref_predictions, cur_predictions)
    affected = [r["name"] for r in feats if r["status"] in ("drift", "warning")]
    statuses = [r["status"] for r in feats] + [pred["status"]]
    overall = "drift" if "drift" in statuses else "warning" if "warning" in statuses else "stable"
    base.update(features=feats, prediction_drift=pred, affected_features=affected, status=overall,
                message="Drift describes a change in data distributions. It does not by itself prove the model got worse.")
    return base
