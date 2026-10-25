"""Monitoring maths over prediction-log dicts. Never invents performance when labels are missing."""
from collections import Counter

import numpy as np

from . import evaluation

MIN_LABELED = 30
DEGRADE_F1_DROP = 0.05      # absolute drop in weighted F1 vs. test baseline
DEGRADE_RMSE_RISE = 0.20    # relative rise in RMSE vs. test baseline


def latency_stats(values):
    v = [x for x in values if x is not None]
    if not v:
        return {"count": 0, "avg_ms": None, "median_ms": None, "p95_ms": None}
    return {"count": len(v), "avg_ms": float(np.mean(v)), "median_ms": float(np.median(v)),
            "p95_ms": float(np.percentile(v, 95))}


def performance(logs, problem_type, baseline=None):
    ok = [l for l in logs if l["status"] == "success" and l.get("actual_label") not in (None, "")]
    n = len(ok)
    if n == 0:
        return {"available": False, "n_labeled": 0,
                "note": "No actual labels have been recorded yet, so live performance cannot be calculated."}
    if problem_type == "classification":
        yt = [str(l["actual_label"]) for l in ok]
        yp = [str(l["output"]) for l in ok]
        m, _ = evaluation.classification_metrics(yt, yp)
        m = {k: m[k] for k in ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted")}
    else:
        pairs = []
        for l in ok:
            try:
                pairs.append((float(l["actual_label"]), float(l["output"])))
            except (TypeError, ValueError):
                continue
        if not pairs:
            return {"available": False, "n_labeled": 0, "note": "Recorded labels are not numeric."}
        n = len(pairs)
        m = evaluation.regression_metrics([p[0] for p in pairs], [p[1] for p in pairs])
    out = {"available": True, "n_labeled": n, "metrics": m, "degradation": None}
    if n < MIN_LABELED:
        out["note"] = f"Only {n} labelled predictions; at least {MIN_LABELED} are needed before judging degradation."
        return out
    if baseline:
        if problem_type == "classification" and baseline.get("f1_weighted") is not None:
            drop = baseline["f1_weighted"] - m["f1_weighted"]
            out["degradation"] = {"metric": "f1_weighted", "baseline": baseline["f1_weighted"],
                                  "live": m["f1_weighted"], "change": -drop, "degraded": drop > DEGRADE_F1_DROP,
                                  "rule": f"degraded if F1 drops by more than {DEGRADE_F1_DROP}"}
        elif problem_type == "regression" and baseline.get("rmse"):
            rise = (m["rmse"] - baseline["rmse"]) / baseline["rmse"]
            out["degradation"] = {"metric": "rmse", "baseline": baseline["rmse"], "live": m["rmse"],
                                  "change": rise, "degraded": rise > DEGRADE_RMSE_RISE,
                                  "rule": f"degraded if RMSE rises by more than {int(DEGRADE_RMSE_RISE * 100)}%"}
    return out


def summarize_logs(logs, problem_type, baseline=None):
    total = len(logs)
    success = [l for l in logs if l["status"] == "success"]
    failed = total - len(success)
    daily = Counter(l["created_at"].strftime("%Y-%m-%d") for l in logs)
    if problem_type == "classification":
        counts = Counter(str(l["output"]) for l in success)
        dist = {"labels": sorted(counts), "values": [counts[k] for k in sorted(counts)]}
    else:
        vals = [float(l["output"]) for l in success if l["output"] not in (None, "")]
        if vals:
            c, edges = np.histogram(vals, bins=min(10, max(1, len(vals))))
            dist = {"labels": [f"{edges[i]:.3g} – {edges[i + 1]:.3g}" for i in range(len(c))], "values": [int(x) for x in c]}
        else:
            dist = {"labels": [], "values": []}
    versions = Counter(l["model_version"] for l in logs)
    return {
        "total": total, "success": len(success), "failed": failed,
        "success_rate": (len(success) / total) if total else None,
        "failure_rate": (failed / total) if total else None,
        "latency": latency_stats([l["latency_ms"] for l in success]),
        "prediction_distribution": dist,
        "daily_volume": {"labels": sorted(daily), "values": [daily[k] for k in sorted(daily)]},
        "version_usage": {f"v{k}": v for k, v in sorted(versions.items())},
        "performance": performance(logs, problem_type, baseline),
    }
