"""Statistics helpers: patient-level (cluster) resampling, calibration, DCA, FDR."""
from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score

EPS = 1e-6


def _cluster_index(ids: np.ndarray):
    uniq, inv = np.unique(ids, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    return members


def cluster_bootstrap(ids, stat_fn, arrays, n_boot=1000, seed=42, alpha=0.05):
    """Percentile CI resampling PATIENTS with replacement. stat_fn(*arrays_subset) -> float."""
    ids = np.asarray(ids)
    members = _cluster_index(ids)
    rng = np.random.default_rng(seed)
    point = stat_fn(*arrays)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[k] for k in pick])
        try:
            v = stat_fn(*(a[idx] for a in arrays))
        except (ValueError, IndexError, np.linalg.LinAlgError):
            continue
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2]) if len(vals) else (np.nan, np.nan)
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot_ok": int(len(vals))}


def patient_block_permutation(y, ids, rng):
    """Permute outcome vectors between patients with the SAME number of rows (keeps within-patient structure)."""
    y = np.asarray(y); ids = np.asarray(ids)
    out = y.copy()
    members = _cluster_index(ids)
    by_size: dict[int, list[np.ndarray]] = {}
    for m in members:
        by_size.setdefault(len(m), []).append(m)
    for size, groups in by_size.items():
        perm = rng.permutation(len(groups))
        for g, pg in zip(groups, perm):
            out[g] = y[groups[pg]]
    return out


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def citl(y, p):
    """Calibration-in-the-large: intercept with logit(p) as offset (slope fixed at 1), unpenalised."""
    X = np.ones((len(y), 1))
    res = sm.GLM(y, X, family=sm.families.Binomial(), offset=logit(p)).fit()
    return float(res.params[0])


def cal_slope(y, p):
    X = sm.add_constant(logit(p))
    res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    return float(res.params[1])


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def brier_decomp(y, p, n_bins=10):
    """Murphy decomposition on quantile bins: reliability (lower=better), resolution (higher=better), uncertainty."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ybar = y.mean(); rel = res = 0.0
    for k in np.unique(b):
        m = b == k; w = m.mean()
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - ybar) ** 2
    return {"reliability": float(rel), "resolution": float(res), "uncertainty": float(ybar * (1 - ybar))}


def net_benefit(y, p, t):
    y = np.asarray(y); n = len(y)
    pos = p >= t
    tp = np.sum(pos & (y == 1)); fp = np.sum(pos & (y == 0))
    return float(tp / n - fp / n * t / (1 - t))


def net_benefit_all(y, t):
    prev = np.mean(y)
    return float(prev - (1 - prev) * t / (1 - t))


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p); ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n); out[order] = np.minimum(q, 1.0)
    return out


# metric shortcuts (signature: y, p)
def auc(y, p): return float(roc_auc_score(y, p))
def auprc(y, p): return float(average_precision_score(y, p))
def r2(y, p): return float(r2_score(y, p))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(y - p)))
