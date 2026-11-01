"""Calibration of out-of-fold probabilities: CITL (slope fixed at 1), calibration slope, Brier + Murphy decomposition,
and flexible-calibration summaries ICI / E50 / E90 (LOWESS), all with patient-level bootstrap CIs (R1-m4)."""
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from _common import setup
from config import N_BOOT, SEED
from lib import stats as S
from lib.io import write_json

a, df, OUT = setup(__doc__)
O = OUT / "oof"; nboot = 200 if a.fast else N_BOOT


def ici_stats(y, p):
    sm = lowess(y, p, frac=2 / 3, it=0, return_sorted=False)
    e = np.abs(sm - p)
    return float(np.mean(e)), float(np.median(e)), float(np.quantile(e, 0.9))


VERSIONS = [("clin", "raw"), ("clin", "cal"), ("ml", "raw"), ("ml_ownshadow", "raw"), ("ml_maxshadow", "raw"), ("lr5", "raw"), ("catboost", "raw"), ("catboost", "cal"), ("catboost_tuned", "raw"),
            ("catboost_fe_tuned", "raw"), ("prior_iv_logistic", "raw")]
res = {}
for out in ["y_pex", "y_decline10"]:
    y = np.load(O / f"clf_{out}_y.npy"); ids = np.load(O / f"clf_{out}_ids.npy", allow_pickle=True); res[out] = {}
    for k, kind in VERSIONS:
        f = O / f"clf_{out}_{k}_{kind}.npy"
        if not f.exists():
            continue
        p = np.load(f)
        r = {n: S.cluster_bootstrap(ids, fn, (y, p), nboot, SEED) for n, fn in [("citl", S.citl), ("slope", S.cal_slope), ("brier", S.brier)]}
        r["ici"] = S.cluster_bootstrap(ids, lambda yy, pp: ici_stats(yy, pp)[0], (y, p), nboot, SEED)
        _, r["e50"], r["e90"] = ici_stats(y, p)
        r["decomp"] = S.brier_decomp(y, p); r["mean_pred"] = float(p.mean()); r["obs_rate"] = float(y.mean())
        res[out][f"{k}_{kind}"] = r
    print(out, {k: (round(v['citl']['point'], 2), round(v['slope']['point'], 2), round(v['ici']['point'], 3)) for k, v in res[out].items()})
write_json(res, OUT / "calibration_summary.json", "04_calibration.py")
