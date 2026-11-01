"""Window of the culture results in the score (co-author review, 1 Nov 2026: P. aeruginosa was taken from the anchor year only,
MRSA from t-1 or t). Every combination of the two windows (anchor year t; either of t-1 and t) for P. aeruginosa and MRSA, with
FEV1 z, BMI z and log(1 + IV days, t-1 + t) fixed: patient-grouped repeated CV (AUC with patient-bootstrap CI, difference from the
two-year/two-year model, log-loss, Brier, calibration), temporal validation in the last outcome year (patient-disjoint, as in
scripts/11_temporal_validation.py), cluster-robust odds ratios,
and the event rate by culture history (positive in t only, t-1 only, both, neither).
Output: results/culture_window.json (Supplementary Table S7b)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from _common import setup
from config import N_BOOT, N_REPEATS, SEED
from lib import stats as S
from lib.cv import run_oof
from lib.io import write_json
from lib.models import lr_unpenalised

a, df, OUT = setup(__doc__)
nrep = 2 if a.fast else N_REPEATS; nboot = 200 if a.fast else N_BOOT
CORE = ["FEV1_Z_lag_1", "bmiZscore_lag_1", "log_iv_burden_2yr"]
PA = {"anchor year": "PseudomAer_lag_1", "t−1 or t": "pa_any_2y"}
MRSA = {"anchor year": "MRSA_lag_1", "t−1 or t": "mrsa_any_2y"}
SPECS = {f"Pa {pk}, MRSA {mk}": [CORE[0], CORE[1], pv, mv, CORE[2]] for pk, pv in PA.items() for mk, mv in MRSA.items()}
REF = "Pa t−1 or t, MRSA t−1 or t"
ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))

res = {"reference": REF, "specs": SPECS, "history": {}}
for out in ["y_pex", "y_decline10"]:
    d = df[df[out].notna()].reset_index(drop=True)
    P = {}
    for k, f in SPECS.items():
        rr = run_oof(d, f, out, lr_unpenalised(), proba=True, native_nan=False, n_repeats=nrep, seed=SEED)
        P[k] = rr["oof"].mean(0); y, ids = rr["y"], rr["ids"]
    ty = int(d.target_year.max()); te = d[d.target_year == ty]; tr = d[(d.target_year < ty) & ~d.PatCode.isin(te.PatCode)]
    blk = {"n": len(d), "events": int(d[out].sum()), "test_year": ty, "models": {}}
    for k, f in SPECS.items():
        p = P[k]
        m = sm.Logit(d[out], sm.add_constant(d[f].astype(float).fillna(d[f].median()))).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.PatCode)[0]})
        pt = lr_unpenalised().fit(tr[f].to_numpy(float), tr[out].to_numpy(float)).predict_proba(te[f].to_numpy(float))[:, 1]
        yt, it = te[out].to_numpy(float), te.PatCode.to_numpy()
        blk["models"][k] = {
            "cv": {"auc": S.cluster_bootstrap(ids, S.auc, (y, p), nboot, SEED), "logloss": ll(y, p), "brier": S.brier(y, p),
                   "citl": S.citl(y, p), "slope": S.cal_slope(y, p)},
            "dauc_vs_ref": None if k == REF else S.cluster_bootstrap(ids, lambda yy, a_, b_: S.auc(yy, a_) - S.auc(yy, b_), (y, p, P[REF]), nboot, SEED),
            "dlogloss_vs_ref": ll(y, p) - ll(y, P[REF]),
            "temporal": {"auc": S.cluster_bootstrap(it, S.auc, (yt, pt), nboot, SEED), "o_e": float(yt.mean() / pt.mean()), "slope": S.cal_slope(yt, pt)},
            "aic": float(m.aic),
            "or": {c: {"or": float(np.exp(m.params[c])), "lo": float(np.exp(m.conf_int().loc[c, 0])), "hi": float(np.exp(m.conf_int().loc[c, 1])),
                       "p": float(m.pvalues[c])} for c in f}}
    res[out] = blk
    print(out, {k: (round(v["cv"]["auc"]["point"], 3), round(v["cv"]["logloss"], 4), round(v["temporal"]["auc"]["point"], 3)) for k, v in blk["models"].items()})
# event rate by culture history (descriptive)
d = df[df.y_pex.notna()]
for org, (c1, c2) in {"P. aeruginosa": ("PseudomAer_lag_1", "PseudomAer_lag_2"), "MRSA": ("MRSA_lag_1", "MRSA_lag_2")}.items():
    h = np.select([(d[c1] > 0) & (d[c2] > 0), d[c1] > 0, d[c2] > 0], ["both years", "anchor year only", "year before only"], "neither")
    g = d.groupby(h).y_pex.agg(["size", "sum", "mean"])
    res["history"][org] = {k: {"n": int(r["size"]), "events": int(r["sum"]), "rate": float(r["mean"])} for k, r in g.iterrows()}
    print(org, {k: (v["n"], round(v["rate"], 3)) for k, v in res["history"][org].items()})
write_json(res, OUT / "culture_window.json", "29_culture_window.py")
