"""Clinical exacerbation model without the severity index (Izzet_sorular D; user request: drop P. aeruginosa if it adds nothing).
(a) candidate index-free model (FEV1 z, BMI z, P. aeruginosa, MRSA, ABPA, IV days t-1+t): cluster-robust coefficients;
(b) backward elimination (cluster-robust Wald p > 0.157, i.e. AIC-equivalent for 1 df) REPEATED INSIDE EVERY TRAINING FOLD, so
    the reported cross-validated performance includes the selection step; selection frequency across 25 folds;
(a') culture results and ABPA counted over t-1 and t for every organism (co-author review, 1 Nov 2026); the anchor-year window
     is recorded by out-of-fold log-loss, the four P. aeruginosa x MRSA combinations in scripts/29_culture_window.py;
(c) comparison with LR-5 (severity index, FEV1 <70%), LR-5 with FEV1 <80% in the index, prior-IV alone, the index-free full
    model, IV days as log(1+days), and the selected model + age; same for the >=10-point decline outcome.
Output: results/model_development.json (CLIN = selected variable list, used by all downstream scripts)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, ClassifierMixin
from _common import setup
from config import N_BOOT, N_REPEATS, SEED
from lib import stats as S
from lib.cv import run_oof
from lib.features import ANCHOR_WINDOW, CLIN_CANDIDATES, LR5, LR5_SEV80, PRIOR_IV
from lib.io import write_json
from lib.models import lr_unpenalised

a, df, OUT = setup(__doc__)
nrep = 2 if a.fast else N_REPEATS; nboot = 200 if a.fast else N_BOOT
P_OUT = 0.157


def cluster_logit(d, feats, y_col):
    X = sm.add_constant(d[feats].fillna(d[feats].median())); y = d[y_col]
    return sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.PatCode)[0]})


def backward(Xdf, y, groups, feats, p_out=P_OUT):
    keep = list(feats)
    while len(keep) > 1:
        X = sm.add_constant(Xdf[keep]); m = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
        pv = m.pvalues.drop("const")
        if pv.max() <= p_out:
            break
        keep.remove(pv.idxmax())
    return keep


class BackwardLR(BaseEstimator, ClassifierMixin):
    """Median imputation + backward elimination (cluster-robust Wald) + unpenalised logistic refit. X has a PatCode column."""
    def __init__(self, feats=None, p_out=P_OUT):
        self.feats, self.p_out = feats, p_out

    def fit(self, X, y):
        g = pd.factorize(X["PatCode"])[0]; Xf = X[self.feats].astype(float)
        self.med_ = Xf.median(); Xf = Xf.fillna(self.med_)
        self.kept_ = backward(Xf.reset_index(drop=True), np.asarray(y), g, self.feats, self.p_out)
        self.m_ = sm.Logit(np.asarray(y), sm.add_constant(Xf[self.kept_].to_numpy())).fit(disp=0)
        self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        Xf = X[self.kept_].astype(float).fillna(self.med_[self.kept_])
        p = self.m_.predict(sm.add_constant(Xf.to_numpy(), has_constant="add")); return np.column_stack([1 - p, p])


res = {"p_out": P_OUT}
# functional form of IV days (linearity check, before selection): linear days vs log(1 + days) in the full index-free model,
# compared by out-of-fold log-loss on the primary outcome
LOG_CAND = [c if c != "iv_burden_2yr_lag" else "log_iv_burden_2yr" for c in CLIN_CANDIDATES]
_d = df[df.y_pex.notna()].reset_index(drop=True)
_ll = lambda yy, pp: float(-np.mean(yy * np.log(np.clip(pp, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - pp, 1e-6, 1))))
form = {}
for lab, f in [("linear_days", CLIN_CANDIDATES), ("log1p_days", LOG_CAND)]:
    rr = run_oof(_d, f, "y_pex", lr_unpenalised(), proba=True, native_nan=False, n_repeats=nrep, seed=SEED)
    form[lab] = {"oof_logloss": _ll(rr["y"], rr["oof"].mean(0)), "aic_full_data": float(cluster_logit(_d, f, "y_pex").aic)}
res["iv_functional_form"] = form
if form["log1p_days"]["oof_logloss"] < form["linear_days"]["oof_logloss"]:
    CLIN_CANDIDATES = LOG_CAND
# window of the culture results and ABPA: one window for every organism, the two years t-1 and t used for IV days (co-author
# review, 1 Nov 2026; MRSA alone had first been set to this window by out-of-fold log-loss, P. aeruginosa kept in the anchor year).
# The anchor-year window is recorded for comparison only; it does not change the candidate set
win = {}
for lab, f in [("anchor_year", [ANCHOR_WINDOW.get(c, c) for c in CLIN_CANDIDATES]), ("either_year", CLIN_CANDIDATES)]:
    rr = run_oof(_d, f, "y_pex", lr_unpenalised(), proba=True, native_nan=False, n_repeats=nrep, seed=SEED)
    win[lab] = {"variables": f, "oof_logloss": _ll(rr["y"], rr["oof"].mean(0)), "aic_full_data": float(cluster_logit(_d, f, "y_pex").aic)}
res["culture_window"] = win
print("culture window:", {k: round(v["oof_logloss"], 4) for k, v in win.items()})
res["candidates"] = CLIN_CANDIDATES
print("IV form:", form, "->", CLIN_CANDIDATES)
for out in ["y_pex", "y_decline10"]:
    d = df[df[out].notna()].reset_index(drop=True)
    full = cluster_logit(d, CLIN_CANDIDATES, out)
    sel = backward(d[CLIN_CANDIDATES].fillna(d[CLIN_CANDIDATES].median()), d[out].to_numpy(), pd.factorize(d.PatCode)[0], CLIN_CANDIDATES)
    blk = {"n": len(d), "events": int(d[out].sum()),
           "full_model": {k: {"or": float(np.exp(full.params[k])), "lo": float(np.exp(full.conf_int().loc[k, 0])), "hi": float(np.exp(full.conf_int().loc[k, 1])),
                              "p": float(full.pvalues[k])} for k in CLIN_CANDIDATES},
           "selected_full_data": sel}
    # (b) selection inside each training fold
    bw = run_oof(d, CLIN_CANDIDATES + ["PatCode"], out, BackwardLR(CLIN_CANDIDATES), proba=True, native_nan=True, n_repeats=nrep, seed=SEED, keep_models=True)
    freq = pd.Series([f for (_, _, _, m) in bw["models"] for f in m.kept_]).value_counts() / len(bw["models"])
    blk["selection_frequency_across_folds"] = {f: float(freq.get(f, 0.0)) for f in CLIN_CANDIDATES}
    clin_sel = sel
    specs = {"lr5_v3": LR5, "lr5_severity80": LR5_SEV80, "index_free_full": CLIN_CANDIDATES, "selected": clin_sel,
             "selected_linear_iv": [c if c != "log_iv_burden_2yr" else "iv_burden_2yr_lag" for c in clin_sel],
             "selected_plus_age": clin_sel + ["AgeAtVisit_lag_1"], "selected_plus_adult": clin_sel + ["adult"],
             "selected_without_pa": [c for c in clin_sel if c != "pa_any_2y"],
             "selected_with_pa": list(dict.fromkeys(clin_sel + ["pa_any_2y"])), "prior_iv_logistic": PRIOR_IV}
    P = {"backward_in_fold": bw["oof"].mean(0)}
    y, ids = bw["y"], bw["ids"]
    for k, f in specs.items():
        rr = run_oof(d, f, out, lr_unpenalised(), proba=True, native_nan=False, n_repeats=nrep, seed=SEED)
        assert (rr["ids"] == ids).all(); P[k] = rr["oof"].mean(0)
    ll = lambda yy, pp: float(-np.mean(yy * np.log(np.clip(pp, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - pp, 1e-6, 1))))
    blk["models"] = {k: {"features": specs.get(k, "backward selection within fold"), "auc": S.cluster_bootstrap(ids, S.auc, (y, p), nboot, SEED),
                         "logloss": ll(y, p), "brier": S.brier(y, p), "citl": S.citl(y, p), "slope": S.cal_slope(y, p)} for k, p in P.items()}
    blk["delta_vs_selected"] = {k: {"dauc": S.cluster_bootstrap(ids, lambda yy, a_, b_: S.auc(yy, a_) - S.auc(yy, b_), (y, p, P["selected"]), nboot, SEED),
                                    "dlogloss": ll(y, p) - ll(y, P["selected"])} for k, p in P.items() if k != "selected"}
    blk["selected_fit"] = {k: {"or": float(v)} for k, v in np.exp(cluster_logit(d, clin_sel, out).params).items()}
    res[out] = blk
    print(out, f"n={len(d)} ev={int(d[out].sum())}")
    print("   full model ORs:", {k: (round(v['or'], 3), round(v['p'], 3)) for k, v in blk["full_model"].items()})
    print("   selected:", sel, "| fold freq:", {k[:10]: round(v, 2) for k, v in blk["selection_frequency_across_folds"].items()})
    print("   AUC:", {k: round(v["auc"]["point"], 3) for k, v in blk["models"].items()})
    print("   logloss:", {k: round(v["logloss"], 4) for k, v in blk["models"].items()})
res["CLIN"] = res["y_pex"]["selected_full_data"]
write_json(res, OUT / "model_development.json", "21_model_development.py")
