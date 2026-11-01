"""Next-year lung function (FEV1 % predicted and z-score; FVC as supplementary), written for the clinician question
"how close will next year's value be to the prediction?" (Izzet_sorular B, Figure 2).
Models: persistence (this year's value carried forward); two-year linear model (t, t-1); ridge on the clinical predictors;
CatBoost on the v3 panel (level target); improved machine learning = CatBoost on the engineered panel predicting the CHANGE
from this year (residual learning over persistence), tuned by inner patient-grouped CV; ridge on the engineered panel (change).
Main-text machine learning (1 Nov 2026): 'ml_change' = curated panel -> in-fold selection of 10–15 features by SHAP rank -> Optuna-tuned LightGBM
on the change (lib/mlx.SelectedLGBMRegressor; conformal interval from the inner out-of-fold residuals of the tuned model).
95% prediction intervals: grouped split-conformal (inner patient-grouped 5-fold residuals inside every outer training set).
Clinician metrics: share of prediction-years with |observed − predicted| <= 5 and <= 10 points (% predicted) or 0.5 / 1.0 z.
Outputs: results/fev1_forecast.json, results/oof/fc_*.npy (repeat-averaged OOF prediction and PI half-width)."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from _common import setup
from config import N_BOOT, N_REPEATS, SEED
from lib import stats as S
from lib.features import FE_PANEL, GENOTYPE_FEATURES, MAIN_PANEL, ML_PANEL
from lib.mlx import SelectedLGBMRegressor
from lib.io import write_json
from lib.models import repeated_group_folds, ridge_pipe
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import json

a, df, OUT = setup(__doc__)
nrep = 2 if a.fast else N_REPEATS; nboot = 200 if a.fast else N_BOOT; NTRIAL = 8 if a.fast else 40; NJOBS = 10
CLIN = json.loads((OUT / "model_development.json").read_text())["CLIN"]
GENO = [c for c in GENOTYPE_FEATURES if c in df.columns]
FE = list(dict.fromkeys(FE_PANEL + GENO))
TARGETS = {"y_fev1_pp": ("FEV1_PP_lag_1", "FEV1_PP_lag_2", (5, 10)), "y_fev1_z": ("FEV1_Z_lag_1", "FEV1_Z_lag_2", (0.5, 1.0)),
           "y_fvc_pp": ("FVC_PP_lag_1", "FVC_PP_lag_2", (5, 10)), "y_fvc_z": ("FVC_Z_lag_1", "FVC_Z_lag_2", (0.5, 1.0))}
GRID = [dict(depth=d, l2_leaf_reg=l2) for d in (3, 5) for l2 in (5, 20)]
ALPHA = 0.05


def conformal_q(resid, alpha=ALPHA):
    n = len(resid); k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(np.abs(resid))[k - 1])


class Model:
    """kind: persistence | ar2 | ridge_clin | cb_level | cb_change | ridge_change | ml_change. fit() stores inner grouped-CV residuals."""
    def __init__(self, kind, anchor, lag2):
        self.kind, self.anchor, self.lag2 = kind, anchor, lag2

    def feats(self):
        return {"ar2": [self.anchor, self.lag2], "ridge_clin": list(dict.fromkeys([self.anchor, self.lag2] + CLIN + ["AgeAtVisit_lag_1"])),
                "cb_level": MAIN_PANEL, "cb_change": FE, "ridge_change": FE, "ml_change": ML_PANEL}.get(self.kind, [self.anchor])

    def _base(self, d):
        return d[self.anchor].fillna(d[self.lag2]).fillna(self.anchor_med_).to_numpy(float)

    def _fit1(self, d, y, prm=None):
        k, X = self.kind, d[self.feats()]
        if k == "persistence":
            return None
        if k in ("ar2", "ridge_clin"):
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))).fit(X.to_numpy(float), y)
        if k == "ridge_change":
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 11))).fit(X.to_numpy(float), y - self._base(d))
        p = dict(iterations=500, learning_rate=0.03, random_seed=SEED, verbose=False, allow_writing_files=False, **(prm or {"depth": 6, "l2_leaf_reg": 3}))
        return CatBoostRegressor(**p).fit(X, y - self._base(d) if k == "cb_change" else y)

    def _pred1(self, m, d):
        k = self.kind
        if k == "persistence":
            return self._base(d)
        X = d[self.feats()]
        if k in ("ar2", "ridge_clin"):
            return m.predict(X.to_numpy(float))
        if k == "ridge_change":
            return self._base(d) + m.predict(X.to_numpy(float))
        if k == "ml_change":
            return self._base(d) + m.predict(X)
        return (self._base(d) + m.predict(X)) if k == "cb_change" else m.predict(X)

    def fit(self, d, y, g):
        self.anchor_med_ = float(d[self.anchor].median())
        if self.kind == "ml_change":   # selection + Optuna inside; conformal residuals = inner out-of-fold residuals of the tuned model
            self.m_ = SelectedLGBMRegressor(feats=ML_PANEL, seed=SEED, ntrial=NTRIAL).fit(d[ML_PANEL + ["PatCode"]], y - self._base(d))
            self.prm_ = self.m_.selected_; self.q_ = conformal_q(self.m_.inner_residuals_); return self
        inner = list(GroupKFold(5).split(d, y, g))
        def oof(prm=None):
            o = np.empty(len(y))
            for tr, te in inner:
                o[te] = self._pred1(self._fit1(d.iloc[tr], y[tr], prm), d.iloc[te])
            return o
        self.prm_ = None
        if self.kind == "cb_change":
            best = None
            for prm in GRID:
                o = oof(prm); e = np.mean((y - o) ** 2)
                if best is None or e < best[0]:
                    best = (e, prm, o)
            self.prm_, inner_oof = best[1], best[2]
        else:
            inner_oof = oof()
        self.q_ = conformal_q(y - inner_oof)
        self.m_ = self._fit1(d, y, self.prm_); return self

    def predict(self, d):
        return self._pred1(self.m_, d)


KINDS = ["persistence", "ar2", "ridge_clin", "cb_level", "cb_change", "ridge_change", "ml_change"]


def _ml_fold(d, y, ids, r, tr, te, anchor, lag2):
    import warnings; warnings.filterwarnings("ignore")
    m = Model("ml_change", anchor, lag2).fit(d.iloc[tr], y[tr], ids[tr])
    return r, te, m.predict(d.iloc[te]), m.q_, m.prm_
res = {"n_repeats": nrep, "alpha": ALPHA, "engineered_panel": FE, "clin": CLIN}
for tgt, (anchor, lag2, tol) in TARGETS.items():
    d = df[df[tgt].notna()].reset_index(drop=True); y = d[tgt].to_numpy(float); ids = d.PatCode.to_numpy()
    pred = {k: np.zeros((nrep, len(d))) for k in KINDS}; hw = {k: np.zeros((nrep, len(d))) for k in KINDS}; prm = []
    folds = list(repeated_group_folds(ids, y, 5, nrep, SEED, regression=True))
    from joblib import Parallel, delayed
    ml_sel = []
    for r, te, p_, q_, s_ in Parallel(n_jobs=NJOBS)(delayed(_ml_fold)(d, y, ids, r, tr, te, anchor, lag2) for r, f, tr, te in folds):
        pred["ml_change"][r, te] = p_; hw["ml_change"][r, te] = q_; ml_sel.append(s_)
    for r, f, tr, te in folds:
        for k in [k for k in KINDS if k != "ml_change"]:
            if tgt.startswith("y_fvc") and k in ("cb_level", "ridge_clin"):
                continue
            m = Model(k, anchor, lag2).fit(d.iloc[tr], y[tr], ids[tr])
            pred[k][r, te] = m.predict(d.iloc[te]); hw[k][r, te] = m.q_
            if k == "cb_change":
                prm.append(m.prm_)
    kinds = [k for k in KINDS if pred[k].any()]
    blk = {"n_rows": len(d), "n_patients": int(d.PatCode.nunique()), "tolerances": tol, "models": {},
           "observed_change": {"median": float(np.median(y - d[anchor])), "iqr": [float(x) for x in np.quantile(y - d[anchor].fillna(d[lag2]), [.25, .75])]},
           "cb_change_selected_params": pd.Series([str(p) for p in prm]).value_counts().to_dict(),
           "ml_change_selection": {"n_folds": len(ml_sel), "frequency": pd.Series([c for s_ in ml_sel for c in s_]).value_counts().div(len(ml_sel)).round(3).to_dict(),
                                   "n_selected": {"median": float(np.median([len(s_) for s_ in ml_sel])), "min": min(map(len, ml_sel)), "max": max(map(len, ml_sel))}}}
    for k in kinds:
        p = pred[k].mean(0); h = hw[k].mean(0); e = np.abs(y - p)
        np.save(OUT / "oof" / f"fc_{tgt}_{k}_pred.npy", p); np.save(OUT / "oof" / f"fc_{tgt}_{k}_hw.npy", h)
        cov_rep = [float(np.mean(np.abs(y - pred[k][r]) <= hw[k][r])) for r in range(nrep)]
        blk["models"][k] = {
            "r2": S.cluster_bootstrap(ids, S.r2, (y, p), nboot, SEED), "mae": S.cluster_bootstrap(ids, S.mae, (y, p), nboot, SEED), "rmse": S.rmse(y, p),
            f"within_{tol[0]}": S.cluster_bootstrap(ids, lambda yy, pp: float(np.mean(np.abs(yy - pp) <= tol[0])), (y, p), nboot, SEED),
            f"within_{tol[1]}": S.cluster_bootstrap(ids, lambda yy, pp: float(np.mean(np.abs(yy - pp) <= tol[1])), (y, p), nboot, SEED),
            "pi95_halfwidth_median": float(np.median(h)), "pi95_coverage": float(np.mean(cov_rep)),
            "r2_across_repeats_sd": float(np.std([S.r2(y, pred[k][r]) for r in range(nrep)], ddof=1)) if nrep > 1 else 0.0,
            "by_age_group": {g: {"n": int(m.sum()), "r2": S.r2(y[m], p[m]), "mae": S.mae(y[m], p[m]),
                                 f"within_{tol[0]}": float(np.mean(e[m] <= tol[0])), "pi95_coverage": float(np.mean(e[m] <= h[m]))}
                             for g, m in [("child", d.adult.to_numpy() == 0), ("adult", d.adult.to_numpy() == 1)]},
            "coverage_by_anchor_tertile": {str(i): float(np.mean((e <= h)[b])) for i, b in
                                           enumerate([d[anchor].rank(pct=True).to_numpy() <= 1 / 3, (d[anchor].rank(pct=True).to_numpy() > 1 / 3) & (d[anchor].rank(pct=True).to_numpy() <= 2 / 3), d[anchor].rank(pct=True).to_numpy() > 2 / 3])}}
    np.save(OUT / "oof" / f"fc_{tgt}_y.npy", y); np.save(OUT / "oof" / f"fc_{tgt}_ids.npy", ids)
    np.save(OUT / "oof" / f"fc_{tgt}_anchor.npy", d[anchor].to_numpy(float)); np.save(OUT / "oof" / f"fc_{tgt}_adult.npy", d.adult.to_numpy())
    P = {k: pred[k].mean(0) for k in kinds}
    blk["delta_vs_persistence"] = {k: {
        "mae": S.cluster_bootstrap(ids, lambda yy, a_, b_: S.mae(yy, a_) - S.mae(yy, b_), (y, P[k], P["persistence"]), nboot, SEED),
        f"within_{tol[0]}": S.cluster_bootstrap(ids, lambda yy, a_, b_: float(np.mean(np.abs(yy - a_) <= tol[0]) - np.mean(np.abs(yy - b_) <= tol[0])), (y, P[k], P["persistence"]), nboot, SEED),
        "r2": S.cluster_bootstrap(ids, lambda yy, a_, b_: S.r2(yy, a_) - S.r2(yy, b_), (y, P[k], P["persistence"]), nboot, SEED)} for k in kinds if k != "persistence"}
    best = max([k for k in kinds if k != "persistence"], key=lambda k: blk["models"][k]["r2"]["point"])
    blk["best_model"] = best
    res[tgt] = blk
    print(tgt, f"n={len(d)}", {k: (round(v["r2"]["point"], 3), round(v["mae"]["point"], 2), round(v[f"within_{tol[0]}"]["point"], 3),
                                    round(v[f"within_{tol[1]}"]["point"], 3), round(v["pi95_halfwidth_median"], 2), round(v["pi95_coverage"], 3)) for k, v in blk["models"].items()}, "best:", best)
write_json(res, OUT / "fev1_forecast.json", "22_fev1_forecast.py")
