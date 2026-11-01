"""Machine-learning experiments (authors' request, 1 Nov 2026: try to make machine learning beat the clinical model, with
nested cross-validation and Optuna/grid tuning). Every tuning step runs inside the outer training fold (nested CV: outer =
the same 5 x 5 patient-grouped folds as the clinical model; inner = 4-fold patient-grouped CV of the outer training data), so
the outer predictions are honest. All configurations below were declared before running and all are reported.

IV-treated exacerbation (y_pex), 71-variable engineered panel unless stated:
  lgbm_optuna        LightGBM, Optuna TPE (40 trials) over trees, depth, leaves, learning rate, sampling and L1/L2 penalties
  lgbm_monotone      the same with monotone constraints in the clinically known direction (FEV1, BMI down; IV days, infection up)
  lgbm_offset        LightGBM boosted FROM the clinical model (its logit as init_score): learns only corrections to it
  lgbm_clin5         LightGBM on the five clinical-model variables only (non-linearity and interactions, no extra data)
  enet_grid          elastic-net logistic regression, grid search over C (9 values) x L1 ratio (0, 0.5, 1)
  ebm                Explainable Boosting Machine (glass-box GAM with 5 pairwise interactions)
  avg_clin_lgbm      mean of the clinical-model and lgbm_optuna probabilities
  avg_clin_ebm       mean of the clinical-model and EBM probabilities
Next-year FEV1 % predicted (change target, prediction = anchor + change):
  lgbm_reg_optuna    LightGBM, Optuna (40 trials) including the loss (squared, Huber, absolute), inner metric MAE
  lgbm_reg_offset    LightGBM boosted from a ridge model of the clinical variables
  avg_ridge_lgbm     mean of the ridge base and lgbm_reg_optuna
Output: results/ml_experiments.json, results/oof/mlx_*.npy."""
import warnings
from functools import partial

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from _common import setup
from config import N_BOOT, N_REPEATS, SEED
from lib import stats as S
from lib.features import CLIN, FE_PANEL, GENOTYPE_FEATURES
from lib.io import write_json
from lib.mlx import fit_ebm as _fit_ebm, fit_lgb_clf as _fit_lgb_clf, inner_splits, lgb_space, ll
from lib.models import lr_unpenalised, repeated_group_folds

warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
a, df, OUT = setup(__doc__)
nrep = 2 if a.fast else N_REPEATS; nboot = 200 if a.fast else N_BOOT; NTRIAL = 8 if a.fast else 40; NJOBS = 10
FE = list(dict.fromkeys(FE_PANEL + [g for g in GENOTYPE_FEATURES if g in df.columns] + CLIN))
fit_lgb_clf = partial(_fit_lgb_clf, ntrial=NTRIAL)
fit_ebm = lambda Xtr, ytr, gtr, Xte, seed: (_fit_ebm(Xtr, ytr, Xte, seed, FE), {})


def fit_enet(Xtr, ytr, gtr, Xte, seed):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000))
    gs = GridSearchCV(pipe, {"logisticregression__C": np.logspace(-3, 1, 9), "logisticregression__l1_ratio": [0.0, 0.5, 1.0]},
                      scoring="neg_log_loss", cv=inner_splits(ytr, gtr, seed), n_jobs=1).fit(Xtr[FE].to_numpy(float), ytr)
    return gs.predict_proba(Xte[FE].to_numpy(float))[:, 1], {k.split("__")[1]: float(v) for k, v in gs.best_params_.items()}


def one_fold_clf(r, f, tr, te, d, y, g):
    optuna.logging.set_verbosity(optuna.logging.WARNING); warnings.filterwarnings("ignore")
    seed = SEED + 100 * r + f; Xtr, Xte = d.iloc[tr], d.iloc[te]; out, prm = {}, {}
    out["clin"] = lr_unpenalised().fit(Xtr[CLIN].to_numpy(float), y[tr]).predict_proba(Xte[CLIN].to_numpy(float))[:, 1]
    out["lgbm_optuna"], prm["lgbm_optuna"] = fit_lgb_clf(Xtr, y[tr], g[tr], Xte, seed, FE)
    out["lgbm_monotone"], prm["lgbm_monotone"] = fit_lgb_clf(Xtr, y[tr], g[tr], Xte, seed, FE, mono=True)
    out["lgbm_offset"], prm["lgbm_offset"] = fit_lgb_clf(Xtr, y[tr], g[tr], Xte, seed, FE, offset=True)
    out["lgbm_clin5"], prm["lgbm_clin5"] = fit_lgb_clf(Xtr, y[tr], g[tr], Xte, seed, CLIN, mono=True)
    out["enet_grid"], prm["enet_grid"] = fit_enet(Xtr, y[tr], g[tr], Xte, seed)
    out["ebm"], _ = fit_ebm(Xtr, y[tr], g[tr], Xte, seed)
    out["avg_clin_lgbm"] = (out["clin"] + out["lgbm_optuna"]) / 2
    out["avg_clin_ebm"] = (out["clin"] + out["ebm"]) / 2
    return r, te, out, prm


d = df[df.y_pex.notna()].reset_index(drop=True); y = d.y_pex.to_numpy(float); ids = d.PatCode.to_numpy(); g = pd.factorize(ids)[0]
folds = list(repeated_group_folds(ids, y, 5, nrep, SEED))
runs = Parallel(n_jobs=NJOBS, verbose=0)(delayed(one_fold_clf)(r, f, tr, te, d, y, g) for r, f, tr, te in folds)
keys = list(runs[0][2]); P = {k: np.full((nrep, len(d)), np.nan) for k in keys}; params = {k: [] for k in keys}
for r, te, out, prm in runs:
    for k in keys:
        P[k][r, te] = out[k]
    for k, v in prm.items():
        params[k].append(v)
P = {k: v.mean(0) for k, v in P.items()}
ref_cb = OUT / "oof" / "clf_y_pex_catboost_fe_tuned_raw.npy"
if ref_cb.exists() and len(np.load(ref_cb)) == len(y):
    P["catboost_fe_tuned (main pipeline)"] = np.load(ref_cb)
res = {"n": len(d), "events": int(y.sum()), "n_features": len(FE), "optuna_trials": NTRIAL, "outer": f"{nrep} x 5 patient-grouped", "inner": "4-fold patient-grouped",
       "classification": {}}
dauc = lambda yy, a_, b_: S.auc(yy, a_) - S.auc(yy, b_)
for k, p in P.items():
    np.save(OUT / "oof" / f"mlx_y_pex_{k.split(' ')[0]}.npy", p)
    res["classification"][k] = {"auc": S.cluster_bootstrap(ids, S.auc, (y, p), nboot, SEED), "logloss": ll(y, p), "brier": S.brier(y, p),
                                "slope": S.cal_slope(y, p), "citl": S.citl(y, p), "nb_020": S.net_benefit(y, p, 0.20),
                                **({} if k == "clin" else {"delta_auc_vs_clin": S.cluster_bootstrap(ids, dauc, (y, p, P["clin"]), nboot, SEED),
                                                           "delta_logloss_vs_clin": ll(y, p) - ll(y, P["clin"]),
                                                           "delta_nb_020_vs_clin": S.cluster_bootstrap(ids, lambda yy, a_, b_: S.net_benefit(yy, a_, 0.2) - S.net_benefit(yy, b_, 0.2), (y, p, P["clin"]), nboot, SEED)})}
res["classification_selected_params"] = {k: pd.Series([str(sorted(v.items())) for v in vs]).value_counts().head(3).to_dict() for k, vs in params.items() if vs}
for k, v in res["classification"].items():
    dd = v.get("delta_auc_vs_clin")
    print(f"{k:34s} AUC {v['auc']['point']:.3f} ({v['auc']['lo']:.3f}-{v['auc']['hi']:.3f}) LL {v['logloss']:.4f} slope {v['slope']:.2f}"
          + (f"  dAUC {dd['point']:+.3f} ({dd['lo']:+.3f},{dd['hi']:+.3f}) dLL {v['delta_logloss_vs_clin']:+.4f}" if dd else ""), flush=True)


# ---------------- next-year FEV1 % predicted
BASE_REG = ["FEV1_PP_lag_1", "FEV1_PP_lag_2", "AgeAtVisit_lag_1"] + [c for c in CLIN if c != "FEV1_Z_lag_1"]


def ridge_base(Xtr, ytr, Xte):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)).fit(Xtr[BASE_REG].to_numpy(float), ytr)
    return m.predict(Xtr[BASE_REG].to_numpy(float)), m.predict(Xte[BASE_REG].to_numpy(float))


def fit_lgb_reg(Xtr, ytr, gtr, Xte, seed, offset=False):
    inner = inner_splits(ytr, gtr, seed, regression=True)
    offs = [ridge_base(Xtr.iloc[i], ytr[i], Xtr.iloc[v]) if offset else (None, None) for i, v in inner]

    def objective(trial):
        prm = lgb_space(trial, seed, reg=True); loss = []
        for (i, v), (o_tr, o_va) in zip(inner, offs):
            m = lgb.LGBMRegressor(**prm).fit(Xtr.iloc[i][FE], ytr[i], init_score=o_tr)
            loss.append(float(np.mean(np.abs(ytr[v] - m.predict(Xtr.iloc[v][FE]) - (o_va if offset else 0)))))
        return float(np.mean(loss))
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)); st.optimize(objective, n_trials=NTRIAL)
    prm = lgb_space(optuna.trial.FixedTrial(st.best_params), seed, reg=True)
    o_tr, o_te = ridge_base(Xtr, ytr, Xte) if offset else (None, 0)
    m = lgb.LGBMRegressor(**prm).fit(Xtr[FE], ytr, init_score=o_tr)
    return m.predict(Xte[FE]) + o_te, st.best_params


def one_fold_reg(r, f, tr, te, d, ych, g):
    optuna.logging.set_verbosity(optuna.logging.WARNING); warnings.filterwarnings("ignore")
    seed = SEED + 100 * r + f; Xtr, Xte = d.iloc[tr], d.iloc[te]; out, prm = {}, {}
    out["ridge_base"] = ridge_base(Xtr, ych[tr], Xte)[1]
    out["lgbm_reg_optuna"], prm["lgbm_reg_optuna"] = fit_lgb_reg(Xtr, ych[tr], g[tr], Xte, seed)
    out["lgbm_reg_offset"], prm["lgbm_reg_offset"] = fit_lgb_reg(Xtr, ych[tr], g[tr], Xte, seed, offset=True)
    out["avg_ridge_lgbm"] = (out["ridge_base"] + out["lgbm_reg_optuna"]) / 2
    return r, te, out, prm


tgt = "y_fev1_pp"
dr = df[df[tgt].notna()].reset_index(drop=True); yr = dr[tgt].to_numpy(float); idr = dr.PatCode.to_numpy(); gr = pd.factorize(idr)[0]
anchor = dr.FEV1_PP_lag_1.fillna(dr.FEV1_PP_lag_2).fillna(dr.FEV1_PP_lag_1.median()).to_numpy(float); ych = yr - anchor   # as 22 (1 row lacks both)
runs = Parallel(n_jobs=NJOBS)(delayed(one_fold_reg)(r, f, tr, te, dr, ych, gr) for r, f, tr, te in repeated_group_folds(idr, yr, 5, nrep, SEED, regression=True))
keys = list(runs[0][2]); PR = {k: np.full((nrep, len(dr)), np.nan) for k in keys}; rparams = {k: [] for k in keys}
for r, te, out, prm in runs:
    for k in keys:
        PR[k][r, te] = out[k]
    for k, v in prm.items():
        rparams[k].append(v)
PR = {k: anchor + v.mean(0) for k, v in PR.items()}
PR = {"persistence": anchor, **PR}
fc_ids = OUT / "oof" / f"fc_{tgt}_ids.npy"
if fc_ids.exists() and (np.load(fc_ids, allow_pickle=True) == idr).all():
    for k, lab in [("ridge_clin", "clinical linear model (main pipeline)"), ("cb_change", "CatBoost change (main pipeline)")]:
        PR[lab] = np.load(OUT / "oof" / f"fc_{tgt}_{k}_pred.npy")
res["fev1"] = {"n": len(dr), "models": {}}
w = lambda t: (lambda yy, pp: float(np.mean(np.abs(yy - pp) <= t)))
for k, p in PR.items():
    np.save(OUT / "oof" / f"mlx_{tgt}_{k.split(' ')[0]}.npy", p)
    res["fev1"]["models"][k] = {"r2": S.cluster_bootstrap(idr, S.r2, (yr, p), nboot, SEED), "mae": S.mae(yr, p),
                                "within_5": S.cluster_bootstrap(idr, w(5), (yr, p), nboot, SEED), "within_10": S.cluster_bootstrap(idr, w(10), (yr, p), nboot, SEED),
                                **({} if k == "persistence" else {"delta_within_5_vs_persistence": S.cluster_bootstrap(idr, lambda yy, a_, b_: w(5)(yy, a_) - w(5)(yy, b_), (yr, p, anchor), nboot, SEED)})}
res["fev1_selected_params"] = {k: pd.Series([str(v.get("objective")) for v in vs]).value_counts().to_dict() for k, vs in rparams.items() if vs}
for k, v in res["fev1"]["models"].items():
    print(f"{k:42s} R2 {v['r2']['point']:.3f} MAE {v['mae']:.2f} within5 {v['within_5']['point']:.3f} within10 {v['within_10']['point']:.3f}", flush=True)
write_json(res, OUT / "ml_experiments.json", script=__file__)
