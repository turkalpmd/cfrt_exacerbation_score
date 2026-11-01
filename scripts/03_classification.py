"""One-year IV-antibiotic-treated exacerbation and >=10-point FEV1 decline. Reference = index-free clinical model (CLIN, from 21).
Models (revision): LR-5 unpenalised (published equation); CatBoost default + nested isotonic; CatBoost tuned by inner grouped CV
(+ isotonic); CV-penalised full-panel logistic; non-redundant LR-5; LR-5 with splines; comparators: prior-IV logistic,
severity-index logistic, binary rules (any IV in t-1/t; severity index >=2). Alternative outcome definitions (LR-5, prior-IV).
Main-text machine-learning model (1 Nov 2026): 'ml' = curated panel -> in-fold selection of 10–15 features (SHAP top 20, number
by inner CV) -> Optuna-tuned LightGBM (lib/mlx.SelectedLGBM); sensitivity: 'ml_ownshadow' (each feature against its own permuted
copy) and 'ml_maxshadow' (classic Boruta rule) (Supplement). Missing-input
experiment: test-fold inputs masked at prediction time for the clinical model (median imputation) and 'ml' (native)."""
import numpy as np
import pandas as pd
from _common import setup
from config import N_BOOT, N_REPEATS, SEED
from lib import stats as S
from lib.cv import run_oof
from lib.features import CLIN, FE_PANEL, GENOTYPE_FEATURES, LR5, LR5_SEV80, MAIN_PANEL, ML_PANEL, PRIOR_IV, RULES_NOV as RULES, SEVERITY
from lib.io import write_json
from lib.mlx import SelectedLGBM
from lib.models import TunedCatBoost, cb_clf, lr_cv_penalised, lr_spline, lr_unpenalised

a, df, OUT = setup(__doc__)
nrep = 2 if a.fast else N_REPEATS; nboot = 200 if a.fast else N_BOOT; NTRIAL = 8 if a.fast else 40; NJOBS = 10
res = {"window": a.window, "n_repeats": nrep, "n_boot": nboot}


FE = list(dict.fromkeys(FE_PANEL + [g for g in GENOTYPE_FEATURES if g in df.columns]))


def specs(outcome):
    sp = {  # name: (features, model, native_nan, nested_cal)
        "clin": (CLIN, lr_unpenalised(), False, True),
        "lr5": (LR5, lr_unpenalised(), False, True),
        "lr5_severity80": (LR5_SEV80, lr_unpenalised(), False, False),
        "catboost": (MAIN_PANEL, cb_clf(), True, True),
        "lr_full_cv": (MAIN_PANEL, lr_cv_penalised(), False, False),
        "clin_spline": (CLIN, lr_spline(2), False, False),
        "prior_iv_logistic": (PRIOR_IV, lr_unpenalised(), False, False),
        "severity_logistic": (SEVERITY, lr_unpenalised(), False, False),
    }
    if outcome in ("y_pex", "y_decline10"):
        sp["catboost_tuned"] = (MAIN_PANEL + ["PatCode"], TunedCatBoost(seed=SEED), True, False)
        sp["catboost_fe_tuned"] = (FE + ["PatCode"], TunedCatBoost(seed=SEED), True, False)   # engineered panel (+ genotype)
    sp["ml"] = (ML_PANEL + ["PatCode"], SelectedLGBM(feats=ML_PANEL, seed=SEED, ntrial=NTRIAL), True, False)
    sp["ml_ownshadow"] = (ML_PANEL + ["PatCode"], SelectedLGBM(feats=ML_PANEL, seed=SEED, ntrial=NTRIAL, rule="own_shadow"), True, False)
    sp["ml_maxshadow"] = (ML_PANEL + ["PatCode"], SelectedLGBM(feats=ML_PANEL, seed=SEED, ntrial=NTRIAL, rule="max_shadow"), True, False)
    return sp


CULTURE = ("PseudomAer", "pa_", "MRSA", "mrsa", "MSSA", "ABPA", "abpa", "NontubMycobacteria", "Afumigatus", "HaemInf")
KEEP_ONLY = ("FEV1_Z_lag_1", "ivDaysTotal_lag_1", "ivDaysTotal_lag_2", "iv_both_years", "log_iv_burden_2yr")
MASKS = {"BMI z and culture results missing": lambda c: c.startswith(("bmi",) + CULTURE),
         "only FEV1 and IV days available": lambda c: c not in KEEP_ONLY}


def masked_oof(rr, X, feats, n, scenario, seed):
    """Re-predict every outer test fold with the fold's own model after masking inputs (np.nan) at prediction time."""
    pr = np.full((nrep, n), np.nan); rng = np.random.default_rng(seed)
    for r, f, te, m in rr["models"]:
        Xt = X.iloc[te].copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X[te], columns=feats)
        cols = [c for c in feats if c != "PatCode"]
        if scenario == "20% of inputs missing at random":
            Xt[cols] = Xt[cols].astype(float).mask(rng.random((len(te), len(cols))) < 0.2)
        else:
            Xt[[c for c in cols if MASKS[scenario](c)]] = np.nan
        pr[r, te] = m.predict_proba(Xt if isinstance(X, pd.DataFrame) else Xt.to_numpy(float))[:, 1]
    return pr.mean(0)


for out in ["y_pex", "y_decline10"]:
    d = df[df[out].notna()].reset_index(drop=True)
    runs = {k: run_oof(d, f, out, m, proba=True, native_nan=nat, n_repeats=nrep, seed=SEED, nested_cal=nc,
                       keep_models=k in ("clin", "ml", "ml_ownshadow", "ml_maxshadow"), n_jobs=NJOBS if k.startswith("ml") else 1)
            for k, (f, m, nat, nc) in specs(out).items()}
    y, ids = runs["clin"]["y"], runs["clin"]["ids"]
    for nm, arr in [("y", y), ("ids", ids), ("target_year", runs["clin"]["target_year"]), ("row_index", runs["clin"]["row_index"])]:
        np.save(OUT / "oof" / f"clf_{out}_{nm}.npy", arr)
    block = {"n_rows": len(d), "n_events": int(y.sum()), "n_patients": int(d.PatCode.nunique()), "prevalence": float(y.mean()),
             "epv_clin": float(y.sum() / len(CLIN)), "epv_main_panel": float(y.sum() / len(MAIN_PANEL)), "epv_fe_panel": float(y.sum() / len(FE)), "models": {}}
    P = {}
    for k, rr in runs.items():
        p = rr["oof"].mean(0); P[k] = p; np.save(OUT / "oof" / f"clf_{out}_{k}_raw.npy", p)
        m = {met: S.cluster_bootstrap(ids, fn, (y, p), nboot, SEED) for met, fn in [("auc", S.auc), ("auprc", S.auprc), ("brier_raw", S.brier)]}
        per = [S.auc(y, rr["oof"][r]) for r in range(nrep)]
        m["auc_across_repeats"] = {"mean": float(np.mean(per)), "sd": float(np.std(per, ddof=1)) if nrep > 1 else 0.0}
        if "oof_cal" in rr:
            pc = rr["oof_cal"].mean(0); np.save(OUT / "oof" / f"clf_{out}_{k}_cal.npy", pc)
            m["brier_cal"] = S.cluster_bootstrap(ids, S.brier, (y, pc), nboot, SEED); m["auc_cal"] = S.auc(y, pc)
        block["models"][k] = m
    # binary rules (no fitting): score = indicator
    for rk, col in RULES.items():
        r_ = d[col].fillna(0).to_numpy(float); P[rk] = r_; np.save(OUT / "oof" / f"clf_{out}_{rk}_raw.npy", r_)
        f = r_ == 1
        block["models"][rk] = {"auc": S.cluster_bootstrap(ids, S.auc, (y, r_), nboot, SEED), "flagged_pct": float(100 * f.mean()),
                               "sensitivity": float(y[f].sum() / y.sum()), "ppv": float(y[f].mean()), "npv": float(1 - y[~f].mean())}
    block["delta_auc"] = {f"{k}_minus_clin": S.cluster_bootstrap(ids, lambda yy, pa, pb: S.auc(yy, pa) - S.auc(yy, pb), (y, P[k], P["clin"]), nboot, SEED)
                          for k in P if k != "clin"}
    # added value of the non-IV LR-5 terms over prior IV: out-of-fold log-loss and Brier skill score
    ll = lambda yy, pp: float(-np.mean(yy * np.log(np.clip(pp, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - pp, 1e-6, 1))))
    block["logloss"] = {k: ll(y, P[k]) for k in P if not k.startswith("rule")}
    block["brier_skill"] = {
        "clin_vs_prevalence": S.cluster_bootstrap(ids, lambda yy, a_: 1 - S.brier(yy, a_) / S.brier(yy, np.full(len(yy), yy.mean())), (y, P["clin"]), nboot, SEED),
        "clin_vs_prior_iv": S.cluster_bootstrap(ids, lambda yy, a_, b_: 1 - S.brier(yy, a_) / S.brier(yy, b_), (y, P["clin"], P["prior_iv_logistic"]), nboot, SEED)}
    block["brier_null"] = float(y.mean() * (1 - y.mean()))
    block["tuned_catboost_note"] = "params selected per outer fold by inner 3-fold grouped CV over depth {3,4,6} x l2 {3,10}"
    # machine-learning model: features retained and hyperparameters chosen in each of the outer training sets
    for k in ("ml", "ml_ownshadow", "ml_maxshadow"):
        fm = [m for *_, m in runs[k]["models"]]
        block[f"{k}_selection"] = {
            "n_folds": len(fm), "frequency": pd.Series([c for m in fm for c in m.selected_]).value_counts().div(len(fm)).round(3).to_dict(),
            "n_selected": {"median": float(np.median([len(m.selected_) for m in fm])), "min": min(len(m.selected_) for m in fm), "max": max(len(m.selected_) for m in fm)},
            "params_median": pd.DataFrame([m.best_params_ for m in fm]).median(numeric_only=True).round(4).to_dict()}
    block["epv_ml_panel"] = float(y.sum() / len(ML_PANEL))
    # missing inputs at prediction time (authors: robustness of the machine-learning model to missing data)
    Xc, Xm = d[CLIN].to_numpy(float), d[ML_PANEL + ["PatCode"]]
    block["masking"] = {}
    for sc in list(MASKS) + ["20% of inputs missing at random"]:
        block["masking"][sc] = {}
        for k, X, feats in [("clin", Xc, CLIN), ("ml", Xm, ML_PANEL + ["PatCode"])]:
            pm = masked_oof(runs[k], X, feats, len(d), sc, SEED)
            block["masking"][sc][k] = {"auc": S.cluster_bootstrap(ids, S.auc, (y, pm), nboot, SEED), "slope": S.cal_slope(y, pm), "citl": S.citl(y, pm)}
        print("   masking", sc, {k: round(v["auc"]["point"], 3) for k, v in block["masking"][sc].items()})
    res[out] = block
    print(out, f"n={len(d)} ev={int(y.sum())}", {k: round(v["auc"]["point"], 3) for k, v in block["models"].items()})
    print("   dAUC vs clin:", {k: round(v["point"], 3) for k, v in block["delta_auc"].items()}, "| BSS vs prior-IV", round(block["brier_skill"]["clin_vs_prior_iv"]["point"], 3))

# alternative outcome definitions (clinical model and prior-IV logistic)
res["alt_outcomes"] = {}
for out in ["y_pex_strict", "y_iv14", "y_hosp"]:
    d = df[df[out].notna()].reset_index(drop=True); blk = {"n": len(d), "events": int(d[out].sum())}
    for k, (f, m) in {"clin": (CLIN, lr_unpenalised()), "prior_iv_logistic": (PRIOR_IV, lr_unpenalised())}.items():
        rr = run_oof(d, f, out, m, proba=True, native_nan=False, n_repeats=nrep, seed=SEED)
        p = rr["oof"].mean(0); y = rr["y"]
        blk[k] = {"auc": S.cluster_bootstrap(rr["ids"], S.auc, (y, p), nboot, SEED), "citl": S.citl(y, p), "slope": S.cal_slope(y, p)}
    res["alt_outcomes"][out] = blk
    print(out, blk["n"], blk["events"], {k: round(blk[k]["auc"]["point"], 3) for k in ["clin", "prior_iv_logistic"]})
write_json(res, OUT / "classification_summary.json", "03_classification.py")
