"""Consistency and data checks for the November protocol (Izzet_sorular C–D).
(a) Prediction-years not used in the two earlier exploratory analyses (April 2026 extract: april_analysis/data/cohort_siniflandirma.csv,
    key (PatCode, anchor year + 1); May 2026 extract: may_analysis/data/cohort.csv, key (PatCode, outcome year)); clinical-model
    performance in prediction-years outside both extracts (out-of-fold, and development on overlap -> test on new).
(b) Added value of the non-IV terms of the clinical model: cluster-robust Wald test and out-of-fold log-likelihood vs prior IV days.
(c) Source export and points-chart axis truncation; CFRD; azithromycin.
(d) BMI z-score: registry values vs WHO 2007 recomputed from registry BMI (adults: reference at 19 y), by age group and registry
    year; clinical model refitted with WHO BMI z (cross-validated and 2024 temporal).
(e) Software versions."""
import importlib
import platform
import numpy as np
import pandas as pd
import statsmodels.api as sm
from _common import setup
from config import DATA, LEGACY_COHORT, N_BOOT, OLD, RAW_XLSX, SEED
from lib import stats as S
from lib.cv import run_oof
from lib.features import CLIN
from lib.io import write_json
from lib.models import lr_unpenalised

bc = importlib.import_module("00_build_cohort")
a, df, OUT = setup(__doc__)
nboot = 200 if a.fast else N_BOOT
res = {}
O = OUT / "oof"; yv = np.load(O / "clf_y_pex_y.npy"); idv = np.load(O / "clf_y_pex_ids.npy", allow_pickle=True)
pl = np.load(O / "clf_y_pex_clin_raw.npy")
d = df[df.y_pex.notna()].reset_index(drop=True)
assert (idv == d.PatCode.to_numpy()).all() and (yv == d.y_pex.to_numpy()).all()


def perf(y, p, ids):
    return {"n": int(len(y)), "events": int(y.sum()), "patients": int(len(np.unique(ids))),
            "auc": S.cluster_bootstrap(ids, S.auc, (y, p), nboot, SEED), "citl": S.citl(y, p), "slope": S.cal_slope(y, p), "o_e": float(y.mean() / p.mean())}


# ---------------- (a) earlier exploratory extracts
APR_COHORT = OLD / "april_analysis" / "data" / "cohort_siniflandirma.csv"
if LEGACY_COHORT.exists() and APR_COHORT.exists():   # earlier internal extracts are not distributed
    may = pd.read_csv(LEGACY_COHORT); k_may = set(zip(may.PatCode, may.DateVisitYear))
    apr = pd.read_csv(APR_COHORT); k_apr = set(zip(apr.PatCode, apr.DateVisitYear_t + 1))
    key = list(zip(d.PatCode, d.target_year))
    d["in_may"] = [k in k_may for k in key]; d["in_april"] = [k in k_apr for k in key]
    new = ~(d.in_may | d.in_april).to_numpy()
    blk = {"n_rows": len(d), "in_may": int(d.in_may.sum()), "in_april": int(d.in_april.sum()), "in_either": int((~new).sum()),
           "in_neither": int(new.sum()), "not_in_may": int((~d.in_may).sum()), "in_neither_by_target_year": d[new].target_year.value_counts().sort_index().to_dict(),
           "in_neither_adult_rows": int(d.adult[new].sum()),
           "extract_dates": {"april": "LR-5 added in the revision of 26 April 2026 (FEV1 % predicted; predictors from t-1, t-2)",
                             "may": "LR-5 as in v3 defined 30 April 2026; report 4 May, manuscript 6 May 2026 (commit 1c04eb4)"},
           "april_extract_rows": len(apr), "may_extract_rows": len(may)}
    if new.sum() >= 30 and yv[new].sum() >= 5:
        blk["oof_in_neither"] = perf(yv[new], pl[new], idv[new]); blk["oof_in_either"] = perf(yv[~new], pl[~new], idv[~new])
        X = d[CLIN].to_numpy(float); y = d.y_pex.to_numpy(float); ids = d.PatCode.to_numpy()
        tr = (~new) & ~np.isin(ids, ids[new]); med = np.nanmedian(X[tr], axis=0); Xi = np.where(np.isnan(X), med, X)
        m = lr_unpenalised().fit(Xi[tr], y[tr])
        blk["develop_on_overlap_test_on_new_patient_disjoint"] = {"n_train": int(tr.sum()), "events_train": int(y[tr].sum()), **perf(y[new], m.predict_proba(Xi[new])[:, 1], ids[new])}
    res["earlier_extracts"] = blk
    print("extracts:", {k: blk[k] for k in ["in_may", "in_april", "in_either", "in_neither", "not_in_may"]},
          {k: (v["n"], round(v["auc"]["point"], 3), round(v["o_e"], 2)) for k, v in blk.items() if isinstance(v, dict) and "auc" in v})
else:
    res["earlier_extracts"] = {"skipped": "earlier internal extracts not available"}

# ---------------- (b) non-IV terms
Xs = sm.add_constant(d[CLIN].fillna(d[CLIN].median())); y = d.y_pex.to_numpy(float)
fit = sm.Logit(y, Xs).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.PatCode)[0]})
ivk = [c for c in CLIN if "iv" in c]; nonIV = [c for c in CLIN if c not in ivk]
R = np.zeros((len(nonIV), Xs.shape[1]))
for i, c in enumerate(nonIV):
    R[i, list(Xs.columns).index(c)] = 1
w = fit.wald_test(R, scalar=True)
pp = np.load(O / "clf_y_pex_prior_iv_logistic_raw.npy")
ll_oof = lambda yy, p: np.mean(yy * np.log(np.clip(p, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - p, 1e-6, 1)))
res["non_iv_terms"] = {"terms": nonIV, "cluster_wald_chi2": float(w.statistic), "df": len(nonIV), "p": float(w.pvalue),
                       "oof_mean_loglik_difference_clin_minus_prior_iv": S.cluster_bootstrap(idv, lambda yy, a_, b_: ll_oof(yy, a_) - ll_oof(yy, b_), (yv, pl, pp), nboot, SEED)}
print("non-IV terms: Wald", round(w.statistic, 1), "df", len(nonIV), "p", w.pvalue)

# ---------------- (c) source export, axis truncation, CFRD, azithromycin
raw = bc.clean_raw(pd.read_excel(RAW_XLSX))
res["source_export"] = {"records": int(len(raw)), "patients": int(raw.PatCode.nunique()), "spirometry_pct": float(100 * raw.spirometry.mean()),
                        "min_age": int(raw.AgeAtVisit.min()), "max_age": int(raw.AgeAtVisit.max()), "years": raw.DateVisitYear.value_counts().sort_index().to_dict(),
                        "patients_ever_modulator": int(raw[raw.Mod != 0].PatCode.nunique()),
                        "records_of_never_modulator_patients": int(raw[~raw.PatCode.isin(raw[raw.Mod != 0].PatCode)].shape[0])}
res["points_axis_truncation"] = {"iv_days_gt120": int((d.iv_burden_2yr_lag > 120).sum()), "iv_days_max": float(d.iv_burden_2yr_lag.max()),
                                 "bmi_z_gt2": int((d.bmiZscore_lag_1 > 2).sum()), "fev1_z_lt_minus6": int((d.FEV1_Z_lag_1 < -6).sum()), "n": int(len(d))}
res["cfrd"] = {"row_prevalence_anchor": float(d.Diabetes_lag_1.mean()), "rows_with_cfrd": int(d.Diabetes_lag_1.sum()),
               "events_in_cfrd_rows": int(d.loc[d.Diabetes_lag_1 == 1, "y_pex"].sum()), "adult_row_prevalence": float(d.loc[d.adult == 1, "Diabetes_lag_1"].mean())}
res["azithro"] = {"row_pct": float(100 * d.Azithro_lag_1.mean())}

# ---------------- (d) BMI z reference (adults: WHO reference at 19 y)
REF = DATA / "reference"
who = {1: pd.read_excel(REF / "who_bmi_boys.xlsx"), 2: pd.read_excel(REF / "who_bmi_girls.xlsx")}
lms = lambda x, L, M, S_: np.where(np.abs(L) > 1e-8, ((x / M) ** L - 1) / (L * S_), np.log(x / M) / S_)


def who_z(df_):
    bm = df_.DateBirthMonth.where(df_.DateBirthMonth.between(1, 12), 6)
    am = np.minimum(((df_.DateVisitYear + 0.5) - (df_.DateBirthYear + (bm - 0.5) / 12)).to_numpy() * 12, 228.0)   # cap at 19 y
    out = np.full(len(df_), np.nan)
    for sx in (1, 2):
        m_ = (df_.Sex == sx).to_numpy() & np.isfinite(am); t = who[sx]; ages = t.Month.to_numpy()
        T = t.iloc[np.clip(np.searchsorted(ages, am[m_]), 0, len(t) - 1)]
        out[m_] = lms(df_.BMI.to_numpy(float)[m_], T.L.to_numpy(), T.M.to_numpy(), T.S.to_numpy())
    return out


rb = raw[(raw.BMI > 0) & raw.bmiZscore.between(-10, 10) & (raw.AgeAtVisit > 5)].copy(); rb["who"] = who_z(rb)
rb["grp"] = np.where(rb.AgeAtVisit >= 18, "adult", "child")
res["bmi_z_reference"] = {
    "agreement_by_group": {g: {"n": int(len(x)), "median_abs_diff": float(np.nanmedian(np.abs(x.who - x.bmiZscore))), "r": float(x.who.corr(x.bmiZscore)),
                               "registry_mean": float(x.bmiZscore.mean()), "who_mean": float(x.who.mean())} for g, x in rb.groupby("grp")},
    "registry_minus_who_by_year": rb.assign(dz=rb.bmiZscore - rb.who).groupby("DateVisitYear").dz.mean().round(3).to_dict(),
    "note": "adult BMI z-scores recomputed against the WHO 2007 reference at 19 years (last available age)"}
rk = raw.set_index(["PatCode", "DateVisitYear"])
sub = rk.loc[pd.MultiIndex.from_arrays([d.PatCode, d.year_lag_1]), ["BMI", "DateBirthYear", "DateBirthMonth", "Sex"]].reset_index()
d["bmiZ_who"] = who_z(sub)
CW = [c if c != "bmiZscore_lag_1" else "bmiZ_who" for c in CLIN]
harm = {}
for lab, feats in [("registry_bmi_z", CLIN), ("who2007_bmi_z", CW)]:
    rr = run_oof(d, feats, "y_pex", lr_unpenalised(), proba=True, native_nan=False, n_repeats=5, seed=SEED)
    p_ = rr["oof"].mean(0); tr_ = (d.target_year < 2024).to_numpy(); te_ = ~tr_
    Xh = d[feats].to_numpy(float); mh = np.nanmedian(Xh[tr_], axis=0); Xh = np.where(np.isnan(Xh), mh, Xh)
    pt = lr_unpenalised().fit(Xh[tr_], y[tr_]).predict_proba(Xh[te_])[:, 1]
    harm[lab] = {"cv_auc": S.cluster_bootstrap(rr["ids"], S.auc, (rr["y"], p_), nboot, SEED), "cv_slope": S.cal_slope(rr["y"], p_),
                 "temporal_2024_auc": S.auc(y[te_], pt), "temporal_2024_o_e": float(y[te_].mean() / pt.mean())}
res["bmi_z_reference"]["clin_harmonised"] = harm
print("BMI z:", {g: (v["n"], round(v["median_abs_diff"], 2), round(v["r"], 3)) for g, v in res["bmi_z_reference"]["agreement_by_group"].items()},
      "| by year", res["bmi_z_reference"]["registry_minus_who_by_year"],
      "| harmonised", {k: (round(v["cv_auc"]["point"], 3), round(v["temporal_2024_auc"], 3), round(v["temporal_2024_o_e"], 2)) for k, v in harm.items()})

# ---------------- (e) software
import catboost, sklearn, statsmodels, shap, scipy, lightgbm, optuna  # noqa: E401
res["software"] = {"python": platform.python_version(), "lightgbm": lightgbm.__version__, "optuna": optuna.__version__, "catboost": catboost.__version__, "scikit_learn": sklearn.__version__,
                   "statsmodels": statsmodels.__version__, "shap": shap.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}
res["fev1_z_t_t1_correlation"] = float(df[["FEV1_Z_lag_1", "FEV1_Z_lag_2"]].corr().iloc[0, 1])
write_json(res, OUT / "consistency_checks.json", "20_consistency_checks.py")
print(res["software"])
