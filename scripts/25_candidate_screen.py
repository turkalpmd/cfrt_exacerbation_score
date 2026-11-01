"""Exploratory screen for a fifth predictor (user question: is there a better variable than MRSA?).
Base = the four predictors kept in every fold (FEV1 z, BMI z, P. aeruginosa, log(1 + IV days)). Each candidate known at the
anchor-year review is added alone; for every base + candidate model: cluster-robust OR and Wald p on the full cohort,
out-of-fold AUC and log-loss (same 5 x 5 patient-grouped folds as the base) and the paired difference from the base with a
patient-cluster bootstrap CI. Benjamini-Hochberg q across candidates. Exploratory: not part of the prespecified pipeline.
Output: results/candidate_screen.json, results/candidate_screen.csv."""
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
BASE = ["FEV1_Z_lag_1", "bmiZscore_lag_1", "pa_any_2y", "log_iv_burden_2yr"]

df["female"] = (df.Sex == 2).astype(int)
for c in ["PExNumber_lag_1", "HospDaysTotal_lag_1", "pex_burden_2yr_lag", "hosp_burden_2yr_lag"]:
    df[f"log_{c}"] = np.log1p(df[c])
CAND = {
    "MRSA (anchor year)": "MRSA_lag_1", "MRSA (either year)": "mrsa_any_2y", "ABPA (anchor year)": "ABPA_lag_1",
    "ABPA (either year)": "abpa_any_2y", "MRSA or ABPA": "mrsa_or_abpa_lag_1", "MSSA": "MSSA_lag_1",
    "P. aeruginosa in both years": "pa_both_years", "Stenotrophomonas": "StenotrophMaltophilia_lag_1",
    "Achromobacter": "Achromobacter_lag_1", "Burkholderia": "BurkhCepacia_lag_1", "Aspergillus": "Afumigatus_lag_1",
    "Haemophilus": "HaemInf_lag_1", "Non-tuberculous mycobacteria": "NontubMycobacteria_lag_1", "Number of pathogens": "n_pathogens_t",
    "Recorded exacerbations, log(1+n)": "log_PExNumber_lag_1", "Exacerbations over 2 years, log(1+n)": "log_pex_burden_2yr_lag",
    "Hospital days, log(1+d)": "log_HospDaysTotal_lag_1", "Hospital days over 2 years, log(1+d)": "log_hosp_burden_2yr_lag",
    "IV antibiotics in both years": "iv_both_years", "Oral corticosteroids": "SteroidOral_lag_1", "Inhaled corticosteroids": "SteroidInhaled_lag_1",
    "Oxygen": "Oxygen_lag_1", "Non-invasive ventilation": "NIPPV_lag_1", "Haemoptysis": "Hemoptysis_lag_1",
    "Azithromycin": "Azithro_lag_1", "Dornase alfa": "rhDNase_lag_1", "Hypertonic saline": "NaCl_lag_1", "Inhaled antibiotics": "Antibiotic_lag_1",
    "Bronchodilator": "Bronchodil_lag_1", "Proton-pump inhibitor": "PPI_lag_1", "Number of maintenance therapies": "treatment_burden_t",
    "CF-related diabetes": "Diabetes_lag_1", "CF-related liver disease": "LiverDisease_lag_1", "Pancreatic enzymes": "Pancreatic_lag_1",
    "FVC z-score": "FVC_Z_lag_1", "FEV1 z change (t-1 to t)": "fev1_z_trend_lag", "FEV1-FVC z gap": "fev1_fvc_z_gap",
    "BMI z change (t-1 to t)": "bmi_trend_lag", "Height z-score": "heightZScore_lag_1", "Age": "AgeAtVisit_lag_1", "Female sex": "female",
    "Age at diagnosis": "AgeDiagnosis", "Sweat chloride": "SweatTestVal", "Meconium ileus": "MeconiumIleus", "Newborn screening": "NeonatalScr",
    "Number of F508del alleles": "cftr_n_f508del", "Two minimal-function alleles": "cftr_mf_mf", "Severe genotype (class I–III)": "cftr_severe",
    "Mild genotype (>=1 class IV–VI)": "cftr_mild", ">=1 residual-function allele": "cftr_any_rf",
    "ETI-eligible genotype": "cftr_eti_eligible",
}
OUTCOME = "y_pex"
d = df[df[OUTCOME].notna()].reset_index(drop=True)
g = pd.factorize(d.PatCode)[0]


def cluster_fit(feats):
    X = sm.add_constant(d[feats].astype(float).fillna(d[feats].astype(float).median()))
    return sm.Logit(d[OUTCOME], X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g})


def oof(feats):
    rr = run_oof(d, feats, OUTCOME, lr_unpenalised(), proba=True, native_nan=False, n_repeats=nrep, seed=SEED)
    return rr["y"], rr["ids"], rr["oof"].mean(0)


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


y, ids, p0 = oof(BASE)
res = {"outcome": OUTCOME, "n": len(d), "events": int(d[OUTCOME].sum()), "base": BASE,
       "base_auc": S.cluster_bootstrap(ids, S.auc, (y, p0), nboot, SEED), "base_logloss": logloss(y, p0), "candidates": {}}
rows = []
for lab, v in CAND.items():
    m = cluster_fit(BASE + [v]); _, _, p1 = oof(BASE + [v])
    x = d[v]; binary = set(x.dropna().unique()) <= {0, 1}
    r = {"variable": v, "label": lab, "missing_pct": float(100 * x.isna().mean()),
         "prevalence_pct": float(100 * x.mean()) if binary else None,
         "or": float(np.exp(m.params[v])), "or_lo": float(np.exp(m.conf_int().loc[v, 0])), "or_hi": float(np.exp(m.conf_int().loc[v, 1])),
         "p": float(m.pvalues[v]), "auc": S.auc(y, p1),
         "delta_auc": S.cluster_bootstrap(ids, lambda yy, a_, b_: S.auc(yy, a_) - S.auc(yy, b_), (y, p1, p0), nboot, SEED),
         "delta_logloss": logloss(y, p1) - res["base_logloss"]}
    res["candidates"][lab] = r
    rows.append({k: r[k] for k in ["label", "variable", "missing_pct", "prevalence_pct", "or", "or_lo", "or_hi", "p", "auc", "delta_logloss"]}
                | {"delta_auc": r["delta_auc"]["point"], "delta_auc_lo": r["delta_auc"]["lo"], "delta_auc_hi": r["delta_auc"]["hi"]})
    print(f"{lab:40s} OR {r['or']:.2f} p {r['p']:.3f}  dAUC {r['delta_auc']['point']:+.4f} ({r['delta_auc']['lo']:+.4f}, {r['delta_auc']['hi']:+.4f})  dLL {r['delta_logloss']:+.4f}")
tab = pd.DataFrame(rows)
tab["q"] = S.bh_fdr(tab.p.to_numpy())
for lab, q in zip(tab.label, tab.q):
    res["candidates"][lab]["q"] = float(q)
tab.sort_values("delta_logloss").to_csv(OUT / "candidate_screen.csv", index=False)
write_json(res, OUT / "candidate_screen.json", script=__file__)
print("base AUC", res["base_auc"], "base logloss", res["base_logloss"])
