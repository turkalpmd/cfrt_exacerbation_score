"""Final clinical model specification exactly as validated (unpenalised ML on the full cohort; cluster-robust SEs): per-unit ORs,
OR per 14-day IV course, imputation medians, missing-input rules, the equation, and a points system for the supplementary
points chart. Variables = CLIN from scripts/21_model_development.py (index-free; IV days entered as log(1 + days))."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from _common import setup
from lib.features import CLIN
from lib.io import write_json

a, df, OUT = setup(__doc__)
LABEL = {"FEV1_Z_lag_1": "FEV1 z-score (anchor year, best of year)", "bmiZscore_lag_1": "BMI z-score (anchor year)",
         "PseudomAer_lag_1": "P. aeruginosa isolated (anchor year)", "pa_any_2y": "P. aeruginosa isolated (anchor year or year before)", "MRSA_lag_1": "MRSA isolated (anchor year)", "mrsa_any_2y": "MRSA isolated (anchor year or year before)", "ABPA_lag_1": "ABPA (anchor year)", "abpa_any_2y": "ABPA (anchor year or year before)",
         "log_iv_burden_2yr": "IV-antibiotic days, anchor year + year before", "iv_burden_2yr_lag": "IV-antibiotic days, anchor year + year before"}
RANGE = {"FEV1_Z_lag_1": (-6, 2), "bmiZscore_lag_1": (-3, 2), "PseudomAer_lag_1": (0, 1), "pa_any_2y": (0, 1), "abpa_any_2y": (0, 1), "MRSA_lag_1": (0, 1), "mrsa_any_2y": (0, 1), "ABPA_lag_1": (0, 1),
         "log_iv_burden_2yr": (0, 120), "iv_burden_2yr_lag": (0, 120)}                     # natural units (days for IV)
TICKS = {"log_iv_burden_2yr": [0, 7, 14, 28, 42, 56, 84, 120], "iv_burden_2yr_lag": [0, 14, 28, 56, 84, 120]}
tf = lambda k, x: np.log1p(x) if k == "log_iv_burden_2yr" else x                       # natural units -> model scale
res = {"variables": CLIN}
for out in ["y_pex", "y_decline10"]:
    d = df[df[out].notna()].copy(); med = d[CLIN].median()
    X = sm.add_constant(d[CLIN].fillna(med)); y = d[out]
    m = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.PatCode)[0]}); ci = m.conf_int()
    coef = {k: {"beta": float(m.params[k]), "se": float(m.bse[k]), "or": float(np.exp(m.params[k])), "or_lo": float(np.exp(ci.loc[k, 0])),
                "or_hi": float(np.exp(ci.loc[k, 1])), "p": float(m.pvalues[k])} for k in m.params.index}
    ivk = [k for k in CLIN if "iv_burden" in k][0]
    b_iv, lo_iv, hi_iv = m.params[ivk], ci.loc[ivk, 0], ci.loc[ivk, 1]
    step = (tf(ivk, 14) - tf(ivk, 0))                                                    # 0 -> 14 days (one course)
    r = {"n": int(len(d)), "events": int(y.sum()), "imputation_medians": med.round(3).to_dict(), "coefficients": coef,
         "or_first_14_iv_days": {"or": float(np.exp(step * b_iv)), "lo": float(np.exp(step * lo_iv)), "hi": float(np.exp(step * hi_iv))},
         "or_14_vs_28_iv_days": float(np.exp((tf(ivk, 28) - tf(ivk, 14)) * b_iv)),
         "formula": "logit(p) = " + f"{m.params['const']:.3f} " + " ".join(
             f"{m.params[k]:+.3f}×{'ln(1 + IV days, t−1 and t)' if k == 'log_iv_burden_2yr' else '[' + LABEL[k] + ']'}" for k in CLIN)}
    contrib = {k: abs(m.params[k]) * abs(tf(k, RANGE[k][1]) - tf(k, RANGE[k][0])) for k in CLIN}; scale = 100 / max(contrib.values())
    pts = {}
    for k in CLIN:
        zero = RANGE[k][0] if m.params[k] > 0 else RANGE[k][1]
        ticks = TICKS.get(k, list(np.arange(RANGE[k][0], RANGE[k][1] + 1e-9, 1 if k in ("FEV1_Z_lag_1", "bmiZscore_lag_1") else 1)))
        pts[k] = {"label": LABEL[k], "range": RANGE[k], "beta": float(m.params[k]), "zero_at": zero, "log1p_days": k == "log_iv_burden_2yr",
                  "points_at_ticks": {str(t): float(scale * m.params[k] * (tf(k, t) - tf(k, zero))) for t in ticks}}
    lp0 = m.params["const"] + sum(m.params[k] * tf(k, pts[k]["zero_at"]) for k in CLIN)
    total = np.arange(0, 301, 10); risk = 1 / (1 + np.exp(-(lp0 + total / scale)))
    r["points"] = {"scale_points_per_logit": float(scale), "linear_predictor_at_zero_points": float(lp0), "variables": pts,
                   "total_points_to_risk": {int(t): float(p) for t, p in zip(total, risk)}}
    ex = {"FEV1_Z_lag_1": -2.0, "bmiZscore_lag_1": -0.5, "PseudomAer_lag_1": 1, "pa_any_2y": 1, "abpa_any_2y": 0, "MRSA_lag_1": 0, "mrsa_any_2y": 0, "ABPA_lag_1": 0, "log_iv_burden_2yr": 14, "iv_burden_2yr_lag": 14}
    lp = m.params["const"] + sum(m.params[k] * tf(k, ex[k]) for k in CLIN)
    r["worked_example"] = {"inputs": {k: ex[k] for k in CLIN}, "linear_predictor": float(lp), "risk": float(1 / (1 + np.exp(-lp))),
                           "points": float(sum(scale * m.params[k] * (tf(k, ex[k]) - tf(k, pts[k]["zero_at"])) for k in CLIN))}
    r["missing_input_rules"] = ("Do not compute the score if anchor-year FEV1 or the IV-antibiotic history is unavailable. Missing BMI z: use the "
                                "cohort median and report the range of risk over BMI z −2 to +1. Unknown culture results are scored as negative and the "
                                "risk is reported as a lower bound.")
    res[out] = r
    print(out, r["formula"], "| OR first 14 d", round(r["or_first_14_iv_days"]["or"], 2), "| example", round(r["worked_example"]["risk"], 3),
          round(r["worked_example"]["points"]), "pts")
    print("   ", {k: (round(v["or"], 2), round(v["or_lo"], 2), round(v["or_hi"], 2), round(v["p"], 4)) for k, v in coef.items() if k != "const"})
write_json(res, OUT / "lr5_model_spec.json", "12_lr5_model_spec.py")
