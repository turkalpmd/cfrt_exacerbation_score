"""Temporal validation (R1-M5, R4-M2, R0-W5): develop on outcome years < last, test on the last year (2024).
(a) all test rows (partial patient overlap); (b) PATIENT-DISJOINT: training excludes every patient with a test-year row.
CatBoost = tuned + isotonic (learned in development data). CITL/slope with patient-bootstrap CIs; development vs test case-mix."""
import numpy as np
import pandas as pd
from _common import setup
from config import N_BOOT, SEED
from lib import stats as S
from lib.features import CLIN, FE_PANEL, GENOTYPE_FEATURES, LR5, MAIN_PANEL, ML_PANEL, PERSISTENCE, PRIOR_IV
from lib.mlx import SelectedLGBM, SelectedLGBMRegressor
from lib.io import write_json
from lib.models import TunedCatBoost, cb_reg, lr_unpenalised, ridge_pipe

a, df, OUT = setup(__doc__)
FE = list(dict.fromkeys(FE_PANEL + [g for g in GENOTYPE_FEATURES if g in df.columns]))
nboot = 200 if a.fast else N_BOOT; NTRIAL = 8 if a.fast else 40
ty = int(df.target_year.max()); res = {"test_year": ty, "train_years": sorted(int(x) for x in df.target_year.unique() if x < ty)}
for design in ["all_test_rows", "patient_disjoint"]:
    res[design] = {}
    for out in ["y_pex", "y_decline10", "y_fev1_z", "y_fev1_pp"]:
        d = df[df[out].notna()]
        te = d[d.target_year == ty]; tr = d[d.target_year < ty]
        if design == "patient_disjoint":
            tr = tr[~tr.PatCode.isin(te.PatCode)]
        ytr, yte, ids = tr[out].to_numpy(float), te[out].to_numpy(float), te.PatCode.to_numpy()
        r = {"n_train": len(tr), "n_test": len(te), "train_patients": int(tr.PatCode.nunique()), "test_patients": int(te.PatCode.nunique()),
             "patients_in_both": len(set(tr.PatCode) & set(te.PatCode))}
        if out.startswith("y_fev1"):
            anchor = PERSISTENCE[out]; lag2 = anchor.replace("_lag_1", "_lag_2")
            preds = {"persistence": te[anchor].fillna(tr[anchor].median()).to_numpy(float),
                     "ar2_linear": ridge_pipe(alpha=1e-6).fit(tr[[anchor, lag2]].to_numpy(float), ytr).predict(te[[anchor, lag2]].to_numpy(float)),
                     "ridge_clin": ridge_pipe().fit(tr[[anchor, lag2] + CLIN].to_numpy(float), ytr).predict(te[[anchor, lag2] + CLIN].to_numpy(float)),
                     "catboost": cb_reg().fit(tr[MAIN_PANEL], ytr).predict(te[MAIN_PANEL]),
                     "catboost_change_fe": te[anchor].fillna(te[lag2]).fillna(tr[anchor].median()).to_numpy(float) + cb_reg(depth=3, l2_leaf_reg=20, iterations=500, learning_rate=0.03).fit(
                         tr[FE], ytr - tr[anchor].fillna(tr[lag2]).fillna(tr[anchor].median()).to_numpy(float)).predict(te[FE]),
                     "ml_change": te[anchor].fillna(te[lag2]).fillna(tr[anchor].median()).to_numpy(float) + SelectedLGBMRegressor(feats=ML_PANEL, seed=SEED, ntrial=NTRIAL).fit(
                         tr[ML_PANEL + ["PatCode"]], ytr - tr[anchor].fillna(tr[lag2]).fillna(tr[anchor].median()).to_numpy(float)).predict(te[ML_PANEL])}
            for k, p in preds.items():
                tol = 5 if out == "y_fev1_pp" else 0.5
                r[k] = {"r2": S.cluster_bootstrap(ids, S.r2, (yte, p), nboot, SEED), "mae": S.mae(yte, p), f"within_{tol}": float(np.mean(np.abs(yte - p) <= tol))}
        else:
            r["events_train"], r["events_test"] = int(ytr.sum()), int(yte.sum())
            preds = {"clin": lr_unpenalised().fit(tr[CLIN].to_numpy(float), ytr).predict_proba(te[CLIN].to_numpy(float))[:, 1],
                     "lr5": lr_unpenalised().fit(tr[LR5].to_numpy(float), ytr).predict_proba(te[LR5].to_numpy(float))[:, 1],
                     "prior_iv_logistic": lr_unpenalised().fit(tr[PRIOR_IV].to_numpy(float), ytr).predict_proba(te[PRIOR_IV].to_numpy(float))[:, 1],
                     "catboost_fe_tuned": TunedCatBoost(seed=SEED).fit(tr[FE + ["PatCode"]], ytr).predict_proba(te[FE + ["PatCode"]])[:, 1],
                     "ml": SelectedLGBM(feats=ML_PANEL, seed=SEED, ntrial=NTRIAL).fit(tr[ML_PANEL + ["PatCode"]], ytr).predict_proba(te[ML_PANEL + ["PatCode"]])[:, 1]}
            for k, p in preds.items():
                r[k] = {"auc": S.cluster_bootstrap(ids, S.auc, (yte, p), nboot, SEED),
                        "citl": S.cluster_bootstrap(ids, S.citl, (yte, p), nboot, SEED),
                        "slope": S.cluster_bootstrap(ids, S.cal_slope, (yte, p), nboot, SEED),
                        "o_e": float(yte.mean() / p.mean()), "brier": S.brier(yte, p)}
            # development vs test case-mix (TRIPOD+AI 20c)
            r["case_mix"] = {c: {"train": float(tr[c].mean()), "test": float(te[c].mean())}
                             for c in ["AgeAtVisit_lag_1", "adult", "FEV1_Z_lag_1", "bmiZscore_lag_1", "PseudomAer_lag_1", "pa_any_2y", "MRSA_lag_1", "mrsa_any_2y", "any_iv_2y"]}
        res[design][out] = r
        print(design, out, f"train {len(tr)} / test {len(te)} (overlap pts {r['patients_in_both']})",
              {k: round(v["auc"]["point"] if "auc" in v else v["r2"]["point"], 3) for k, v in r.items() if isinstance(v, dict) and ("auc" in v or "r2" in v)})
write_json(res, OUT / "temporal_validation.json", "11_temporal_validation.py")
