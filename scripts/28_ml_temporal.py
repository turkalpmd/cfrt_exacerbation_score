"""Temporal check of the machine-learning experiments (scripts/27_ml_experiments.py): the configurations that improved on the
five-variable model in nested cross-validation are re-developed on outcome years before 2024 (tuning by Optuna within the
development data only) and tested on 2024, in the two designs of scripts/11_temporal_validation.py (all test rows; patient-
disjoint). Paired AUC and net-benefit differences from the five-variable model with patient-bootstrap CIs.
Output: results/ml_temporal.json."""
import warnings

import numpy as np
import optuna
import pandas as pd
from _common import setup
from config import N_BOOT, SEED
from lib import stats as S
from lib.features import CLIN, FE_PANEL, GENOTYPE_FEATURES
from lib.io import write_json
from lib.mlx import fit_ebm, fit_lgb_clf
from lib.models import lr_unpenalised

warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
a, df, OUT = setup(__doc__)
nboot = 200 if a.fast else N_BOOT; NTRIAL = 8 if a.fast else 40
FE = list(dict.fromkeys(FE_PANEL + [g for g in GENOTYPE_FEATURES if g in df.columns] + CLIN))
ty = int(df.target_year.max()); res = {"test_year": ty}
d = df[df.y_pex.notna()]
for design in ["all_test_rows", "patient_disjoint"]:
    te = d[d.target_year == ty]; tr = d[d.target_year < ty]
    if design == "patient_disjoint":
        tr = tr[~tr.PatCode.isin(te.PatCode)]
    tr = tr.reset_index(drop=True); te = te.reset_index(drop=True)
    ytr, yte, ids = tr.y_pex.to_numpy(float), te.y_pex.to_numpy(float), te.PatCode.to_numpy(); gtr = pd.factorize(tr.PatCode)[0]
    P = {"clin": lr_unpenalised().fit(tr[CLIN].to_numpy(float), ytr).predict_proba(te[CLIN].to_numpy(float))[:, 1]}
    P["lgbm_optuna"], _ = fit_lgb_clf(tr, ytr, gtr, te, SEED, FE, NTRIAL)
    P["lgbm_offset"], _ = fit_lgb_clf(tr, ytr, gtr, te, SEED, FE, NTRIAL, offset=True)
    P["ebm"] = fit_ebm(tr, ytr, te, SEED, FE)
    P["avg_clin_lgbm"] = (P["clin"] + P["lgbm_optuna"]) / 2
    P["avg_clin_ebm"] = (P["clin"] + P["ebm"]) / 2
    r = {"n_train": len(tr), "events_train": int(ytr.sum()), "n_test": len(te), "events_test": int(yte.sum()), "models": {}}
    for k, p in P.items():
        r["models"][k] = {"auc": S.cluster_bootstrap(ids, S.auc, (yte, p), nboot, SEED), "o_e": float(yte.mean() / p.mean()),
                          "slope": S.cal_slope(yte, p), "nb_020": S.net_benefit(yte, p, 0.2)}
        if k != "clin":
            r["models"][k]["delta_auc_vs_clin"] = S.cluster_bootstrap(ids, lambda yy, a_, b_: S.auc(yy, a_) - S.auc(yy, b_), (yte, p, P["clin"]), nboot, SEED)
            r["models"][k]["delta_nb_020_vs_clin"] = S.cluster_bootstrap(ids, lambda yy, a_, b_: S.net_benefit(yy, a_, 0.2) - S.net_benefit(yy, b_, 0.2), (yte, p, P["clin"]), nboot, SEED)
    res[design] = r
    print(design, f"train {len(tr)}/{int(ytr.sum())} test {len(te)}/{int(yte.sum())}")
    for k, v in r["models"].items():
        dd = v.get("delta_auc_vs_clin")
        print(f"   {k:16s} AUC {v['auc']['point']:.3f} ({v['auc']['lo']:.3f}-{v['auc']['hi']:.3f}) O/E {v['o_e']:.2f} slope {v['slope']:.2f}"
              + (f"  dAUC {dd['point']:+.3f} ({dd['lo']:+.3f},{dd['hi']:+.3f})" if dd else ""), flush=True)
write_json(res, OUT / "ml_temporal.json", script=__file__)
