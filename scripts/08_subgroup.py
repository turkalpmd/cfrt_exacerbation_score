"""Subgroup performance and fairness (R1-m10, R3-W4): discrimination AND calibration (O/E, CITL), flag rate and sensitivity
at threshold 0.20 for LR-5 (published) and tuned CatBoost; patient-level bootstrap CIs; <30 events flagged as underpowered."""
import numpy as np
from _common import setup
from config import N_BOOT, SEED
from lib import stats as S
from lib.io import write_json

a, df, OUT = setup(__doc__)
O = OUT / "oof"; nboot = 200 if a.fast else N_BOOT
res = {}
for out in ["y_pex", "y_decline10"]:
    y = np.load(O / f"clf_{out}_y.npy"); ids = np.load(O / f"clf_{out}_ids.npy", allow_pickle=True)
    rows = df[df[out].notna()].reset_index(drop=True)
    P = {k: np.load(O / f"clf_{out}_{f}.npy") for k, f in [("clin", "clin_raw"), ("ml", "ml_raw"), ("lr5", "lr5_raw"), ("catboost_fe_tuned", "catboost_fe_tuned_raw")] if (O / f"clf_{out}_{f}.npy").exists()}
    vc = rows.institution.value_counts()
    G = {
        "Children (<18 y)": rows.adult == 0, "Adults (≥18 y)": rows.adult == 1,
        "Age 6–9 y": rows.AgeAtVisit_lag_1.between(6, 9), "Age 10–13 y": rows.AgeAtVisit_lag_1.between(10, 13), "Age 14–17 y": rows.AgeAtVisit_lag_1.between(14, 17),
        "Male": rows.Sex == 1, "Female": rows.Sex == 2,
        "FEV1 <70% predicted": rows.FEV1_PP_lag_1 < 70, "FEV1 ≥70% predicted": rows.FEV1_PP_lag_1 >= 70,
        "P. aeruginosa positive": rows.pa_any_2y == 1, "P. aeruginosa negative": rows.pa_any_2y == 0,
        "IV antibiotics in t−1/t": rows.iv_burden_2yr_lag > 0, "No IV antibiotics in t−1/t": rows.iv_burden_2yr_lag.fillna(0) == 0,
        **({"Genotype complete": rows.cftr_incomplete == 0, "Genotype incomplete": rows.cftr_incomplete == 1,
            "F508del (≥1 allele)": rows.cftr_n_f508del >= 1, "No F508del": (rows.cftr_n_f508del == 0) & (rows.cftr_incomplete == 0),
            "ETI-eligible genotype": rows.cftr_eti_eligible == 1, "Not ETI-eligible": rows.cftr_eti_eligible == 0,
            "Severe genotype (class I–III)": rows.cftr_class == "severe", "Mild genotype (class IV–VI)": rows.cftr_class == "mild"} if "cftr_incomplete" in rows else {}),
        "Pancreatic insufficient": rows.Pancreatic_lag_1 == 1, "Pancreatic sufficient": rows.Pancreatic_lag_1 == 0,
        "Largest centre": rows.institution == vc.index[0], "Other centres": rows.institution != vc.index[0],
        "Small centres (<40 rows)": rows.institution.map(vc) < 40,
    }
    r = {}
    for g, m in G.items():
        m = m.fillna(False).to_numpy(); ev = int(y[m].sum()); n = int(m.sum())
        r[g] = {"n": n, "events": ev, "underpowered": ev < 30}
        if ev >= 5 and n - ev >= 5:
            for k, p in P.items():
                f = p[m] >= 0.20
                r[g][k] = {"auc": S.cluster_bootstrap(ids[m], S.auc, (y[m], p[m]), nboot, SEED),
                           "o_e": S.cluster_bootstrap(ids[m], lambda yy, pp: yy.mean() / pp.mean(), (y[m], p[m]), nboot, SEED),
                           "citl": S.citl(y[m], p[m]), "flag_pct_0.20": float(100 * f.mean()),
                           "sens_0.20": float(y[m][f].sum() / ev)}
    res[out] = r
    print(out); [print(f"   {g:34s} n={v['n']:4d} ev={v['events']:3d}", {k: (round(v[k]['auc']['point'], 2), round(v[k]['o_e']['point'], 2)) for k in P if k in v}) for g, v in r.items()]
write_json(res, OUT / "subgroup_summary.json", "08_subgroup.py")
