"""Leave-one-centre-out for next-year FEV1 % predicted and z-score (same centre groups as 19_extra_validation): carried forward,
two-year linear, clinical linear, machine learning (CatBoost on the engineered panel predicting the change; fixed parameters) and
the main-text machine-learning model (SelectedLGBMRegressor on the change; selection and Optuna inside each development set)."""
import numpy as np
from catboost import CatBoostRegressor
from _common import setup
from config import N_BOOT, SEED
from lib import stats as S
from lib.features import CLIN, FE_PANEL, GENOTYPE_FEATURES
from lib.io import write_json
from lib.mlx import SelectedLGBMRegressor
from lib.models import ridge_pipe
from lib.features import ML_PANEL
from joblib import Parallel, delayed

a, df, OUT = setup(__doc__)
nboot = 200 if a.fast else N_BOOT; NTRIAL = 8 if a.fast else 40
FE = list(dict.fromkeys(FE_PANEL + [g for g in GENOTYPE_FEATURES if g in df.columns]))
ev = df.groupby("institution").y_pex.sum(); big = ev[ev >= 10].index
df["loco_group"] = np.where(df.institution.isin(big), df.institution, "Other centres (pooled)")
res = {"n_groups": int(df.loco_group.nunique())}
for tgt, anchor, lag2, tol in [("y_fev1_pp", "FEV1_PP_lag_1", "FEV1_PP_lag_2", (5, 10)), ("y_fev1_z", "FEV1_Z_lag_1", "FEV1_Z_lag_2", (0.5, 1.0))]:
    d = df[df[tgt].notna()].reset_index(drop=True); y = d[tgt].to_numpy(float); ids = d.PatCode.to_numpy()
    RC = list(dict.fromkeys([anchor, lag2] + CLIN + ["AgeAtVisit_lag_1"]))
    pr = {k: np.full(len(d), np.nan) for k in ["persistence", "ar2", "ridge_clin", "cb_change", "ml_change"]}

    def ml_group(g):
        te = (d.loco_group == g).to_numpy(); tr = ~te
        base = lambda m: d.loc[m, anchor].fillna(d.loc[m, lag2]).fillna(d.loc[tr, anchor].median()).to_numpy(float)
        m = SelectedLGBMRegressor(feats=ML_PANEL, seed=SEED, ntrial=NTRIAL).fit(d.loc[tr, ML_PANEL + ["PatCode"]], y[tr] - base(tr))
        return te, base(te) + m.predict(d.loc[te, ML_PANEL])
    for te, p_ in Parallel(n_jobs=10)(delayed(ml_group)(g) for g in d.loco_group.unique()):
        pr["ml_change"][te] = p_
    for g in d.loco_group.unique():
        te = (d.loco_group == g).to_numpy(); tr = ~te
        base = lambda m: d.loc[m, anchor].fillna(d.loc[m, lag2]).fillna(d.loc[tr, anchor].median()).to_numpy(float)
        pr["persistence"][te] = base(te)
        pr["ar2"][te] = ridge_pipe(alpha=1e-6).fit(d.loc[tr, [anchor, lag2]].to_numpy(float), y[tr]).predict(d.loc[te, [anchor, lag2]].to_numpy(float))
        pr["ridge_clin"][te] = ridge_pipe().fit(d.loc[tr, RC].to_numpy(float), y[tr]).predict(d.loc[te, RC].to_numpy(float))
        m = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=3, l2_leaf_reg=20, random_seed=SEED, verbose=False, allow_writing_files=False)
        pr["cb_change"][te] = base(te) + m.fit(d.loc[tr, FE], y[tr] - base(tr)).predict(d.loc[te, FE])
    res[tgt] = {k: {"r2": S.cluster_bootstrap(ids, S.r2, (y, p), nboot, SEED), "mae": S.mae(y, p),
                    f"within_{tol[0]}": float(np.mean(np.abs(y - p) <= tol[0])), f"within_{tol[1]}": float(np.mean(np.abs(y - p) <= tol[1]))} for k, p in pr.items()}
    print(tgt, {k: (round(v["r2"]["point"], 3), round(v["mae"], 2), round(v[f"within_{tol[0]}"], 3)) for k, v in res[tgt].items()})
write_json(res, OUT / "loco_fev1.json", "24_loco_fev1.py")
