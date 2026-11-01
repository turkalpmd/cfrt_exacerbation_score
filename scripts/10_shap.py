"""SHAP drivers of the deployed machine-learning models (main text Fig. 3C; Supplement Figs S3/S4). The deployed model is the
procedure validated in cross-validation applied to the whole cohort: curated panel -> 10–15 features by SHAP rank (top 20,
number by inner CV) -> Optuna-tuned LightGBM (lib/mlx.SelectedLGBM). Exacerbation classifier (log-odds) and next-year FEV1 % predicted CHANGE model (percentage points;
what moves next year's value away from this year's). TreeSHAP from LightGBM (pred_contrib). Per-feature mean |SHAP|, direction
(Spearman of feature value with SHAP), grouped SHAP. SHAP values explain the model; they are not effect estimates."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from _common import setup
from config import SEED
from lib.features import LABELS, ML_PANEL
from lib.io import write_json
from lib.mlx import SelectedLGBM, SelectedLGBMRegressor

a, df, OUT = setup(__doc__)
NTRIAL = 8 if a.fast else 40
GROUPS = {"Lung function": ["FEV1_Z_lag_1", "fev1_z_trend_lag", "fev1_fvc_z_gap"],
          "IV antibiotics and exacerbations": ["ivDaysTotal_lag_1", "ivDaysTotal_lag_2", "iv_both_years", "PExNumber_lag_1", "PExNumber_lag_2"],
          "Growth": ["bmiZscore_lag_1", "bmi_trend_lag", "heightZScore_lag_1"],
          "Microbiology and ABPA": ["PseudomAer_lag_1", "PseudomAer_lag_2", "MRSA_lag_1", "MRSA_lag_2", "MSSA_lag_1", "ABPA_lag_1", "ABPA_lag_2",
                                    "NontubMycobacteria_lag_1", "Afumigatus_lag_1", "HaemInf_lag_1"],
          "Age, diagnosis and genotype": ["AgeAtVisit_lag_1", "Sex", "AgeDiagnosis", "SweatTestVal", "NeonatalScr", "MeconiumIleus",
                                          "cftr_n_f508del", "cftr_mild", "cftr_incomplete"],
          "Comorbidity and treatment": ["Pancreatic_lag_1", "Diabetes_lag_1", "LiverDisease_lag_1", "Azithro_lag_1", "rhDNase_lag_1", "NaCl_lag_1",
                                        "Antibiotic_lag_1", "Oxygen_lag_1", "NIPPV_lag_1", "SteroidOral_lag_1", "PPI_lag_1", "Bronchodil_lag_1",
                                        "SteroidInhaled_lag_1", "Ursodeoxyc_lag_1"]}
assert sorted(c for v in GROUPS.values() for c in v) == sorted(ML_PANEL)

res = {}
for out, proba in [("y_pex", True), ("y_fev1_pp_change", False)]:
    d = df[df[out].notna()].reset_index(drop=True)
    est = SelectedLGBM if proba else SelectedLGBMRegressor
    m = est(feats=ML_PANEL, seed=SEED, ntrial=NTRIAL).fit(d[ML_PANEL + ["PatCode"]], d[out])
    contrib = m.contributions(d[ML_PANEL])[:, :-1]
    sv = np.zeros((len(d), len(ML_PANEL)))
    for j, f in enumerate(m.selected_):
        sv[:, ML_PANEL.index(f)] = contrib[:, j]
    imp = pd.Series(np.abs(sv).mean(0), index=ML_PANEL).sort_values(ascending=False)
    imp = imp[imp > 0]
    direction = {f: float(spearmanr(d[f], sv[:, ML_PANEL.index(f)], nan_policy="omit")[0]) for f in imp.index[:15]}
    grouped = {g: float(np.abs(sv[:, [ML_PANEL.index(c) for c in cols]].sum(1)).mean()) for g, cols in GROUPS.items()}
    res[out] = {"mean_abs_shap": imp.round(4).to_dict(), "scale": "log-odds" if proba else "FEV1 % predicted points (change from anchor year)",
                "direction_spearman": direction, "grouped_mean_abs_shap": dict(sorted(grouped.items(), key=lambda kv: -kv[1])),
                "deployed_model": {"n": len(d), "selected": m.selected_, "n_selected": len(m.selected_), "shap_rank_candidate_panel": m.selection_freq_,
                                   "inner_loss_by_n_features": m.k_loss_, "best_params": m.best_params_},
                "features": ML_PANEL}
    np.save(OUT / "oof" / f"shap_{out}.npy", sv)
    # trained model for the code repository (no patient data): LightGBM text model + the inputs it needs
    md_ = OUT / "models"; md_.mkdir(exist_ok=True)
    m.model_.booster_.save_model(str(md_ / f"ml_{out}.lgbm.txt"))
    (md_ / f"ml_{out}.json").write_text(__import__("json").dumps(
        {"outcome": out, "output": "probability" if proba else "change in FEV1 % predicted from the anchor year",
         "inputs_in_order": m.selected_, "input_labels": {f: LABELS[f] for f in m.selected_}, "missing_values": "allowed (LightGBM native handling)",
         "hyperparameters": m.best_params_, "n_development": len(d)}, indent=1, ensure_ascii=False))
    print(out, f"selected {len(m.selected_)}:", m.selected_)
    print("   ", list(imp.head(8).round(3).items()), "| grouped:", {k[:18]: round(v, 3) for k, v in res[out]["grouped_mean_abs_shap"].items()})
write_json(res, OUT / "shap_summary.json", "10_shap.py")
