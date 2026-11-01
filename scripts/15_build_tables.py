"""Main Tables 1–2 and supplementary tables (CSV), built only from results JSON (November, JCF: <=5 display items)."""
import json
import numpy as np
import pandas as pd
from _common import setup
from config import DATA, TABLES
from lib.features import CLIN, FE_PANEL, GENOTYPE_FEATURES, LABELS, MAIN_PANEL, ML_PANEL

a, df, OUT = setup(__doc__)
J = lambda n: {k: v for k, v in json.loads((OUT / n).read_text().replace("NaN", "null")).items() if not k.startswith("_")}
fx = lambda v, k=2: (f"{v:.{k}f}" if round(v, k) != 0 else f"{0:.{k}f}").replace("-", "−")
ci = lambda d, k=2: f"{fx(d['point'], k)} ({fx(d['lo'], k)}–{fx(d['hi'], k)})" if min(d["lo"], d["hi"]) >= 0 else f"{fx(d['point'], k)} ({fx(d['lo'], k)} to {fx(d['hi'], k)})"
pc = lambda v: f"{100 * v:.0f}%"
save = lambda name, t: t.to_csv(TABLES / f"{name}.csv", index=False)

# ---------- Table 1
save("Table1_baseline", pd.read_csv(TABLES / "table1_patients.csv"))

# ---------- Table 2: performance, three outcomes, three validation designs
fc, cl, ca, tv, ex, lf = J("fev1_forecast.json"), J("classification_summary.json"), J("calibration_summary.json"), J("temporal_validation.json"), J("extra_validation.json"), J("loco_fev1.json")
lo = ex["leave_institution_out"]; ty = tv["test_year"]; pdj = tv["patient_disjoint"]
COLS = ["Outcome and model", "Cross-validation (95% CI)", "What it means for a patient", f"Validation in {ty} (patients not used for development)", "Leave-one-centre-out"]
rows = [[f"**Next-year FEV1 % predicted** (n={fc['y_fev1_pp']['n_rows']}): R²", "", "", "", ""]]
FCN = {"persistence": "This year's value carried forward", "ar2": "Two-year linear model (FEV1 in t and t−1)", "ridge_clin": "Clinical linear model",
       "cb_level": "CatBoost, previous panel (level)", "cb_change": "CatBoost, engineered features (change)", "ridge_change": "Linear, engineered features (change)",
       "ml_change": "Machine learning (main model)"}
TVK = {"persistence": "persistence", "ar2": "ar2_linear", "ridge_clin": "ridge_clin", "cb_change": "catboost_change_fe", "ml_change": "ml_change"}
# main text: the two tools only (comparators in Supplementary Tables S3b, S6, S13)
for k, lab in [("persistence", "This year's value carried forward"), ("ml_change", "Machine learningᵇ")]:
    m = fc["y_fev1_pp"]["models"][k]; t = pdj["y_fev1_pp"][TVK[k]]; l_ = lf["y_fev1_pp"][k]
    rows.append([lab, ci(m["r2"]), f"Within ±5 / ±10 points: {pc(m['within_5']['point'])} / {pc(m['within_10']['point'])}; mean error {m['mae']['point']:.1f}; 95% PI ±{m['pi95_halfwidth_median']:.0f}",
                 f"R² {t['r2']['point']:.2f}; mean error {t['mae']:.1f}", f"R² {l_['r2']['point']:.2f}; mean error {l_['mae']:.1f}"])
CALK = {"clin": "clin_raw", "ml": "ml_raw", "catboost_fe_tuned": "catboost_fe_tuned_raw", "lr5": "lr5_raw", "prior_iv_logistic": "prior_iv_logistic_raw"}
TOOLS = [("clin", "CFRT exacerbation scoreᵃ", "CFRT exacerbation score"), ("ml", "Machine learningᵇ", "Machine learning")]
for out, head in [("y_pex", "IV-treated exacerbation in the next year"), ("y_decline10", "Fall of ≥10 points in FEV1 % predicted")]:
    c = cl[out]; opx = J("dca_summary.json")[out]["operating_points"]
    rows.append([f"**{head}** (n={c['n_rows']}, events={c['n_events']}): AUC", "", "", "", ""])
    for k, lab, dlab in TOOLS:
        lab = "Five score variables, refitted to this outcomeᵃ" if (out == "y_decline10" and k == "clin") else lab
        m = c["models"][k]; cv = ca[out][CALK[k]]; o = opx.get(f"{dlab} @ 0.2"); t = pdj[out].get(k); l_ = lo[out]["meta"].get(k)
        mean = (f"At risk ≥0.20: flags {o['flagged_pct']:.0f}%, finds {100 * o['sensitivity']:.0f}% of events (PPV {o['ppv']:.2f}, NPV {o['npv']:.2f}); "
                f"calibration slope {cv['slope']['point']:.2f}") if out == "y_pex" else f"Calibration slope {cv['slope']['point']:.2f}"
        rows.append([lab, ci(m["auc"]), mean, f"AUC {t['auc']['point']:.2f}; O/E {t['o_e']:.2f}" if t else "–",
                     f"AUC {l_['auc']['pooled']:.2f} (PI {l_['auc']['pi_lo']:.2f}–{l_['auc']['pi_hi']:.2f})" if l_ else "–"])
save("Table2_performance", pd.DataFrame(rows, columns=COLS))

# ---------- Supplementary tables
md = J("model_development.json"); sp = J("lr5_model_spec.json")
NM = {"FEV1_Z_lag_1": "FEV1 z-score, anchor year (per 1 z)", "bmiZscore_lag_1": "BMI z-score, anchor year (per 1 z)", "PseudomAer_lag_1": "P. aeruginosa, anchor year",
      "pa_any_2y": "P. aeruginosa, t−1 or t", "MRSA_lag_1": "MRSA, anchor year", "mrsa_any_2y": "MRSA, t−1 or t", "ABPA_lag_1": "ABPA, anchor year",
      "abpa_any_2y": "ABPA, t−1 or t",
      "log_iv_burden_2yr": "IV-antibiotic days in t−1 and t (per unit of ln(1 + days))", "iv_burden_2yr_lag": "IV-antibiotic days in t−1 and t (per day)"}
rows = []
for k, lab in ((k, NM[k]) for k in md["candidates"]):
    f = md["y_pex"]["full_model"][k]; s = sp["y_pex"]["coefficients"].get(k)
    rows.append([lab, f"{f['or']:.2f} ({f['lo']:.2f}–{f['hi']:.2f}); " + ("p<0.001" if f["p"] < 0.001 else f"p={f['p']:.3f}"), pc(md["y_pex"]["selection_frequency_across_folds"][k]),
                 f"{s['or']:.2f} ({s['or_lo']:.2f}–{s['or_hi']:.2f})" if s else "removed",
                 (lambda s2: f"{s2['or']:.2f} ({s2['or_lo']:.2f}–{s2['or_hi']:.2f})" if s2 else "–")(sp["y_decline10"]["coefficients"].get(k))])
save("TableS3a_model_development", pd.DataFrame(rows, columns=["Candidate predictor", "Exacerbation: OR, all six candidates (95% CI)", "Retained in training folds",
                                                              "Exacerbation: OR, final model (95% CI)", "≥10-point fall: OR, same variables (95% CI)"]))
MDN = {"selected": "CFRT exacerbation score (final five-variable model)", "backward_in_fold": "Backward elimination repeated in each training fold", "index_free_full": "All six candidates",
       "selected_linear_iv": "IV days as a linear term", "selected_without_pa": "Without P. aeruginosa", "selected_plus_age": "Plus age", "selected_plus_adult": "Plus adult (≥18 y)",
       "lr5_v3": "LR-5 (severity index, FEV1 <70%)", "lr5_severity80": "LR-5 (severity index, FEV1 <80%)", "prior_iv_logistic": "IV-antibiotic days alone"}
rows = []
for k, lab in MDN.items():
    a_, b_ = md["y_pex"]["models"][k], md["y_decline10"]["models"][k]
    dd = md["y_pex"]["delta_vs_selected"].get(k)
    rows.append([lab, ci(a_["auc"], 3), ci(dd["dauc"], 3) if dd else "reference", f"{a_['logloss']:.4f}", f"{a_['slope']:.2f}", ci(b_["auc"], 3)])
if (OUT / "pa_vs_genotype.json").exists():          # authors' question: CFTR class genotype instead of / in addition to P. aeruginosa
    pg = J("pa_vs_genotype.json")["all"]
    for k, lab in [("genotype instead of P. aeruginosa (mild, unknown vs severe)", "CFTR class genotype instead of P. aeruginosa"),
                   ("P. aeruginosa and genotype (mild, unknown vs severe)", "Plus CFTR class genotype")]:
        v = pg[k]; rows.append([lab, ci(v["auc"], 3), ci(v["delta_auc_vs_clinical"], 3), f"{v['logloss']:.4f}", f"{v['slope']:.2f}", "–"])
for k, lab in [("ml", "Machine learning (main model: curated panel, 10–15 features by SHAP rank, Optuna-tuned LightGBM)"),
               ("catboost_fe_tuned", "CatBoost, 71-variable engineered panel, grid-tuned and recalibrated")]:
    rows.append([lab, ci(cl["y_pex"]["models"][k]["auc"], 3), ci(cl["y_pex"]["delta_auc"][f"{k}_minus_clin"], 3), f"{cl['y_pex']['logloss'][k]:.4f}",
                 f"{ca['y_pex'][CALK[k]]['slope']['point']:.2f}", ci(cl["y_decline10"]["models"][k]["auc"], 3)])
save("TableS3b_model_comparison", pd.DataFrame(rows, columns=["Model", "Exacerbation AUC (95% CI)", "Δ AUC vs final model", "Log-loss (lower = better)", "Calibration slope", "≥10-point fall AUC"]))

# ---------- Table S3d: window of the culture results (co-author review: P. aeruginosa and MRSA counted over the same years)
cw = J("culture_window.json"); rows = []
for k, v in cw["y_pex"]["models"].items():
    pa, mr = (next(v["or"][c] for c in v["or"] if c in grp) for grp in (("PseudomAer_lag_1", "pa_any_2y"), ("MRSA_lag_1", "mrsa_any_2y")))
    rows.append([k.replace("Pa ", "P. aeruginosa ") + (" (final score)" if k == cw["reference"] else ""), ci(v["cv"]["auc"], 3),
                 ci(v["dauc_vs_ref"], 3) if v["dauc_vs_ref"] else "reference", f"{v['cv']['logloss']:.4f}", f"{v['cv']['slope']:.2f}",
                 f"{v['temporal']['auc']['point']:.3f}; {v['temporal']['o_e']:.2f}",
                 f"{pa['or']:.2f} ({pa['lo']:.2f}–{pa['hi']:.2f})", f"{mr['or']:.2f} ({mr['lo']:.2f}–{mr['hi']:.2f})"])
save("TableS3d_culture_window", pd.DataFrame(rows, columns=["Culture window", "AUC (95% CI)", "Δ AUC vs final score", "Log-loss", "Calibration slope",
                                                            f"{cw['y_pex']['test_year']}: AUC; O/E", "OR P. aeruginosa (95% CI)", "OR MRSA (95% CI)"]))

# ---------- Table S3c: feature selection of the machine-learning model (share of the 25 training sets; deployed model on the full cohort)
from lib.features import LABELS, ML_PANEL
shs = J("shap_summary.json"); fsel = fc["y_fev1_pp"].get("ml_change_selection", {})
fr = lambda dct, f: pc(dct.get(f, 0.0))
rows = [[LABELS[f], fr(cl["y_pex"]["ml_selection"]["frequency"], f), "✓" if f in shs["y_pex"]["deployed_model"]["selected"] else "",
         fr(cl["y_decline10"]["ml_selection"]["frequency"], f), fr(fsel.get("frequency", {}), f), "✓" if f in shs["y_fev1_pp_change"]["deployed_model"]["selected"] else ""]
        for f in ML_PANEL]
rows.sort(key=lambda r_: -float(r_[1].rstrip("%")))
nsel = lambda b: f"{b['n_selected']['median']:.0f} ({b['n_selected']['min']}–{b['n_selected']['max']})"
rows.append(["Features selected per training set, median (range)", nsel(cl["y_pex"]["ml_selection"]), str(shs["y_pex"]["deployed_model"]["n_selected"]),
             nsel(cl["y_decline10"]["ml_selection"]), nsel(fsel) if fsel else "–", str(shs["y_fev1_pp_change"]["deployed_model"]["n_selected"])])
hp = lambda d_: "; ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in d_.items())
rows.append(["Hyperparameters of the deployed model", "", hp(shs["y_pex"]["deployed_model"]["best_params"]), "", "", hp(shs["y_fev1_pp_change"]["deployed_model"]["best_params"])])
save("TableS3c_ml_feature_selection", pd.DataFrame(rows, columns=["Candidate feature", "Exacerbation: selected in training sets", "Exacerbation: deployed model",
                                                                  "≥10-point fall: selected in training sets", "Next-year FEV1: selected in training sets", "Next-year FEV1: deployed model"]))

# ---------- Table S15: case mix of the development and 2024 validation data (patient-disjoint design; TRIPOD+AI 16, 20c)
_r = tv["patient_disjoint"]["y_pex"]; _cm = _r["case_mix"]
_pct = lambda k: [f"{100 * _cm[k]['train']:.1f}", f"{100 * _cm[k]['test']:.1f}"]
rows = [["Prediction-years / patients", f"{_r['n_train']} / {_r['train_patients']}", f"{_r['n_test']} / {_r['test_patients']}"],
        ["IV-treated exacerbation, %", f"{100 * _r['events_train'] / _r['n_train']:.1f}", f"{100 * _r['events_test'] / _r['n_test']:.1f}"],
        ["Age in the anchor year, mean", f"{_cm['AgeAtVisit_lag_1']['train']:.1f}", f"{_cm['AgeAtVisit_lag_1']['test']:.1f}"],
        ["Adults, %", *_pct("adult")],
        ["FEV1 z-score, mean", f"{_cm['FEV1_Z_lag_1']['train']:.2f}", f"{_cm['FEV1_Z_lag_1']['test']:.2f}"],
        ["BMI z-score, mean", f"{_cm['bmiZscore_lag_1']['train']:.2f}", f"{_cm['bmiZscore_lag_1']['test']:.2f}"],
        ["P. aeruginosa in t−1 or t, %", *_pct("pa_any_2y")],
        ["MRSA in t−1 or t, %", *_pct("mrsa_any_2y")],
        ["IV antibiotics in t−1 or t, %", *_pct("any_iv_2y")]]
save("TableS15_case_mix_2024", pd.DataFrame([[r_[0], r_[1].replace("-", "−"), r_[2].replace("-", "−")] for r_ in rows],
                                            columns=["Characteristic", f"Development (outcome years before {ty})", f"Validation ({ty}, patients not used for development)"]))

# ---------- Table S14: pre-specified curation of the machine-learning panel (pairs with |rho| >= 0.8 in the 71-variable panel; sparse items)
_fe71 = list(dict.fromkeys(FE_PANEL + GENOTYPE_FEATURES + CLIN)); _X = df[_fe71].astype(float); _r = _X.corr(method="spearman").abs()
rows = []
for i, a_ in enumerate(_fe71):
    for b_ in _fe71[i + 1:]:
        if _r.loc[a_, b_] >= 0.8:
            kept = [v for v in (a_, b_) if v in ML_PANEL]
            rows.append([a_, b_, f"{_r.loc[a_, b_]:.2f}", ", ".join(LABELS.get(v, v) for v in kept) or "neither (derived or duplicate)"])
for v in _fe71:
    if _X[v].nunique() <= 2 and min(_X[v].mean(), 1 - _X[v].mean()) < 0.02:
        rows.append([v, "–", f"{100 * _X[v].mean():.1f}% positive", "excluded (<2% positive)"])
_mx = df[ML_PANEL].astype(float).corr(method="spearman").abs().where(lambda m_: ~np.eye(len(ML_PANEL), dtype=bool)).max().max()
rows.append(["Curated panel", f"{len(ML_PANEL)} variables", f"largest |ρ| {_mx:.2f}", "–"])
save("TableS14_ml_panel_curation", pd.DataFrame(rows, columns=["Variable 1", "Variable 2", "Spearman |ρ| or prevalence", "Kept in the curated panel"]))

# ---------- Table S13: machine-learning experiments (nested cross-validation; scripts/27_ml_experiments.py)
if (OUT / "ml_experiments.json").exists():
    mx = J("ml_experiments.json")
    MXL = {"clin": "CFRT exacerbation score (reference)", "lgbm_optuna": "LightGBM, Optuna-tuned", "lgbm_monotone": "LightGBM, Optuna-tuned, monotone constraints",
           "lgbm_offset": "LightGBM boosted from the score", "lgbm_clin5": "LightGBM on the five score variables",
           "enet_grid": "Elastic-net logistic regression, grid search", "ebm": "Explainable boosting machine",
           "avg_clin_lgbm": "Average of the score and LightGBM", "avg_clin_ebm": "Average of the score and EBM",
           "catboost_fe_tuned (main pipeline)": "CatBoost, grid-tuned (main analysis)"}
    rows = [[lab, ci(v["auc"], 3), ci(v["delta_auc_vs_clin"], 3) if "delta_auc_vs_clin" in v else "reference", f"{v['logloss']:.4f}", f"{v['slope']:.2f}",
             (lambda d_: f"{100 * d_['point']:.1f} ({100 * d_['lo']:.1f} to {100 * d_['hi']:.1f})")(v["delta_nb_020_vs_clin"]).replace("-", "−") if "delta_nb_020_vs_clin" in v else "reference"]
            for k, lab in MXL.items() if (v := mx["classification"].get(k))]
    _kt = J("dca_summary.json")["y_pex"]["key_thresholds"]["0.2"]["CFRT exacerbation score minus Machine learning"]
    rows.insert(1, ["Machine learning (main model: curated panel, 10–15 features by SHAP rank, Optuna-tuned LightGBM)", ci(cl["y_pex"]["models"]["ml"]["auc"], 3),
                    ci(cl["y_pex"]["delta_auc"]["ml_minus_clin"], 3), f"{cl['y_pex']['logloss']['ml']:.4f}", f"{ca['y_pex']['ml_raw']['slope']['point']:.2f}",
                    f"{-100 * _kt['point']:.1f} ({-100 * _kt['hi']:.1f} to {-100 * _kt['lo']:.1f})".replace("-", "−"),
                    f"AUC {pdj['y_pex']['ml']['auc']['point']:.3f}"])
    for i_, (k_, lab_) in enumerate([("ml_ownshadow", "Same, features kept when better than their own permuted copy (Boruta-type)"),
                                      ("ml_maxshadow", "Same, classic Boruta rule (gain above the largest permuted copy)")]):
        sel_ = cl["y_pex"][f"{k_}_selection"]["n_selected"]
        rows.insert(2 + i_, [f"{lab_}; {sel_['min']}–{sel_['max']} features", ci(cl["y_pex"]["models"][k_]["auc"], 3), ci(cl["y_pex"]["delta_auc"][f"{k_}_minus_clin"], 3),
                             f"{cl['y_pex']['logloss'][k_]:.4f}", f"{ca['y_pex'][k_ + '_raw']['slope']['point']:.2f}", "–", "not tested"])
    mt = J("ml_temporal.json")["patient_disjoint"]["models"] if (OUT / "ml_temporal.json").exists() else {}
    for row, k in zip([r_ for i_, r_ in enumerate(rows) if i_ not in (1, 2, 3)], [k for k in MXL if k in mx["classification"]]):
        t_ = mt.get(k); row.append(("AUC " + f"{t_['auc']['point']:.3f}" + (f"; Δ {ci(t_['delta_auc_vs_clin'], 3)}" if "delta_auc_vs_clin" in t_ else "")) if t_ else "not tested")
    save("TableS13a_ml_exacerbation", pd.DataFrame(rows, columns=["Model (71-variable engineered panel unless stated)", "AUC (95% CI)", "Δ AUC vs the score", "Log-loss",
                                                                   "Calibration slope", "Δ net benefit at 0.20, per 100", "2024, patients not used for development"]))
    MXR = {"persistence": "This year's value carried forward", "clinical linear model (main pipeline)": "Clinical linear model (main analysis)",
           "CatBoost change (main pipeline)": "CatBoost, change target (main analysis)", "ridge_base": "Ridge base model (clinical variables, FEV1 in t and t−1)",
           "lgbm_reg_optuna": "LightGBM, Optuna-tuned (loss tuned)", "lgbm_reg_offset": "LightGBM boosted from the ridge base model", "avg_ridge_lgbm": "Average of ridge base and LightGBM"}
    rows = [[lab, ci(v["r2"], 2), f"{v['mae']:.2f}", f"{100 * v['within_5']['point']:.1f}", f"{100 * v['within_10']['point']:.1f}",
             (lambda d_: f"{100 * d_['point']:+.1f} ({100 * d_['lo']:+.1f} to {100 * d_['hi']:+.1f})")(v["delta_within_5_vs_persistence"]).replace("-", "−") if "delta_within_5_vs_persistence" in v else "reference"]
            for k, lab in MXR.items() if (v := mx["fev1"]["models"].get(k))]
    _m, _d = fc["y_fev1_pp"]["models"]["ml_change"], fc["y_fev1_pp"]["delta_vs_persistence"]["ml_change"]["within_5"]
    rows.insert(1, ["Machine learning (main model: curated panel, 10–15 features by SHAP rank, Optuna-tuned LightGBM)", ci(_m["r2"], 2), f"{_m['mae']['point']:.2f}",
                    f"{100 * _m['within_5']['point']:.1f}", f"{100 * _m['within_10']['point']:.1f}",
                    f"{100 * _d['point']:+.1f} ({100 * _d['lo']:+.1f} to {100 * _d['hi']:+.1f})".replace("-", "−")])
    save("TableS13b_ml_fev1", pd.DataFrame(rows, columns=["Model (next-year FEV1 % predicted)", "R² (95% CI)", "Mean absolute error", "Within ±5, %", "Within ±10, %",
                                                          "Δ within ±5 vs carried forward, percentage points"]))
dc = J("dca_summary.json")["y_pex"]; op = dc["operating_points"]; kt = dc["key_thresholds"]
rows = []
for t in [0.15, 0.2, 0.25, 0.3]:
    o = op[f"CFRT exacerbation score @ {t}"]; k = kt[str(t)]
    rows.append([f"CFRT exacerbation score, risk ≥{t:.2f}", f"{o['flagged_pct']:.0f}", f"{100 * o['sensitivity']:.0f}", f"{100 * o['specificity']:.0f}", f"{o['ppv']:.2f}", f"{o['npv']:.2f}",
                 f"{o['flagged_per_event_found']:.1f}", ci(k["CFRT exacerbation score minus Rule: any IV (t−1/t)"], 3), ci(k["CFRT exacerbation score minus Rule: severity index ≥2"], 3),
                 ci(k["CFRT exacerbation score minus Machine learning"], 3)])
for lab in ["Rule: any IV (t−1/t)", "Rule: severity index ≥2", "Rule: severity index (FEV1 <80%) ≥2"]:
    o = op[lab]
    rows.append([lab, f"{o['flagged_pct']:.0f}", f"{100 * o['sensitivity']:.0f}", f"{100 * o['specificity']:.0f}", f"{o['ppv']:.2f}", f"{o['npv']:.2f}",
                 f"{o['flagged_per_event_found']:.1f}", "–", "–", "–"])
save("TableS5_operating_points", pd.DataFrame(rows, columns=["Strategy", "Flagged, %", "Sensitivity, %", "Specificity, %", "PPV", "NPV", "Flagged per event found",
                                                             "ΔNB vs any-IV rule", "ΔNB vs severity ≥2 rule", "ΔNB vs machine learning"]))
rows = []
for tgt, lab in [("y_fev1_pp", "FEV1 % predicted"), ("y_fev1_z", "FEV1 z-score"), ("y_fvc_pp", "FVC % predicted"), ("y_fvc_z", "FVC z-score")]:
    b = fc[tgt]; t1, t2 = b["tolerances"]
    for k, m in b["models"].items():
        rows.append([lab, {"cb_level": "CatBoost, previous panel (level)", "ridge_change": "Linear, engineered features (change)", **FCN}[k], ci(m["r2"]), f"{m['mae']['point']:.2f}",
                     pc(m[f"within_{t1}"]["point"]), pc(m[f"within_{t2}"]["point"]), f"±{m['pi95_halfwidth_median']:.2f}", pc(m["pi95_coverage"]),
                     f"{m['by_age_group']['child']['r2']:.2f} / {m['by_age_group']['adult']['r2']:.2f}"])
save("TableS6_lung_function", pd.DataFrame(rows, columns=["Outcome", "Model", "R² (95% CI)", "Mean absolute error", "Within tolerance 1ᵃ", "Within tolerance 2ᵃ",
                                                          "95% PI half-width", "PI coverage", "R² children / adults"]))
ccx = J("consistency_checks.json"); s_ = ex["sensitivity_including_outcome_year_modulator_starters"]; bmi = ccx["bmi_z_reference"]["clin_harmonised"]
rows = [["Main analysis", f"{cl['y_pex']['n_rows']} / {cl['y_pex']['n_events']}", ci(cl["y_pex"]["models"]["clin"]["auc"]), "–"],
        ["Outcome: IV antibiotics with a recorded exacerbation", f"{cl['alt_outcomes']['y_pex_strict']['n']} / {cl['alt_outcomes']['y_pex_strict']['events']}", ci(cl["alt_outcomes"]["y_pex_strict"]["clin"]["auc"]), "–"],
        ["Outcome: ≥14 IV days", f"{cl['alt_outcomes']['y_iv14']['n']} / {cl['alt_outcomes']['y_iv14']['events']}", ci(cl["alt_outcomes"]["y_iv14"]["clin"]["auc"]), "–"],
        ["Outcome: any hospital days", f"{cl['alt_outcomes']['y_hosp']['n']} / {cl['alt_outcomes']['y_hosp']['events']}", ci(cl["alt_outcomes"]["y_hosp"]["clin"]["auc"]), "–"],
        ["Outcome-year modulator starters retained", f"{s_['n']} / {s_['events']}", ci(s_["auc"]), f"O/E in starters {s_['o_e_starters']:.2f}"],
        ["No 2020 (pandemic) year among predictor years", f"{ex['sensitivity_no_2020_predictor_year']['n']} / {ex['sensitivity_no_2020_predictor_year']['events']}", ci(ex["sensitivity_no_2020_predictor_year"]["auc"]), "–"],
        ["Adjusted for calendar year", "–", ci(ex["sensitivity_calendar_year_adjusted"]["auc"]), "–"],
        ["Weighted for selection into the cohort", "–", f"{ex['ipw_complete_history']['weighted_auc']:.2f}", f"O/E {ex['ipw_complete_history']['weighted_o_e']:.2f}"],
        ["BMI z recomputed with WHO 2007 (adults: 19-year reference)", "–", ci(bmi["who2007_bmi_z"]["cv_auc"]), f"2024 AUC {bmi['who2007_bmi_z']['temporal_2024_auc']:.2f}; O/E {bmi['who2007_bmi_z']['temporal_2024_o_e']:.2f}"],
        ["Only the anchor year's IV days (one year of history)", "–", ci(ex["one_prior_year_clin"]["auc"]), "–"],
        ["Four-year window: adding IV days in t−2", f"{ex['clin_plus_lag3_iv']['n']} / –", f"{ex['clin_plus_lag3_iv']['clin_lag3_auc']:.2f}", f"Δ {ci(ex['clin_plus_lag3_iv']['delta'], 3)}"],
        ["Bootstrap optimism correction (Harrell)", "–", f"{ex['harrell_optimism_clin']['corrected_auc']:.2f}", f"Corrected slope {ex['harrell_optimism_clin']['corrected_slope']:.2f}"]]
if "oof_in_neither" in ccx["earlier_extracts"]:   # needs the earlier internal extracts (not distributed)
    _e = ccx["earlier_extracts"]; rows.insert(len(rows) - 1, ["Prediction-years outside both earlier exploratory extracts", f"{_e['in_neither']} / {_e['oof_in_neither']['events']}",
                                          ci(_e["oof_in_neither"]["auc"]), f"O/E {_e['oof_in_neither']['o_e']:.2f}"])
rows.insert(5, ["Outcome-year modulator starters retained: machine learning", f"{s_['n']} / {s_['events']}", ci(s_["ml"]["auc"]), f"O/E in starters {s_['ml']['o_e_starters']:.2f}"])
for sc, v in cl["y_pex"]["masking"].items():
    rows.append([f"Missing inputs at prediction ({sc}): score / machine learning", f"{cl['y_pex']['n_rows']} / {cl['y_pex']['n_events']}",
                 f"{v['clin']['auc']['point']:.2f} / {v['ml']['auc']['point']:.2f}", f"Slope {v['clin']['slope']:.2f} / {v['ml']['slope']:.2f}"])
save("TableS8_sensitivity", pd.DataFrame(rows, columns=["Analysis (IV-treated exacerbation; CFRT exacerbation score unless stated)", "Prediction-years / events", "AUC (95% CI)", "Calibration"]))
ge = J("gee_associations.json"); bh = {(b["outcome"], b["term"]): b["q"] for b in ge["bh_family"]}
rows = []
LBL = {"f508del": "F508del (vs no F508del)", "genotype_class": "CFTR class genotype (vs severe, both variants class I–III)", "eti_eligible": "ETI-eligible genotype (FDA label)",
       "cfrd_anchor": "CF-related diabetes, anchor year", "cfld_anchor": "CF liver disease, anchor year", "cfld_ever": "CF liver disease, t−1 or t"}
for k, lab in LBL.items():
    for term, v in ge["y_pex"][k]["total_effect"].items():
        w = ge["y_decline10"][k]["total_effect"][term]
        nm = term.split("[T.")[-1].rstrip("]") if "[T." in term else ""
        rows.append([lab + (f": {nm}" if nm else ""), f"{v['est']:.2f} ({v['lo']:.2f}–{v['hi']:.2f})", f"{bh[('y_pex', term)]:.3f}",
                     f"{w['est']:.2f} ({w['lo']:.2f}–{w['hi']:.2f})", f"{bh[('y_decline10', term)]:.3f}"])
save("TableS10_genotype_comorbidity", pd.DataFrame(rows, columns=["Exposure", "Exacerbation OR (95% CI)", "q (BH)", "≥10-point fall OR (95% CI)", "q (BH)"]))
rv = ex["riley_sample_size"]; mi = J("missingness.json")
save("TableS11_sample_size", pd.DataFrame([[lab, rv[k]["parameters"], f"{rv[k]['n_required']:.0f}", f"{rv[k]['n_required_with_design_effect']:.0f}", rv[k]["n_available"], rv[k]["events_available"]]
                                           for k, lab in [("clin", "CFRT exacerbation score"), ("ml_panel", "Machine learning: curated candidate panel"), ("fe_panel", "Comparators: 71-variable engineered panel")]],
                                          columns=["Model", "Parameters", "Required prediction-years (Riley)", "Required, allowing for repeated years per patient", "Available", "Events"]))
lo_ = lo["y_pex"]["meta"]
save("TableS9_leave_centre_out", pd.DataFrame([[{"clin": "CFRT exacerbation score", "lr5": "LR-5 (severity index)", "prior_iv_logistic": "IV-antibiotic days alone", "ml": "Machine learning", "catboost_fe_tuned": "CatBoost (comparator)"}[k],
                                                f"{v['auc']['pooled']:.3f} ({v['auc']['lo']:.3f}–{v['auc']['hi']:.3f})", f"{v['auc']['pi_lo']:.2f}–{v['auc']['pi_hi']:.2f}",
                                                f"{v['o_e']['pooled']:.2f} ({v['o_e']['lo']:.2f}–{v['o_e']['hi']:.2f})", f"{v['o_e']['pi_lo']:.2f}–{v['o_e']['pi_hi']:.2f}"] for k, v in lo_.items()],
                                              columns=["Model", "Pooled AUC (95% CI)", "AUC 95% prediction interval", "Pooled O/E (95% CI)", "O/E 95% prediction interval"]))
print(sorted(p.name for p in TABLES.iterdir()))
print(pd.read_csv(TABLES / "Table2_performance.csv").to_string())

# ---------- Table S2: predictors and missingness; Table S4: equation; Table S12: CFTR variants in the cohort
from lib.features import CLIN, FE_PANEL, GENOTYPE_FEATURES, MAIN_PANEL
miss = mi["pct_missing"]
use = lambda f: " · ".join(u for u, ok in [("score", f in CLIN), ("machine learning (candidate)", f in ML_PANEL),
                                           ("comparators", f in FE_PANEL or f in GENOTYPE_FEATURES)] if ok)
rows = [[LABELS.get(f, f), f, use(f), f"{miss.get(f, 0.0):.1f}"] for f in dict.fromkeys(CLIN + ML_PANEL + FE_PANEL + GENOTYPE_FEATURES)]
save("TableS2_predictors", pd.DataFrame(rows, columns=["Variable", "Analysis name", "Used in", "Missing, %"]))
rows = []
for o, lab in [("y_pex", "IV-treated exacerbation"), ("y_decline10", "≥10-point fall in FEV1 % predicted")]:
    s_ = sp[o]; rows.append([lab, __import__("re").sub(r"(?<![\w])-(?=\d)", "−", s_["formula"]), f"{s_['n']} / {s_['events']}",
                             ", ".join(f"{k}: {v:.2f}" for k, v in s_["imputation_medians"].items())])
save("TableS4_equation", pd.DataFrame(rows, columns=["Outcome", "Equation (p = 1 / (1 + e^−logit))", "Prediction-years / events", "Imputation medians"]))
fv = pd.read_csv(DATA / "cohort_lag2_first_visit.csv")
vm = pd.read_csv(DATA / "reference" / "cftr_variant_map.csv", dtype=str).drop_duplicates("normalised_legacy_name").set_index("normalised_legacy_name")
al = pd.concat([fv.cftr_allele1, fv.cftr_allele2]).value_counts()
al = al[~al.index.str.startswith("(")].head(20)
from lib.cftr import CLASS_CURATED, class_severity
def _cls(n):
    fc = CLASS_CURATED.get(n, vm.loc[n, "functional_class"] if n in vm.index else "")
    sv = class_severity(fc); return f"{str(fc).split(' ')[0]} ({sv})" if sv != "unclassified" else "not classified"
rows = [[n, int(c), vm.loc[n, "cftr2_status"] if n in vm.index else "–", _cls(n),
         vm.loc[n, "eti_fda_eligible"] if n in vm.index else "–"] for n, c in al.items()]
save("TableS12_cftr_variants", pd.DataFrame(rows, columns=["Variant (legacy name)", "Alleles, n (patients' first prediction-year)", "CFTR2 status", "Functional class", "ETI-eligible (FDA 2024)"]))
