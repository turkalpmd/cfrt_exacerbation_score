"""Number sheet: every figure quoted in the manuscript and the Turkish report, read from results/*.json (never retyped by hand)."""
import json
from _common import setup
from config import RESULTS

a, df, OUT = setup(__doc__)
J = lambda n: json.loads((OUT / n).read_text().replace("NaN", "null"))
L = []
p = lambda *x: L.append(" ".join(str(v) for v in x))
c3 = lambda d: f"{d['point']:.3f} ({d['lo']:.3f}–{d['hi']:.3f})"
c2 = lambda d: f"{d['point']:.2f} ({d['lo']:.2f}–{d['hi']:.2f})"
pc = lambda d: f"{100 * d['point']:.1f}% ({100 * d['lo']:.1f}–{100 * d['hi']:.1f})"

fl = J("cohort_flow_lag2.json"); p("## COHORT")
p("export", fl["export_rows"], fl["export_patients"], "| candidates", fl["candidate_anchor_rows"], fl["candidate_anchor_patients"])
for s in fl["exclusion_steps"]:
    p("  ", s["step"], "excl", s["excluded_rows"], "-> remaining", s["remaining_rows"], s["remaining_patients"], "| pts losing all rows", s["patients_losing_all_rows"])
p("final", fl["final_rows"], fl["final_patients"], "centres", fl["centres"], "| adult rows", fl["adult_rows"], "adult pts", fl["adult_patients_any_row"], "| age", fl["anchor_age_min_max"])
p("PEx", fl["events_y_pex"], "/", fl["n_y_pex"], "adult ev", fl["events_y_pex_adult"], "| decline", fl["events_y_decline10"], "/", fl["n_y_decline10"],
  "| strict", fl["events_y_pex_strict"], "iv14", fl["events_y_iv14"], "hosp", fl["events_y_hosp"])
p("years", fl["target_years"], "rows/pt", fl["rows_per_patient"], "male%", fl["male_pct_patient_level"], "adult% first", fl["adult_pct_patient_level_first_row"])
au = J("data_audit.json")["v3_vs_november"]; p("v3 vs Nov", au)
t1 = J("table1.json"); p("## TABLE1 outcomes", t1["outcomes"])
md = J("model_development.json"); p("## MODEL DEVELOPMENT  CLIN =", md["CLIN"], "| IV form", md["iv_functional_form"])
for o in ["y_pex", "y_decline10"]:
    b = md[o]; p(o, "full ORs", {k: (round(v["or"], 2), round(v["lo"], 2), round(v["hi"], 2), round(v["p"], 4)) for k, v in b["full_model"].items()})
    p("   sel freq", b["selection_frequency_across_folds"]); p("   AUC", {k: c3(v["auc"]) for k, v in b["models"].items()})
    p("   logloss", {k: round(v["logloss"], 4) for k, v in b["models"].items()}); p("   slope", {k: round(v["slope"], 2) for k, v in b["models"].items()})
    p("   dAUC vs selected", {k: c3(v["dauc"]) for k, v in b["delta_vs_selected"].items()})
p("culture window (oof logloss)", {k: round(v["oof_logloss"], 4) for k, v in md["culture_window"].items()})
cw = J("culture_window.json"); p("## CULTURE WINDOW (29)  reference =", cw["reference"])
for o in ["y_pex", "y_decline10"]:
    for k, v in cw[o]["models"].items():
        p(f"   {o} {k}: AUC", c3(v["cv"]["auc"]), "dAUC", c3(v["dauc_vs_ref"]) if v["dauc_vs_ref"] else "ref", "ll", round(v["cv"]["logloss"], 4),
          "slope", round(v["cv"]["slope"], 2), "| temporal AUC", c3(v["temporal"]["auc"]), "O/E", round(v["temporal"]["o_e"], 2),
          "| ORs", {c: (round(x["or"], 2), round(x["lo"], 2), round(x["hi"], 2)) for c, x in v["or"].items()})
p("   event rate by culture history", {org: {h: (x["n"], x["events"], round(100 * x["rate"], 1)) for h, x in g.items()} for org, g in cw["history"].items()})
cl = J("classification_summary.json"); p("## CLASSIFICATION")
for o in ["y_pex", "y_decline10"]:
    b = cl[o]; p(o, "n", b["n_rows"], "ev", b["n_events"], "pts", b["n_patients"], "prev", round(b["prevalence"], 3), "EPV clin", round(b["epv_clin"], 1))
    for k, v in b["models"].items():
        p("   ", k, "AUC", c3(v["auc"]), {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("flagged_pct", "sensitivity", "ppv", "npv")})
    p("   dAUC vs clin", {k: c3(v) for k, v in b["delta_auc"].items()}); p("   BSS", {k: c3(v) for k, v in b["brier_skill"].items()})
    for k in ("ml_selection", "ml_ownshadow_selection", "ml_maxshadow_selection"):
        p("   ", k, b[k]["n_selected"], b[k]["frequency"]); p("      params median", b[k]["params_median"])
    p("   masking", {sc: {k: (c3(v["auc"]), round(v["slope"], 2)) for k, v in m_.items()} for sc, m_ in b["masking"].items()})
p("alt outcomes", {o: {k: (v if not isinstance(v, dict) else c3(v["auc"])) for k, v in b.items()} for o, b in cl["alt_outcomes"].items()})
ca = J("calibration_summary.json"); p("## CALIBRATION")
for o in ca:
    if o.startswith("_"): continue
    p(o, {k: (c2(v["citl"]), c2(v["slope"]), round(v["ici"]["point"], 3), round(v["mean_pred"], 3), round(v["obs_rate"], 3)) for k, v in ca[o].items()})
dc = J("dca_summary.json"); p("## DCA")
for o in ["y_pex"]:
    for t, k in dc[o]["key_thresholds"].items():
        p(" t", t, {kk: (round(vv, 4) if isinstance(vv, float) else c3(vv)) for kk, vv in k.items()})
    p(" ops", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in dc[o]["operating_points"].items()})
    p(" capacity", dc[o]["capacity_constrained"])
fc = J("fev1_forecast.json"); p("## FEV1 FORECAST")
for t in ["y_fev1_pp", "y_fev1_z", "y_fvc_pp", "y_fvc_z"]:
    b = fc[t]; t1_, t2_ = b["tolerances"]; p(t, "n", b["n_rows"], "best", b["best_model"], "obs change", b["observed_change"], "cb params", b["cb_change_selected_params"])
    for k, v in b["models"].items():
        p("   ", k, "R2", c3(v["r2"]), "MAE", c2(v["mae"]), f"w{t1_}", pc(v[f"within_{t1_}"]), f"w{t2_}", pc(v[f"within_{t2_}"]),
          "PIhw", round(v["pi95_halfwidth_median"], 2), "cov", round(v["pi95_coverage"], 3), "| child/adult", {g: (gg["n"], round(gg["r2"], 3), round(gg["mae"], 2), round(gg["pi95_coverage"], 3)) for g, gg in v["by_age_group"].items()})
    p("   delta vs persistence", {k: {m: c3(vv) for m, vv in v.items()} for k, v in b["delta_vs_persistence"].items()})
    if "ml_change_selection" in b: p("   ml_change selection", b["ml_change_selection"]["n_selected"], b["ml_change_selection"]["frequency"])
tv = J("temporal_validation.json"); p("## TEMPORAL", tv["test_year"])
for des in ["all_test_rows", "patient_disjoint"]:
    for o, r in tv[des].items():
        p(des, o, "train", r["n_train"], "test", r["n_test"], {k: (c3(v["auc"]) + f" O/E {v['o_e']:.2f} slope {v['slope']['point']:.2f}" if "auc" in v else c3(v["r2"]) + f" MAE {v['mae']:.2f}")
                                                               for k, v in r.items() if isinstance(v, dict) and ("auc" in v or "r2" in v)})
ex = J("extra_validation.json"); lo = ex["leave_institution_out"]; p("## LOCO", lo["n_groups"], "groups", lo["n_institutions"], "centres")
for o in ["y_pex", "y_decline10"]:
    p(o, "pooled heldout", {k: (c3(v["auc"]), round(v["citl"], 2), round(v["slope"], 2)) for k, v in lo[o]["pooled_heldout"].items()})
    p("   RE meta", {k: (round(v["auc"]["pooled"], 3), round(v["auc"]["lo"], 3), round(v["auc"]["hi"], 3), "PI", round(v["auc"]["pi_lo"], 3), round(v["auc"]["pi_hi"], 3),
                       "O/E", round(v["o_e"]["pooled"], 2), "PI", round(v["o_e"]["pi_lo"], 2), round(v["o_e"]["pi_hi"], 2), "I2", round(v["auc"]["I2"], 2)) for k, v in lo[o]["meta"].items()})
    p("   local update", lo[o]["local_update"])
p("fev1 LOCO (z)", {k: c3(v) for k, v in lo["y_fev1_z"].items()})
lf = J("loco_fev1.json"); p("fev1 LOCO (% pred)", {k: (c3(v["r2"]), round(v["mae"], 2), round(v["within_5"], 3), round(v["within_10"], 3)) for k, v in lf["y_fev1_pp"].items()})
p("by year", ex["clin_by_outcome_year"]); p("no2020", c3(ex["sensitivity_no_2020_predictor_year"]["auc"]), "year-adj", c3(ex["sensitivity_calendar_year_adjusted"]["auc"]))
p("IPW", ex["ipw_complete_history"]); s_ = ex["sensitivity_including_outcome_year_modulator_starters"]
p("incl t+1 modulator", {k: (c3(v) if isinstance(v, dict) and "point" in v else v) for k, v in s_.items()})
p("Harrell", ex["harrell_optimism_clin"]); p("Riley", ex["riley_sample_size"]); p("one prior year", c3(ex["one_prior_year_clin"]["auc"]))
p("lag3", {k: (c3(v) if isinstance(v, dict) else v) for k, v in ex["clin_plus_lag3_iv"].items()})
p("selection SMD", {k: {kk: round(vv["smd"], 2) for kk, vv in v["smd"].items()} for k, v in ex["selection"].items() if "smd" in v})
sg = J("subgroup_summary.json"); p("## SUBGROUPS")
for o in sg:
    if o.startswith("_"): continue
    for g, v in sg[o].items():
        p(o, g, v["n"], v["events"], {k: (c3(v[k]["auc"]), c2(v[k]["o_e"])) for k in ["clin", "ml", "catboost_fe_tuned", "lr5"] if k in v})
sp = J("lr5_model_spec.json"); p("## MODEL SPEC")
for o in ["y_pex", "y_decline10"]:
    s = sp[o]; p(o, s["formula"]); p("   ORs", {k: (round(v["or"], 3), round(v["or_lo"], 3), round(v["or_hi"], 3), round(v["p"], 4)) for k, v in s["coefficients"].items() if k != "const"})
    p("   first 14 d", s["or_first_14_iv_days"], "14 vs 28", round(s["or_14_vs_28_iv_days"], 3), "example", s["worked_example"])
cc = J("consistency_checks.json"); p("## CONSISTENCY")
p("extracts", {k: v for k, v in cc["earlier_extracts"].items() if not isinstance(v, dict) or k.startswith("in_")})
p("  perf", {k: (v["n"], c3(v["auc"]), round(v["o_e"], 2), round(v["slope"], 2)) for k, v in cc["earlier_extracts"].items() if isinstance(v, dict) and "auc" in v})
p("non-IV", cc["non_iv_terms"]["cluster_wald_chi2"], cc["non_iv_terms"]["df"], cc["non_iv_terms"]["p"], c3(cc["non_iv_terms"]["oof_mean_loglik_difference_clin_minus_prior_iv"]))
p("BMI", cc["bmi_z_reference"]["agreement_by_group"], cc["bmi_z_reference"]["registry_minus_who_by_year"])
p("BMI harm", {k: (c3(v["cv_auc"]), round(v["temporal_2024_auc"], 3), round(v["temporal_2024_o_e"], 2)) for k, v in cc["bmi_z_reference"]["clin_harmonised"].items()})
p("source", cc["source_export"]); p("axis", cc["points_axis_truncation"]); p("cfrd", cc["cfrd"]); p("software", cc["software"])
mi = J("missingness.json"); p("## MISSING", {k: v for k, v in mi["pct_missing"].items() if v > 0}, "clin any", mi["clin_any_missing_rows_pct"], "clin max", mi["clin_max_missing_pct"])
p("   ML panel any", mi.get("ml_panel_any_missing_rows_pct"), "max", mi.get("ml_panel_max_missing_pct"), "| export missing score input", mi.get("export_records_missing_score_input"))
try:
    ge = J("gee_associations.json"); p("## GEE")
    for o in ["y_pex", "y_decline10"]:
        p(o, {k: {t.split('[')[-1][:24]: (round(v['est'], 2), round(v['lo'], 2), round(v['hi'], 2), round(v['p'], 3)) for t, v in ge[o][k]["total_effect"].items()} for k in ge[o] if isinstance(ge[o][k], dict) and "total_effect" in ge[o][k]})
        p("   events by", {k: v for k, v in ge[o].items() if k.startswith("events_by")})
    p("BH", [(b["outcome"], b["model"], b["term"][-30:], round(b["p"], 4), round(b["q"], 3)) for b in ge["bh_family"]])
    p("CFLD ANCOVA", ge["cfld_ancova_fev1z"])
except FileNotFoundError:
    pass
sh = J("shap_summary.json"); p("## SHAP", {o: list(sh[o]["mean_abs_shap"].items())[:10] for o in sh if not o.startswith("_")})
p("SHAP deployed", {o: (sh[o]["deployed_model"]["n_selected"], sh[o]["deployed_model"]["selected"], sh[o]["deployed_model"]["best_params"]) for o in sh if not o.startswith("_")})
p("SHAP grouped", {o: sh[o]["grouped_mean_abs_shap"] for o in sh if not o.startswith("_")})
(RESULTS / "number_sheet.txt").write_text("\n".join(L)); print("\n".join(L))
