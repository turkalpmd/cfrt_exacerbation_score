"""Export tidy plotting data (CSV/JSON) for analysis/R/figures.R (November). No plotting here."""
import json
import numpy as np
import pandas as pd
from _common import setup
from config import RESULTS
from lib.features import CLIN, LABELS

a, df, OUT = setup(__doc__)
FD = RESULTS / "figdata"; FD.mkdir(exist_ok=True)
O = OUT / "oof"; J = lambda n: {k: v for k, v in json.loads((OUT / n).read_text().replace("NaN", "null")).items() if not k.startswith("_")}
CLF = {"clin_raw": "CFRT exacerbation score", "ml_raw": "Machine learning", "catboost_fe_tuned_raw": "CatBoost (comparator)", "prior_iv_logistic_raw": "IV-antibiotic days alone",
       "lr5_raw": "LR-5 (severity index)"}
FC = {"persistence": "This year's value carried forward", "ar2": "Two-year linear model", "ridge_clin": "Clinical linear model",
      "cb_level": "CatBoost, previous panel", "cb_change": "CatBoost, engineered features", "ridge_change": "Linear, engineered features", "ml_change": "Machine learning"}

# calibration rows and statistics
rows = []
for out in ["y_pex", "y_decline10"]:
    y = np.load(O / f"clf_{out}_y.npy")
    for k, lab in CLF.items():
        f = O / f"clf_{out}_{k}.npy"
        if f.exists():
            rows.append(pd.DataFrame({"outcome": out, "model": lab, "y": y, "p": np.load(f)}))
pd.concat(rows).to_csv(FD / "calibration_rows.csv", index=False)
cal = J("calibration_summary.json")
pd.DataFrame([{"outcome": o, "model": CLF[k], "citl": v["citl"]["point"], "citl_lo": v["citl"]["lo"], "citl_hi": v["citl"]["hi"],
               "slope": v["slope"]["point"], "slope_lo": v["slope"]["lo"], "slope_hi": v["slope"]["hi"], "ici": v["ici"]["point"]}
              for o in cal for k, v in cal[o].items() if k in CLF]).to_csv(FD / "calibration_stats.csv", index=False)
# decision curves
dca = J("dca_summary.json")
pd.DataFrame([{"outcome": o, "strategy": s, "threshold": t, "nb": nb} for o in dca for s, cv in dca[o]["curves"].items()
              for t, nb in zip(dca[o]["thresholds"], cv)]).to_csv(FD / "dca.csv", index=False)
# next-year lung function: per-row predictions and summary
fc = J("fev1_forecast.json"); rows, summ = [], []
for tgt in ["y_fev1_pp", "y_fev1_z", "y_fvc_pp", "y_fvc_z"]:
    b = fc[tgt]; y = np.load(O / f"fc_{tgt}_y.npy"); adult = np.load(O / f"fc_{tgt}_adult.npy")
    t1, t2 = b["tolerances"]
    for k, m in b["models"].items():
        p = np.load(O / f"fc_{tgt}_{k}_pred.npy"); h = np.load(O / f"fc_{tgt}_{k}_hw.npy")
        rows.append(pd.DataFrame({"target": tgt, "model": FC[k], "key": k, "y": y, "pred": p, "hw": h, "adult": adult}))
        e = np.abs(y - p)
        summ.append({"target": tgt, "model": FC[k], "key": k, "r2": m["r2"]["point"], "r2_lo": m["r2"]["lo"], "r2_hi": m["r2"]["hi"], "mae": m["mae"]["point"],
                     "tol1": t1, "tol2": t2, "w1": m[f"within_{t1}"]["point"], "w1_lo": m[f"within_{t1}"]["lo"], "w1_hi": m[f"within_{t1}"]["hi"],
                     "w2": m[f"within_{t2}"]["point"], "w2_lo": m[f"within_{t2}"]["lo"], "w2_hi": m[f"within_{t2}"]["hi"],
                     "pi_hw": m["pi95_halfwidth_median"], "coverage": m["pi95_coverage"], "best": k == b["best_model"],
                     "share_le1": float(np.mean(e <= t1)), "share_1_2": float(np.mean((e > t1) & (e <= t2))), "share_gt2": float(np.mean(e > t2))})
pd.concat(rows).to_csv(FD / "fev1_rows.csv", index=False); pd.DataFrame(summ).to_csv(FD / "fev1_summary.csv", index=False)
# SHAP rows + summaries (engineered panel)
sh = J("shap_summary.json")
for out in ["y_pex", "y_fev1_pp_change"]:
    feats = sh[out]["features"]; sv = np.load(O / f"shap_{out}.npy"); X = df[df[out].notna()].reset_index(drop=True)[feats]
    top = list(sh[out]["mean_abs_shap"])[:25]
    pd.concat([pd.DataFrame({"feature": f, "label": LABELS[f], "shap": sv[:, feats.index(f)], "value_pct": pd.Series(X[f].to_numpy(float)).rank(pct=True)}) for f in top]).to_csv(FD / f"shap_rows_{out}.csv", index=False)
    pd.Series(sh[out]["mean_abs_shap"]).rename("mean_abs").rename_axis("feature").reset_index().assign(label=lambda t: t.feature.map(LABELS)).to_csv(FD / f"shap_mean_{out}.csv", index=False)
    pd.Series(sh[out]["grouped_mean_abs_shap"]).rename("mean_abs").rename_axis("group").reset_index().to_csv(FD / f"shap_grouped_{out}.csv", index=False)
# subgroups
sub = J("subgroup_summary.json"); rows = []
for out in sub:
    for g, v in sub[out].items():
        for k, lab in [("clin", "CFRT exacerbation score"), ("ml", "Machine learning")]:
            if k in v:
                rows.append({"outcome": out, "group": g, "model": lab, "auc": v[k]["auc"]["point"], "auc_lo": v[k]["auc"]["lo"], "auc_hi": v[k]["auc"]["hi"],
                             "oe": v[k]["o_e"]["point"], "oe_lo": v[k]["o_e"]["lo"], "oe_hi": v[k]["o_e"]["hi"], "n": v["n"], "events": v["events"]})
pd.DataFrame(rows).to_csv(FD / "subgroups.csv", index=False)
# leave-one-centre-out (anonymised: Centre A, B, ... by size)
x = J("extra_validation.json")["leave_institution_out"]
grp = [g for g in x["y_pex"]["groups"] if not g.startswith("Other")]
names = {g: f"Centre {chr(65 + i)}" for i, g in enumerate(sorted(grp, key=lambda g: -x["y_pex"]["groups"][g]["clin"]["n"]))}
ML = {"clin": "CFRT exacerbation score", "ml": "Machine learning", "catboost_fe_tuned": "CatBoost (comparator)", "prior_iv_logistic": "IV-antibiotic days alone",
      "lr5": "LR-5 (severity index)"}
rows = []
for out in ["y_pex", "y_decline10"]:
    for g, v in x[out]["groups"].items():
        for k, r in v.items():
            if "auc" in r:
                rows.append({"outcome": out, "group": names.get(g, "Other centres (pooled)"), "model": ML[k],
                             "auc": r["auc"]["point"], "lo": r["auc"]["lo"], "hi": r["auc"]["hi"], "oe": r["o_e"], "n": r["n"], "events": r["events"]})
    for k, m in x[out]["meta"].items():
        rows.append({"outcome": out, "group": "Random-effects pooled", "model": ML[k], "auc": m["auc"]["pooled"], "lo": m["auc"]["lo"], "hi": m["auc"]["hi"],
                     "pi_lo": m["auc"]["pi_lo"], "pi_hi": m["auc"]["pi_hi"], "oe": m["o_e"]["pooled"], "oe_lo": m["o_e"]["lo"], "oe_hi": m["o_e"]["hi"],
                     "oe_pi_lo": m["o_e"]["pi_lo"], "oe_pi_hi": m["o_e"]["pi_hi"]})
pd.DataFrame(rows).to_csv(FD / "loco.csv", index=False)
(RESULTS / "centre_key_CONFIDENTIAL.json").write_text(json.dumps(names, ensure_ascii=False, indent=1))
ex = J("extra_validation.json")
pd.DataFrame([{"year": int(y_), **v} for y_, v in ex["by_calendar_year_registry"].items()]).to_csv(FD / "calendar_year.csv", index=False)
# points chart: tick points per variable + worked example contributions
spec = J("lr5_model_spec.json")["y_pex"]; pts = spec["points"]
tf = lambda k, v: np.log1p(v) if k == "log_iv_burden_2yr" else v
ex_pts = {k: pts["scale_points_per_logit"] * pts["variables"][k]["beta"] * (tf(k, spec["worked_example"]["inputs"][k]) - tf(k, pts["variables"][k]["zero_at"])) for k in CLIN}
(FD / "nomogram.json").write_text(json.dumps({**pts, "example_inputs": spec["worked_example"]["inputs"], "example_points": ex_pts,
                                              "example_risk": spec["worked_example"]["risk"], "formula": spec["formula"]}, ensure_ascii=False))
(FD / "flow.json").write_text(json.dumps(J(f"cohort_flow_lag{a.window}.json"), default=str))
print("figure data:", sorted(p.name for p in FD.iterdir()))
