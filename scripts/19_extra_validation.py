"""Additional validation requested in review round 1 (see reviews/round1/EDITORIAL_DECISION_and_ROADMAP.md).
(a) leave-one-institution-out (internal–external) with random-effects meta-analysis of logit AUC, log(O/E) and slope + local
    intercept updating demo (P1.7); (b) calendar-year / COVID-19 sensitivity (P2.1); (c) selection: IPW for complete history,
    modulator-excluded and lost-to-follow-up anchors (P1.5, R1-m12); (d) Harrell optimism bootstrap (P2.4); (e) Riley sample
    size (P1.9); (f) only one prior registry year (P2.10); (g) LR-5 + lag-3 IV history on the four-year cohort (P2.12)."""
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from _common import setup
from config import DATA, MIN_AGE_EXCL, N_BOOT, RAW_XLSX, SEED
from lib import stats as S
from lib.cv import run_oof
from lib.features import CLIN, FE_PANEL, GENOTYPE_FEATURES, LR5, MAIN_PANEL, ML_PANEL, PRIOR_IV
from lib.mlx import SelectedLGBM
from joblib import Parallel, delayed
from lib.io import write_json
from lib.models import TunedCatBoost, cb_reg, lr_unpenalised, ridge_pipe
import importlib
bc = importlib.import_module("00_build_cohort")

a, df, OUT = setup(__doc__)
FE = list(dict.fromkeys(FE_PANEL + [g for g in GENOTYPE_FEATURES if g in df.columns]))
nboot = 200 if a.fast else N_BOOT; rng = np.random.default_rng(SEED); NTRIAL = 8 if a.fast else 40
res = {}
logit = lambda v: np.log(np.clip(v, 1e-3, 1 - 1e-3) / (1 - np.clip(v, 1e-3, 1 - 1e-3)))
expit = lambda v: float(1 / (1 + np.exp(-v)))


def re_meta(est, se):
    est, se = np.asarray(est, float), np.maximum(np.asarray(se, float), 1e-3); w = 1 / se ** 2; k = len(est)
    fe = np.sum(w * est) / w.sum(); q = np.sum(w * (est - fe) ** 2)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - np.sum(w ** 2) / w.sum()))
    ws = 1 / (se ** 2 + tau2); mu = np.sum(ws * est) / ws.sum(); sm = np.sqrt(1 / ws.sum()); t = st.t.ppf(0.975, max(k - 2, 1))
    return {"pooled": mu, "lo": mu - 1.96 * sm, "hi": mu + 1.96 * sm, "pi_lo": mu - t * np.sqrt(tau2 + sm ** 2),
            "pi_hi": mu + t * np.sqrt(tau2 + sm ** 2), "tau": float(np.sqrt(tau2)), "I2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0, "k": k}


# ---------------- (a) leave-one-institution-out
d0 = df.copy(); ev = d0.groupby("institution").y_pex.sum()
big = ev[ev >= 10].index
d0["loco_group"] = np.where(d0.institution.isin(big), d0.institution, "Other institutions (pooled)")
loco = {"groups_definition": "institutions with >=10 exacerbation events held out singly; the rest held out as one pooled group",
        "n_groups": int(d0.loco_group.nunique()), "n_institutions": int(d0.institution.nunique())}
for out, models in [("y_pex", ["clin", "ml", "lr5", "prior_iv_logistic", "catboost_fe_tuned"]), ("y_decline10", ["clin", "ml"])]:
    dd = d0[d0[out].notna()].reset_index(drop=True); y = dd[out].to_numpy(float); ids = dd.PatCode.to_numpy()
    pred = {k: np.full(len(dd), np.nan) for k in models}
    for g in dd.loco_group.unique():
        te = (dd.loco_group == g).to_numpy(); tr = ~te
        if "clin" in pred: pred["clin"][te] = lr_unpenalised().fit(dd.loc[tr, CLIN].to_numpy(float), y[tr]).predict_proba(dd.loc[te, CLIN].to_numpy(float))[:, 1]
        if "lr5" in pred: pred["lr5"][te] = lr_unpenalised().fit(dd.loc[tr, LR5].to_numpy(float), y[tr]).predict_proba(dd.loc[te, LR5].to_numpy(float))[:, 1]
        if "prior_iv_logistic" in pred: pred["prior_iv_logistic"][te] = lr_unpenalised().fit(dd.loc[tr, PRIOR_IV].to_numpy(float), y[tr]).predict_proba(dd.loc[te, PRIOR_IV].to_numpy(float))[:, 1]
        if "catboost_fe_tuned" in pred: pred["catboost_fe_tuned"][te] = TunedCatBoost(seed=SEED).fit(dd.loc[tr, FE + ["PatCode"]], y[tr]).predict_proba(dd.loc[te, FE + ["PatCode"]])[:, 1]
    if "ml" in pred:   # selection + Optuna inside each development set, groups in parallel
        def _ml(g):
            te = (dd.loco_group == g).to_numpy()
            return te, SelectedLGBM(feats=ML_PANEL, seed=SEED, ntrial=NTRIAL).fit(dd.loc[~te, ML_PANEL + ["PatCode"]], y[~te]).predict_proba(dd.loc[te, ML_PANEL + ["PatCode"]])[:, 1]
        for te, p_ in Parallel(n_jobs=10)(delayed(_ml)(g) for g in dd.loco_group.unique()):
            pred["ml"][te] = p_
    blk = {"pooled_heldout": {}, "groups": {}, "meta": {}, "local_update": {}}
    for k, p in pred.items():
        blk["pooled_heldout"][k] = {"auc": S.cluster_bootstrap(ids, S.auc, (y, p), nboot, SEED), "citl": S.citl(y, p), "slope": S.cal_slope(y, p)}
        rows = []
        for g in dd.loco_group.unique():
            m = (dd.loco_group == g).to_numpy(); e = int(y[m].sum()); n = int(m.sum())
            r = {"n": n, "events": e, "patients": int(dd.PatCode[m].nunique()), "o_e": float(y[m].mean() / p[m].mean())}
            if e >= 5 and n - e >= 5:
                bs_auc = S.cluster_bootstrap(ids[m], lambda yy, pp: logit(S.auc(yy, pp)), (y[m], p[m]), nboot, SEED)
                bs_oe = S.cluster_bootstrap(ids[m], lambda yy, pp: np.log(yy.mean() / pp.mean()), (y[m], p[m]), nboot, SEED)
                bs_sl = S.cluster_bootstrap(ids[m], S.cal_slope, (y[m], p[m]), nboot, SEED)
                r.update({"auc": S.cluster_bootstrap(ids[m], S.auc, (y[m], p[m]), nboot, SEED), "citl": S.citl(y[m], p[m]),
                          "slope": bs_sl["point"], "slope_lo": bs_sl["lo"], "slope_hi": bs_sl["hi"],
                          "_lauc": bs_auc["point"], "_lauc_se": (bs_auc["hi"] - bs_auc["lo"]) / 3.92,
                          "_loe": bs_oe["point"], "_loe_se": (bs_oe["hi"] - bs_oe["lo"]) / 3.92, "_sl_se": (bs_sl["hi"] - bs_sl["lo"]) / 3.92})
                rows.append(r)
            blk["groups"].setdefault(g, {})[k] = r
        ma = re_meta([r["_lauc"] for r in rows], [r["_lauc_se"] for r in rows])
        mo = re_meta([r["_loe"] for r in rows], [r["_loe_se"] for r in rows])
        ms = re_meta([r["slope"] for r in rows], [r["_sl_se"] for r in rows])
        blk["meta"][k] = {"auc": {kk: (expit(v) if kk in ("pooled", "lo", "hi", "pi_lo", "pi_hi") else v) for kk, v in ma.items()},
                          "o_e": {kk: (float(np.exp(v)) if kk in ("pooled", "lo", "hi", "pi_lo", "pi_hi") else v) for kk, v in mo.items()},
                          "slope": ms}
        # local intercept updating: within each held-out institution, estimate the CITL on half the patients, apply to the other half
        oe_before, oe_after = [], []
        for g in dd.loco_group.unique():
            m = np.flatnonzero((dd.loco_group == g).to_numpy()); pts = np.unique(ids[m])
            if len(pts) < 10 or y[m].sum() < 5: continue
            half = set(rng.choice(pts, len(pts) // 2, replace=False)); A = np.array([i in half for i in ids[m]])
            for fit_, ev_ in [(A, ~A), (~A, A)]:
                if y[m][fit_].sum() == 0 or y[m][ev_].sum() == 0: continue
                c = S.citl(y[m][fit_], p[m][fit_]); pu = 1 / (1 + np.exp(-(logit(p[m][ev_]) + c)))
                oe_before.append(y[m][ev_].mean() / p[m][ev_].mean()); oe_after.append(y[m][ev_].mean() / pu.mean())
        blk["local_update"][k] = {"o_e_before_range": [float(min(oe_before)), float(max(oe_before))], "o_e_after_range": [float(min(oe_after)), float(max(oe_after))],
                                  "abs_log_oe_before_mean": float(np.mean(np.abs(np.log(oe_before)))), "abs_log_oe_after_mean": float(np.mean(np.abs(np.log(oe_after))))}
    for g in blk["groups"]:
        for k in blk["groups"][g]:
            blk["groups"][g][k] = {kk: vv for kk, vv in blk["groups"][g][k].items() if not kk.startswith("_")}
    loco[out] = blk
    print(out, "LOCO", {k: (round(v["auc"]["point"], 3), round(v["citl"], 2), round(v["slope"], 2)) for k, v in blk["pooled_heldout"].items()},
          "| RE AUC", {k: (round(v["auc"]["pooled"], 3), "PI", round(v["auc"]["pi_lo"], 2), round(v["auc"]["pi_hi"], 2), "O/E PI", round(v["o_e"]["pi_lo"], 2), round(v["o_e"]["pi_hi"], 2)) for k, v in blk["meta"].items()},
          "| local update", {k: (np.round(v["o_e_before_range"], 2).tolist(), np.round(v["o_e_after_range"], 2).tolist()) for k, v in blk["local_update"].items()})
dd = d0[d0.y_fev1_z.notna()].reset_index(drop=True); y = dd.y_fev1_z.to_numpy(); ids = dd.PatCode.to_numpy()
pr = {k: np.full(len(dd), np.nan) for k in ["persistence", "ar2_linear", "ridge_clin"]}
for g in dd.loco_group.unique():
    te = (dd.loco_group == g).to_numpy(); tr = ~te
    pr["persistence"][te] = dd.loc[te, "FEV1_Z_lag_1"].fillna(dd.loc[tr, "FEV1_Z_lag_1"].median())
    pr["ar2_linear"][te] = ridge_pipe(alpha=1e-6).fit(dd.loc[tr, ["FEV1_Z_lag_1", "FEV1_Z_lag_2"]].to_numpy(float), y[tr]).predict(dd.loc[te, ["FEV1_Z_lag_1", "FEV1_Z_lag_2"]].to_numpy(float))
    RC = ["FEV1_Z_lag_1", "FEV1_Z_lag_2"] + [c for c in CLIN if c != "FEV1_Z_lag_1"] + ["AgeAtVisit_lag_1"]
    pr["ridge_clin"][te] = ridge_pipe().fit(dd.loc[tr, RC].to_numpy(float), y[tr]).predict(dd.loc[te, RC].to_numpy(float))
loco["y_fev1_z"] = {k: S.cluster_bootstrap(ids, S.r2, (y, p), nboot, SEED) for k, p in pr.items()}
loco["outcome_rate_by_institution"] = d0.groupby("institution").y_pex.agg(["mean", "count"]).round(3).sort_values("count", ascending=False).to_dict("index")
res["leave_institution_out"] = loco

# ---------------- (b) calendar year / COVID-19
raw = bc.clean_raw(pd.read_excel(RAW_XLSX)); rp = raw[(raw.AgeAtVisit > MIN_AGE_EXCL) & (raw.Mod == 0)]
res["by_calendar_year_registry"] = rp.groupby("DateVisitYear").agg(
    rows=("PatCode", "size"), iv_any=("ivDaysTotal", lambda s: float((s > 0).mean())), iv_days_median_if_any=("ivDaysTotal", lambda s: float(s[s > 0].median())),
    spirometry_recorded=("FEV1_PP", lambda s: float(s.notna().mean())), pa_any=("PseudomAer", "mean")).round(3).to_dict("index")
O = OUT / "oof"; yv = np.load(O / "clf_y_pex_y.npy"); idv = np.load(O / "clf_y_pex_ids.npy", allow_pickle=True)
tyv = np.load(O / "clf_y_pex_target_year.npy"); pl = np.load(O / "clf_y_pex_clin_raw.npy")
res["clin_by_outcome_year"] = {int(t): {"n": int((tyv == t).sum()), "events": int(yv[tyv == t].sum()), "auc": S.auc(yv[tyv == t], pl[tyv == t]),
                                       "o_e": float(yv[tyv == t].mean() / pl[tyv == t].mean())} for t in np.unique(tyv)}
nocovid = df[(df.year_lag_1 != 2020) & (df.year_lag_2 != 2020)]
rr = run_oof(nocovid, CLIN, "y_pex", lr_unpenalised(), proba=True, native_nan=False, n_repeats=2, seed=SEED)
res["sensitivity_no_2020_predictor_year"] = {"n": len(nocovid), "events": int(nocovid.y_pex.sum()),
                                             "auc": S.cluster_bootstrap(rr["ids"], S.auc, (rr["y"], rr["oof"].mean(0)), nboot, SEED)}
dyr = df.copy(); dyr = pd.concat([dyr, pd.get_dummies(dyr.target_year, prefix="yr", drop_first=True).astype(float)], axis=1)
yrc = [c for c in dyr.columns if c.startswith("yr_")]
rr = run_oof(dyr, CLIN + yrc, "y_pex", lr_unpenalised(), proba=True, native_nan=False, n_repeats=2, seed=SEED)
res["sensitivity_calendar_year_adjusted"] = {"auc": S.cluster_bootstrap(rr["ids"], S.auc, (rr["y"], rr["oof"].mean(0)), nboot, SEED)}
print("by year", {t: (v["n"], round(v["auc"], 3), round(v["o_e"], 2)) for t, v in res["clin_by_outcome_year"].items()},
      "| no-2020", round(res["sensitivity_no_2020_predictor_year"]["auc"]["point"], 3), "| year-adj", round(res["sensitivity_calendar_year_adjusted"]["auc"]["point"], 3))

# ---------------- (c) selection
L = a.window; v = raw.set_index(["PatCode", "DateVisitYear"])
anc = raw[(raw.AgeAtVisit > MIN_AGE_EXCL) & (raw.DateVisitYear - (L - 1) >= raw.DateVisitYear.min()) & (raw.DateVisitYear + 1 <= raw.DateVisitYear.max())].copy()
ix = lambda pc, yr: pd.MultiIndex.from_arrays([np.asarray(pc), np.asarray(yr)])
anc["mod_anchor"] = (anc.Mod != 0) | (v.reindex(ix(anc.PatCode, anc.DateVisitYear - 1)).Mod.fillna(0).to_numpy() != 0) | \
                    (v.reindex(ix(anc.PatCode, anc.DateVisitYear + 1)).Mod.fillna(0).to_numpy() != 0)
anc["has_prev"] = ix(anc.PatCode, anc.DateVisitYear - 1).isin(v.index); anc["has_next"] = ix(anc.PatCode, anc.DateVisitYear + 1).isin(v.index)
anc["included"] = ix(anc.PatCode, anc.DateVisitYear).isin(ix(df.PatCode, df.year_lag_1))
anc["iv_any"] = (anc.ivDaysTotal > 0).astype(float)
gfn = anc.apply(bc.legacy_genotype, axis=1, result_type="expand"); anc["het_or_hom"] = gfn[0].isin(["het", "hom"]).astype(float); anc["genotype_unknown"] = (gfn[0] == "unknown").astype(float)
VARS = ["AgeAtVisit", "FEV1_Z", "bmiZscore", "PseudomAer", "iv_any", "LiverDisease", "Pancreatic", "het_or_hom", "genotype_unknown"]


def smd(a_, b_):
    out = {}
    for c in VARS:
        x1, x0 = a_[c].dropna(), b_[c].dropna(); sd = np.sqrt((x1.var() + x0.var()) / 2)
        out[c] = {"group_mean": float(x1.mean()), "comparison_mean": float(x0.mean()), "smd": float((x1.mean() - x0.mean()) / sd) if sd > 0 else 0.0}
    return out


res["selection"] = {
    "modulator_t_minus1_t_or_t_plus1_vs_eligible": {"n": int(anc.mod_anchor.sum()), "smd": smd(anc[anc.mod_anchor], anc[~anc.mod_anchor])},
    "lost_no_outcome_year_vs_included": {"n": int((~anc.mod_anchor & anc.has_prev & ~anc.has_next).sum()),
                                         "smd": smd(anc[~anc.mod_anchor & anc.has_prev & ~anc.has_next], anc[anc.included])},
    "included_vs_eligible_not_included": {"n_included": int(anc.included.sum()), "n_not": int((~anc.mod_anchor & ~anc.included).sum()),
                                          "smd": smd(anc[anc.included], anc[~anc.mod_anchor & ~anc.included])},
}
# treatment-policy sensitivity: outcome-year modulator starters retained (cohort_lag2_inclmod.csv)
dm = pd.read_csv(DATA / "cohort_lag2_inclmod.csv")
rr = run_oof(dm, CLIN, "y_pex", lr_unpenalised(), proba=True, native_nan=False, n_repeats=5, seed=SEED)
pm = rr["oof"].mean(0); st = dm[dm.y_pex.notna()].mod_start_outcome_year.to_numpy() == 1
rml = run_oof(dm, ML_PANEL + ["PatCode"], "y_pex", SelectedLGBM(feats=ML_PANEL, seed=SEED, ntrial=NTRIAL), proba=True, native_nan=True, n_repeats=5, seed=SEED, n_jobs=10)
pml = rml["oof"].mean(0)
s_ = lambda c: float((dm.loc[dm.mod_start_outcome_year == 1, c].mean() - dm.loc[dm.mod_start_outcome_year == 0, c].mean()) /
                     np.sqrt((dm.loc[dm.mod_start_outcome_year == 1, c].var() + dm.loc[dm.mod_start_outcome_year == 0, c].var()) / 2))
res["sensitivity_including_outcome_year_modulator_starters"] = {
    "n": len(rr["y"]), "events": int(rr["y"].sum()), "n_starters": int(st.sum()), "events_starters": int(rr["y"][st].sum()),
    "auc": S.cluster_bootstrap(rr["ids"], S.auc, (rr["y"], pm), nboot, SEED), "citl": S.citl(rr["y"], pm), "slope": S.cal_slope(rr["y"], pm),
    "o_e_starters": float(rr["y"][st].mean() / pm[st].mean()), "o_e_others": float(rr["y"][~st].mean() / pm[~st].mean()),
    "ml": {"auc": S.cluster_bootstrap(rml["ids"], S.auc, (rml["y"], pml), nboot, SEED), "slope": S.cal_slope(rml["y"], pml),
           "o_e_starters": float(rml["y"][st].mean() / pml[st].mean())},
    "smd_starters_vs_others": {c: s_(c) for c in ["AgeAtVisit_lag_1", "FEV1_Z_lag_1", "any_iv_2y", "pa_any_2y"]}}
# IPW: probability that an eligible (paediatric, modulator-free) anchor enters the cohort
el = anc[~anc.mod_anchor].copy(); Xs = el[["AgeAtVisit", "FEV1_Z", "bmiZscore", "PseudomAer", "iv_any", "LiverDisease", "Pancreatic", "genotype_unknown"]]
Xs = Xs.assign(fev1_missing=Xs.FEV1_Z.isna().astype(float)).fillna(Xs.median(numeric_only=True))
ps = LogisticRegression(max_iter=5000).fit(Xs, el.included).predict_proba(Xs)[:, 1]
el["w"] = 1 / np.clip(ps, 0.02, 1)
wmap = dict(zip(zip(el.PatCode, el.DateVisitYear), el.w)); rows = df[df.y_pex.notna()].reset_index(drop=True)
w = np.array([wmap.get((p_, y_), np.nan) for p_, y_ in zip(rows.PatCode, rows.year_lag_1)])
res["ipw_complete_history"] = {"weighted_auc": float(roc_auc_score(yv, pl, sample_weight=w)), "weighted_o_e": float(np.sum(w * yv) / np.sum(w * pl)),
                               "weight_range": [float(np.nanmin(w)), float(np.nanmax(w))], "selection_model_auc": float(roc_auc_score(el.included, ps))}
print("selection: IPW AUC", round(res["ipw_complete_history"]["weighted_auc"], 3), "O/E", round(res["ipw_complete_history"]["weighted_o_e"], 2),
      "| incl t+1 modulator starters", {k: res["sensitivity_including_outcome_year_modulator_starters"][k] for k in ["n", "n_starters", "o_e_starters"]},
      round(res["sensitivity_including_outcome_year_modulator_starters"]["auc"]["point"], 3))

# ---------------- (d) Harrell optimism bootstrap for the clinical model (patient resampling)
dd = df[df.y_pex.notna()].reset_index(drop=True); X = dd[CLIN].to_numpy(float); y = dd.y_pex.to_numpy(float); ids = dd.PatCode.to_numpy()
mem = [np.flatnonzero(ids == u) for u in np.unique(ids)]
app = lr_unpenalised().fit(X, y); pa = app.predict_proba(X)[:, 1]; app_auc, app_sl = S.auc(y, pa), S.cal_slope(y, pa)
opt_auc, opt_sl = [], []
for _ in range(200 if not a.fast else 30):
    b = np.concatenate([mem[k] for k in rng.integers(0, len(mem), len(mem))])
    m = lr_unpenalised().fit(X[b], y[b]); pb, po = m.predict_proba(X[b])[:, 1], m.predict_proba(X)[:, 1]
    opt_auc.append(S.auc(y[b], pb) - S.auc(y, po)); opt_sl.append(S.cal_slope(y[b], pb) - S.cal_slope(y, po))
res["harrell_optimism_clin"] = {"apparent_auc": app_auc, "optimism_auc": float(np.mean(opt_auc)), "corrected_auc": app_auc - float(np.mean(opt_auc)),
                               "apparent_slope": app_sl, "optimism_slope": float(np.mean(opt_sl)), "corrected_slope": app_sl - float(np.mean(opt_sl))}
print("Harrell", {k: round(v, 3) for k, v in res["harrell_optimism_clin"].items()})

CLIN_CAND_N = __import__("json").loads((OUT / "model_development.json").read_text())["candidates"]
# ---------------- (e) Riley sample size (binary outcome; Riley et al. BMJ 2020)
phi = y.mean(); n = len(y)
ll1 = np.sum(y * np.log(pa) + (1 - y) * np.log(1 - pa)); ll0 = n * (phi * np.log(phi) + (1 - phi) * np.log(1 - phi))
r2cs = 1 - np.exp(2 * (ll0 - ll1) / n); maxr2 = 1 - np.exp(2 * ll0 / n)
import statsmodels.api as sm
icc = float(sm.GEE(y, np.ones((n, 1)), groups=ids, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit().model.cov_struct.dep_params)
mbar = n / len(mem); de = 1 + (mbar - 1) * icc
riley = {"prevalence": float(phi), "r2cs_anticipated": float(r2cs), "max_r2cs": float(maxr2), "icc_outcome": icc, "mean_rows_per_patient": float(mbar), "design_effect": float(de)}
for P_, lab in [(len(CLIN), "clin"), (len(CLIN_CAND_N), "candidates_backward"), (len(ML_PANEL), "ml_panel"), (len(FE), "fe_panel")]:
    n1 = P_ / ((0.9 - 1) * np.log(1 - r2cs / 0.9))
    s_vh = r2cs / (r2cs + 0.05 * maxr2); n2 = P_ / ((s_vh - 1) * np.log(1 - r2cs / s_vh))
    n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi); nreq = float(max(n1, n2, n3))
    riley[lab] = {"parameters": P_, "n_shrinkage": float(n1), "n_optimism": float(n2), "n_intercept": float(n3), "n_required": nreq,
                  "events_required": nreq * phi, "n_required_with_design_effect": nreq * de, "n_available": n, "events_available": int(y.sum())}
res["riley_sample_size"] = riley
print("Riley", {k: (round(v["n_required"]), round(v["n_required_with_design_effect"])) for k, v in riley.items() if isinstance(v, dict)}, "available", n)

# ---------------- (f) only one prior registry year (new transfer / newly enrolled)
d1 = df.assign(iv_anchor_only=np.log1p(df.ivDaysTotal_lag_1))
LR5_1Y = [c if "iv_burden" not in c else "iv_anchor_only" for c in CLIN]
rr = run_oof(d1, LR5_1Y, "y_pex", lr_unpenalised(), proba=True, native_nan=False, n_repeats=2, seed=SEED)
res["one_prior_year_clin"] = {"auc": S.cluster_bootstrap(rr["ids"], S.auc, (rr["y"], rr["oof"].mean(0)), nboot, SEED)}

# ---------------- (g) LR-5 + lag-3 IV history (four-year cohort)
d3 = pd.read_csv(DATA / "cohort_lag3.csv")
d3["log_iv_lag3"] = np.log1p(d3.ivDaysTotal_lag_3)
r_a = run_oof(d3, CLIN, "y_pex", lr_unpenalised(), proba=True, native_nan=False, n_repeats=5, seed=SEED)
r_b = run_oof(d3, CLIN + ["log_iv_lag3"], "y_pex", lr_unpenalised(), proba=True, native_nan=False, n_repeats=5, seed=SEED)
res["clin_plus_lag3_iv"] = {"n": len(d3), "clin_auc": S.auc(r_a["y"], r_a["oof"].mean(0)), "clin_lag3_auc": S.auc(r_b["y"], r_b["oof"].mean(0)),
                           "delta": S.cluster_bootstrap(r_a["ids"], lambda yy, pa_, pb_: S.auc(yy, pb_) - S.auc(yy, pa_), (r_a["y"], r_a["oof"].mean(0), r_b["oof"].mean(0)), nboot, SEED)}
print("one-year history AUC", round(res["one_prior_year_clin"]["auc"]["point"], 3), "| clin+lag3 Δ", round(res["clin_plus_lag3_iv"]["delta"]["point"], 3))
write_json(res, OUT / "extra_validation.json", "19_extra_validation.py")
