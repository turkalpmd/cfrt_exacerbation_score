"""Decision-curve analysis (R4-C1, R0-W1, R3-W2): LR-5 (published, uncalibrated) vs tuned/isotonic CatBoost vs clinical
comparators (prior-IV logistic; binary rules 'any IV in t-1/t' and 'severity index >=2'), treat-all/none.
ΔNB with patient-bootstrap CIs; operating points at matched thresholds; capacity-constrained net benefit (top 10/20/30%)."""
import numpy as np
from _common import setup
from config import N_BOOT, SEED
from lib import stats as S
from lib.io import write_json

a, df, OUT = setup(__doc__)
O = OUT / "oof"; nboot = 200 if a.fast else N_BOOT
TH = np.round(np.arange(0.05, 0.51, 0.01), 2); KEY = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
STRAT = {"CFRT exacerbation score": ("clin", "raw"), "Machine learning": ("ml", "raw"), "LR-5 (severity index)": ("lr5", "raw"),
         "IV-antibiotic days alone": ("prior_iv_logistic", "raw"), "Rule: any IV (t−1/t)": ("rule_any_iv_2y", "raw"),
         "Rule: severity index ≥2": ("rule_severity_ge2", "raw"), "Rule: severity index (FEV1 <80%) ≥2": ("rule_severity80_ge2", "raw")}
REF = "CFRT exacerbation score"


def nb_rule(y, flag, t):
    n = len(y); return float(np.sum(flag & (y == 1)) / n - np.sum(flag & (y == 0)) / n * t / (1 - t))


def nb(y, p, t, is_rule):
    return nb_rule(y, p >= 0.5, t) if is_rule else S.net_benefit(y, p, t)


res = {}
for out in ["y_pex", "y_decline10"]:
    y = np.load(O / f"clf_{out}_y.npy"); ids = np.load(O / f"clf_{out}_ids.npy", allow_pickle=True)
    P = {lab: np.load(O / f"clf_{out}_{k}_{v}.npy") for lab, (k, v) in STRAT.items() if (O / f"clf_{out}_{k}_{v}.npy").exists()}
    isr = {lab: lab.startswith("Rule") for lab in P}
    curves = {lab: [nb(y, p, t, isr[lab]) for t in TH] for lab, p in P.items()}
    curves["Treat all"] = [S.net_benefit_all(y, t) for t in TH]
    key = {}
    for t in KEY:
        k_ = {lab: nb(y, p, t, isr[lab]) for lab, p in P.items()}; k_["Treat all"] = S.net_benefit_all(y, t)
        for lab in P:
            if lab == REF: continue
            k_[f"{REF} minus {lab}"] = S.cluster_bootstrap(ids, lambda yy, a_, b_, il=isr[lab]: nb(yy, a_, t, False) - nb(yy, b_, t, il),
                                                          (y, P[REF], P[lab]), nboot, SEED)
        k_[f"{REF} minus Treat all"] = S.cluster_bootstrap(ids, lambda yy, a_: nb(yy, a_, t, False) - S.net_benefit_all(yy, t), (y, P[REF]), nboot, SEED)
        key[str(t)] = k_
    ops = {}
    for lab, p in P.items():
        for t in ([0.5] if isr[lab] else [0.15, 0.20, 0.25, 0.30]):
            f = p >= t
            ops[f"{lab} @ {t}" if not isr[lab] else lab] = {
                "threshold": t, "flagged_pct": float(100 * f.mean()), "sensitivity": float(y[f].sum() / y.sum()), "specificity": float(np.mean(~f[y == 0])),
                "ppv": float(y[f].mean()) if f.any() else np.nan, "npv": float(1 - y[~f].mean()) if (~f).any() else np.nan,
                "events_found_per_100": float(100 * np.sum(f & (y == 1)) / len(y)), "flagged_per_event_found": float(f.sum() / max(np.sum(f & (y == 1)), 1))}
    cap = {}
    for frac in [0.10, 0.20, 0.30]:  # resource-constrained: flag the top x% by predicted risk (R3 cross-disciplinary)
        cap[str(frac)] = {}
        for lab in [l for l in P if not isr[l]]:
            thr = np.quantile(P[lab], 1 - frac); f = P[lab] >= thr
            cap[str(frac)][lab] = {"sensitivity": float(y[f].sum() / y.sum()), "ppv": float(y[f].mean()), "flagged_pct": float(100 * f.mean())}
    res[out] = {"thresholds": TH.tolist(), "curves": curves, "key_thresholds": key, "operating_points": ops,
                "capacity_constrained": cap, "prevalence": float(y.mean())}
    print(out, {lab: round(v, 3) for lab, v in key["0.2"].items() if isinstance(v, float)})
    print("   ΔNB five-variable vs others @0.20:", {k: (round(v["point"], 3), round(v["lo"], 3), round(v["hi"], 3)) for k, v in key["0.2"].items() if isinstance(v, dict)})
write_json(res, OUT / "dca_summary.json", "05_dca.py")
