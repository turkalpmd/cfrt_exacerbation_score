"""Authors' question (1 Nov 2026): replace P. aeruginosa in the clinical model by CFTR class genotype (severe = both variants
class I–III; mild = at least one class IV–VI variant)? Same 5 x 5 patient-grouped folds for every specification; out-of-fold
AUC, log-loss, calibration slope and paired differences from the clinical model (patient-cluster bootstrap), cluster-robust
ORs on the full cohort. Genotype unknown (incomplete or unclassified) is coded as its own category, and the comparison is
repeated in patients with a classified genotype. Exploratory. Output: results/pa_vs_genotype.json."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from _common import setup
from config import N_BOOT, N_REPEATS, SEED
from lib import stats as S
from lib.cv import run_oof
from lib.features import CLIN
from lib.io import write_json
from lib.models import lr_unpenalised

a, df, OUT = setup(__doc__)
nrep = 2 if a.fast else N_REPEATS; nboot = 200 if a.fast else N_BOOT
OUTCOME = "y_pex"
df["cftr_unknown"] = df.cftr_class.isin(["incomplete", "unclassified"]).astype(int)
NO_PA = [c for c in CLIN if c != "pa_any_2y"]
SPECS = {
    "clinical model (P. aeruginosa)": CLIN,
    "genotype instead of P. aeruginosa (severe vs other)": NO_PA + ["cftr_severe"],
    "genotype instead of P. aeruginosa (mild, unknown vs severe)": NO_PA + ["cftr_mild", "cftr_unknown"],
    "P. aeruginosa and genotype (mild, unknown vs severe)": CLIN + ["cftr_mild", "cftr_unknown"],
    "neither (four variables)": NO_PA,
}


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(d, specs):
    g = pd.factorize(d.PatCode)[0]; P, out = {}, {}
    for k, f in specs.items():
        rr = run_oof(d, f, OUTCOME, lr_unpenalised(), proba=True, native_nan=False, n_repeats=nrep, seed=SEED)
        y, ids, P[k] = rr["y"], rr["ids"], rr["oof"].mean(0)
        X = sm.add_constant(d[f].astype(float).fillna(d[f].astype(float).median()))
        m = sm.Logit(d[OUTCOME], X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": g}); ci = m.conf_int()
        out[k] = {"features": f, "auc": S.cluster_bootstrap(ids, S.auc, (y, P[k]), nboot, SEED), "logloss": logloss(y, P[k]),
                  "slope": S.cal_slope(y, P[k]), "aic_full_data": float(m.aic),
                  "or": {c: {"or": float(np.exp(m.params[c])), "lo": float(np.exp(ci.loc[c, 0])), "hi": float(np.exp(ci.loc[c, 1])), "p": float(m.pvalues[c])} for c in f}}
    ref = list(specs)[0]
    for k in specs:
        if k != ref:
            out[k]["delta_auc_vs_clinical"] = S.cluster_bootstrap(ids, lambda yy, a_, b_: S.auc(yy, a_) - S.auc(yy, b_), (y, P[k], P[ref]), nboot, SEED)
            out[k]["delta_logloss_vs_clinical"] = out[k]["logloss"] - out[ref]["logloss"]
    return out


d = df[df[OUTCOME].notna()].reset_index(drop=True)
res = {"outcome": OUTCOME, "n": len(d), "events": int(d[OUTCOME].sum()), "clin": CLIN,
       "genotype_rows": d.cftr_class.value_counts().to_dict(),
       "events_by_genotype": d.groupby("cftr_class")[OUTCOME].agg(["sum", "count"]).astype(int).to_dict("index"),
       "pa_by_genotype_pct": (100 * d.groupby("cftr_class").pa_any_2y.mean()).round(1).to_dict(),
       "all": evaluate(d, SPECS)}
dc = d[d.cftr_class.isin(["severe", "mild"])].reset_index(drop=True)
res["classified_only"] = {"n": len(dc), "events": int(dc[OUTCOME].sum()),
                          "results": evaluate(dc, {"clinical model (P. aeruginosa)": CLIN, "genotype instead of P. aeruginosa": NO_PA + ["cftr_mild"],
                                                   "P. aeruginosa and genotype": CLIN + ["cftr_mild"]})}
write_json(res, OUT / "pa_vs_genotype.json", script=__file__)
for blk, r in [("all", res["all"]), ("classified", res["classified_only"]["results"])]:
    print("==", blk)
    for k, v in r.items():
        dd = v.get("delta_auc_vs_clinical")
        print(f"{k:62s} AUC {v['auc']['point']:.3f} ({v['auc']['lo']:.3f}-{v['auc']['hi']:.3f}) LL {v['logloss']:.4f} slope {v['slope']:.2f}"
              + (f" dAUC {dd['point']:+.3f} ({dd['lo']:+.3f},{dd['hi']:+.3f}) dLL {v['delta_logloss_vs_clinical']:+.4f}" if dd else ""))
        print("     ", {c: f"{o['or']:.2f} ({o['lo']:.2f}-{o['hi']:.2f}) p={o['p']:.3f}" for c, o in v["or"].items() if "cftr" in c or "Pseudom" in c})
print(res["events_by_genotype"], res["pa_by_genotype_pct"])
