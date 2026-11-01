"""EXPLORATORY associations of genotype and comorbidity with one-year outcomes (GEE, exchangeable, patient clusters).
Revision (R1-M8, R2-M4/M5, R4-M3/M4): primary = TOTAL-effect model adjusted for age, sex and centre only; secondary =
mediator-adjusted (adds anchor FEV1 z, P. aeruginosa, pancreatic insufficiency). Functional genotype classification,
PTC allele count, events per group, Benjamini–Hochberg across the whole family, CFLD by centre and ANCOVA on next-year FEV1 z."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from _common import setup
from lib import stats as S
from lib.io import write_json

a, df, OUT = setup(__doc__)
d = df.copy(); L = a.window
big = d.institution.value_counts(); keep = big[big >= 40].index
d["centre"] = np.where(d.institution.isin(keep), d.institution, "Other")
# curated CFTR classification (lib/cftr.py): F508del dose and functional genotype; incomplete = unknown / not done / non-CF-causing slot
d["f508"] = pd.Categorical(d.cftr_f508del_cat, categories=["none", "heterozygous", "homozygous", "incomplete"])
d["gcls"] = pd.Categorical(d.cftr_class, categories=["severe", "mild", "unclassified", "incomplete"])   # class I–III vs IV–VI
d["cfld_ever"] = d[[f"LiverDisease_lag_{k}" for k in range(1, L + 1)]].max(axis=1)
d["cfrd_ever"] = d[[f"Diabetes_lag_{k}" for k in range(1, L + 1)]].max(axis=1)
d["male"] = (d.Sex == 1).astype(int)
TOTAL = "AgeAtVisit_lag_1 + male + C(centre)"
DIRECT = TOTAL + " + FEV1_Z_lag_1 + pa_any_2y + Pancreatic_lag_1"


def gee(formula, data, family=sm.families.Binomial()):
    m = smf.gee(formula, groups="PatCode", data=data, family=family, cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = m.conf_int(); tf = (lambda v: float(np.exp(v))) if isinstance(family, sm.families.Binomial) else float
    return {t: {"est": tf(m.params[t]), "lo": tf(ci.loc[t, 0]), "hi": tf(ci.loc[t, 1]), "p": float(m.pvalues[t])}
            for t in m.params.index if not t.startswith(("Intercept", "C(centre)"))}


EXPOSURES = {"f508del": "C(f508, Treatment('none'))", "genotype_class": "C(gcls, Treatment('severe'))",
             "eti_eligible": "cftr_eti_eligible", "cfrd_anchor": "Diabetes_lag_1", "cfld_anchor": "LiverDisease_lag_1", "cfld_ever": "cfld_ever"}
res, family_p = {}, []
for out in ["y_pex", "y_decline10"]:
    base = d.dropna(subset=[out, "AgeAtVisit_lag_1", "FEV1_Z_lag_1", "pa_any_2y", "Pancreatic_lag_1"])
    r = {}
    for name, expo in EXPOSURES.items():
        tot = gee(f"{out} ~ {expo} + {TOTAL}", base); dire = gee(f"{out} ~ {expo} + {DIRECT}", base)
        keep_ = lambda dd: {k: v for k, v in dd.items() if any(s in k for s in ["f508", "gcls", "eti", "Diabetes", "Liver", "ever"])}
        r[name] = {"total_effect": keep_(tot), "direct_effect": keep_(dire), "n_rows": len(base), "n_patients": int(base.PatCode.nunique())}
        for t, v in keep_(tot).items():
            family_p.append((out, name, t, v["p"]))
    for g, col in [("f508del", "f508"), ("genotype_class", "gcls"), ("eti_eligible", "cftr_eti_eligible")]:
        r[f"events_by_{g}"] = base.groupby(col, observed=True)[out].agg(["sum", "count"]).astype(int).rename(columns={"sum": "events", "count": "rows"}).to_dict("index")
    res[out] = r
# CFLD by centre and centre-adjusted ANCOVA on continuous next-year FEV1 z (R2-M5, R1-M8e)
res["cfld_prevalence_by_centre"] = d.groupby("centre").LiverDisease_lag_1.agg(["mean", "count"]).round(3).to_dict("index")
res["incomplete_genotype_by_centre"] = d.groupby("centre").cftr_incomplete.mean().round(3).to_dict()
anc = d.dropna(subset=["y_fev1_z", "FEV1_Z_lag_1"])
res["cfld_ancova_fev1z"] = gee("y_fev1_z ~ LiverDisease_lag_1 + FEV1_Z_lag_1 + " + TOTAL, anc, family=sm.families.Gaussian())
# round 2 (NEW-4): the ANCOVA belongs to the same exploratory family
family_p.append(("y_fev1_z", "cfld_ancova", "LiverDisease_lag_1", res["cfld_ancova_fev1z"]["LiverDisease_lag_1"]["p"]))
# Benjamini–Hochberg across the full family of total-effect tests (both binary outcomes + CFLD ANCOVA)
q = S.bh_fdr(np.array([x[3] for x in family_p]))
res["bh_family"] = [{"outcome": o, "model": m, "term": t, "p": p, "q": float(qq)} for (o, m, t, p), qq in zip(family_p, q)]
res["modulator_excluded_rows"] = "see extra_validation.json"
for out in ["y_pex", "y_decline10"]:
    print(out, {k: {t.split('[')[-1][:14]: (round(v['est'], 2), round(v['lo'], 2), round(v['hi'], 2)) for t, v in res[out][k]["total_effect"].items()} for k in ["f508del", "genotype_class", "eti_eligible", "cfld_anchor"]})
print("CFLD ANCOVA:", {k: (round(v["est"], 3), round(v["p"], 3)) for k, v in res["cfld_ancova_fev1z"].items() if "Liver" in k})
print("BH min q:", round(min(x["q"] for x in res["bh_family"]), 3))
write_json(res, OUT / "gee_associations.json", "07_gee_associations.py")
