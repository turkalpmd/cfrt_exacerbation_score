"""Table 1: patient level (first prediction-year) and all prediction-years, with children (<18 y) and adults (>=18 y) shown
separately at the prediction-year level (Izzet_sorular A: children and adults reported separately). Genotype from lib/cftr.py."""
import pandas as pd
from _common import setup
from config import DATA
from lib.io import write_json

a, df, OUT = setup(__doc__)
first = pd.read_csv(DATA / f"cohort_lag{a.window}_first_visit.csv")


def med(s):
    s = s.dropna()
    if not len(s): return "–"
    f = lambda v: f"{v:.1f}".replace("-", "−")
    q1, q3 = s.quantile(.25), s.quantile(.75)
    return f"{f(s.median())} [{f(q1)}{' to ' if min(q1, q3) < 0 else '–'}{f(q3)}]"


def pct(s, val=1):
    s = s.dropna(); return f"{int((s == val).sum())} ({100 * (s == val).mean():.1f}%)" if len(s) else "–"


def describe(d):
    comp = d[d.cftr_incomplete == 0]
    kp = lambda m: f"{int(m.sum())} ({100 * m.mean():.1f}%)" if len(m) else "–"
    return [
        ("Age in the anchor year, y, median [IQR]; range", f"{med(d.AgeAtVisit_lag_1)}; {int(d.AgeAtVisit_lag_1.min())}–{int(d.AgeAtVisit_lag_1.max())}"),
        ("Adults (≥18 years)", pct(d.adult, 1)),
        ("Male sex", pct(d.Sex, 1)),
        ("Age at diagnosis, y, median [IQR]", med(d.AgeDiagnosis)),
        ("Diagnosed by newborn screening", pct(d.NeonatalScr, 1)),
        ("Sweat chloride, mmol/L", med(d.SweatTestVal)),
        ("Both CFTR variants identified and CF-causing", f"{len(comp)} ({100 * len(comp) / len(d):.1f}%)"),
        ("  F508del homozygous, % of identified", kp(comp.cftr_f508del_cat == "homozygous")),
        ("  F508del heterozygous, % of identified", kp(comp.cftr_f508del_cat == "heterozygous")),
        ("  No F508del allele, % of identified", kp(comp.cftr_f508del_cat == "none")),
        ("  Severe genotype (both variants class I–III), % of identified", kp(comp.cftr_class == "severe")),
        ("  Mild genotype (≥1 class IV–VI variant), % of identified", kp(comp.cftr_class == "mild")),
        ("Genotype eligible for elexacaftor/tezacaftor/ivacaftor (FDA label)", pct(d.cftr_eti_eligible, 1)),
        ("Pancreatic enzymes (pancreatic insufficiency)", pct(d.Pancreatic_lag_1, 1)),
        ("FEV1 % predicted, best of anchor year", med(d.FEV1_PP_lag_1)),
        ("FEV1 z-score (GLI-2012), anchor year", med(d.FEV1_Z_lag_1)),
        ("FEV1 <70% predicted, anchor year", pct(d.fev1_pp_lt70, 1)),
        ("BMI z-score, anchor year", med(d.bmiZscore_lag_1)),
        ("BMI, kg/m², anchor year (adults)", med(d.loc[d.adult == 1, "BMI_lag_1"]) if (d.adult == 1).any() else "–"),
        ("P. aeruginosa isolated in anchor year (chronic or ≥1 positive culture)", pct(d.PseudomAer_lag_1, 1)),
        ("P. aeruginosa isolated in t−1 or t", pct(d.pa_any_2y, 1)),
        ("MRSA isolated in anchor year", pct(d.MRSA_lag_1, 1)),
        ("MRSA isolated in t−1 or t", pct(d.mrsa_any_2y, 1)),
        ("ABPA, anchor year", pct(d.ABPA_lag_1, 1)),
        ("IV antibiotics in anchor year", pct(d.any_iv_t, 1)),
        ("IV antibiotics in anchor year or year before", pct(d.any_iv_2y, 1)),
        ("IV-antibiotic days in t−1 and t, median [IQR] among treated", med(d.loc[d.iv_burden_2yr_lag > 0, "iv_burden_2yr_lag"])),
        ("CF-related diabetes, anchor year", pct(d.Diabetes_lag_1, 1)),
        ("CF-related liver disease (ECFSPR definition), anchor year", pct(d.LiverDisease_lag_1, 1)),
        ("Continuous azithromycin", pct(d.Azithro_lag_1, 1)),
        ("Continuous inhaled antibiotic", pct(d.Antibiotic_lag_1, 1)),
        ("Continuous dornase alfa", pct(d.rhDNase_lag_1, 1)),
        ("Centres", str(d.institution.nunique())),
    ]


cols = [(first, f"Patients, first prediction-year (n={len(first)})"), (df, f"All prediction-years (n={len(df)})"),
        (df[df.adult == 0], f"Children <18 y (n={int((df.adult == 0).sum())})"), (df[df.adult == 1], f"Adults ≥18 y (n={int((df.adult == 1).sum())})")]
t1 = None
for d, lab in cols:
    t = pd.DataFrame(describe(d), columns=["Characteristic", lab])
    t1 = t if t1 is None else t1.merge(t, on="Characteristic", how="left", sort=False)
(OUT / "tables").mkdir(exist_ok=True, parents=True)
t1.to_csv(OUT / "tables" / "table1_patients.csv", index=False)
chg = df.y_fev1_pp - df.FEV1_PP_lag_1
out = {"pex": {"events": int(df.y_pex.sum()), "n": int(df.y_pex.notna().sum()),
               "children": [int(df.loc[df.adult == 0, "y_pex"].sum()), int((df.adult == 0).sum())], "adults": [int(df.loc[df.adult == 1, "y_pex"].sum()), int((df.adult == 1).sum())]},
       "decline10": {"events": int(df.y_decline10.sum()), "n": int(df.y_decline10.notna().sum())},
       "fev1_z_target": {"n": int(df.y_fev1_z.notna().sum()), "median": float(df.y_fev1_z.median())},
       "fev1_pp_change_median": float(chg.median()), "fev1_pp_change_iqr": [float(x) for x in chg.quantile([.25, .75])],
       "fev1_pp_change_within5_pct": float(100 * (chg.abs() <= 5).mean()), "fev1_pp_change_within10_pct": float(100 * (chg.abs() <= 10).mean()),
       "fev1_pp_change_by_group": {g: {"median": float(chg[df.adult == k].median()), "iqr": [float(x) for x in chg[df.adult == k].quantile([.25, .75])]}
                                   for g, k in [("children", 0), ("adults", 1)]}}
write_json({"table1": t1.values.tolist(), "columns": list(t1.columns), "outcomes": out}, OUT / "table1.json", "01_table1.py")
print(t1.to_string(index=False)); print(out)
