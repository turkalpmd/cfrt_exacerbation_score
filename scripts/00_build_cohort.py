"""Build calendar-aligned prediction cohorts from the raw registry export (November protocol, JCF manuscript).

Row = one prediction-year: anchor year t (prediction made at its year-end review), history year t-1 (and t-2 for the
four-year window), outcome year t+1.
Eligibility (MANUSCRIPT ÖFİ §2.2): age > 5 years in the anchor year (no upper limit); annual records with spirometry in
t-1, t and t+1; NO CFTR modulator recorded in t-1, t or t+1. No exclusion for lung transplantation (authors' decision, 1 Nov 2026).
Outputs: data/cohort_lag{L}.csv, data/cohort_lag{L}_first_visit.csv, results/cohort_flow_lag{L}.json and, for the
treatment-policy sensitivity analysis, data/cohort_lag2_inclmod.csv (outcome-year modulator starters kept).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, DECLINE_PP, MIN_AGE_EXCL, PED_AGE_MAX, RAW_XLSX, RESULTS  # noqa: E402
from lib.centres import patient_institution  # noqa: E402
from lib.features import DYNAMIC, STATIC  # noqa: E402
from lib.io import write_json  # noqa: E402

try:                                   # curated CFTR variant classification (lib/cftr.py, data/reference/cftr_variant_map.csv)
    from lib import cftr  # noqa: E402
except Exception:                      # pragma: no cover
    cftr = None

UNKNOWN = {"unknown", "not done", "nan", "", "none"}
F508_RE = re.compile(r"^(\[?delta\]?)?f508(del)?$", re.I)          # excludes E508X
PTC_RE = re.compile(r"(X$|X\b|del[ACGT]+$|ins[ACGT]+|AA-?\s*>\s*G|CFTRdele)", re.I)
PATHOGENS = ["PseudomAer", "MRSA", "MSSA", "BurkhCepacia", "StenotrophMaltophilia", "Achromobacter", "NontubMycobacteria", "Afumigatus", "HaemInf"]
THERAPIES = ["NaCl", "rhDNase", "Antibiotic", "Azithro", "Bronchodil", "SteroidInhaled"]


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NIPPV"] = pd.to_numeric(df.NIPPV, errors="coerce").fillna(1).gt(0).astype(int)   # 0 / 2 / 'BiPAP' -> 0/1
    df["Bronchodil"] = df.Bronchodil.map(lambda x: np.nan if pd.isna(x) else int(str(x).strip().lower() not in ("no", "0")))
    num = df.select_dtypes("number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    for c in ["SweatTestVal", "AgeDiagnosis", "ValFEV1", "ValFVC", "PExNumber", "ivDaysTotal", "HospDaysTotal"]:
        df.loc[df[c] < 0, c] = np.nan                          # DataBook: -1 = unknown
    for c in ["FEV1_PP", "FVC_PP"]:
        df.loc[(df[c] <= 10) | (df[c] > 160), c] = np.nan      # implausible spirometry
    for c in ["FEV1_Z", "FVC_Z"]:
        df.loc[df[c].abs() > 10, c] = np.nan
    df.loc[~(df.BMI > 0), "BMI"] = np.nan
    df["AgeAtVisit"] = df["DateVisitYear"] - df["DateBirthYear"]
    df["spirometry"] = (df.FEV1Percent > 0) | df.FEV1_PP.notna() | df.FEV1_Z.notna()
    return df


# ---- legacy genotype coding (kept only to document the change; the analysis uses lib/cftr.py)
def allele_status(mut, other):
    s = str(mut).strip(); first = s.split(";")[0].strip()
    if F508_RE.match(first):
        return "F508"
    if s.lower() in UNKNOWN:
        return "UNK"
    if s.lower() == "other":
        o = str(other).strip()
        return "F508" if F508_RE.match(o.split(";")[0].strip()) else "NON"
    return "NON"


def legacy_genotype(row):
    a1 = allele_status(row["Mutation1"], row["OtherMutation1"]); a2 = allele_status(row["Mutation2"], row["OtherMutation2"])
    n_f = (a1 == "F508") + (a2 == "F508"); n_u = (a1 == "UNK") + (a2 == "UNK")
    g4 = {2: "hom", 1: "het", 0: "non"}[n_f] if n_u == 0 else "unknown"
    ptc = sum(bool(PTC_RE.search(str(m))) and not F508_RE.match(str(m).split(";")[0].strip()) for m in (row["Mutation1"], row["Mutation2"]))
    return g4, ptc


def patient_genotype(raw: pd.DataFrame) -> pd.DataFrame:
    """One genotype per patient from the curated CFTR map (lib/cftr.py). Genotype is time-invariant but often completed in a
    later registry year, so the patient's most informative record is used: fewest incomplete slots, then the most recent."""
    g = cftr.classify_frame(raw)
    g["PatCode"] = raw.PatCode.to_numpy(); g["year"] = raw.DateVisitYear.to_numpy()
    g["n_incomplete_slots"] = (~g.slot1_status.isin(cftr.COMPLETE)).astype(int) + (~g.slot2_status.isin(cftr.COMPLETE)).astype(int)
    g = g.sort_values(["PatCode", "n_incomplete_slots", "year"], ascending=[True, True, False]).groupby("PatCode").first()
    std = pd.DataFrame(index=g.index)
    std["cftr_f508del_cat"] = g.f508del_genotype.map({"F508del homozygous": "homozygous", "F508del heterozygous": "heterozygous",
                                                      "no F508del": "none"}).fillna("incomplete")
    std["cftr_functional"] = g.functional_genotype.map({"two MF alleles": "two minimal-function alleles", "≥1 RF allele": "≥1 residual-function allele",
                                                        "other or unresolved": "other/unresolved"}).fillna("incomplete")
    std["cftr_incomplete"] = (std.cftr_f508del_cat == "incomplete").astype(int)
    std["cftr_n_f508del"] = g.n_f508del.astype(float)
    std["cftr_mf_mf"] = (std.cftr_functional == "two minimal-function alleles").astype(int)
    std["cftr_any_rf"] = (std.cftr_functional == "≥1 residual-function allele").astype(int)
    std["cftr_class"] = g.class_genotype                                          # severe (class I–III) / mild (IV–VI) / incomplete / unclassified
    std["cftr_severe"] = (std.cftr_class == "severe").astype(int)
    std["cftr_mild"] = (std.cftr_class == "mild").astype(int)
    std["cftr_class1"], std["cftr_class2"] = g.class1, g.class2
    std["cftr_eti_eligible"] = (g.eti_eligible_cf == "yes").astype(int)          # >=1 label-listed allele that is CF-causing/VCC
    std["cftr_allele1"], std["cftr_allele2"] = g.allele1, g.allele2
    std["cftr_polymorphism_slot"] = g.any_non_cf_polymorphism.astype(int)
    std["cftr_genotype_year"] = g.year
    return std


def add_genotype(out: pd.DataFrame) -> pd.DataFrame:
    lg = out.apply(legacy_genotype, axis=1, result_type="expand"); out["genotype4_legacy"], out["n_ptc_alleles_legacy"] = lg[0], lg[1]
    if cftr is None:                    # curated map not yet available: legacy coding only (re-run when lib/cftr.py exists)
        print("WARNING: lib/cftr.py missing - legacy genotype coding only"); return out
    return out.join(GENO, on="PatCode")


def engineered(out: pd.DataFrame, L: int) -> pd.DataFrame:
    """Features derived only from information available at the anchor-year review (t-1, t; t-2 if L=3)."""
    o = out
    o["fev1_pp_trend_lag"] = o.FEV1_PP_lag_1 - o.FEV1_PP_lag_2
    o["fev1_z_trend_lag"] = o.FEV1_Z_lag_1 - o.FEV1_Z_lag_2
    o["fev1_z_mean2"] = o[["FEV1_Z_lag_1", "FEV1_Z_lag_2"]].mean(axis=1)
    o["fev1_pp_mean2"] = o[["FEV1_PP_lag_1", "FEV1_PP_lag_2"]].mean(axis=1)
    o["fvc_z_trend_lag"] = o.FVC_Z_lag_1 - o.FVC_Z_lag_2
    o["fev1_fvc_z_gap"] = o.FEV1_Z_lag_1 - o.FVC_Z_lag_1                   # obstruction proxy
    o["bmi_trend_lag"] = o.bmiZscore_lag_1 - o.bmiZscore_lag_2
    o["height_z_trend_lag"] = o.heightZScore_lag_1 - o.heightZScore_lag_2
    ivb = o[["ivDaysTotal_lag_1", "ivDaysTotal_lag_2"]]
    o["iv_burden_2yr_lag"] = ivb.sum(axis=1, min_count=1)
    o["log_iv_burden_2yr"] = np.log1p(o.iv_burden_2yr_lag)
    o["any_iv_2y"] = (o.iv_burden_2yr_lag > 0).astype(float).where(o.iv_burden_2yr_lag.notna())
    o["any_iv_t"] = (o.ivDaysTotal_lag_1 > 0).astype(float).where(o.ivDaysTotal_lag_1.notna())
    o["iv_both_years"] = ((o.ivDaysTotal_lag_1 > 0) & (o.ivDaysTotal_lag_2 > 0)).astype(float)
    o["pex_burden_2yr_lag"] = o[["PExNumber_lag_1", "PExNumber_lag_2"]].sum(axis=1, min_count=1)
    o["hosp_burden_2yr_lag"] = o[["HospDaysTotal_lag_1", "HospDaysTotal_lag_2"]].sum(axis=1, min_count=1)
    pa1, pa2 = o.PseudomAer_lag_1.fillna(0) > 0, o.PseudomAer_lag_2.fillna(0) > 0
    o["pa_both_years"] = (pa1 & pa2).astype(int); o["pa_new_t"] = (pa1 & ~pa2).astype(int); o["pa_any_2y"] = (pa1 | pa2).astype(int)
    o["mrsa_any_2y"] = ((o.MRSA_lag_1.fillna(0) > 0) | (o.MRSA_lag_2.fillna(0) > 0)).astype(int)
    o["abpa_any_2y"] = ((o.ABPA_lag_1.fillna(0) > 0) | (o.ABPA_lag_2.fillna(0) > 0)).astype(int)
    o["n_pathogens_t"] = sum((o[f"{p}_lag_1"].fillna(0) > 0).astype(int) for p in PATHOGENS)
    o["treatment_burden_t"] = sum((o[f"{p}_lag_1"].fillna(0) > 0).astype(int) for p in THERAPIES)
    o["resp_support_t"] = ((o.Oxygen_lag_1.fillna(0) > 0) | (o.NIPPV_lag_1.fillna(0) > 0)).astype(int)
    o["years_since_diagnosis"] = (o.AgeAtVisit_lag_1 - o.AgeDiagnosis).clip(lower=0)
    o["adult"] = (o.AgeAtVisit_lag_1 >= PED_AGE_MAX).astype(int)
    o["age_x_fev1z"] = o.AgeAtVisit_lag_1 * o.FEV1_Z_lag_1
    if L >= 3:
        o["fev1_pp_trend_lag23"] = o.FEV1_PP_lag_2 - o.FEV1_PP_lag_3
    hr = (o.PseudomAer_lag_1.fillna(0) > 0) | (o.MRSA_lag_1.fillna(0) > 0) | (o.ABPA_lag_1.fillna(0) > 0)
    o["high_risk_colonization_lag_1"] = hr.astype(int)
    o["mrsa_or_abpa_lag_1"] = ((o.MRSA_lag_1.fillna(0) > 0) | (o.ABPA_lag_1.fillna(0) > 0)).astype(int)
    for cut in (70, 80):                                                     # severity index: main <70, sensitivity <80
        si = ((o.FEV1_PP_lag_1 < cut).fillna(False).astype(int) + (o.ivDaysTotal_lag_1.fillna(0) > 0).astype(int) + hr.astype(int))
        sfx = "" if cut == 70 else "_80"
        o[f"severity_index_lag_1{sfx}"] = si; o[f"severity_ge2{sfx}"] = (si >= 2).astype(float)
    o["fev1_pp_lt70"] = (o.FEV1_PP_lag_1 < 70).astype(float).where(o.FEV1_PP_lag_1.notna())
    o["fev1_pp_lt80"] = (o.FEV1_PP_lag_1 < 80).astype(float).where(o.FEV1_PP_lag_1.notna())
    return o


def build(raw: pd.DataFrame, L: int, exclude_outcome_year_modulator: bool = True):
    """L = number of predictor years (2 = t-1 and anchor t; 3 adds t-2). Outcome year t+1. Exclusions applied in order."""
    flow = {"window_L": L, "export_rows": len(raw), "export_patients": int(raw.PatCode.nunique()),
            "export_years": [int(raw.DateVisitYear.min()), int(raw.DateVisitYear.max())],
            "export_rows_with_spirometry_pct": float(100 * raw.spirometry.mean())}
    v = raw.set_index(["PatCode", "DateVisitYear"]); assert v.index.is_unique, "duplicate PatCode-year"
    idx = lambda pc, yr: pd.MultiIndex.from_arrays([np.asarray(pc), np.asarray(yr)])
    y0, y1 = raw.DateVisitYear.min(), raw.DateVisitYear.max()
    c = raw[(raw.DateVisitYear - (L - 1) >= y0) & (raw.DateVisitYear + 1 <= y1)][["PatCode", "DateVisitYear", "AgeAtVisit", "Mod"]]
    c = c.rename(columns={"DateVisitYear": "year_lag_1"}).reset_index(drop=True)
    flow["anchor_years"] = [int(c.year_lag_1.min()), int(c.year_lag_1.max())]
    steps = []

    def step(name, keep):
        nonlocal c
        before = c.PatCode.nunique(); keep = keep.to_numpy()
        c = c[keep].reset_index(drop=True)
        steps.append({"step": name, "excluded_rows": int((~keep).sum()), "patients_losing_all_rows": int(before - c.PatCode.nunique())})
        steps[-1].update({"remaining_rows": len(c), "remaining_patients": int(c.PatCode.nunique())})

    flow["candidate_anchor_rows"], flow["candidate_anchor_patients"] = len(c), int(c.PatCode.nunique())
    step("age_le_5_at_anchor", c.AgeAtVisit > MIN_AGE_EXCL)
    hist = np.ones(len(c), bool)
    for k in range(2, L + 1):
        c[f"year_lag_{k}"] = c.year_lag_1 - (k - 1)
        ix = idx(c.PatCode, c[f"year_lag_{k}"]); hist &= ix.isin(v.index) & v.reindex(ix).spirometry.astype("boolean").fillna(False).to_numpy(bool)
    step("no_spirometry_record_in_history_year", pd.Series(hist))
    modp = c.Mod.fillna(0).to_numpy() != 0
    for k in range(2, L + 1):
        modp |= v.reindex(idx(c.PatCode, c[f"year_lag_{k}"])).Mod.fillna(0).to_numpy() != 0
    step("modulator_in_history_or_anchor_year", pd.Series(~modp))
    c["target_year"] = c.year_lag_1 + 1
    ixt = idx(c.PatCode, c.target_year)
    step("no_spirometry_record_in_outcome_year", pd.Series(ixt.isin(v.index) & v.reindex(ixt).spirometry.astype("boolean").fillna(False).to_numpy(bool)))
    mod_t1 = v.reindex(idx(c.PatCode, c.target_year)).Mod.fillna(0).to_numpy() != 0
    c["mod_start_outcome_year"] = mod_t1.astype(int)
    if exclude_outcome_year_modulator:
        step("modulator_in_outcome_year", pd.Series(~mod_t1))
    flow["exclusion_steps"] = steps

    out = c.drop(columns=["AgeAtVisit", "Mod"]).copy()
    for k in range(1, L + 1):
        blk = v.reindex(idx(out.PatCode, out[f"year_lag_{k}"]))[DYNAMIC].reset_index(drop=True)
        blk.columns = [f"{x}_lag_{k}" for x in DYNAMIC]
        out = pd.concat([out, blk], axis=1)
    out = pd.concat([out, v.reindex(idx(out.PatCode, out.year_lag_1))[STATIC].reset_index(drop=True)], axis=1)
    pat = raw.groupby("PatCode").agg(SweatTestVal_any=("SweatTestVal", "median"), AgeDiagnosis_any=("AgeDiagnosis", "median"))
    out["SweatTestVal"] = out.SweatTestVal.fillna(out.PatCode.map(pat.SweatTestVal_any))
    out["AgeDiagnosis"] = out.AgeDiagnosis.fillna(out.PatCode.map(pat.AgeDiagnosis_any))
    tcols = ["FEV1_Z", "FEV1_PP", "FVC_Z", "FVC_PP", "ivDaysTotal", "PExNumber", "HospDaysTotal", "PseudomAer", "bmiZscore"]
    target = v.reindex(idx(out.PatCode, out.target_year))[tcols].reset_index(drop=True)
    target.columns = ["y_fev1_z", "y_fev1_pp", "y_fvc_z", "y_fvc_pp", "target_ivDaysTotal", "target_PExNumber", "target_HospDaysTotal",
                      "target_PseudomAer", "target_bmiZscore"]
    out = pd.concat([out, target], axis=1)

    iv = out.target_ivDaysTotal
    out["y_pex"] = np.where(iv.isna(), np.nan, (iv > 0).astype(float))
    out["y_pex_strict"] = np.where(iv.isna() | out.target_PExNumber.isna(), np.nan, ((iv > 0) & (out.target_PExNumber >= 1)).astype(float))
    out["y_iv14"] = np.where(iv.isna(), np.nan, (iv >= 14).astype(float))
    out["y_hosp"] = np.where(out.target_HospDaysTotal.isna(), np.nan, (out.target_HospDaysTotal > 0).astype(float))
    dpp = out["FEV1_PP_lag_1"] - out["y_fev1_pp"]
    out["y_decline10"] = np.where(dpp.isna(), np.nan, (dpp >= DECLINE_PP).astype(float))
    out["y_fev1_pp_change"] = out.y_fev1_pp - out.FEV1_PP_lag_1
    out["y_fev1_z_change"] = out.y_fev1_z - out.FEV1_Z_lag_1

    out = engineered(out, L)
    out["institution"] = out.PatCode.map(patient_institution(RAW_FOR_CENTRES))      # = centre (harmonised registry centre)
    out = add_genotype(out)

    flow["final_rows"] = len(out); flow["final_patients"] = int(out.PatCode.nunique()); flow["centres"] = int(out.institution.nunique())
    flow["anchor_age_min_max"] = [int(out.AgeAtVisit_lag_1.min()), int(out.AgeAtVisit_lag_1.max())]
    flow["adult_rows"] = int(out.adult.sum()); flow["adult_patients_any_row"] = int(out[out.adult == 1].PatCode.nunique())
    flow["rows_per_patient"] = out.PatCode.value_counts().value_counts().sort_index().to_dict()
    flow["target_years"] = out.target_year.value_counts().sort_index().to_dict()
    for o_ in ["y_pex", "y_pex_strict", "y_iv14", "y_hosp", "y_decline10"]:
        flow[f"events_{o_}"] = int(np.nansum(out[o_])); flow[f"n_{o_}"] = int(out[o_].notna().sum())
        flow[f"events_{o_}_adult"] = int(np.nansum(out.loc[out.adult == 1, o_]))
    flow["n_fev1_z_target"] = int(out.y_fev1_z.notna().sum()); flow["n_fev1_pp_target"] = int(out.y_fev1_pp.notna().sum())
    flow["outcome_year_modulator_starts_in_cohort"] = int(out.mod_start_outcome_year.sum())
    first = out.sort_values(["PatCode", "target_year"]).groupby("PatCode", as_index=False).first()
    flow["male_pct_patient_level"] = round(100 * (first.Sex == 1).mean(), 1)
    flow["adult_pct_patient_level_first_row"] = round(100 * first.adult.mean(), 1)
    for gcol in ["cftr_f508del_cat", "cftr_functional", "cftr_class", "cftr_eti_eligible", "cftr_polymorphism_slot"]:
        flow[f"{gcol}_patient_level"] = first[gcol].value_counts().to_dict()
    flow["rows_by_centre"] = out.institution.value_counts().to_dict()
    return out, first, flow


RAW_FOR_CENTRES = None
GENO = None


def main():
    global RAW_FOR_CENTRES, GENO
    RAW_FOR_CENTRES = pd.read_excel(RAW_XLSX)
    raw = clean_raw(RAW_FOR_CENTRES)
    GENO = patient_genotype(RAW_FOR_CENTRES)
    for L, excl, tag in [(2, True, "lag2"), (3, True, "lag3"), (2, False, "lag2_inclmod")]:
        coh, first, flow = build(raw, L, excl)
        coh.to_csv(DATA / f"cohort_{tag}.csv", index=False)
        first.to_csv(DATA / f"cohort_{tag}_first_visit.csv", index=False)
        write_json(flow, RESULTS / f"cohort_flow_{tag}.json", Path(__file__).name)
        print(f"{tag}: {flow['final_rows']} rows / {flow['final_patients']} pts / {flow['centres']} centres | PEx {flow['events_y_pex']}/{flow['n_y_pex']}"
              f" | decline {flow['events_y_decline10']}/{flow['n_y_decline10']} | adult rows {flow['adult_rows']} | age {flow['anchor_age_min_max']}")
        print("   candidates", flow["candidate_anchor_rows"], flow["candidate_anchor_patients"], "| steps:",
              [(s["step"], s["excluded_rows"], s["remaining_rows"], s["remaining_patients"]) for s in flow["exclusion_steps"]])
        print("   years:", flow["target_years"], "| rows/pt:", flow["rows_per_patient"])


if __name__ == "__main__":
    main()
