"""Audit: November cohort vs legacy May cohort and vs the October v3 paediatric cohort (what changed with the new eligibility)."""
import numpy as np
import pandas as pd
from _common import ROOT  # noqa: F401
from config import DATA, LEGACY_COHORT, OCT_COHORT, RAW_XLSX, RESULTS
from lib.io import write_json
import importlib
bc = importlib.import_module("00_build_cohort")

if not (LEGACY_COHORT.exists() and OCT_COHORT.exists()):   # earlier internal extracts are not distributed
    write_json({"skipped": "earlier internal extracts not available", "v3_vs_november": None}, RESULTS / "data_audit.json", "00b_data_audit.py")
    raise SystemExit(0)
leg = pd.read_csv(LEGACY_COHORT)
raw = bc.clean_raw(pd.read_excel(RAW_XLSX)).set_index(["PatCode", "DateVisitYear"])
new = pd.read_csv(DATA / "cohort_lag2.csv")   # primary cohort (three-year window)
res = {"legacy_rows": len(leg), "legacy_patients": int(leg.PatCode.nunique())}
# legacy lag alignment check: legacy FEV1_Z_lag_k vs raw FEV1_Z at calendar year t-k
for k in (1, 2, 3):
    idx = pd.MultiIndex.from_arrays([leg.PatCode, leg.DateVisitYear - k])
    present = idx.isin(raw.index)
    rv = np.full(len(leg), np.nan); rv[present] = raw.loc[idx[present], "FEV1_Z"].to_numpy(float)
    lv = leg[f"FEV1_Z_lag_{k}"].to_numpy(float)
    match = np.isclose(lv, rv, equal_nan=False) | (np.isnan(lv) & np.isnan(rv))
    res[f"legacy_lag{k}_calendar_year_present"] = int(present.sum())
    res[f"legacy_lag{k}_FEV1_Z_matches_calendar"] = int(match.sum())
age = leg.get("AgeAtVisit_lag_1")
res["legacy_anchor_age_ge18"] = int((age >= 18).sum()); res["legacy_anchor_age_nan"] = int(age.isna().sum())
res["legacy_sweat_negative"] = int((leg.SweatTestVal < 0).sum())
k_leg = set(zip(leg.PatCode, leg.DateVisitYear)); k_new = set(zip(new.PatCode, new.target_year))
res["overlap_rows_new_in_legacy"] = len(k_new & k_leg); res["new_only"] = len(k_new - k_leg); res["legacy_only"] = len(k_leg - k_new)
# October v3 (paediatric, outcome-year modulator starters kept, age >= 3) vs November
oc = pd.read_csv(OCT_COHORT); k_oct = set(zip(oc.PatCode, oc.target_year)); kn = dict(zip(zip(new.PatCode, new.target_year), new.AgeAtVisit_lag_1))
ok = dict(zip(zip(oc.PatCode, oc.target_year), zip(oc.AgeAtVisit_lag_1, oc.mod_start_outcome_year)))
only_oct = k_oct - set(kn); only_new = set(kn) - k_oct
res["v3_vs_november"] = {"v3_rows": len(oc), "v3_patients": int(oc.PatCode.nunique()), "november_rows": len(new), "november_patients": int(new.PatCode.nunique()),
                         "shared_rows": len(k_oct & set(kn)), "v3_only_rows": len(only_oct), "november_only_rows": len(only_new),
                         "v3_only_outcome_year_modulator": int(sum(ok[k][1] == 1 for k in only_oct)), "v3_only_age_le5": int(sum(ok[k][0] <= 5 for k in only_oct)),
                         "november_only_adult": int(sum(kn[k] >= 18 for k in only_new)), "november_only_child": int(sum(kn[k] < 18 for k in only_new))}
write_json(res, RESULTS / "data_audit.json", "00b_data_audit.py")
for k, v in res.items(): print(f"{k}: {v}")
