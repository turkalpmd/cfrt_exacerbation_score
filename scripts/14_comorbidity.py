"""Comorbidity descriptives: CFRD / CF-LD prevalence at anchor and ever-in-window, crude outcome rates (replaces orphan JSON, A12)."""
import pandas as pd
from _common import setup
from lib.io import write_json

a, df, OUT = setup(__doc__)
L = a.window
res = {}
for name, var in [("cfrd", "Diabetes"), ("cfld", "LiverDisease")]:
    anchor = df[f"{var}_lag_1"]; ever = df[[f"{var}_lag_{k}" for k in range(1, L + 1)]].max(axis=1)
    incident = (ever == 1) & (df[f"{var}_lag_{L}"] == 0)
    res[name] = {"anchor_prev_pct": round(100 * anchor.mean(), 1), "ever_prev_pct": round(100 * ever.mean(), 1),
                 "incident_within_window_rows": int(incident.sum()),
                 "pex_rate": {"with": round(df.y_pex[anchor == 1].mean(), 3), "without": round(df.y_pex[anchor == 0].mean(), 3)},
                 "decline_rate": {"with": round(df.y_decline10[anchor == 1].mean(), 3), "without": round(df.y_decline10[anchor == 0].mean(), 3)},
                 "n_with_anchor": int(anchor.sum())}
first = df.sort_values(["PatCode", "target_year"]).groupby("PatCode").first()
res["cfld_by_pancreatic_patient_level"] = pd.crosstab(first.Pancreatic_lag_1, first.LiverDisease_lag_1).to_dict()
write_json(res, OUT / "comorbidity.json", "14_comorbidity.py")
print(res)
