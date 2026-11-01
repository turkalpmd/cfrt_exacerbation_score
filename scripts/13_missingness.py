"""Missingness per predictor/outcome and by target year; missingness vs outcome (replaces orphan missingness.json, A12)."""
import numpy as np
import pandas as pd
from _common import setup
from config import RAW_XLSX
from lib.features import CLIN, FE_PANEL, LAG3_EXTRA, LR5, MAIN_PANEL, ML_PANEL
from lib.io import write_json

a, df, OUT = setup(__doc__)
cols = list(dict.fromkeys(CLIN + LR5 + ML_PANEL + FE_PANEL + (LAG3_EXTRA if a.window == 3 else []) + ["y_fev1_z", "y_fev1_pp", "y_pex", "y_decline10"]))
miss = df[cols].isna().mean().mul(100).round(1).sort_values(ascending=False)
by_year = df.groupby("target_year")[cols].apply(lambda g: g.isna().mean().mul(100).round(1)).T
res = {"pct_missing": miss.to_dict(), "clin_any_missing_rows_pct": round(100 * df[CLIN].isna().any(axis=1).mean(), 1),
       "clin_max_missing_pct": float(df[CLIN].isna().mean().max() * 100), "any_missing_main_panel_rows_pct": round(100 * df[MAIN_PANEL].isna().any(axis=1).mean(), 1),
       "by_target_year": by_year.loc[miss[miss > 0].index].to_dict()}
res["ml_panel_any_missing_rows_pct"] = round(100 * df[ML_PANEL].isna().any(axis=1).mean(), 1)
res["ml_panel_max_missing_pct"] = float(df[ML_PANEL].isna().mean().max() * 100)
# registry export: annual records lacking an input of the score (FEV1 z, BMI z, P. aeruginosa, MRSA, IV days)
raw = pd.read_excel(RAW_XLSX)
sc = ["FEV1_Z", "bmiZscore", "PseudomAer", "MRSA", "ivDaysTotal"]
res["export_records_missing_score_input"] = {"n_records": len(raw), "any_pct": round(100 * raw[sc].isna().any(axis=1).mean(), 1),
                                             "by_item_pct": raw[sc].isna().mean().mul(100).round(1).to_dict()}
m = df.SweatTestVal.isna()
res["pex_rate_by_sweat_missing"] = {"missing": float(df.y_pex[m].mean()), "observed": float(df.y_pex[~m].mean())}
write_json(res, OUT / "missingness.json", "13_missingness.py")
print(miss[miss > 0].to_string()); print("rows with any missing main-panel value:", res["any_missing_main_panel_rows_pct"], "%")
