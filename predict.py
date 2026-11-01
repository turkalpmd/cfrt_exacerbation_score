"""Apply the CFRT exacerbation score and the two machine-learning models to annual reviews.

Input: a CSV file with one row per annual review (anchor year t); empty cells are missing values.
  Score:     FEV1_Z_lag_1, bmiZscore_lag_1, pa_any_2y, mrsa_any_2y, iv_days_2yr (IV-antibiotic days in t-1 and t, summed)
  ML models: the inputs listed in model/ml_exacerbation.json and model/ml_fev1_change.json
  FEV1:      FEV1_PP_lag_1 (this year's FEV1 % predicted), to which the predicted change is added
Output: the input with score_risk, ml_risk and ml_fev1_pp_next_year added (NaN where the required inputs are missing).
Usage: python predict.py reviews.csv predictions.csv"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

MODEL = Path(__file__).resolve().parent / "model"


def score_risk(d):
    """CFRT exacerbation score. Not computed without FEV1 or the IV-antibiotic history; missing BMI z takes the
    development median; an unknown culture result counts as negative (the risk is then a lower bound)."""
    s = json.loads((MODEL / "cfrt_exacerbation_score.json").read_text())
    med = s["missing_values"]["imputation_medians"]
    x = {"FEV1_Z_lag_1": d.FEV1_Z_lag_1, "bmiZscore_lag_1": d.bmiZscore_lag_1.fillna(med["bmiZscore_lag_1"]),
         "pa_any_2y": d.pa_any_2y.fillna(0), "mrsa_any_2y": d.mrsa_any_2y.fillna(0), "log_iv_burden_2yr": np.log1p(d.iv_days_2yr)}
    logit = s["intercept"] + sum(v["beta"] * x[k].astype(float) for k, v in s["predictors"].items())
    return 1 / (1 + np.exp(-logit))


def ml_predict(d, name):
    meta = json.loads((MODEL / f"{name}.json").read_text())
    booster = lgb.Booster(model_file=str(MODEL / meta["model_file"]))
    return booster.predict(d.reindex(columns=meta["inputs_in_order"]).astype(float).to_numpy())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    d = pd.read_csv(sys.argv[1])
    for c in ["FEV1_Z_lag_1", "bmiZscore_lag_1", "pa_any_2y", "mrsa_any_2y", "iv_days_2yr", "FEV1_PP_lag_1"]:
        if c not in d:
            d[c] = np.nan
    d["score_risk"] = score_risk(d)
    d["ml_risk"] = ml_predict(d, "ml_exacerbation")
    d["ml_fev1_pp_next_year"] = d.FEV1_PP_lag_1.astype(float) + ml_predict(d, "ml_fev1_change")
    d.to_csv(sys.argv[2], index=False)
    print(f"{len(d)} reviews -> {sys.argv[2]}")
