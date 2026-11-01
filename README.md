# CFRT exacerbation score

Analysis code for the study *One-year prediction of lung function and pulmonary exacerbation in CFTR modulator-untreated people with cystic fibrosis: a national registry study*, based on the Cystic Fibrosis Registry of Türkiye (CFRT). Citation details will be added on publication.

The repository holds the analysis code, the equation of the CFRT exacerbation score and the two trained machine-learning models. **It contains no data of any kind**: no registry data, no data derived from the registry and no reference tables. Registry data are held by the CFRT and are available from the registry steering committee on reasonable request.

## Contents

| Path | Content |
|---|---|
| `model/cfrt_exacerbation_score.json` | CFRT exacerbation score, version 1.0: equation, odds ratios, imputation medians, points chart and worked example |
| `model/ml_exacerbation.lgbm.txt` + `.json` | Trained LightGBM model for IV-treated exacerbation in the next year, with its input list |
| `model/ml_fev1_change.lgbm.txt` + `.json` | Trained LightGBM model for the change in FEV1 % predicted to the next year, with its input list |
| `predict.py` | Applies the score and both models to a CSV file of annual reviews |
| `run_all.sh` | Runs the complete analysis in order |
| `scripts/` | Analysis steps (see the table below) |
| `lib/` | Shared code: feature panels, patient-grouped cross-validation, models, statistics, CFTR variant classification, centre harmonisation |
| `R/figures.R` | Figures (ggplot2) |
| `tests/` | Tests of the CFTR classification and of cohort integrity |

Not included: the registry export, the analysis cohort, the CFTR variant map, reference tables, out-of-fold predictions, results and figures; `data/` and `results/` are created locally when the analysis is run and are excluded from version control. The per-variable minimum and maximum values that LightGBM stores in a model file were removed because they are single patients' values; predictions are unchanged (checked when the repository was built).

## Study design in brief

The unit of analysis is the prediction-year: the anchor year *t*, at whose year-end review the prediction is made; the year before, *t*−1; and the outcome year *t*+1 (anchor years 2020–2023). Patients older than 5 years with spirometry in all three years and no CFTR modulator therapy in any of them were eligible. Outcomes were an IV-antibiotic-treated pulmonary exacerbation in *t*+1, next-year FEV1 % predicted and a fall of at least 10 points. Every model was validated by 5 × 5 patient-grouped cross-validation, with variable selection and tuning repeated inside each training set, by temporal validation in 2024 (patients with a 2024 record excluded from development) and by leave-one-centre-out validation.

## CFRT exacerbation score

```
logit(p) = -2.933 -0.266 × [FEV1 z-score (anchor year, best of year)] -0.225 × [BMI z-score (anchor year)] +0.686 × [P. aeruginosa isolated (anchor year or year before)] +0.595 × [MRSA isolated (anchor year or year before)] +0.527 × ln(1 + IV days, t−1 and t)
risk = 1 / (1 + exp(−logit))
```

| Input (column in `predict.py`) | Definition | Odds ratio (95% CI) |
|---|---|---|
| `FEV1_Z_lag_1` | FEV1 z-score (GLI-2012), best of the anchor year t | 0.77 (0.69–0.86) per unit |
| `bmiZscore_lag_1` | BMI z-score of the anchor year t, as recorded by the registry | 0.80 (0.66–0.96) per unit |
| `pa_any_2y` | *P. aeruginosa* recorded (chronic infection or at least one positive culture) in year t−1 or t: 1 = yes, 0 = no | 1.99 (1.37–2.88) |
| `mrsa_any_2y` | MRSA recorded (chronic infection or at least one positive culture) in year t−1 or t: 1 = yes, 0 = no | 1.81 (1.11–2.95) |
| `iv_days_2yr` | IV-antibiotic days in years t−1 and t, summed (enters as ln(1 + days)) | 4.17 (3.03–5.73) for the first 14 days |

Missing inputs: Do not compute the score if anchor-year FEV1 or the IV-antibiotic history is unavailable. Missing BMI z: use the cohort median and report the range of risk over BMI z −2 to +1. Unknown culture results are scored as negative and the risk is reported as a lower bound.

Worked example: FEV1 z-score −2, BMI z-score −0.5, *P. aeruginosa* yes, MRSA no and 14 IV-antibiotic days in years t−1 and t give a risk of 0.46 (148 points on the points chart).

Intended use: to identify modulator-untreated patients who may benefit from closer review in the coming year. The score is not intended for de-escalation of care or for allocating modulators. The name "CFRT exacerbation score" is a working name.

## Machine-learning models

LightGBM gradient boosting. Within every training set, the variables were ranked by SHAP value, 10–15 of the 20 highest-ranked were kept, and the hyperparameters were tuned with Optuna in patient-grouped inner cross-validation (details in the Supplementary Methods of the article). Missing inputs are allowed and are handled by LightGBM. The exacerbation model returns a probability; the FEV1 model returns the change in FEV1 % predicted, which is added to this year's value.

| Input | Definition | Exacerbation model | FEV1 model |
|---|---|---|---|
| `FEV1_Z_lag_1` | FEV1 z-score (GLI-2012), best of year t | yes | yes |
| `ivDaysTotal_lag_1` | IV-antibiotic days, year t | yes | yes |
| `bmiZscore_lag_1` | BMI z-score, year t | yes | yes |
| `AgeDiagnosis` | Age at diagnosis, years | yes | yes |
| `fev1_z_trend_lag` | FEV1 z-score of year t minus that of year t−1 | yes | yes |
| `heightZScore_lag_1` | Height z-score, year t | yes | yes |
| `bmi_trend_lag` | BMI z-score of year t minus that of year t−1 | yes | yes |
| `SweatTestVal` | Sweat chloride, mmol/L | yes | yes |
| `fev1_fvc_z_gap` | FEV1 z-score minus FVC z-score, year t | yes | yes |
| `ivDaysTotal_lag_2` | IV-antibiotic days, year t−1 | yes | yes |
| `AgeAtVisit_lag_1` | Age in year t, years | yes | yes |
| `MRSA_lag_2` | MRSA recorded in year t−1 (1/0) | yes | yes |
| `PseudomAer_lag_2` | *P. aeruginosa* recorded in year t−1 (1/0) | yes |  |
| `cftr_n_f508del` | Number of F508del alleles (0, 1, 2) | yes |  |
| `PseudomAer_lag_1` | *P. aeruginosa* recorded in year t (1/0) | yes |  |
| `NontubMycobacteria_lag_1` | Non-tuberculous mycobacteria recorded in year t (1/0) |  | yes |
| `cftr_incomplete` | CFTR genotype incomplete (1/0) |  | yes |

```python
import json
import lightgbm as lgb

meta = json.load(open("model/ml_exacerbation.json"))
booster = lgb.Booster(model_file="model/ml_exacerbation.lgbm.txt")
risk = booster.predict(df[meta["inputs_in_order"]].to_numpy(float))   # df: one row per annual review
```

## Applying the score and the models

```
python predict.py reviews.csv predictions.csv
```

`reviews.csv` has one row per annual review. Use the column names in the tables above; for the score, give the IV-antibiotic days of years *t*−1 and *t* summed in a column `iv_days_2yr`, and for the FEV1 forecast this year's FEV1 % predicted in `FEV1_PP_lag_1`. Empty cells are treated as missing. The output adds `score_risk`, `ml_risk` and `ml_fev1_pp_next_year`.

## Reproducing the analysis

Requirements: Python 3.12 with `requirements.txt`, and R 4.5 with ggplot2, patchwork, dplyr, tidyr, readr, jsonlite and scales.

1. `cp .pyenv.example .pyenv` and set the Python and Rscript paths.
2. `export CFRT_EXPORT=/path/to/export.xlsx`, the CFRT annual export (one row per patient-year, registry column names).
3. Download the WHO 2007 BMI-for-age LMS tables for boys and girls, 5–19 years (https://www.who.int/tools/growth-reference-data-for-5to19-years), and save them as `data/reference/who_bmi_boys.xlsx` and `data/reference/who_bmi_girls.xlsx`; they are used by one sensitivity analysis.
4. `export NCBI_EMAIL=you@example.org`. The first step, `cftr_variant_map.py`, builds the CFTR variant map from the export and downloads the public references it needs (CFTR2 variant list, FDA labels, NCBI RefSeq and ClinVar) into `data/reference/`; this needs internet access.
5. `bash run_all.sh`. Tables are written to `results/tables/`, figures to `results/figures/` and all reported numbers to `results/number_sheet.txt`. The run takes about 40 minutes on 10 cores.

`00b_data_audit.py` and part of `20_consistency_checks.py` compare the cohort with earlier internal extracts; these steps are skipped when the extracts are absent. `16_figure_data.py` writes `results/centre_key_CONFIDENTIAL.json`, which links anonymised centre codes to centre names; `results/` is excluded from version control and must not be shared. Tests: `pytest tests` (the cohort tests are skipped without the export).

| Step | Script | Output |
|---|---|---|
| CFTR variants | `cftr_variant_map.py` | Variant classification map (CFTR2, FDA label, clinician assignments), written to `data/reference/` |
| Cohort | `00_build_cohort.py` | Prediction-years and outcomes (`data/`, not shared) |
| Score development | `21_model_development.py`, `29_culture_window.py`, `25_candidate_screen.py`, `12_lr5_model_spec.py` | Variable selection, culture window, candidate screen, equation and points chart |
| Exacerbation and decline | `03_classification.py`, `04_calibration.py`, `05_dca.py`, `08_subgroup.py`, `26_pa_vs_genotype.py` | Discrimination, calibration, decision curves, subgroups, genotype |
| Next-year FEV1 | `22_fev1_forecast.py`, `24_loco_fev1.py` | Forecast models and prediction intervals |
| Validation | `11_temporal_validation.py`, `19_extra_validation.py` | 2024 validation, leave-one-centre-out, sensitivity analyses |
| Machine learning | `10_shap.py`, `27_ml_experiments.py`, `28_ml_temporal.py` | Final models and SHAP values; other configurations |
| Descriptive | `01_table1.py`, `07_gee_associations.py`, `13_missingness.py`, `14_comorbidity.py`, `20_consistency_checks.py` | Table 1, genotype and comorbidity associations, missing data, checks |
| Output | `15_build_tables.py`, `16_figure_data.py`, `23_number_sheet.py`, `R/figures.R` | Tables, figure data, number sheet, figures |

## Licence and citation

Licence and citation will be added by the authors.
