"""Feature panels — single source for every analysis (fixes A11 / panel drift)."""

# Dynamic variables carried from each lag year (calendar-aligned)
DYNAMIC = [
    "Weight", "Height", "weightZScore", "heightZScore", "bmiZscore",
    "PExNumber", "ivDaysTotal", "HospDaysTotal",
    "NaCl", "Bronchodil", "rhDNase", "SteroidInhaled", "Azithro", "Ursodeoxyc", "Pancreatic",
    "PseudomAer", "MSSA", "MRSA", "NontubMycobacteria", "ABPA",
    "Diabetes", "LiverDisease",
    "FEV1_Z", "FVC_Z", "FEV1_PP", "FVC_PP", "AgeAtVisit", "Antibiotic", "Mod",
    "BMI", "Oxygen", "NIPPV", "SteroidOral", "Hemoptysis", "PPI", "BurkhCepacia", "StenotrophMaltophilia", "Achromobacter",
    "Afumigatus", "HaemInf", "LungTransp",
]
# Patient-level (time-invariant) variables, taken from the anchor row
STATIC = ["Sex", "AgeDiagnosis", "SweatTestVal", "NeonatalScr", "MeconiumIleus",
          "DateBirthYear", "CentreCode", "Mutation1", "Mutation2", "OtherMutation1", "OtherMutation2"]

# Five-variable clinical logistic / ridge baseline — SAME set for every outcome
LR5 = ["FEV1_Z_lag_1", "bmiZscore_lag_1", "PseudomAer_lag_1", "iv_burden_2yr_lag", "severity_index_lag_1"]

# Main CatBoost panel: lag-1 + lag-2 only (identical in main, subgroup, lag-sensitivity, temporal)
MAIN_PANEL = [
    "AgeAtVisit_lag_1", "Sex", "AgeDiagnosis", "SweatTestVal",
    "FEV1_Z_lag_1", "FEV1_Z_lag_2", "FVC_Z_lag_1", "FVC_Z_lag_2",
    "bmiZscore_lag_1", "bmiZscore_lag_2", "weightZScore_lag_1", "weightZScore_lag_2",
    "heightZScore_lag_1", "heightZScore_lag_2",
    "PExNumber_lag_1", "PExNumber_lag_2", "ivDaysTotal_lag_1", "ivDaysTotal_lag_2",
    "PseudomAer_lag_1", "MRSA_lag_1", "MSSA_lag_1", "ABPA_lag_1",
    "Pancreatic_lag_1", "Diabetes_lag_1", "LiverDisease_lag_1",
    "Azithro_lag_1", "rhDNase_lag_1", "NaCl_lag_1",
    "fev1_pp_trend_lag", "iv_burden_2yr_lag", "severity_index_lag_1", "high_risk_colonization_lag_1",
    "Antibiotic_lag_1", "HospDaysTotal_lag_1", "NontubMycobacteria_lag_1",   # added after review (R2 Methods)
]

# Non-redundant LR-5 parameterisation (R1-M3, R4-M7): no double counting of IV / P. aeruginosa
LR5_NR = ["FEV1_Z_lag_1", "bmiZscore_lag_1", "PseudomAer_lag_1", "mrsa_or_abpa_lag_1", "log_iv_burden_2yr"]
# Clinical heuristics / persistence analogues for the exacerbation outcome (R4-C1)
PRIOR_IV = ["log_iv_burden_2yr"]
SEVERITY = ["severity_index_lag_1"]
RULES = {"rule_any_iv_2y": "any_iv_2y", "rule_severity_ge2": "severity_ge2"}

# Lag-3 additions (sensitivity only; exact list stated in the manuscript)
LAG3_EXTRA = ["FEV1_Z_lag_3", "FVC_Z_lag_3", "bmiZscore_lag_3", "ivDaysTotal_lag_3",
              "PExNumber_lag_3", "PseudomAer_lag_3", "fev1_pp_trend_lag23"]

OUTCOMES_REG = ["y_fev1_z", "y_fev1_pp"]
OUTCOMES_CLF = ["y_pex", "y_decline10"]
OUTCOMES_SENS = ["y_pex_strict", "y_iv14", "y_hosp"]   # alternative exacerbation definitions (R1-M7, R2-M1)
PERSISTENCE = {"y_fev1_z": "FEV1_Z_lag_1", "y_fev1_pp": "FEV1_PP_lag_1"}


# ---------------- November protocol (JCF manuscript)
# Index-free clinical candidate set (Izzet_sorular D: no severity index; MRSA and ABPA as separate binary predictors).
# Measurements from the anchor year; history items over t-1 and t: cultures and ABPA count when recorded in either year, IV days
# are summed (co-author review, 1 Nov 2026: one window for both organisms; anchor-year windows in scripts/29_culture_window.py)
CLIN_CANDIDATES = ["FEV1_Z_lag_1", "bmiZscore_lag_1", "pa_any_2y", "mrsa_any_2y", "abpa_any_2y", "iv_burden_2yr_lag"]
ANCHOR_WINDOW = {"pa_any_2y": "PseudomAer_lag_1", "mrsa_any_2y": "MRSA_lag_1", "abpa_any_2y": "ABPA_lag_1"}
LR5_SEV80 = ["FEV1_Z_lag_1", "bmiZscore_lag_1", "PseudomAer_lag_1", "iv_burden_2yr_lag", "severity_index_lag_1_80"]
RULES_NOV = {"rule_any_iv_2y": "any_iv_2y", "rule_severity_ge2": "severity_ge2", "rule_severity80_ge2": "severity_ge2_80"}

# Engineered panel for the machine-learning models (all features known at the anchor-year review)
FE_PANEL = MAIN_PANEL + [
    "fev1_z_trend_lag", "fev1_z_mean2", "fev1_pp_mean2", "fvc_z_trend_lag", "fev1_fvc_z_gap", "bmi_trend_lag", "height_z_trend_lag",
    "log_iv_burden_2yr", "iv_both_years", "pex_burden_2yr_lag", "hosp_burden_2yr_lag", "pa_both_years", "pa_new_t",
    "mrsa_any_2y", "abpa_any_2y", "n_pathogens_t", "treatment_burden_t", "resp_support_t", "years_since_diagnosis",
    "StenotrophMaltophilia_lag_1", "Achromobacter_lag_1", "Afumigatus_lag_1", "HaemInf_lag_1", "BurkhCepacia_lag_1",
    "Oxygen_lag_1", "NIPPV_lag_1", "Hemoptysis_lag_1", "SteroidOral_lag_1", "PPI_lag_1", "Bronchodil_lag_1", "SteroidInhaled_lag_1",
]
GENOTYPE_FEATURES = ["cftr_n_f508del", "cftr_severe", "cftr_mild", "cftr_eti_eligible", "cftr_incomplete"]   # class I–III severe, IV–VI mild

# Curated candidate panel of the machine-learning model (authors' decision, 1 Nov 2026). Pre-specified rules: one
# representation per clinical concept (z-scores, not % predicted; yearly counts, not two-year sums or logs), no composite
# or derived item whose components are already in the panel (severity index, pathogen/treatment counts, "any IV"), no item
# with <2% positives, no pair with Spearman |rho| >= 0.8 (checked in 15_build_tables, Table S14). Hospital days dropped
# (|rho| 0.83 with IV days; also an alternative outcome). Features are selected further inside every training set.
ML_PANEL = [
    "FEV1_Z_lag_1", "fev1_z_trend_lag", "fev1_fvc_z_gap",                                   # lung function
    "bmiZscore_lag_1", "bmi_trend_lag", "heightZScore_lag_1",                                # growth
    "ivDaysTotal_lag_1", "ivDaysTotal_lag_2", "iv_both_years", "PExNumber_lag_1", "PExNumber_lag_2",   # IV antibiotics, exacerbations
    "PseudomAer_lag_1", "PseudomAer_lag_2", "MRSA_lag_1", "MRSA_lag_2", "MSSA_lag_1", "ABPA_lag_1", "ABPA_lag_2",
    "NontubMycobacteria_lag_1", "Afumigatus_lag_1", "HaemInf_lag_1",                         # microbiology, ABPA
    "Diabetes_lag_1", "LiverDisease_lag_1", "Oxygen_lag_1", "NIPPV_lag_1",                   # comorbidity, support
    "NaCl_lag_1", "Bronchodil_lag_1", "rhDNase_lag_1", "SteroidInhaled_lag_1", "SteroidOral_lag_1", "Azithro_lag_1",
    "Ursodeoxyc_lag_1", "Pancreatic_lag_1", "Antibiotic_lag_1", "PPI_lag_1",                 # maintenance therapy
    "AgeAtVisit_lag_1", "AgeDiagnosis", "Sex", "SweatTestVal", "NeonatalScr", "MeconiumIleus",   # demographics, diagnosis
    "cftr_n_f508del", "cftr_mild", "cftr_incomplete",                                        # genotype
]

# Final clinical model = output of scripts/21_model_development.py (index-free, backward elimination); read, never retyped
import json as _json
from pathlib import Path as _Path
_MD = _Path(__file__).resolve().parents[1] / "results" / "model_development.json"
CLIN = _json.loads(_MD.read_text())["CLIN"] if _MD.exists() else None

# Readable names for tables and figures (anchor year = t; t−1 = the year before)
LABELS = {
    "FEV1_Z_lag_1": "FEV1 z-score", "fev1_z_trend_lag": "FEV1 z change from t−1", "fev1_fvc_z_gap": "FEV1 z minus FVC z",
    "bmiZscore_lag_1": "BMI z-score", "bmi_trend_lag": "BMI z change from t−1", "heightZScore_lag_1": "Height z-score",
    "ivDaysTotal_lag_1": "IV-antibiotic days, t", "ivDaysTotal_lag_2": "IV-antibiotic days, t−1", "iv_both_years": "IV antibiotics in both years",
    "PExNumber_lag_1": "Recorded exacerbations, t", "PExNumber_lag_2": "Recorded exacerbations, t−1",
    "PseudomAer_lag_1": "P. aeruginosa, t", "PseudomAer_lag_2": "P. aeruginosa, t−1", "MRSA_lag_1": "MRSA, t", "MRSA_lag_2": "MRSA, t−1", "MSSA_lag_1": "MSSA",
    "ABPA_lag_1": "ABPA, t", "ABPA_lag_2": "ABPA, t−1", "NontubMycobacteria_lag_1": "Non-tuberculous mycobacteria", "Afumigatus_lag_1": "A. fumigatus",
    "HaemInf_lag_1": "H. influenzae", "Diabetes_lag_1": "CF-related diabetes", "LiverDisease_lag_1": "CF liver disease", "Oxygen_lag_1": "Oxygen therapy",
    "NIPPV_lag_1": "Non-invasive ventilation", "NaCl_lag_1": "Hypertonic saline", "Bronchodil_lag_1": "Bronchodilator", "rhDNase_lag_1": "Dornase alfa",
    "SteroidInhaled_lag_1": "Inhaled corticosteroid", "SteroidOral_lag_1": "Oral corticosteroid", "Azithro_lag_1": "Azithromycin",
    "Ursodeoxyc_lag_1": "Ursodeoxycholic acid", "Pancreatic_lag_1": "Pancreatic enzymes", "Antibiotic_lag_1": "Inhaled antibiotic", "PPI_lag_1": "Proton-pump inhibitor",
    "AgeAtVisit_lag_1": "Age", "AgeDiagnosis": "Age at diagnosis", "Sex": "Sex", "SweatTestVal": "Sweat chloride", "NeonatalScr": "Newborn screening",
    "MeconiumIleus": "Meconium ileus", "cftr_n_f508del": "F508del alleles", "cftr_mild": "Mild CFTR genotype (class IV–VI)", "cftr_incomplete": "CFTR genotype incomplete",
    "pa_any_2y": "P. aeruginosa, t−1 or t", "mrsa_any_2y": "MRSA, t−1 or t", "abpa_any_2y": "ABPA, t−1 or t", "log_iv_burden_2yr": "IV-antibiotic days, t−1 + t",
}
