"""Cohort integrity tests — must pass before any modelling."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, MIN_AGE_EXCL, RAW_XLSX  # noqa: E402
from lib.features import DYNAMIC, LAG3_EXTRA, LR5, MAIN_PANEL  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import importlib  # noqa: E402
bc = importlib.import_module("00_build_cohort")

if not RAW_XLSX.exists():   # registry export not distributed with the code
    pytest.skip("registry export not available (set CFRT_EXPORT)", allow_module_level=True)
RAW = bc.clean_raw(pd.read_excel(RAW_XLSX)).set_index(["PatCode", "DateVisitYear"])


@pytest.fixture(params=[2, 3])
def coh(request):
    return request.param, pd.read_csv(DATA / f"cohort_lag{request.param}.csv")


def test_calendar_alignment_and_horizon(coh):
    L, df = coh
    for k in range(1, L + 1):
        assert (df[f"year_lag_{k}"] == df.target_year - k).all()
    assert (df.target_year - df.year_lag_1 == 1).all()


def test_lag_values_equal_raw(coh):
    L, df = coh
    for k in range(1, L + 1):
        idx = pd.MultiIndex.from_arrays([df.PatCode, df[f"year_lag_{k}"]])
        for c in ["FEV1_Z", "FEV1_PP", "ivDaysTotal", "PseudomAer", "bmiZscore"]:
            a = df[f"{c}_lag_{k}"].to_numpy(float); b = RAW.loc[idx, c].to_numpy(float)
            assert np.allclose(a, b, equal_nan=True), f"{c}_lag_{k} mismatch"


def test_targets_equal_raw(coh):
    _, df = coh
    idx = pd.MultiIndex.from_arrays([df.PatCode, df.target_year])
    assert np.allclose(df.y_fev1_z, RAW.loc[idx, "FEV1_Z"].to_numpy(float), equal_nan=True)
    assert np.allclose(df.target_ivDaysTotal, RAW.loc[idx, "ivDaysTotal"].to_numpy(float), equal_nan=True)


def test_modulator_free_and_age(coh):
    """November protocol: age > 5 years in the anchor year (adults included); no modulator in any predictor year. The main
    cohort also excludes outcome-year modulator starts, so mod_start_outcome_year must be 0 there."""
    L, df = coh
    assert (df.AgeAtVisit_lag_1 > MIN_AGE_EXCL).all()
    for yr in [f"year_lag_{k}" for k in range(1, L + 1)]:
        assert (RAW.loc[pd.MultiIndex.from_arrays([df.PatCode, df[yr]]), "Mod"].values == 0).all()
    tm = RAW.loc[pd.MultiIndex.from_arrays([df.PatCode, df.target_year]), "Mod"].values
    assert np.array_equal(tm, df.mod_start_outcome_year.values)


def test_sensitivity_outcomes_nested(coh):
    _, df = coh
    m = df.y_pex_strict.notna()
    assert (df.loc[m, "y_pex_strict"] <= df.loc[m, "y_pex"]).all() and (df.y_iv14.dropna() <= df.y_pex[df.y_iv14.notna()]).all()


def test_no_target_year_leakage_in_panels(coh):
    L, df = coh
    panel = set(MAIN_PANEL) | set(LR5) | (set(LAG3_EXTRA) if L >= 3 else set())
    assert not {c for c in panel if c.startswith(("y_", "target_"))}
    assert not panel & set(DYNAMIC), "un-lagged dynamic variable in a feature panel"
    assert panel <= set(df.columns), panel - set(df.columns)


def test_genotype_rules():
    f = bc.allele_status
    assert f("deltaF508", None) == "F508" and f("[delta]F508", None) == "F508" and f("F508", None) == "F508"
    assert f("deltaF508; M470V", None) == "F508" and f("E508X", None) == "NON"
    assert f("unknown", None) == "UNK" and f("not done", None) == "UNK" and f(np.nan, None) == "UNK"
    assert f("other", np.nan) == "NON" and f("other", "F508del") == "F508"


def test_sweat_minus_one_cleaned(coh):
    _, df = coh
    assert (df.SweatTestVal.dropna() >= 0).all()
