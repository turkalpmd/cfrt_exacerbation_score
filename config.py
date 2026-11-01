"""Single source of truth for paths and analysis constants (November pipeline: JCF manuscript, modulator-untreated, age >5 y, no upper limit)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # .../UKKS
OLD = REPO / "Old_Files"
RAW_XLSX = Path(os.environ.get("CFRT_EXPORT", OLD / "finaldata.xlsx"))   # raw registry export (single source; not distributed)
DATABOOK = OLD / "DataBook.xlsx"
LEGACY_COHORT = OLD / "may_analysis" / "data" / "cohort.csv"   # 765-row May cohort (for audit/diff only)
LEGACY_RESULTS = OLD / "may_analysis" / "results"

ANALYSIS = Path(__file__).resolve().parent         # this folder (data/, results/ live here)
OCT_COHORT = REPO / "October" / "analysis" / "data" / "cohort_lag2.csv"   # v3 paediatric cohort (consistency checks only)
DATA = ANALYSIS / "data"
RESULTS = ANALYSIS / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
OOF = RESULTS / "oof"
LOGS = RESULTS / "logs"
for _d in (DATA, TABLES, FIGURES, OOF, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
N_REPEATS = 5
N_BOOT = 1000
N_PERM = 200
WINDOW_MAIN = 2          # three-year window: t-1, anchor t -> outcome t+1 (primary; revision R1-M1)
WINDOW_ROBUST = 3        # four-year window, used only for the lag-3 sensitivity analysis
PED_AGE_MAX = 18         # child = anchor-year age < 18; adult = >= 18 (subgroups only; no upper age limit for eligibility)
MIN_AGE_EXCL = 5         # eligible if anchor-year age > 5 years
SEVERITY_FEV1_CUT = 70   # severity-index FEV1 % predicted cut (main); 80 = sensitivity
DECLINE_PP = 10.0        # >=10 percentage-point FEV1 decline (secondary outcome)

CATBOOST_PARAMS = dict(iterations=600, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                       random_seed=SEED, verbose=False, allow_writing_files=False)
