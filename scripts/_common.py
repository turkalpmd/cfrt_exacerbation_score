"""Shared CLI/window handling for pipeline scripts."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from config import DATA, RESULTS, WINDOW_MAIN  # noqa: E402


def setup(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--window", type=int, default=WINDOW_MAIN)
    ap.add_argument("--fast", action="store_true", help="fewer repeats/boots for smoke tests")
    a = ap.parse_args()
    out = RESULTS if a.window == WINDOW_MAIN else RESULTS / f"lag{a.window}"
    (out / "oof").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / f"cohort_lag{a.window}.csv")
    return a, df, out
