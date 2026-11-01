"""Repeated patient-grouped CV engine producing out-of-fold (OOF) predictions.

Nested isotonic calibration: inside each outer training set an inner grouped 5-fold CV produces
inner-OOF probabilities on which an isotonic map is fitted, then applied to the outer test fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedGroupKFold

from .models import repeated_group_folds


def _fit_predict(model, Xtr, ytr, Xte, proba):
    m = clone(model) if not hasattr(model, "copy") else model.copy()
    m.fit(Xtr, ytr)
    return (m.predict_proba(Xte)[:, 1] if proba else m.predict(Xte)), m


def _X(df, feats, native_nan):
    X = df[feats]
    return X if native_nan else X.to_numpy(float)


def _one_fold(model, X, y, ids, r, f, tr, te, proba, native_nan, nested_cal, seed):
    Xtr = X.iloc[tr] if native_nan else X[tr]; Xte = X.iloc[te] if native_nan else X[te]
    pred, m = _fit_predict(model, Xtr, y[tr], Xte, proba)
    cal = None
    if nested_cal and proba:
        inner = np.full(len(tr), np.nan)
        sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 100 * r + f)
        for itr, ite in sg.split(np.zeros(len(tr)), y[tr], ids[tr]):
            Xi_tr = Xtr.iloc[itr] if native_nan else Xtr[itr]
            Xi_te = Xtr.iloc[ite] if native_nan else Xtr[ite]
            inner[ite], _ = _fit_predict(model, Xi_tr, y[tr][itr], Xi_te, True)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(inner, y[tr])
        cal = iso.predict(pred)
    return r, f, te, pred, cal, m


def run_oof(df: pd.DataFrame, feats, y_col, model, *, proba, native_nan, n_splits=5, n_repeats=5,
            seed=42, nested_cal=False, keep_models=False, n_jobs=1):
    """Returns dict(y, ids, oof[n_rep,n], oof_cal[n_rep,n] (if nested_cal), folds[n_rep,n], models).
    n_jobs > 1 fits the outer folds in parallel processes (identical folds and results)."""
    d = df[df[y_col].notna()].reset_index(drop=True)
    y = d[y_col].to_numpy(float); ids = d.PatCode.to_numpy()
    X = _X(d, feats, native_nan)
    oof = np.full((n_repeats, len(d)), np.nan); cal = oof.copy(); fold_id = np.full((n_repeats, len(d)), -1)
    folds = list(repeated_group_folds(ids, y, n_splits, n_repeats, seed, regression=not proba))
    args = [(model, X, y, ids, r, f, tr, te, proba, native_nan, nested_cal, seed) for r, f, tr, te in folds]
    if n_jobs == 1:
        runs = [_one_fold(*a) for a in args]
    else:
        from joblib import Parallel, delayed
        runs = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(*a) for a in args)
    models = []
    for r, f, te, pred, c, m in runs:
        oof[r, te] = pred; fold_id[r, te] = f
        if c is not None:
            cal[r, te] = c
        if keep_models:
            models.append((r, f, te, m))
    assert not np.isnan(oof).any()
    out = {"y": y, "ids": ids, "oof": oof, "folds": fold_id, "row_index": d.index.to_numpy(),
           "target_year": d.target_year.to_numpy()}
    if nested_cal and proba:
        out["oof_cal"] = cal
    if keep_models:
        out["models"] = models
    return out
