"""Model factories + repeated patient-grouped folds. All preprocessing lives INSIDE the fold (fixes A7)."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CATBOOST_PARAMS  # noqa: E402


def lr_pipe(class_weight=None):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight=class_weight))


def ridge_pipe(alpha=1.0):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def cb_clf(class_weights=False, **over):
    p = dict(CATBOOST_PARAMS, **over)
    if class_weights:
        p["auto_class_weights"] = "Balanced"
    return CatBoostClassifier(**p)


def cb_reg(**over):
    return CatBoostRegressor(**dict(CATBOOST_PARAMS, **over))


def repeated_group_folds(ids, y, n_splits=5, n_repeats=5, seed=42, regression=False):
    """Yield (repeat, fold, train_idx, test_idx). Patients never split across train/test.
    Stratified on y (binary) or on y-quintiles (regression)."""
    ids = np.asarray(ids)
    strat = np.asarray(y)
    if regression:
        strat = np.digitize(strat, np.nanquantile(strat, [0.2, 0.4, 0.6, 0.8]))
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(sgkf.split(np.zeros(len(ids)), strat, ids)):
            assert not set(ids[tr]) & set(ids[te]), "patient leakage across folds"
            yield r, f, tr, te


# ---------------- revision additions (R1-M9, R1-M3) ----------------
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import SplineTransformer


def lr_unpenalised():
    """LR-5 exactly as published: unpenalised maximum likelihood (median imputation inside the fold)."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(penalty=None, max_iter=5000))


def lr_cv_penalised():
    """Full-panel logistic with the L2 penalty chosen by inner 5-fold CV (log-loss)."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegressionCV(Cs=12, cv=5, scoring="neg_log_loss", max_iter=5000))


def lr_spline(n_cont):
    """LR-5 with restricted-cubic-type splines (4 knots) for the first n_cont continuous inputs (nonlinearity test)."""
    ct = ColumnTransformer([("spl", SplineTransformer(n_knots=4, degree=3), list(range(n_cont)))], remainder="passthrough")
    return make_pipeline(SimpleImputer(strategy="median"), ct, StandardScaler(), LogisticRegression(penalty=None, max_iter=5000))


class TunedCatBoost(BaseEstimator, ClassifierMixin):
    """CatBoost tuned by inner patient-grouped CV (log-loss) and isotonic-recalibrated on the inner out-of-fold predictions
    of the selected configuration. X must contain a 'PatCode' column (used for grouping, dropped before fitting)."""
    GRID = [dict(depth=d, l2_leaf_reg=l2, learning_rate=0.05, iterations=300) for d in (3, 4, 6) for l2 in (3, 10)]

    def __init__(self, seed=42, inner=3):
        self.seed, self.inner = seed, inner

    def fit(self, X, y):
        from sklearn.metrics import log_loss
        g = X["PatCode"].to_numpy(); Xf = X.drop(columns="PatCode"); y = np.asarray(y)
        folds = list(StratifiedGroupKFold(self.inner, shuffle=True, random_state=self.seed).split(Xf, y, g))
        best, best_ll, best_oof = None, np.inf, None
        for prm in self.GRID:
            oof = np.empty(len(y))
            for tr, te in folds:
                m = CatBoostClassifier(**prm, random_seed=self.seed, verbose=False, allow_writing_files=False)
                oof[te] = m.fit(Xf.iloc[tr], y[tr]).predict_proba(Xf.iloc[te])[:, 1]
            ll = log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))
            if ll < best_ll:
                best, best_ll, best_oof = prm, ll, oof
        self.best_params_ = best
        self.model_ = CatBoostClassifier(**best, random_seed=self.seed, verbose=False, allow_writing_files=False).fit(Xf, y)
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(best_oof, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.iso_.predict(self.model_.predict_proba(X.drop(columns="PatCode", errors="ignore"))[:, 1])
        return np.column_stack([1 - p, p])
