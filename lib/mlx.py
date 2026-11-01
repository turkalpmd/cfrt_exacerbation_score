"""Machine-learning helpers. Main-text model: SelectedLGBM / SelectedLGBMRegressor (curated panel, in-fold shadow-feature
selection, Optuna-tuned LightGBM). Supplement experiments (scripts/27_ml_experiments.py, 28_ml_temporal.py): Optuna-tuned
LightGBM with patient-grouped inner CV (optionally monotone or boosted from the clinical model) and the explainable boosting machine."""
import warnings

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import StratifiedGroupKFold

from lib.features import CLIN
from lib.models import lr_unpenalised

MONO = {"FEV1_Z_lag_1": -1, "fev1_z_mean2": -1, "FEV1_PP_lag_1": -1, "fev1_pp_mean2": -1, "bmiZscore_lag_1": -1, "log_iv_burden_2yr": 1,
        "iv_burden_2yr_lag": 1, "ivDaysTotal_lag_1": 1, "ivDaysTotal_lag_2": 1, "any_iv_2y": 1, "any_iv_t": 1, "iv_both_years": 1,
        "PseudomAer_lag_1": 1, "pa_any_2y": 1, "mrsa_any_2y": 1}
sig = lambda z: 1 / (1 + np.exp(-z))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def lgb_space(trial, seed, reg=False):
    p = dict(n_estimators=trial.suggest_int("n_estimators", 50, 800, log=True), learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
             num_leaves=trial.suggest_int("num_leaves", 3, 31, log=True), max_depth=trial.suggest_int("max_depth", 2, 6),
             min_child_samples=trial.suggest_int("min_child_samples", 10, 100, log=True), subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
             colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0), reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50, log=True),
             reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True), verbose=-1, n_jobs=1, random_state=seed)
    if reg:
        p["objective"] = trial.suggest_categorical("objective", ["regression", "huber", "regression_l1"])
    return p


def inner_splits(y, g, seed, regression=False):
    strat = np.digitize(y, np.nanquantile(y, [0.25, 0.5, 0.75])) if regression else y
    return list(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(np.zeros(len(y)), strat, g))


def clin_offset(Xtr, ytr, Xte):
    m = lr_unpenalised().fit(Xtr[CLIN].to_numpy(float), ytr)
    return logit(m.predict_proba(Xtr[CLIN].to_numpy(float))[:, 1]), logit(m.predict_proba(Xte[CLIN].to_numpy(float))[:, 1])


def fit_lgb_clf(Xtr, ytr, gtr, Xte, seed, feats, ntrial, mono=False, offset=False):
    """Optuna (TPE) over lgb_space on the inner patient-grouped log-loss, refit on all of Xtr; returns (test probabilities, best params)."""
    inner = inner_splits(ytr, gtr, seed)
    offs = [clin_offset(Xtr.iloc[i], ytr[i], Xtr.iloc[v]) if offset else (None, None) for i, v in inner]
    mc = [MONO.get(f, 0) for f in feats] if mono else None

    def objective(trial):
        prm = lgb_space(trial, seed); loss = []
        for (i, v), (o_tr, o_va) in zip(inner, offs):
            m = lgb.LGBMClassifier(**prm, monotone_constraints=mc).fit(Xtr.iloc[i][feats], ytr[i], init_score=o_tr)
            p = sig(m.predict(Xtr.iloc[v][feats], raw_score=True) + (o_va if offset else 0))
            loss.append(ll(ytr[v], p))
        return float(np.mean(loss))
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)); st.optimize(objective, n_trials=ntrial)
    prm = {**lgb_space(optuna.trial.FixedTrial(st.best_params), seed)}
    o_tr, o_te = clin_offset(Xtr, ytr, Xte) if offset else (None, 0)
    m = lgb.LGBMClassifier(**prm, monotone_constraints=mc).fit(Xtr[feats], ytr, init_score=o_tr)
    return sig(m.predict(Xte[feats], raw_score=True) + o_te), st.best_params


def fit_ebm(Xtr, ytr, Xte, seed, feats):
    m = ExplainableBoostingClassifier(interactions=5, random_state=seed, n_jobs=1).fit(Xtr[feats], ytr)
    return m.predict_proba(Xte[feats])[:, 1]


# ---------------- main-text machine-learning model (authors' decision, 1 Nov 2026): curated panel -> in-fold selection of
# 10–15 features (SHAP top 20, then the number with the lowest inner CV loss; rule "shap_top") -> Optuna-tuned LightGBM.
# Sensitivity rules: "own_shadow" and "max_shadow" (Boruta-type shadow features). Everything happens inside fit(), so any validation that refits the estimator
# (cross-validation, 2024, leave-one-centre-out) repeats selection and tuning on its own development data.
SHADOW_PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, max_depth=4, min_child_samples=20, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)


def boruta_select(X, y, seed, n_iter=20, task="clf", rule="own_shadow"):
    """Shadow-feature (Boruta-type) selection. Each column gets a permuted 'shadow' copy and LightGBM is fitted n_iter times.
    rule 'own_shadow' (used): a feature scores a hit when its mean |SHAP| exceeds that of its own shadow, and is kept when the
    hits pass a one-sided binomial test (alpha 0.025; >= 15 of 20), i.e. it is reliably better than noise with the same
    distribution. rule 'max_shadow' (classic Boruta, gain importance): a hit when its gain exceeds the largest shadow gain,
    kept in > 50% of fits; strongly favours continuous variables. Returns (kept, hit frequency)."""
    from scipy.stats import binom
    rng = np.random.default_rng(seed); cols = list(X.columns); hits = np.zeros(len(cols))
    est = lgb.LGBMClassifier if task == "clf" else lgb.LGBMRegressor
    Xr = X.reset_index(drop=True)
    for i in range(n_iter):
        sh = pd.DataFrame({f"shadow_{j}": rng.permutation(Xr[c].to_numpy()) for j, c in enumerate(cols)})
        Z = pd.concat([Xr, sh], axis=1)
        m = est(**SHADOW_PARAMS, random_state=seed + i).fit(Z, y)
        if rule == "own_shadow":
            imp = np.abs(m.predict(Z, pred_contrib=True)[:, :-1]).mean(0)
            hits += imp[:len(cols)] > imp[len(cols):]
        else:
            imp = m.booster_.feature_importance(importance_type="gain")
            hits += imp[:len(cols)] > imp[len(cols):].max()
    freq = hits / n_iter
    k = int(binom.ppf(0.975, n_iter, 0.5)) + 1 if rule == "own_shadow" else n_iter // 2 + 1
    kept = [c for c, h in zip(cols, hits) if h >= k] or [cols[i] for i in np.argsort(-freq)[:5]]
    return kept, dict(zip(cols, freq.tolist()))


def shap_rank(X, y, seed, task="clf", n_fit=5):
    """Mean |SHAP| per feature of LightGBM (SHADOW_PARAMS), averaged over n_fit fits on random 80% subsamples (stable ranking)."""
    rng = np.random.default_rng(seed); est = lgb.LGBMClassifier if task == "clf" else lgb.LGBMRegressor
    Xr = X.reset_index(drop=True); imp = np.zeros(X.shape[1])
    for i in range(n_fit):
        idx = rng.choice(len(Xr), int(0.8 * len(Xr)), replace=False)
        m = est(**SHADOW_PARAMS, random_state=seed + i).fit(Xr.iloc[idx], y[idx])
        imp += np.abs(m.predict(Xr.iloc[idx], pred_contrib=True)[:, :-1]).mean(0)
    return pd.Series(imp / n_fit, index=X.columns).sort_values(ascending=False)


def shap_top_select(X, y, g, seed, task="clf", top=20, kmin=10, kmax=15):
    """Authors' rule (1 Nov 2026): the 20 features with the largest mean |SHAP| are short-listed; they are re-ranked on the
    short list and the number kept (10–15) is the one with the lowest inner patient-grouped cross-validated loss (log-loss or
    absolute error; ties -> fewer features). Returns (kept, {feature: mean |SHAP| on the full candidate panel}, loss by k)."""
    reg = task != "clf"; est = lgb.LGBMRegressor if reg else lgb.LGBMClassifier
    r1 = shap_rank(X, y, seed, task); short = list(r1.index[:top])
    order = list(shap_rank(X[short], y, seed + 100, task).index)
    inner = inner_splits(y, g, seed, regression=reg); loss = {}
    for k in range(kmin, min(kmax, len(order)) + 1):
        f = order[:k]; o = np.empty(len(y))
        for i, v in inner:
            m = est(**SHADOW_PARAMS, random_state=seed).fit(X.iloc[i][f], y[i])
            o[v] = m.predict(X.iloc[v][f]) if reg else m.predict_proba(X.iloc[v][f])[:, 1]
        loss[k] = float(np.mean(np.abs(y - o))) if reg else ll(y, o)
    kbest = min(loss, key=lambda k: (round(loss[k], 4), k))
    return order[:kbest], r1.round(5).to_dict(), loss


def tune_lgb(X, y, g, seed, ntrial, task="clf"):
    """Optuna (TPE) over lgb_space on pooled inner patient-grouped out-of-fold log-loss (clf) or absolute error (reg).
    Returns (LightGBM params, best trial params, inner out-of-fold predictions of the best configuration)."""
    reg = task != "clf"; est = lgb.LGBMRegressor if reg else lgb.LGBMClassifier
    inner = inner_splits(y, g, seed, regression=reg)

    def oof_of(prm):
        o = np.empty(len(y))
        for i, v in inner:
            m = est(**prm).fit(X.iloc[i], y[i])
            o[v] = m.predict(X.iloc[v]) if reg else m.predict_proba(X.iloc[v])[:, 1]
        return o

    def objective(trial):
        o = oof_of(lgb_space(trial, seed, reg=reg))
        return float(np.mean(np.abs(y - o))) if reg else ll(y, o)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)); st.optimize(objective, n_trials=ntrial)
    prm = lgb_space(optuna.trial.FixedTrial(st.best_params), seed, reg=reg)
    return prm, st.best_params, oof_of(prm)


class _SelectedBase(BaseEstimator):
    """X: DataFrame containing the candidate features and a 'PatCode' column (inner folds are grouped by patient).
    Missing values are left to LightGBM (no imputation)."""
    task = "clf"

    def __init__(self, feats=None, seed=42, ntrial=40, n_shadow=20, rule="shap_top"):
        self.feats, self.seed, self.ntrial, self.n_shadow, self.rule = feats, seed, ntrial, n_shadow, rule

    def fit(self, X, y):
        warnings.filterwarnings("ignore")
        y = np.asarray(y, float); g = pd.factorize(X["PatCode"])[0]
        Xf = X[list(self.feats)].astype(float).reset_index(drop=True)
        if self.rule == "shap_top":
            self.selected_, self.selection_freq_, self.k_loss_ = shap_top_select(Xf, y, g, self.seed, self.task)
        else:
            self.selected_, self.selection_freq_ = boruta_select(Xf, y, self.seed, self.n_shadow, self.task, self.rule)
        prm, self.best_params_, inner_oof = tune_lgb(Xf[self.selected_], y, g, self.seed, self.ntrial, self.task)
        self.inner_residuals_ = y - inner_oof
        est = lgb.LGBMClassifier if self.task == "clf" else lgb.LGBMRegressor
        self.model_ = est(**prm).fit(Xf[self.selected_], y)
        return self

    def contributions(self, X):
        """TreeSHAP contributions of the selected features (last column = expected value), on the raw-score scale."""
        return self.model_.predict(X[self.selected_].astype(float), pred_contrib=True)


class SelectedLGBM(ClassifierMixin, _SelectedBase):
    task = "clf"

    def fit(self, X, y):
        super().fit(X, y); self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        p = self.model_.predict_proba(X[self.selected_].astype(float))[:, 1]
        return np.column_stack([1 - p, p])


class SelectedLGBMRegressor(RegressorMixin, _SelectedBase):
    task = "reg"

    def predict(self, X):
        return self.model_.predict(X[self.selected_].astype(float))
