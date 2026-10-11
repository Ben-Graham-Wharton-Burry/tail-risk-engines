"""Phase 3 additional risk metrics and the LASSO challenger baseline.

Three small, self-contained additions:

    conditional_expected_drawdown(mdd_samples, alpha)
        CED (Goldberg & Mahmoud, 2017) - the tail mean of the maximum-drawdown
        distribution: the average of the worst (1 - alpha) simulated MDDs.
        It is to drawdowns what Expected Shortfall is to single-day losses.

    downside_deviation / semivariance / sortino_ratio
        Downside-only dispersion measures. Unlike standard deviation they
        penalise only returns below a target, matching how a family office
        actually experiences risk.

    lasso_exceedance_probabilities(X_train, y_train, X_test, C)
        L1-penalised (LASSO) logistic regression - the linear baseline that
        the notebook-07 gradient-boosting challenger should be compared
        against, with features standardised before the fit.

Loss / drawdown convention: drawdowns and MDDs are positive fractions.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Conditional Expected Drawdown
# ---------------------------------------------------------------------------

def conditional_expected_drawdown(mdd_samples, alpha=0.95):
    """Conditional Expected Drawdown: the tail mean of the MDD distribution.

    Given a sample of maximum drawdowns (e.g. from a Monte-Carlo or bootstrap
    simulation) CED_alpha is the mean of those at or above the alpha-quantile
    - the average of the worst (1 - alpha) drawdown outcomes.
    """
    m = np.asarray(mdd_samples, dtype=float)
    m = m[np.isfinite(m)]
    if m.size == 0:
        return float('nan')
    threshold = float(np.quantile(m, alpha))
    tail = m[m >= threshold]
    return float(tail.mean()) if tail.size else threshold


# ---------------------------------------------------------------------------
# Downside-risk measures
# ---------------------------------------------------------------------------

def semivariance(returns, target=0.0):
    """Mean squared shortfall below ``target`` (downside semivariance)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    downside = np.minimum(r - target, 0.0)
    return float(np.mean(downside ** 2))


def downside_deviation(returns, target=0.0, periods_per_year=None):
    """Downside deviation = sqrt(semivariance below target).

    If ``periods_per_year`` is given the result is annualised by sqrt(p).
    """
    dd = np.sqrt(semivariance(returns, target))
    if periods_per_year:
        dd *= np.sqrt(periods_per_year)
    return float(dd)


def sortino_ratio(returns, target=0.0, periods_per_year=252):
    """Annualised Sortino ratio: excess return per unit of downside deviation."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    excess = float(np.mean(r)) - target
    dd = downside_deviation(r, target)            # per-period
    if dd == 0.0:
        return float('nan')
    return float(excess / dd * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# LASSO logistic challenger baseline
# ---------------------------------------------------------------------------

def lasso_exceedance_probabilities(X_train, y_train, X_test, C=1.0, seed=0):
    """L1-penalised logistic regression: predicted exceedance probabilities.

    Features are standardised on the training set (LASSO is scale-sensitive).
    Returns the predicted probability of class 1 (a VaR exception) for each
    row of ``X_test``. This is the linear baseline for the gradient-boosting
    challenger in notebook 07.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=C,
                             random_state=seed)
    clf.fit(scaler.transform(X_train), np.asarray(y_train).astype(int))
    return clf.predict_proba(scaler.transform(X_test))[:, 1]
