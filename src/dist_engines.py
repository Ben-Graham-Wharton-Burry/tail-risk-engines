"""Phase 3 richer-distribution VaR/ES engines.

Phase 1 fixed the Student-t degrees of freedom at nu in {4, 6, 10}. Phase 3
adds two data-driven alternatives:

    rolling_t_mle_var_es : Student-t with nu re-estimated by maximum
        likelihood on each rolling window instead of fixed. Answers the
        question "does letting the data choose the tail thickness help?".

    rolling_skewt_var_es : Hansen (1994) skewed Student-t. Captures BOTH
        heavy tails (eta) and asymmetry (lambda) - the negative skew that a
        symmetric Student-t cannot represent. The skew-t shape is estimated
        by maximum likelihood via the `arch` package's SkewStudent; VaR/ES
        use the standardised skew-t quantile function.

Loss convention: losses are positive; VaR and ES are positive numbers.

Reference
---------
Hansen, B.E. (1994). "Autoregressive Conditional Density Estimation."
    International Economic Review 35(3), 705-730.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from arch.univariate import SkewStudent

from risk import student_t_var_es
from garch_fhs import _SCALE


# ---------------------------------------------------------------------------
# MLE Student-t degrees of freedom
# ---------------------------------------------------------------------------

def estimate_t_nu(window, nu_min=2.5, nu_max=100.0):
    """Maximum-likelihood estimate of the Student-t degrees of freedom.

    Uses scipy's Student-t MLE on the window and clamps the result to
    [nu_min, nu_max]. nu_min > 2 keeps the variance finite; nu_max caps the
    near-Gaussian case in which the degrees of freedom are weakly identified.
    """
    data = np.asarray(window, dtype=float)
    data = data[np.isfinite(data)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df, _loc, _scale = stats.t.fit(data)
    return float(min(max(df, nu_min), nu_max))


def rolling_t_mle_var_es(returns, window, alpha, nu_min=2.5, nu_max=100.0):
    """Rolling 1-day-ahead VaR/ES from a Student-t with MLE-estimated nu.

    Location and scale use the rolling-window mean and standard deviation -
    identical to the Phase 1 t-engine - so the only change is that nu is
    estimated from the data rather than fixed.

    Returns a DataFrame with columns ['VaR', 'ES', 'nu'] indexed by the
    forecast date.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    rows = []
    for t in range(window, len(returns)):
        win = returns.iloc[t - window:t]
        nu = estimate_t_nu(win, nu_min, nu_max)
        var, es = student_t_var_es(float(win.mean()), float(win.std()),
                                   nu, alpha)
        rows.append({'date': returns.index[t], 'VaR': float(var),
                     'ES': float(es), 'nu': nu})
    if rows:
        return pd.DataFrame(rows).set_index('date')
    return pd.DataFrame(columns=['VaR', 'ES', 'nu'])


# ---------------------------------------------------------------------------
# Hansen skewed Student-t
# ---------------------------------------------------------------------------

def fit_skewt_shape(window):
    """Estimate the Hansen skew-t shape (eta, lambda) on a return window.

    A constant-mean / constant-variance model with a skew-t innovation is
    fitted by maximum likelihood; only the standardised shape parameters are
    returned. ``eta`` is the tail (degrees-of-freedom) parameter; ``lambda``
    in (-1, 1) is the skewness, with lambda < 0 a heavier left tail.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        am = arch_model(np.asarray(window, dtype=float) * _SCALE,
                        mean='Constant', vol='Constant', dist='skewt')
        res = am.fit(disp='off', show_warning=False, update_freq=0)
    return float(res.params['eta']), float(res.params['lambda'])


def skewt_var_es(mu, sigma, eta, lam, alpha, n_grid=400):
    """VaR and ES for a Hansen skew-t return distribution.

    The return is mu + sigma * Z with Z a standardised Hansen skew-t. VaR is
    minus the (1 - alpha) quantile; ES is minus the average of the quantiles
    below it, ES = -(1/p) * integral_0^p F^{-1}(u) du.
    """
    p = 1.0 - alpha
    dist = SkewStudent()
    par = np.array([eta, lam], dtype=float)
    q = float(dist.ppf(p, par))
    var = -(mu + sigma * q)
    grid = np.linspace(p / n_grid, p, n_grid)
    es_std = float(np.mean(dist.ppf(grid, par)))
    es = -(mu + sigma * es_std)
    return float(var), float(es)


def rolling_skewt_var_es(returns, window, alpha, refit_freq=20):
    """Rolling 1-day-ahead VaR/ES from a Hansen skew-t engine.

    The skew-t shape (eta, lambda) is re-estimated every ``refit_freq`` days;
    location and scale use the rolling-window mean and standard deviation and
    update daily. Returns a DataFrame with columns ['VaR', 'ES', 'eta', 'lam']
    indexed by the forecast date.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    rows = []
    eta = lam = None
    since_fit = refit_freq
    for t in range(window, len(returns)):
        win = returns.iloc[t - window:t]
        if since_fit >= refit_freq:
            try:
                eta, lam = fit_skewt_shape(win)
                since_fit = 0
            except Exception:  # noqa: BLE001
                pass
        if eta is None:
            since_fit += 1
            continue
        var, es = skewt_var_es(float(win.mean()), float(win.std()),
                               eta, lam, alpha)
        rows.append({'date': returns.index[t], 'VaR': var, 'ES': es,
                     'eta': eta, 'lam': lam})
        since_fit += 1
    if rows:
        return pd.DataFrame(rows).set_index('date')
    return pd.DataFrame(columns=['VaR', 'ES', 'eta', 'lam'])
