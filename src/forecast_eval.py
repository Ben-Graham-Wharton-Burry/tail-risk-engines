"""Forecast-to-risk-metric evaluation (NB 11, extension 1).

Makes the volatility-to-risk relationship explicit: does the GARCH-FHS forecast
actually track the risk the family office subsequently experiences? This is the
formal justification for *trading on* the forecast.

Tools
-----
realized_vol_forward(returns, horizon)
    Forward realised volatility: at each date, the (annualised) standard
    deviation of the next ``horizon`` daily returns -- the quantity the forecast
    is meant to predict.

mincer_zarnowitz(realized, forecast)
    The Mincer-Zarnowitz forecast-evaluation regression
        realised_t = alpha + beta * forecast_t + e_t,
    with Newey-West (HAC) standard errors. A forecast is *unbiased* iff
    alpha = 0 and beta = 1; R^2 measures how much realised risk variation the
    forecast explains. Reports two-sided p-values for H0: alpha = 0 and
    H0: beta = 1.

regress_metric_on_forecast(metric, forecast)
    Thin wrapper to regress any realised downside metric (e.g. forward
    drawdown, forward downside deviation) on the forecast, same HAC inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

TRADING_DAYS = 252


def realized_vol_forward(returns, horizon=21, annualize=True):
    """At each date t, the std of returns over (t, t+horizon]; annualised by default.

    The last ``horizon`` dates have no full forward window and are NaN.
    """
    r = pd.Series(returns).astype(float)
    arr = r.to_numpy()
    n = len(arr)
    out = np.full(n, np.nan)
    if n > horizon:
        windows = sliding_window_view(arr[1:], horizon)     # row i = arr[i+1 : i+1+h]
        sd = windows.std(axis=1, ddof=1)
        out[: len(sd)] = sd
    fv = pd.Series(out, index=r.index, name="realized_vol_fwd")
    return fv * np.sqrt(TRADING_DAYS) if annualize else fv


def _ols_hac(y, X, lags=None):
    """OLS with Newey-West HAC covariance. Returns (beta, se, resid)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))   # Newey-West rule of thumb
    u = X * resid[:, None]
    S = u.T @ u
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = u[l:].T @ u[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.diag(cov)), resid


def mincer_zarnowitz(realized, forecast):
    """Mincer-Zarnowitz regression realised = alpha + beta*forecast (HAC inference).

    Returns a dict with alpha, beta, their HAC standard errors, R^2, n, and the
    two-sided p-values for H0: alpha = 0 and H0: beta = 1 (joint unbiasedness is
    alpha = 0 AND beta = 1).
    """
    df = (pd.concat({"y": pd.Series(realized), "f": pd.Series(forecast)}, axis=1)
          .replace([np.inf, -np.inf], np.nan).dropna())
    y = df["y"].to_numpy()
    f = df["f"].to_numpy()
    n = len(df)
    X = np.column_stack([np.ones(n), f])
    beta, se, resid = _ols_hac(y, X)
    a, b = beta
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    t_a = a / se[0]
    t_b = (b - 1.0) / se[1]
    return {
        "alpha": float(a), "beta": float(b),
        "se_alpha": float(se[0]), "se_beta": float(se[1]),
        "R2": float(r2), "n": int(n),
        "p_alpha_eq0": float(2 * (1 - stats.norm.cdf(abs(t_a)))),
        "p_beta_eq1": float(2 * (1 - stats.norm.cdf(abs(t_b)))),
    }


def regress_metric_on_forecast(metric, forecast):
    """Regress any realised metric on the forecast (same HAC inference as MZ).

    Returns alpha, beta, se_beta, R^2, n and the p-value for H0: beta = 0
    (i.e. 'does the forecast carry information about this metric at all?').
    """
    df = (pd.concat({"y": pd.Series(metric), "f": pd.Series(forecast)}, axis=1)
          .replace([np.inf, -np.inf], np.nan).dropna())
    y = df["y"].to_numpy()
    f = df["f"].to_numpy()
    n = len(df)
    X = np.column_stack([np.ones(n), f])
    beta, se, resid = _ols_hac(y, X)
    a, b = beta
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)
    return {"alpha": float(a), "beta": float(b), "se_beta": float(se[1]),
            "R2": float(r2), "n": int(n),
            "p_beta_eq0": float(2 * (1 - stats.norm.cdf(abs(b / se[1]))))}
