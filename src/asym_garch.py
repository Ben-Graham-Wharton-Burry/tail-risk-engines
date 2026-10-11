"""Phase 3 asymmetric-GARCH extension of the GARCH-FHS engine.

Phase 2 used a symmetric GARCH(1,1): a negative and a positive return of equal
size raise next-day volatility equally. Equity returns violate this - the
leverage effect: bad news raises volatility more than good news. Phase 3 adds
two standard asymmetric volatility models and runs them through the same
Filtered Historical Simulation (FHS) step as Phase 2.

    GJR-GARCH(1,1,1)  (Glosten, Jagannathan & Runkle, 1993)
        sigma2_t = omega + alpha r2_{t-1}
                   + gamma r2_{t-1} 1{r_{t-1} < 0} + beta sigma2_{t-1}

    EGARCH(1,1,1)  (Nelson, 1991)
        ln sigma2_t = omega + alpha (|z_{t-1}| - E|z|)
                      + gamma z_{t-1} + beta ln sigma2_{t-1}

GJR is updated with an explicit one-step recursion between refits. EGARCH is
refit on every forecast date and stepped one day ahead with the arch package's
own forecast machinery, with a finiteness guard - EGARCH log-variance fits can
return pathological parameters on near-homoskedastic windows.

References
----------
Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993). Journal of Finance 48(5).
Nelson, D.B. (1991). Econometrica 59(2), 347-370.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from garch_fhs import garch_step_var_es, _SCALE

_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))   # E|z| for a standard normal
_MAX_PCT_VARIANCE = 1.0e4                       # sigma > 100%/day => failed fit


def fit_asymmetric_window(returns_window, model='gjr', dist='normal'):
    """Fit a zero-mean GJR-GARCH(1,1,1) or EGARCH(1,1,1) model via `arch`."""
    model = model.lower()
    if model == 'gjr':
        am = arch_model(returns_window * _SCALE, mean='Zero', vol='Garch',
                        p=1, o=1, q=1, dist=dist, rescale=False)
    elif model == 'egarch':
        am = arch_model(returns_window * _SCALE, mean='Zero', vol='EGARCH',
                        p=1, o=1, q=1, dist=dist, rescale=False)
    else:
        raise ValueError("model must be 'gjr' or 'egarch', got %r" % model)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = am.fit(disp='off', show_warning=False, update_freq=0)
    return res


def gjr_step_sigma2(omega, alpha, gamma, beta, r_prev, sigma2_prev):
    """One-step GJR-GARCH conditional-variance recursion (percent scale).

    The leverage term gamma*r_prev^2 is added only when the previous return
    was negative, so for gamma > 0 a negative shock raises next-day variance
    more than a positive shock of equal magnitude.
    """
    leverage = gamma * r_prev ** 2 if r_prev < 0 else 0.0
    return omega + alpha * r_prev ** 2 + leverage + beta * sigma2_prev


def egarch_step_lnsigma2(omega, alpha, gamma, beta, z_prev, lnsigma2_prev):
    """One-step EGARCH log-variance recursion (reference implementation).

    Documents the EGARCH dynamics and is used to verify the leverage property
    in the unit tests. With gamma < 0 a negative standardised shock raises
    log-variance more than a positive one.
    """
    return (omega + alpha * (abs(z_prev) - _SQRT_2_OVER_PI)
            + gamma * z_prev + beta * lnsigma2_prev)


def _rolling_gjr(returns, window, alpha, refit_freq, verbose):
    """Rolling GJR-GARCH + FHS: refit every refit_freq days, recurse between."""
    n = len(returns)
    out_rows = []
    params = None
    residual_pool = None
    sigma2_current = None
    days_since_fit = refit_freq

    for t in range(window, n):
        if days_since_fit >= refit_freq:
            try:
                res = fit_asymmetric_window(returns.iloc[t - window:t],
                                            model='gjr')
                p = res.params
                params = (float(p['omega']), float(p['alpha[1]']),
                          float(p['gamma[1]']), float(p['beta[1]']))
                sigma2_current = float(res.conditional_volatility.iloc[-1] ** 2)
                residual_pool = np.asarray(res.std_resid.values, dtype=float)
                days_since_fit = 0
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print('  t=%d  gjr fit failed: %s' % (t, exc))

        if params is None:
            days_since_fit += 1
            continue

        omega, a1, g1, b1 = params
        r_last = float(returns.iloc[t - 1]) * _SCALE
        sigma2_next = gjr_step_sigma2(omega, a1, g1, b1, r_last, sigma2_current)
        sigma_next = float(np.sqrt(sigma2_next) / _SCALE)
        var, es = garch_step_var_es(sigma_next, residual_pool, alpha)
        out_rows.append({'date': returns.index[t], 'VaR': var, 'ES': es,
                         'sigma': sigma_next})
        sigma2_current = sigma2_next
        days_since_fit += 1

    if out_rows:
        return pd.DataFrame(out_rows).set_index('date')
    return pd.DataFrame(columns=['VaR', 'ES', 'sigma'])


def _rolling_egarch(returns, window, alpha, verbose):
    """Rolling EGARCH + FHS: refit daily, one-step forecast via arch, guarded."""
    n = len(returns)
    out_rows = []
    last_sigma = None
    last_pool = None

    for t in range(window, n):
        sigma_next = None
        pool = None
        try:
            res = fit_asymmetric_window(returns.iloc[t - window:t],
                                        model='egarch')
            fc = res.forecast(horizon=1, reindex=False)
            var_scaled = float(fc.variance.iloc[-1, 0])
            if np.isfinite(var_scaled) and 0.0 < var_scaled < _MAX_PCT_VARIANCE:
                sigma_next = float(np.sqrt(var_scaled) / _SCALE)
                pool = np.asarray(res.std_resid.values, dtype=float)
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print('  t=%d  egarch fit failed: %s' % (t, exc))

        if sigma_next is None:           # pathological fit: carry last good
            sigma_next, pool = last_sigma, last_pool
        if sigma_next is None:           # no good fit yet
            continue

        last_sigma, last_pool = sigma_next, pool
        var, es = garch_step_var_es(sigma_next, pool, alpha)
        out_rows.append({'date': returns.index[t], 'VaR': var, 'ES': es,
                         'sigma': sigma_next})

    if out_rows:
        return pd.DataFrame(out_rows).set_index('date')
    return pd.DataFrame(columns=['VaR', 'ES', 'sigma'])


def rolling_asymmetric_fhs(returns, window, alpha, model='gjr',
                           refit_freq=20, verbose=False):
    """Rolling 1-day-ahead VaR/ES from an asymmetric GARCH + FHS engine.

    Returns a DataFrame with columns ['VaR', 'ES', 'sigma'] indexed by the
    forecast date, in decimal units - mirroring rolling_garch_fhs. For GJR the
    refit cadence is refit_freq; EGARCH is refit on every forecast date.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    model = model.lower()
    if model == 'gjr':
        return _rolling_gjr(returns, window, alpha, refit_freq, verbose)
    if model == 'egarch':
        return _rolling_egarch(returns, window, alpha, verbose)
    raise ValueError("model must be 'gjr' or 'egarch', got %r" % model)
