"""GARCH(1,1) + Filtered Historical Simulation engine for the Phase 2 project.

The Phase 1 engines in ``risk.py`` assumed that the W-day rolling sample of
returns was an iid draw from a Gaussian or Student-t distribution. The
empirical evidence in notebook 02 strongly contradicted that assumption:
returns show heavy volatility clustering (Ljung-Box on r^2 with LR > 4,000;
ARCH-LM LM > 2,000). Notebook 04 then showed that exception clustering
breaks the iid assumption for *every* Phase 1 engine, regardless of
distributional family (Christoffersen independence test rejects in every
case at p < 0.0001).

This module implements the standard remedy:

1. Fit a GARCH(1,1) model with zero mean to the rolling window. This gives a
   time-varying *conditional* variance sigma_t^2 that adapts to volatility
   regimes within days, not years.

2. Compute the standardised residuals z_t = r_t / sigma_t. These are
   approximately iid by construction.

3. Form a 1-day-ahead VaR/ES forecast by **filtering** -- multiplying the
   1-step variance forecast sigma_{t+1} by the empirical quantile of the
   residual pool. This is *Filtered Historical Simulation* (Barone-Adesi,
   Giannopoulos & Vosper 1999): no parametric assumption on z, the empirical
   shape is whatever the data say.

Why FHS rather than parametric GARCH-t? Two reasons. (1) FHS lets the
distributional shape be whatever the standardised residuals empirically are,
without imposing a parametric family. (2) For the dual-test backtest battery
of notebook 04, FHS combines an adaptive variance with an empirical tail --
hitting both the unconditional-coverage and independence calibration
questions simultaneously, which is exactly the test we set up in section 4.6
of notebook 04.

Reference:
    Barone-Adesi, G., Giannopoulos, K., & Vosper, L. (1999).
    "VaR Without Correlations for Non-Linear Portfolios."
    Journal of Futures Markets, 19(5), 583-602.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

# Internal: scale daily returns into percent units for the GARCH optimiser.
# The arch package's likelihood is much better conditioned when r is O(1) than
# when it is O(0.01). We rescale back to decimal when reporting VaR/ES.
_SCALE = 100.0


# ---------------------------------------------------------------------------
# Single-window fit and FHS step
# ---------------------------------------------------------------------------

def fit_garch_window(returns_window, p=1, q=1, dist='normal'):
    """Fit a zero-mean GARCH(p, q) model to a window of daily returns.

    Parameters
    ----------
    returns_window : pd.Series
        Daily returns (decimal scale, e.g., 0.012 = 1.2%).
    p, q : int
        Orders of the ARCH and GARCH terms (default 1, 1).
    dist : str
        Distribution of innovations passed to ``arch_model`` (default 'normal').
        Note: this only affects the GARCH likelihood; the FHS step does not
        depend on it because the residual pool is used non-parametrically.

    Returns
    -------
    arch.univariate.ARCHModelResult
    """
    am = arch_model(
        returns_window * _SCALE,
        mean='Zero',
        vol='Garch',
        p=p, q=q,
        dist=dist,
        rescale=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = am.fit(disp='off', show_warning=False, update_freq=0)
    return res


def garch_step_var_es(sigma_next, residual_pool, alpha):
    """Compute FHS VaR and ES from a 1-step-ahead sigma and a residual pool.

    Parameters
    ----------
    sigma_next : float
        Conditional volatility forecast in DECIMAL units (e.g. 0.012 = 1.2%/day).
    residual_pool : array-like
        Standardised residuals from the most recent GARCH fit (mean ~0,
        std ~1, in standardised units).
    alpha : float
        Confidence level for VaR/ES (e.g., 0.99).

    Returns
    -------
    (VaR, ES) : tuple of float, in DECIMAL units.

    Notes
    -----
    Let z be the residual pool. Define q = quantile(z, 1 - alpha) -- the
    lower-tail quantile. Then:

        VaR_alpha = -sigma_next * q
        ES_alpha  = -sigma_next * mean(z | z <= q)

    The negative signs convert the lower-tail of returns into a positive loss.
    """
    z = np.asarray(residual_pool, dtype=float)
    if z.size == 0:
        return float('nan'), float('nan')
    q = float(np.quantile(z, 1.0 - alpha))
    var = -sigma_next * q
    tail = z[z <= q]
    es = -sigma_next * float(tail.mean()) if tail.size else var
    return float(var), float(es)


# ---------------------------------------------------------------------------
# Rolling GARCH-FHS engine
# ---------------------------------------------------------------------------

def rolling_garch_fhs(returns, window, alpha, refit_freq=20, verbose=False):
    """Rolling 1-day-ahead GARCH-FHS VaR and ES forecasts.

    For each forecast date d, the engine produces (VaR_d, ES_d) using:

    - the GARCH(1,1) parameters most recently estimated on the trailing
      W-day window (re-estimated every ``refit_freq`` business days);
    - the GARCH(1,1) recursion applied between refits to update
      sigma_{t+1}^2 daily based on r_t;
    - the standardised-residual pool from the most recent fit, used as the
      non-parametric distribution of innovations.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns in decimal units, date-indexed.
    window : int
        Length of the rolling estimation window (e.g., 1250 for ~5y).
    alpha : float
        Confidence level for VaR/ES (e.g., 0.99).
    refit_freq : int, default 20
        Re-estimate GARCH parameters every N business days. Between refits,
        sigma_{t+1} is updated daily via the GARCH recursion using the most
        recent estimated parameters; the residual pool is held fixed.
        ``refit_freq = 1`` means refit on every forecast date (most expensive
        but theoretically cleanest).
    verbose : bool
        If True, print progress every 500 forecast days.

    Returns
    -------
    pd.DataFrame
        Columns ``['VaR', 'ES', 'sigma']`` indexed by the FORECAST date.
        Sigma is reported in decimal units to allow direct comparison with
        the Phase 1 rolling-sigma estimates.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    n = len(returns)
    out_rows = []

    # GARCH state held between refits
    omega = alpha_p = beta_p = None
    residual_pool = None
    sigma2_current = None   # in % scale (since arch was fitted to r * SCALE)

    days_since_fit = refit_freq  # force a refit at the very first forecast date

    for t in range(window, n):
        # ---- (Re)fit GARCH if needed ----
        if days_since_fit >= refit_freq:
            window_data = returns.iloc[t - window:t]
            try:
                res = fit_garch_window(window_data)
                params = res.params
                omega = float(params['omega'])
                alpha_p = float(params['alpha[1]'])
                beta_p = float(params['beta[1]'])
                # Last fitted conditional sigma (in % scale)
                sigma2_current = float(res.conditional_volatility.iloc[-1] ** 2)
                residual_pool = np.asarray(res.std_resid.values, dtype=float)
                days_since_fit = 0
            except Exception as exc:
                if verbose:
                    print(f'  t={t}  GARCH fit failed: {exc}')
                # Keep the previous state if a fit fails

        if omega is None:
            # We never managed a successful fit, skip until we do
            days_since_fit += 1
            continue

        # ---- One-step-ahead GARCH recursion ----
        # In % scale: sigma2_next = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}
        r_last_scaled = float(returns.iloc[t - 1]) * _SCALE
        sigma2_next = omega + alpha_p * r_last_scaled ** 2 + beta_p * sigma2_current
        sigma_next_decimal = float(np.sqrt(sigma2_next) / _SCALE)

        # ---- FHS VaR/ES ----
        var, es = garch_step_var_es(sigma_next_decimal, residual_pool, alpha)

        out_rows.append({
            'date': returns.index[t],
            'VaR': var,
            'ES': es,
            'sigma': sigma_next_decimal,
        })

        # ---- Roll state forward ----
        sigma2_current = sigma2_next
        days_since_fit += 1

        if verbose and (t - window) % 500 == 0:
            print(f'  t={t}/{n}  date={returns.index[t].date()}  '
                  f'sigma={sigma_next_decimal * 100:.3f}%  '
                  f'VaR={var * 100:.3f}%')

    df = pd.DataFrame(out_rows).set_index('date') if out_rows else pd.DataFrame(
        columns=['VaR', 'ES', 'sigma']
    )
    return df
