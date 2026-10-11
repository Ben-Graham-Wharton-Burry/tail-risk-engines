"""Drawdown metrics for Phase 2 of the tail-risk study.

Path-dependent tail-risk measures derived from cumulative wealth:

    compute_wealth_drawdowns(returns) -> DataFrame[wealth, peak, drawdown]
        Wealth path, running maximum (high-water mark), and drawdown
        D_t = W_t / M_t - 1 (non-positive).

    max_drawdown(returns) -> float
        |min D_t| -- the deepest realised drawdown in the sample (non-neg).

    drawdown_episodes(returns, min_depth=0) -> DataFrame
        One row per drawdown episode with peak/trough/recovery dates,
        depth, days-to-trough, days-to-recovery, and total underwater
        days.

    time_under_water_stats(returns) -> dict
        Aggregate TuW statistics: % time underwater, max TuW spell length,
        mean TuW spell length, number of episodes.

    cdar(returns, alpha=0.95) -> float
        Conditional Drawdown-at-Risk: mean |drawdown| in the alpha-tail.

    simulate_mdd_distribution(...) -> ndarray
        Forward-simulate paths under a chosen engine and return the
        empirical distribution of horizon-MDD. Engines span parametric
        Monte-Carlo (Gaussian, Student-t), the filtered residual bootstrap
        (FHS), and -- as the non-parametric resampling alternative to
        parametric Monte-Carlo -- the i.i.d. and circular block bootstraps.

The unconditional return-distribution case (sample MDD over the historical
series) is a single value -- a *realised* statistic.  The simulated-MDD
distribution lets us compare engine-implied MDD risk, parallel to the way
the rolling VaR/ES analysis compared engine-implied VaR.

The block bootstrap is included specifically for the path-dependent layer:
i.i.d. resampling reproduces the marginal distribution of returns but
destroys volatility clustering, which systematically *under*-states MDD
because clustered losses compound into deeper drawdowns. Resampling blocks
of consecutive days retains that dependence.

References:
    Magdon-Ismail, M., Atiya, A.F., et al. (2004).
        "On the Maximum Drawdown of a Brownian Motion."
        Journal of Applied Probability 41(1), 147-161.
    Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005).
        "Drawdown Measure in Portfolio Optimization."
        International Journal of Theoretical and Applied Finance 8(1), 13-58.
    Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife."
        Annals of Statistics 7(1), 1-26.
    Kuensch, H.R. (1989). "The Jackknife and the Bootstrap for General
        Stationary Observations." Annals of Statistics 17(3), 1217-1241.
    Politis, D.N., & Romano, J.P. (1992). "A Circular Block-Resampling
        Procedure for Stationary Data." In Exploring the Limits of the
        Bootstrap, Wiley, 263-270.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Realised drawdown analytics
# ---------------------------------------------------------------------------

def compute_wealth_drawdowns(returns):
    """Compute the wealth path, running maximum, and drawdown series.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns (decimal, e.g., 0.01 = 1%).

    Returns
    -------
    pd.DataFrame
        Columns ``wealth``, ``peak``, ``drawdown`` indexed by the same
        dates as the input. ``drawdown`` is non-positive; -0.20 means the
        path is 20% below its running peak.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return pd.DataFrame({
        'wealth'  : wealth,
        'peak'    : peak,
        'drawdown': drawdown,
    })


def max_drawdown(returns):
    """Return the maximum drawdown as a non-negative percentage."""
    dd = compute_wealth_drawdowns(returns)['drawdown']
    return float(-dd.min())


def drawdown_episodes(returns, min_depth=0.0):
    """Identify all drawdown episodes in the return series.

    An episode begins the first day the wealth path falls below a prior peak
    and ends either (a) the first subsequent day the path equals or exceeds
    that peak (recovery), or (b) the last day of the sample (unrecovered).

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    min_depth : float, default 0.0
        Filter out episodes with depth (positive) less than this threshold.
        E.g., ``min_depth=0.05`` keeps only episodes with at least a 5%
        drawdown. Useful for excluding tiny noise drawdowns.

    Returns
    -------
    pd.DataFrame
        One row per episode, columns:
            ``peak_date``           Date of the peak preceding the episode.
            ``trough_date``         Date of the deepest point in the episode.
            ``recovery_date``       Date the prior peak is recovered (NaT if not).
            ``depth``               Magnitude of the trough drawdown (non-neg).
            ``days_to_trough``      Trading days from peak to trough.
            ``days_to_recovery``    Trading days from peak to recovery (NaN if not).
            ``total_underwater_days`` Total trading days underwater for this episode.
    """
    d = compute_wealth_drawdowns(returns)
    drawdown = d['drawdown']
    underwater = (drawdown < 0).values
    n = len(drawdown)

    episodes = []
    i = 0
    while i < n:
        if underwater[i]:
            ep_start = i
            trough_depth = 0.0
            trough_idx = ep_start
            j = i
            while j < n and underwater[j]:
                if drawdown.iloc[j] < trough_depth:
                    trough_depth = float(drawdown.iloc[j])
                    trough_idx = j
                j += 1
            recovered = j < n
            recovery_idx = j if recovered else None

            peak_idx = ep_start - 1 if ep_start > 0 else 0
            depth = -trough_depth
            if depth >= min_depth:
                episodes.append({
                    'peak_date'             : drawdown.index[peak_idx],
                    'trough_date'           : drawdown.index[trough_idx],
                    'recovery_date'         : drawdown.index[recovery_idx] if recovered else pd.NaT,
                    'depth'                 : depth,
                    'days_to_trough'        : trough_idx - peak_idx,
                    'days_to_recovery'      : (recovery_idx - peak_idx) if recovered else np.nan,
                    'total_underwater_days' : (j - ep_start),
                })
            i = j + 1
        else:
            i += 1

    return pd.DataFrame(episodes)


def time_under_water_stats(returns):
    """Aggregate time-under-water statistics for a return series.

    Returns
    -------
    dict
        ``n_total_days``         Total observation days.
        ``n_underwater_days``    Days when wealth was strictly below the running peak.
        ``pct_time_underwater``  Fraction of days underwater (0-1).
        ``max_tuw_days``         Length of the longest single underwater spell.
        ``mean_tuw_days``        Mean spell length across all episodes.
        ``n_episodes``           Number of distinct drawdown episodes.
    """
    d = compute_wealth_drawdowns(returns)
    underwater = (d['drawdown'] < 0)
    n_total = int(len(underwater))
    n_under = int(underwater.sum())

    eps = drawdown_episodes(returns)
    if len(eps):
        max_tuw = int(eps['total_underwater_days'].max())
        mean_tuw = float(eps['total_underwater_days'].mean())
    else:
        max_tuw = 0
        mean_tuw = 0.0

    return {
        'n_total_days'        : n_total,
        'n_underwater_days'   : n_under,
        'pct_time_underwater' : n_under / n_total if n_total else float('nan'),
        'max_tuw_days'        : max_tuw,
        'mean_tuw_days'       : mean_tuw,
        'n_episodes'          : int(len(eps)),
    }


def cdar(returns, alpha=0.95):
    """Conditional Drawdown-at-Risk.

    DaR_alpha is the alpha-quantile of the (positive) drawdown distribution.
    CDaR_alpha is the conditional mean of drawdowns in the alpha-tail:
        CDaR_alpha = E[|D| | |D| >= DaR_alpha].

    Parameters
    ----------
    returns : pd.Series
    alpha : float, default 0.95
        Tail probability. For alpha=0.95, CDaR is the mean of the worst
        5% of all observed drawdowns.

    Returns
    -------
    float
        CDaR_alpha as a non-negative magnitude.
    """
    abs_dd = -compute_wealth_drawdowns(returns)['drawdown'].values
    if abs_dd.size == 0:
        return float('nan')
    threshold = float(np.quantile(abs_dd, alpha))
    tail = abs_dd[abs_dd >= threshold]
    return float(tail.mean()) if len(tail) else 0.0


# ---------------------------------------------------------------------------
# Forward Monte-Carlo simulation
# ---------------------------------------------------------------------------

def _circular_block_bootstrap(pool, n_sims, horizon, block_length, rng):
    """Circular (moving) block bootstrap of a 1-D return pool.

    Resamples *blocks* of ``block_length`` consecutive observations rather
    than individual observations, so short-range serial dependence -- in
    particular volatility clustering -- is preserved inside each block. The
    pool is treated as circular (wrapped) so every observation, including
    those near the end, can start a block with equal probability.

    Parameters
    ----------
    pool : ndarray
        1-D array of historical returns to resample from.
    n_sims, horizon : int
        Number of paths and path length.
    block_length : int
        Number of consecutive observations per block.
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of shape (n_sims, horizon)

    Reference
    ---------
    Politis, D.N., & Romano, J.P. (1992). "A Circular Block-Resampling
    Procedure for Stationary Data." In *Exploring the Limits of the
    Bootstrap*, Wiley, 263-270. (Block bootstrap: Kuensch, H.R. (1989),
    "The Jackknife and the Bootstrap for General Stationary Observations,"
    Annals of Statistics 17(3), 1217-1241.)
    """
    n = len(pool)
    n_blocks = int(np.ceil(horizon / block_length))
    starts = rng.integers(0, n, size=(n_sims, n_blocks))
    offsets = np.arange(block_length)
    # (n_sims, n_blocks, block_length) absolute indices, wrapped modulo n
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    idx = idx.reshape(n_sims, n_blocks * block_length)[:, :horizon]
    return pool[idx]


def simulate_mdd_distribution(engine, sigma=None, horizon=252, n_sims=10_000,
                              mu=0.0, nu=None, residual_pool=None,
                              return_pool=None, block_length=10, seed=42):
    """Monte-Carlo / bootstrap simulate forward MDDs under a chosen engine.

    For each of ``n_sims`` simulations, generate a path of ``horizon`` daily
    returns from the chosen engine, compute the realised MDD of that path,
    and return the array of MDDs across all sims.

    The engines span two families:

    * **Parametric Monte-Carlo** (``gaussian``, ``t``) -- draws are generated
      from an *assumed* distribution whose parameters are estimated from data.
    * **Non-parametric bootstrap** (``iid_bootstrap``, ``block_bootstrap``)
      and the **filtered residual bootstrap** (``fhs``) -- no distributional
      family is assumed; the empirical sample *is* the distribution. This is
      the resampling alternative to parametric Monte-Carlo that the project
      design document specified for the path-dependent (drawdown) layer.

    Parameters
    ----------
    engine : {'gaussian', 't', 'fhs', 'iid_bootstrap', 'block_bootstrap'}
        ``gaussian``       : i.i.d. N(mu, sigma^2) draws.
        ``t``              : i.i.d. mu + sigma_eff * X with X ~ t(nu),
                             sigma_eff = sigma * sqrt((nu-2)/nu) so Var = sigma^2.
        ``fhs``            : i.i.d. bootstrap of standardised residuals from
                             ``residual_pool``, scaled by sigma (filtered
                             historical simulation).
        ``iid_bootstrap``  : i.i.d. resampling of *raw historical returns*
                             from ``return_pool``. No distributional
                             assumption and no volatility scaling -- the
                             empirical marginal is reproduced exactly, but
                             serial dependence is destroyed.
        ``block_bootstrap``: circular block resampling of ``return_pool`` in
                             blocks of ``block_length`` consecutive days, so
                             volatility clustering is preserved. This is the
                             appropriate resampling engine for path-dependent
                             metrics such as MDD and time-under-water.
    sigma : float or None
        Per-day standard deviation in decimal units (e.g., 0.012). Required
        for ``gaussian``, ``t`` and ``fhs``; ignored by the raw-return
        bootstrap engines (their scale comes from ``return_pool`` itself).
    horizon : int, default 252
        Path length in days.
    n_sims : int, default 10000
        Number of simulated paths.
    mu : float, default 0.0
        Per-day mean return (parametric / fhs engines only).
    nu : int or None
        Required if engine='t'. Degrees of freedom (>2).
    residual_pool : array-like or None
        Required if engine='fhs'. Standardised residuals (mean ~0, std ~1).
    return_pool : array-like or None
        Required if engine in {'iid_bootstrap', 'block_bootstrap'}. A pool of
        raw historical returns (decimal units) to resample from.
    block_length : int, default 10
        Block length for engine='block_bootstrap'. A common rule of thumb is
        ~ n^(1/3); for daily data 10-20 days captures volatility clustering.
    seed : int, default 42
        RNG seed for reproducibility.

    Returns
    -------
    ndarray of shape (n_sims,)
        Realised MDD (non-negative) for each simulated path.
    """
    rng = np.random.default_rng(seed)

    if engine in ('gaussian', 't', 'fhs') and sigma is None:
        raise ValueError(f"engine={engine!r} requires sigma.")

    if engine == 'gaussian':
        z = rng.standard_normal((n_sims, horizon))
        rets = mu + sigma * z
    elif engine == 't':
        if nu is None or nu <= 2:
            raise ValueError("engine='t' requires nu > 2.")
        # Standardised t to unit variance, then scale to total sigma.
        sigma_eff = sigma * np.sqrt((nu - 2.0) / nu)
        z = rng.standard_t(df=nu, size=(n_sims, horizon))
        rets = mu + sigma_eff * z
    elif engine == 'fhs':
        if residual_pool is None or len(residual_pool) == 0:
            raise ValueError("engine='fhs' requires a non-empty residual_pool.")
        pool = np.asarray(residual_pool, dtype=float)
        z = rng.choice(pool, size=(n_sims, horizon), replace=True)
        rets = mu + sigma * z
    elif engine == 'iid_bootstrap':
        if return_pool is None or len(return_pool) == 0:
            raise ValueError("engine='iid_bootstrap' requires a non-empty return_pool.")
        pool = np.asarray(return_pool, dtype=float)
        rets = rng.choice(pool, size=(n_sims, horizon), replace=True)
    elif engine == 'block_bootstrap':
        if return_pool is None or len(return_pool) == 0:
            raise ValueError("engine='block_bootstrap' requires a non-empty return_pool.")
        if block_length < 1:
            raise ValueError("block_length must be >= 1.")
        pool = np.asarray(return_pool, dtype=float)
        rets = _circular_block_bootstrap(pool, n_sims, horizon,
                                         int(block_length), rng)
    else:
        raise ValueError(
            f"Unknown engine: {engine!r}. Use 'gaussian', 't', 'fhs', "
            f"'iid_bootstrap', or 'block_bootstrap'."
        )

    # MDD path-by-path (vectorised: cumulative product, running max, min drawdown)
    wealth = np.cumprod(1.0 + rets, axis=1)
    peak = np.maximum.accumulate(wealth, axis=1)
    drawdown = wealth / peak - 1.0
    mdd = -drawdown.min(axis=1)
    return mdd
