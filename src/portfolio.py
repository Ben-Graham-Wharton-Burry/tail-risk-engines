"""Multi-asset portfolio construction with periodic rebalancing and costs.

Builds the daily return series of a fixed-weight *policy* portfolio that is
rebalanced on a chosen cadence (default monthly), with weights drifting between
rebalances and a one-way turnover cost charged at each rebalance.

This replaces the original daily-costless rebalancing (design review point 2) with
a realistic cadence, and produces two things the responsive strategy needs:

* the **static FO benchmark** (the policy portfolio itself), and
* the **risky-sleeve** return series (equity + gold + REIT, renormalised) that
  the GARCH-FHS overlay scales toward cash.

Turnover convention: one-way turnover = 0.5 * sum |w_target - w_drifted|, and
the cost charged on a rebalance day is ``cost * turnover`` (so ``cost`` is the
round-trip-consistent one-way rate, e.g. 0.0010 = 10 bps). This matches the
cost convention in ``strategy.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy import month_end_flags        # reuse the month-end detector


def rebalance_flags(index, cadence="M"):
    """Boolean array marking rebalance days for the given cadence ('M','Q','D')."""
    idx = pd.DatetimeIndex(index)
    if cadence == "D":
        return np.ones(len(idx), dtype=bool)
    if cadence == "M":
        return month_end_flags(idx)
    if cadence == "Q":
        per = idx.to_period("Q")
        last = {}
        for i, p in enumerate(per):
            last[p] = i
        flags = np.zeros(len(idx), dtype=bool)
        for i in last.values():
            flags[i] = True
        return flags
    raise ValueError(f"unknown cadence {cadence!r} (use 'M', 'Q' or 'D')")


def build_rebalanced_portfolio(asset_returns, target_weights, cadence="M",
                               cost=0.0010, return_detail=False):
    """Daily net return of a periodically-rebalanced fixed-weight portfolio.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Daily simple returns, one column per asset.
    target_weights : Mapping[str, float]
        Policy weight per column (need not sum to 1; they are renormalised).
    cadence : {'M', 'Q', 'D'}
        Rebalance cadence. Weights drift between rebalances.
    cost : float
        One-way transaction cost per unit turnover (0.0010 = 10 bps).
    return_detail : bool
        If True also return a DataFrame of held weights, gross return and cost.

    Returns
    -------
    pd.Series (port_ret), or (pd.Series, pd.DataFrame) if ``return_detail``.
    """
    cols = list(asset_returns.columns)
    w_target = np.array([float(target_weights[c]) for c in cols], dtype=float)
    if w_target.sum() <= 0:
        raise ValueError("target weights must sum to a positive number")
    w_target = w_target / w_target.sum()

    R = asset_returns.to_numpy(dtype=float)
    n, k = R.shape
    reb = rebalance_flags(asset_returns.index, cadence)

    w = w_target.copy()
    port = np.empty(n)
    gross = np.empty(n)
    costs = np.zeros(n)
    held = np.empty((n, k))

    for t in range(n):
        if reb[t] or t == 0:
            turnover = 0.5 * np.abs(w_target - w).sum()      # one-way
            costs[t] = cost * turnover
            w = w_target.copy()
        held[t] = w
        g = float(w @ R[t])
        gross[t] = g
        port[t] = g - costs[t]
        grown = w * (1.0 + R[t])
        s = grown.sum()
        if s > 0:
            w = grown / s                                    # drift to next day

    port_ret = pd.Series(port, index=asset_returns.index, name="port_ret")
    if not return_detail:
        return port_ret
    detail = pd.DataFrame(held, index=asset_returns.index,
                          columns=[f"w_{c}" for c in cols])
    detail["gross_ret"] = gross
    detail["cost"] = costs
    detail["port_ret"] = port
    return port_ret, detail


def sleeve_returns(asset_returns, sleeve_weights):
    """Renormalised return series of a sub-sleeve (e.g. the risky assets).

    Given policy weights restricted to the sleeve's assets, returns the daily
    return of that sleeve held at its *internal* (renormalised) weights with no
    rebalancing cost -- used as the 'risky book' the overlay scales.
    """
    cols = list(sleeve_weights.keys())
    w = np.array([float(sleeve_weights[c]) for c in cols], dtype=float)
    w = w / w.sum()
    return (asset_returns[cols].to_numpy(dtype=float) @ w)
