"""Responsive risk-targeting overlay for the family-office strategy (NB 11).

Turns the GARCH-FHS forecasts into a risk-controlled allocation: hold the risky
book while forecast risk is within the office's limits, and de-risk toward cash
when a limit would be breached. De-risk only: never lever.

Two engines live here:

* ``run_risk_targeted_strategy`` -- the original monthly multi-limit overlay
  (vol + VaR + ES caps, drawdown ladder, optional DCC correlation trigger). Kept
  for backward compatibility with the earlier notebooks.

* ``run_vol_target_strategy`` -- the revised NB11 design: a single volatility
  *instrument* with asymmetric (cut-fast / add-slow) rebalancing. VaR and
  max-drawdown are treated as *objectives* (calibrated via a sensitivity
  frontier), not as binding caps. This is the engine NB11 now uses.

No look-ahead: the weight on day t uses the one-day-ahead forecast formed at t-1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Soft-deleverage ladder defending the max-drawdown limit: (drawdown, multiplier).
# Per documented FO practice (cut exposure as the drawdown deepens). Tuned to a
# ~20% MDD ceiling: trim to 0.75 at 10% DD, 0.50 at 15%, 0.25 at 20%.
DEFAULT_DD_LADDER = [(0.10, 0.75), (0.15, 0.50), (0.20, 0.25)]

# Correlation-breakdown ladder: (equity-bond rho threshold, exposure multiplier).
# Equity-bond correlation is normally negative; positive values signal the bond
# cushion failing. Grounded in the DCC rho_t distribution (rho>0.25 ~ top 19%
# of days; rho>0.40 ~ top 8.5%, around the 2022 peak).
DEFAULT_CORR_LADDER = [(0.25, 0.85), (0.40, 0.70)]


def annual_vol_to_daily(sigma_annual, trading_days=TRADING_DAYS):
    """Convert an annualised volatility to a daily volatility."""
    return sigma_annual / np.sqrt(trading_days)


def _cap(target, forecast, w_max):
    """Exposure cap target/forecast, with NaN/inf forecasts mapped to w_max."""
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.asarray(target, float) / np.asarray(forecast, float)
    return np.where(np.isfinite(c), c, w_max)


def vol_target_weight(sigma_forecast, sigma_target_daily, w_max=1.0, w_min=0.0):
    """Exposure weight that targets a daily volatility (de-risk only by default)."""
    return np.clip(_cap(sigma_target_daily, sigma_forecast, w_max), w_min, w_max)


def drawdown_multiplier(current_dd, ladder=DEFAULT_DD_LADDER):
    """Exposure multiplier for a non-negative running drawdown via the ladder."""
    mult = 1.0
    for thresh, m in sorted(ladder):
        if current_dd >= thresh:
            mult = m
    return mult


def correlation_multiplier(rho, ladder=DEFAULT_CORR_LADDER):
    """Exposure multiplier for the equity-bond correlation regime via the ladder."""
    if rho is None or not np.isfinite(rho):
        return 1.0
    mult = 1.0
    for thresh, m in sorted(ladder):
        if rho >= thresh:
            mult = m
    return mult


def month_end_flags(index):
    """Boolean array: True on the last available trading day of each month."""
    idx = pd.DatetimeIndex(index)
    periods = idx.to_period("M")
    last_pos = {}
    for i, p in enumerate(periods):
        last_pos[p] = i
    flags = np.zeros(len(idx), dtype=bool)
    for i in last_pos.values():
        flags[i] = True
    return flags


def run_risk_targeted_strategy(risky_ret, safe_ret, sigma_forecast,
                               sigma_target_annual=0.10,
                               var_forecast=None, var_limit=None,
                               es_forecast=None, es_limit=None,
                               cost=0.0010,
                               dd_ladder=DEFAULT_DD_LADDER,
                               corr_signal=None,
                               corr_ladder=DEFAULT_CORR_LADDER,
                               w_max=1.0, w_min=0.0):
    """Backtest the monthly multi-limit risk-targeted overlay (legacy engine).

    Kept for backward compatibility with earlier notebooks. NB11 now uses
    ``run_vol_target_strategy`` instead.

    Parameters
    ----------
    risky_ret, safe_ret : pd.Series
        Daily simple returns of the risky book and the safe asset (cash).
    sigma_forecast : pd.Series
        One-day-ahead conditional volatility forecast (GARCH-FHS 'sigma'), daily.
    sigma_target_annual : float
        Annualised volatility budget.
    var_forecast, es_forecast : pd.Series or None
        One-day-ahead 99% VaR / ES forecasts (GARCH-FHS), as positive loss
        fractions. If given with the matching limit, they add a cap.
    var_limit, es_limit : float or None
        VaR / ES limits as positive loss fractions of capital (e.g. 0.025).
    cost : float
        One-way transaction cost per unit turnover (0.0010 = 10 bps).
    dd_ladder : list[(dd, mult)] or None
        Soft-deleverage ladder defending the MDD limit; None disables it.
    corr_signal : pd.Series or None
        DCC equity-bond correlation rho_t; enables the regime-aware variant.
    corr_ladder : list[(rho, mult)]
        Correlation-breakdown ladder used when corr_signal is given.
    w_max, w_min : float
        Exposure bounds (w_max = 1 -> never lever).

    Returns
    -------
    pd.DataFrame indexed by date with: cap_vol, cap_var, cap_es, corr_mult,
        weight, risky_ret, safe_ret, gross_ret, cost, net_ret, wealth, drawdown.
    """
    parts = {"risky": risky_ret, "safe": safe_ret, "sigma": sigma_forecast}
    if var_forecast is not None and var_limit is not None:
        parts["var"] = var_forecast
    if es_forecast is not None and es_limit is not None:
        parts["es"] = es_forecast
    df = pd.concat(parts, axis=1).dropna()
    idx = df.index
    n = len(df)
    rr = df["risky"].to_numpy()
    rs = df["safe"].to_numpy()

    sig_target_daily = annual_vol_to_daily(sigma_target_annual)
    cap_vol = _cap(sig_target_daily, df["sigma"].to_numpy(), w_max)
    cap_var = (_cap(var_limit, df["var"].to_numpy(), w_max)
               if "var" in df else np.full(n, np.inf))
    cap_es = (_cap(es_limit, df["es"].to_numpy(), w_max)
              if "es" in df else np.full(n, np.inf))
    cap_min = np.minimum.reduce([cap_vol, cap_var, cap_es])      # most binding

    reb = month_end_flags(idx)
    if corr_signal is not None:
        ca = pd.Series(corr_signal).reindex(idx).ffill().to_numpy()
        corr_mult_arr = np.array([correlation_multiplier(c, corr_ladder) for c in ca])
    else:
        corr_mult_arr = np.ones(n)

    weight = np.empty(n)
    gross = np.empty(n)
    costs = np.zeros(n)
    wealth = np.empty(n)
    ddown = np.empty(n)

    w_held = 0.0
    W = 1.0
    peak = 1.0
    for t in range(n):
        cur_dd = max(0.0, 1.0 - W / peak)
        if reb[t] or t == 0:
            w_star = cap_min[t]
            if dd_ladder is not None:
                w_star *= drawdown_multiplier(cur_dd, dd_ladder)
            w_star *= corr_mult_arr[t]
            w_star = float(np.clip(w_star, w_min, w_max))
            costs[t] = cost * abs(w_star - w_held)
            w_held = w_star
        weight[t] = w_held
        g = w_held * rr[t] + (1.0 - w_held) * rs[t]
        gross[t] = g
        net = g - costs[t]
        W *= (1.0 + net)
        wealth[t] = W
        peak = max(peak, W)
        ddown[t] = max(0.0, 1.0 - W / peak)
        denom = w_held * (1.0 + rr[t]) + (1.0 - w_held) * (1.0 + rs[t])
        if denom > 0:
            w_held = w_held * (1.0 + rr[t]) / denom

    return pd.DataFrame(
        {
            "cap_vol": np.clip(cap_vol, w_min, w_max),
            "cap_var": np.clip(cap_var, w_min, w_max),
            "cap_es": np.clip(cap_es, w_min, w_max),
            "corr_mult": corr_mult_arr,
            "weight": weight,
            "risky_ret": rr,
            "safe_ret": rs,
            "gross_ret": gross,
            "cost": costs,
            "net_ret": gross - costs,
            "wealth": wealth,
            "drawdown": ddown,
        },
        index=idx,
    )


def run_vol_target_strategy(risky_ret, safe_ret, sigma_forecast,
                            sigma_target_annual=0.10, cost=0.0010,
                            w_max=1.0, w_min=0.0, deadband=0.10):
    """Responsive vol-target overlay with asymmetric (cut-fast / add-slow) rebalancing.

    A single risk *instrument* drives the allocation: the GARCH-FHS one-day-ahead
    conditional volatility. The target risky-book exposure is the vol-budget cap

        w* = sigma_target_daily / sigma_forecast,   clipped to [w_min, w_max]

    holding the fixed policy portfolio (the risky book) while forecast vol is
    inside budget and de-risking toward cash (the safe leg) when it is not.
    Exposure is never levered (w_max = 1).

    Rebalancing is deliberately ASYMMETRIC, because volatility spikes are sudden
    and clustered while calm returns gradually:

    * De-risk (cut): checked EVERY DAY. If today's target cap falls below the
      currently held weight by more than ``deadband`` (in relative terms), the
      book is cut immediately to the cap -- no waiting for month-end.
    * Re-risk (add): only at the scheduled month-end, when the weight is reset
      to that day's cap. This is the sole channel for adding exposure back, so
      there is no fixed holding period in cash: the book stays defensive until
      the next month-end and then re-risks only to the extent the forecast has
      actually normalised. The cut-fast / add-slow asymmetry prevents whipsaw
      on transient calm.

    The VaR and maximum-drawdown limits are NOT binding constraints here. Under
    FHS they are (slow-moving) multiples of sigma, so adding them as caps would
    largely duplicate the vol budget. Instead they are treated as *objectives*
    used to calibrate ``sigma_target_annual`` via the sensitivity frontier in
    notebook 11, and reported as realised outcomes.

    No look-ahead: the weight on day t uses sigma_t (the one-day-ahead forecast
    formed at t-1); returns on day t are then applied to the held weight.

    Parameters
    ----------
    risky_ret, safe_ret : pd.Series
        Daily simple returns of the risky book and the safe asset (cash).
    sigma_forecast : pd.Series
        One-day-ahead conditional volatility forecast (GARCH-FHS 'sigma'), daily.
    sigma_target_annual : float
        Annualised volatility budget (the instrument that is calibrated).
    cost : float
        One-way transaction cost per unit turnover (0.0010 = 10 bps).
    w_max, w_min : float
        Exposure bounds (w_max = 1 -> never lever).
    deadband : float
        Relative tolerance for the intra-month de-risk trigger. A cut fires only
        when the target cap is below the held weight by more than this fraction
        (0.10 = 10%), avoiding churn on negligible breaches.

    Returns
    -------
    pd.DataFrame indexed by date with: cap_vol, weight, traded, risky_ret,
        safe_ret, gross_ret, cost, net_ret, wealth, drawdown.
    """
    df = pd.concat({"risky": risky_ret, "safe": safe_ret,
                    "sigma": sigma_forecast}, axis=1).dropna()
    idx = df.index
    n = len(df)
    rr = df["risky"].to_numpy()
    rs = df["safe"].to_numpy()

    sig_target_daily = annual_vol_to_daily(sigma_target_annual)
    cap_vol = np.clip(_cap(sig_target_daily, df["sigma"].to_numpy(), w_max),
                      w_min, w_max)
    reb = month_end_flags(idx)

    weight = np.empty(n)
    traded = np.zeros(n, dtype=bool)
    gross = np.empty(n)
    costs = np.zeros(n)
    wealth = np.empty(n)
    ddown = np.empty(n)

    w_held = 0.0
    W = 1.0
    peak = 1.0
    for t in range(n):
        target = cap_vol[t]
        # Decide whether to trade today.
        if t == 0 or reb[t]:
            do_trade = True                          # scheduled reset (up or down)
        elif target < w_held * (1.0 - deadband):
            do_trade = True                          # intra-month breach -> cut only
        else:
            do_trade = False
        if do_trade:
            costs[t] = cost * abs(target - w_held)
            w_held = float(target)
            traded[t] = True

        weight[t] = w_held
        g = w_held * rr[t] + (1.0 - w_held) * rs[t]
        gross[t] = g
        net = g - costs[t]
        W *= (1.0 + net)
        wealth[t] = W
        peak = max(peak, W)
        ddown[t] = max(0.0, 1.0 - W / peak)
        # Passive drift of the held weight between trades.
        denom = w_held * (1.0 + rr[t]) + (1.0 - w_held) * (1.0 + rs[t])
        if denom > 0:
            w_held = w_held * (1.0 + rr[t]) / denom

    return pd.DataFrame(
        {
            "cap_vol": cap_vol,
            "weight": weight,
            "traded": traded,
            "risky_ret": rr,
            "safe_ret": rs,
            "gross_ret": gross,
            "cost": costs,
            "net_ret": gross - costs,
            "wealth": wealth,
            "drawdown": ddown,
        },
        index=idx,
    )


def performance_summary(net_ret, rf_daily=0.00008979, trading_days=TRADING_DAYS):
    """CAGR, annualised vol, Sharpe, Sortino and max drawdown for a return series."""
    r = pd.Series(net_ret).dropna()
    n = len(r)
    if n == 0:
        return {k: float("nan") for k in
                ("CAGR", "ann_vol", "sharpe", "sortino", "max_drawdown", "n")}
    cagr = (1.0 + r).prod() ** (trading_days / n) - 1.0
    sd = r.std(ddof=1)
    vol = sd * np.sqrt(trading_days)
    excess = r.mean() - rf_daily
    sharpe = excess / sd * np.sqrt(trading_days) if sd > 0 else float("nan")
    downside = np.minimum(r.to_numpy() - rf_daily, 0.0)
    dd_dev = np.sqrt(np.mean(downside ** 2))
    sortino = excess / dd_dev * np.sqrt(trading_days) if dd_dev > 0 else float("nan")
    w = (1.0 + r).cumprod()
    mdd = float(-(w / w.cummax() - 1.0).min())
    return {"CAGR": float(cagr), "ann_vol": float(vol), "sharpe": float(sharpe),
            "sortino": float(sortino), "max_drawdown": mdd, "n": int(n)}
