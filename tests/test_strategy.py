"""Unit tests for src/strategy.py (responsive multi-limit risk-targeting overlay).

Run with:

    python tests/test_strategy.py

Covers the exposure-scaling rule, the drawdown and correlation ladders,
month-end detection, the no-leverage cap, the multi-limit (vol / VaR / ES)
min-cap logic, that de-risking lowers realised volatility, that costs reduce
returns, and that weights contain no look-ahead. Bare asserts, no framework.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import numpy as np
import pandas as pd

from strategy import (
    annual_vol_to_daily,
    vol_target_weight,
    drawdown_multiplier,
    correlation_multiplier,
    month_end_flags,
    run_risk_targeted_strategy,
    performance_summary,
    DEFAULT_DD_LADDER,
    DEFAULT_CORR_LADDER,
)


def _bdays(n, start="2000-01-03"):
    return pd.date_range(start, periods=n, freq="B")


# --- exposure-scaling rule ------------------------------------------------

def test_annual_to_daily():
    assert abs(annual_vol_to_daily(0.10) - 0.10 / np.sqrt(252)) < 1e-15
    print("  PASS  test_annual_to_daily")


def test_vol_target_exact_in_range():
    assert abs(vol_target_weight(np.array([0.02]), 0.01, w_max=1.0)[0] - 0.5) < 1e-12
    print("  PASS  test_vol_target_exact_in_range")


def test_vol_target_no_leverage_cap():
    assert vol_target_weight(np.array([0.001]), 0.01, w_max=1.0)[0] == 1.0
    print("  PASS  test_vol_target_no_leverage_cap")


def test_vol_target_monotone_decreasing():
    w = vol_target_weight(np.array([0.005, 0.01, 0.02, 0.04]), 0.01, w_max=1.0)
    assert w[0] >= w[1] >= w[2] >= w[3] and w[3] < w[0]
    print("  PASS  test_vol_target_monotone_decreasing")


def test_vol_target_nan_maps_to_wmax():
    w = vol_target_weight(np.array([np.nan, 0.0]), 0.01, w_max=1.0)
    assert w[0] == 1.0 and w[1] == 1.0
    print("  PASS  test_vol_target_nan_maps_to_wmax")


# --- ladders --------------------------------------------------------------

def test_drawdown_ladder_steps():
    assert (drawdown_multiplier(0.0) == 1.0 and drawdown_multiplier(0.10) == 0.75
            and drawdown_multiplier(0.16) == 0.50 and drawdown_multiplier(0.25) == 0.25)
    print("  PASS  test_drawdown_ladder_steps")


def test_correlation_ladder_steps():
    assert (correlation_multiplier(-0.30) == 1.0 and correlation_multiplier(0.25) == 0.85
            and correlation_multiplier(0.40) == 0.70 and correlation_multiplier(float("nan")) == 1.0)
    print("  PASS  test_correlation_ladder_steps")


def test_month_end_flags_count_and_position():
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    flags = month_end_flags(idx)
    assert flags.sum() == 12
    for d in idx[flags]:
        m = idx[(idx.year == d.year) & (idx.month == d.month)]
        assert d == m[-1]
    print("  PASS  test_month_end_flags_count_and_position")


# --- strategy backtest ----------------------------------------------------

def _synthetic(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    sigma = np.full(n, 0.008); sigma[n // 3: n // 3 + 120] = 0.030
    risky = pd.Series(rng.normal(0.0003, 1.0, n) * sigma, index=_bdays(n))
    safe = pd.Series(np.full(n, 0.00008), index=risky.index)
    return risky, safe, pd.Series(sigma, index=risky.index)


def test_weights_never_lever():
    risky, safe, sig = _synthetic()
    res = run_risk_targeted_strategy(risky, safe, sig, sigma_target_annual=0.10)
    assert res["weight"].max() <= 1.0 + 1e-9 and res["weight"].min() >= -1e-9
    print("  PASS  test_weights_never_lever")


def test_derisking_lowers_realised_vol():
    risky, safe, sig = _synthetic()
    res = run_risk_targeted_strategy(risky, safe, sig, 0.10, cost=0.0, dd_ladder=None)
    assert performance_summary(res["net_ret"])["ann_vol"] < performance_summary(risky)["ann_vol"]
    print("  PASS  test_derisking_lowers_realised_vol")


def test_costs_reduce_return():
    risky, safe, sig = _synthetic()
    free = run_risk_targeted_strategy(risky, safe, sig, cost=0.0)
    paid = run_risk_targeted_strategy(risky, safe, sig, cost=0.0020)
    assert paid["net_ret"].sum() < free["net_ret"].sum() and paid["cost"].sum() > 0
    print("  PASS  test_costs_reduce_return")


def test_no_lookahead_in_weights():
    risky, safe, sig = _synthetic(seed=1)
    base = run_risk_targeted_strategy(risky, safe, sig, dd_ladder=DEFAULT_DD_LADDER)
    k = len(risky) - 50
    bumped = risky.copy(); bumped.iloc[k] = -0.20
    pert = run_risk_targeted_strategy(bumped, safe, sig, dd_ladder=DEFAULT_DD_LADDER)
    assert np.allclose(base["weight"].values[: k + 1], pert["weight"].values[: k + 1])
    print("  PASS  test_no_lookahead_in_weights")


def test_wealth_matches_net_returns():
    risky, safe, sig = _synthetic()
    res = run_risk_targeted_strategy(risky, safe, sig)
    assert np.allclose((1.0 + res["net_ret"]).cumprod().values, res["wealth"].values)
    print("  PASS  test_wealth_matches_net_returns")


def test_high_target_stays_fully_invested():
    risky, safe, sig = _synthetic()
    res = run_risk_targeted_strategy(risky, safe, sig, sigma_target_annual=5.0, cost=0.0, dd_ladder=None)
    assert np.allclose(res["weight"].values, 1.0)
    print("  PASS  test_high_target_stays_fully_invested")


# --- regime-correlation feature ------------------------------------------

def test_corr_none_reproduces_baseline():
    risky, safe, sig = _synthetic()
    a = run_risk_targeted_strategy(risky, safe, sig)
    b = run_risk_targeted_strategy(risky, safe, sig, corr_signal=None)
    assert np.allclose(a["weight"].values, b["weight"].values) and np.allclose(a["corr_mult"].values, 1.0)
    print("  PASS  test_corr_none_reproduces_baseline")


def test_regime_aware_derisks_when_corr_high():
    risky, safe, sig = _synthetic(seed=2)
    n = len(risky)
    rho = pd.Series(np.full(n, -0.20), index=risky.index); rho.iloc[n // 3: n // 3 + 150] = 0.50
    base = run_risk_targeted_strategy(risky, safe, sig, cost=0.0, dd_ladder=None, corr_signal=None)
    regime = run_risk_targeted_strategy(risky, safe, sig, cost=0.0, dd_ladder=None, corr_signal=rho)
    assert regime["weight"].mean() < base["weight"].mean()
    print("  PASS  test_regime_aware_derisks_when_corr_high")


# --- multi-limit (vol / VaR / ES) min-cap logic --------------------------

def test_var_cap_binds_below_vol():
    n = 200; idx = _bdays(n)
    sigma = pd.Series(0.005, index=idx)           # vol cap clips to 1
    var_fc = pd.Series(0.03, index=idx)           # VaR cap = 0.015/0.03 = 0.5
    z = pd.Series(0.0, index=idx)
    volonly = run_risk_targeted_strategy(z, z, sigma, 0.10, dd_ladder=None, cost=0.0)
    multi = run_risk_targeted_strategy(z, z, sigma, 0.10, var_forecast=var_fc, var_limit=0.015,
                                       dd_ladder=None, cost=0.0)
    assert (multi["weight"].values <= volonly["weight"].values + 1e-9).all()
    assert abs(multi["cap_var"].iloc[-1] - 0.5) < 1e-9
    assert multi["weight"].iloc[-1] < volonly["weight"].iloc[-1]
    print("  PASS  test_var_cap_binds_below_vol")


def test_caps_take_minimum():
    n = 60; idx = _bdays(n)
    sigma = pd.Series(0.006, index=idx)           # vol cap ~1.05 -> clip 1.0
    var_fc = pd.Series(0.02, index=idx)           # VaR cap = 0.015/0.02 = 0.75
    es_fc = pd.Series(0.03, index=idx)            # ES cap  = 0.018/0.03 = 0.60 (binds)
    z = pd.Series(0.0, index=idx)
    m = run_risk_targeted_strategy(z, z, sigma, 0.10, var_forecast=var_fc, var_limit=0.015,
                                   es_forecast=es_fc, es_limit=0.018, dd_ladder=None, cost=0.0)
    assert abs(m["weight"].iloc[-1] - 0.60) < 1e-9       # min(1.0, 0.75, 0.60)
    print("  PASS  test_caps_take_minimum")


def _all_tests():
    return [
        test_annual_to_daily, test_vol_target_exact_in_range, test_vol_target_no_leverage_cap,
        test_vol_target_monotone_decreasing, test_vol_target_nan_maps_to_wmax,
        test_drawdown_ladder_steps, test_correlation_ladder_steps,
        test_month_end_flags_count_and_position,
        test_weights_never_lever, test_derisking_lowers_realised_vol, test_costs_reduce_return,
        test_no_lookahead_in_weights, test_wealth_matches_net_returns, test_high_target_stays_fully_invested,
        test_corr_none_reproduces_baseline, test_regime_aware_derisks_when_corr_high,
        test_var_cap_binds_below_vol, test_caps_take_minimum,
    ]


def run_all():
    tests = _all_tests()
    print(f"Running {len(tests)} tests for src/strategy.py...\n")
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append((t.__name__, e)); print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"All {len(tests)} tests passed." if not failed else f"{len(failed)} of {len(tests)} FAILED.")
    return len(failed)


if __name__ == "__main__":
    sys.exit(0 if run_all() == 0 else 1)
