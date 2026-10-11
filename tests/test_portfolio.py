"""Unit tests for src/portfolio.py (rebalanced multi-asset construction).

Run with:  python tests/test_portfolio.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import numpy as np
import pandas as pd

from portfolio import rebalance_flags, build_rebalanced_portfolio, sleeve_returns


def _bdays(n, start="2000-01-03"):
    return pd.date_range(start, periods=n, freq="B")


def test_rebalance_flags_cadences():
    idx = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    assert rebalance_flags(idx, "D").all()
    assert rebalance_flags(idx, "M").sum() == 12
    assert rebalance_flags(idx, "Q").sum() == 4
    print("  PASS  test_rebalance_flags_cadences")


def test_single_asset_equals_itself():
    n = 300
    R = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.01, n),
                      "B": np.random.default_rng(1).normal(0, 0.01, n)},
                     index=_bdays(n))
    port = build_rebalanced_portfolio(R, {"A": 1.0, "B": 0.0}, cost=0.0)
    assert np.allclose(port.values, R["A"].values)
    print("  PASS  test_single_asset_equals_itself")


def test_weights_are_renormalised():
    n = 50
    R = pd.DataFrame({"A": np.full(n, 0.01), "B": np.full(n, 0.01)}, index=_bdays(n))
    # unnormalised weights (sum=4) -> 50/50 -> constant 1% return
    port = build_rebalanced_portfolio(R, {"A": 2.0, "B": 2.0}, cost=0.0)
    assert np.allclose(port.values, 0.01)
    print("  PASS  test_weights_are_renormalised")


def test_equal_weight_constant_returns():
    n = 120
    R = pd.DataFrame({"A": np.full(n, 0.005), "B": np.full(n, 0.005)}, index=_bdays(n))
    port = build_rebalanced_portfolio(R, {"A": 0.5, "B": 0.5}, cost=0.0)
    assert np.allclose(port.values, 0.005)
    print("  PASS  test_equal_weight_constant_returns")


def test_weights_drift_then_reset_on_rebalance():
    n = 130
    rng = np.random.default_rng(3)
    R = pd.DataFrame({"A": rng.normal(0.001, 0.02, n),
                      "B": rng.normal(0.0, 0.001, n)}, index=_bdays(n))
    _, detail = build_rebalanced_portfolio(R, {"A": 0.6, "B": 0.4},
                                           cadence="M", cost=0.0, return_detail=True)
    reb = rebalance_flags(R.index, "M")
    # On every rebalance day the held weight is reset to the 0.6 target.
    assert np.allclose(detail["w_A"].values[reb], 0.6)
    # Between rebalances the weight drifts away from 0.6 at least once.
    assert not np.allclose(detail["w_A"].values[~reb], 0.6)
    print("  PASS  test_weights_drift_then_reset_on_rebalance")


def test_costs_reduce_return():
    n = 260
    rng = np.random.default_rng(4)
    R = pd.DataFrame({"A": rng.normal(0.0005, 0.02, n),
                      "B": rng.normal(0.0001, 0.004, n)}, index=_bdays(n))
    free = build_rebalanced_portfolio(R, {"A": 0.6, "B": 0.4}, cost=0.0)
    paid = build_rebalanced_portfolio(R, {"A": 0.6, "B": 0.4}, cost=0.0030)
    assert paid.sum() < free.sum()
    print("  PASS  test_costs_reduce_return")


def test_sleeve_returns_renormalises():
    n = 40
    R = pd.DataFrame({"EQ": np.full(n, 0.02), "GLD": np.full(n, 0.02),
                      "BND": np.full(n, -0.01)}, index=_bdays(n))
    s = sleeve_returns(R, {"EQ": 0.3, "GLD": 0.1})    # both 2% -> sleeve 2%
    assert np.allclose(s, 0.02)
    print("  PASS  test_sleeve_returns_renormalises")


def _all_tests():
    return [
        test_rebalance_flags_cadences,
        test_single_asset_equals_itself,
        test_weights_are_renormalised,
        test_equal_weight_constant_returns,
        test_weights_drift_then_reset_on_rebalance,
        test_costs_reduce_return,
        test_sleeve_returns_renormalises,
    ]


def run_all():
    tests = _all_tests()
    print(f"Running {len(tests)} tests for src/portfolio.py...\n")
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append((t.__name__, e))
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"All {len(tests)} tests passed." if not failed
          else f"{len(failed)} of {len(tests)} tests FAILED.")
    return len(failed)


if __name__ == "__main__":
    sys.exit(0 if run_all() == 0 else 1)
