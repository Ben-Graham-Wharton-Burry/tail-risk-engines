"""Unit tests for src/forecast_eval.py (forecast-to-risk-metric regression).

Run with:  python tests/test_forecast_eval.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import numpy as np
import pandas as pd

from forecast_eval import (
    realized_vol_forward,
    mincer_zarnowitz,
    regress_metric_on_forecast,
)


def test_realized_vol_forward_shape():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0, 0.01, 1000))
    fv = realized_vol_forward(s, horizon=21)
    assert fv.isna().sum() == 21               # last horizon dates have no window
    assert 0.10 < fv.mean() < 0.25             # ~ 0.01*sqrt(252) annualised
    print("  PASS  test_realized_vol_forward_shape")


def test_mz_exact_identity():
    # Regressing y on itself: beta = 1, alpha = 0, R2 = 1 exactly. Residuals are
    # zero so the HAC SE -> 0 and the p-value is undefined; that is expected,
    # hence the p-value is exercised in the near-unbiased test below instead.
    y = pd.Series(np.abs(np.random.default_rng(1).normal(0.15, 0.05, 2000)))
    mz = mincer_zarnowitz(y, y)
    assert abs(mz["beta"] - 1.0) < 1e-6 and abs(mz["alpha"]) < 1e-8
    assert mz["R2"] > 0.999
    print("  PASS  test_mz_exact_identity")


def test_mz_unbiased_forecast_not_rejected():
    # y = f + small noise -> an unbiased forecast: beta ~ 1, alpha ~ 0, and the
    # unbiasedness restriction (beta = 1) is NOT rejected.
    rng = np.random.default_rng(1)
    f = pd.Series(np.abs(rng.normal(0.15, 0.05, 4000)))
    y = f + rng.normal(0, 0.002, 4000)
    mz = mincer_zarnowitz(y, f)
    assert abs(mz["beta"] - 1.0) < 0.05 and abs(mz["alpha"]) < 0.02
    assert mz["R2"] > 0.9 and mz["p_beta_eq1"] > 0.05
    print("  PASS  test_mz_unbiased_forecast_not_rejected")


def test_mz_recovers_known_coefficients():
    rng = np.random.default_rng(2)
    f = pd.Series(np.abs(rng.normal(0.12, 0.04, 5000)))
    y = 0.5 + 1.2 * f + rng.normal(0, 1e-4, 5000)
    mz = mincer_zarnowitz(y, f)
    assert abs(mz["beta"] - 1.2) < 0.01 and abs(mz["alpha"] - 0.5) < 0.01
    assert mz["R2"] > 0.99
    print("  PASS  test_mz_recovers_known_coefficients")


def test_mz_se_positive():
    rng = np.random.default_rng(3)
    f = pd.Series(np.abs(rng.normal(0.1, 0.03, 1500)))
    y = 0.02 + 0.8 * f + rng.normal(0, 0.02, 1500)
    mz = mincer_zarnowitz(y, f)
    assert mz["se_alpha"] > 0 and mz["se_beta"] > 0
    print("  PASS  test_mz_se_positive")


def test_regress_metric_detects_information():
    rng = np.random.default_rng(4)
    f = pd.Series(np.abs(rng.normal(0.1, 0.03, 3000)))
    y = 3.0 * f                                # perfectly informative
    r = regress_metric_on_forecast(y, f)
    assert abs(r["beta"] - 3.0) < 1e-6 and r["R2"] > 0.999
    assert r["p_beta_eq0"] < 1e-6              # forecast clearly carries information
    print("  PASS  test_regress_metric_detects_information")


def _all_tests():
    return [
        test_realized_vol_forward_shape,
        test_mz_exact_identity,
        test_mz_unbiased_forecast_not_rejected,
        test_mz_recovers_known_coefficients,
        test_mz_se_positive,
        test_regress_metric_detects_information,
    ]


def run_all():
    tests = _all_tests()
    print(f"Running {len(tests)} tests for src/forecast_eval.py...\n")
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
