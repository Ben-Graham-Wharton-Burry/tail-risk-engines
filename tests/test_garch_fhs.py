"""Unit tests for src/garch_fhs.py.

Run with:

    python tests/test_garch_fhs.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from garch_fhs import (
    fit_garch_window,
    garch_step_var_es,
    rolling_garch_fhs,
)


# ---------------------------------------------------------------------------
# garch_step_var_es
# ---------------------------------------------------------------------------

def test_step_var_es_gaussian_residuals():
    """For standard-normal residuals and sigma=1, VaR99 ~ 2.33, ES99 ~ 2.67."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(50_000)
    var, es = garch_step_var_es(sigma_next=1.0, residual_pool=z, alpha=0.99)
    assert abs(var - 2.326) < 0.05, f"VaR={var}"
    assert abs(es - 2.665) < 0.05, f"ES={es}"
    assert es > var
    print("  PASS  test_step_var_es_gaussian_residuals")


def test_step_var_es_scales_with_sigma():
    """Both VaR and ES are linear in sigma_next."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal(20_000)
    v1, e1 = garch_step_var_es(0.01, z, alpha=0.99)
    v2, e2 = garch_step_var_es(0.05, z, alpha=0.99)
    assert abs(v2 - 5 * v1) < 1e-12
    assert abs(e2 - 5 * e1) < 1e-12
    print("  PASS  test_step_var_es_scales_with_sigma")


def test_step_var_es_heavier_tails_give_larger_es():
    """Replacing Gaussian residuals with heavy-tailed (t_4) ones increases ES."""
    rng = np.random.default_rng(2)
    z_norm = rng.standard_normal(50_000)
    # Standardise t(4) sample to unit variance for fair comparison
    from scipy import stats
    z_t = stats.t.rvs(df=4, size=50_000, random_state=2)
    z_t = z_t * np.sqrt(2 / 4)  # var of t(4) = 4/(4-2) = 2; scale to unit var
    _, es_norm = garch_step_var_es(1.0, z_norm, alpha=0.99)
    _, es_t = garch_step_var_es(1.0, z_t, alpha=0.99)
    assert es_t > es_norm, f"es_t={es_t} should exceed es_norm={es_norm}"
    print("  PASS  test_step_var_es_heavier_tails_give_larger_es")


def test_step_var_es_empty_pool_returns_nan():
    var, es = garch_step_var_es(0.01, [], alpha=0.99)
    assert np.isnan(var) and np.isnan(es)
    print("  PASS  test_step_var_es_empty_pool_returns_nan")


# ---------------------------------------------------------------------------
# fit_garch_window
# ---------------------------------------------------------------------------

def test_fit_garch_window_recovers_persistence():
    """On simulated GARCH(1,1) data, the fit should recover near-truth params."""
    # Simulate GARCH(1,1) with omega=0.02, alpha=0.08, beta=0.90 (so persistence 0.98)
    rng = np.random.default_rng(3)
    n = 5000
    omega, a, b = 0.02, 0.08, 0.90
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = omega / (1 - a - b)
    for t in range(n):
        z = rng.standard_normal()
        r[t] = np.sqrt(sigma2[t]) * z
        if t + 1 < n:
            sigma2[t + 1] = omega + a * r[t] ** 2 + b * sigma2[t]
    # Convert to "decimal returns" by dividing by 100, then fit (the fit
    # internally re-scales by 100, so we'll recover params close to the truth)
    s = pd.Series(r / 100.0, index=pd.date_range('2010-01-01', periods=n, freq='B'))
    res = fit_garch_window(s)
    a_hat = res.params['alpha[1]']
    b_hat = res.params['beta[1]']
    # Allow generous tolerances given finite-sample noise
    assert abs(a_hat - a) < 0.03, f"alpha_hat={a_hat}, true={a}"
    assert abs(b_hat - b) < 0.03, f"beta_hat={b_hat}, true={b}"
    print(f"  PASS  test_fit_garch_window_recovers_persistence  "
          f"(a_hat={a_hat:.3f}, b_hat={b_hat:.3f})")


def test_fit_garch_window_std_resid_unit_variance():
    """The standardised residuals from a GARCH fit should have ~unit variance."""
    rng = np.random.default_rng(4)
    n = 2000
    s = pd.Series(rng.standard_normal(n) * 0.01,
                  index=pd.date_range('2010-01-01', periods=n, freq='B'))
    res = fit_garch_window(s)
    z = res.std_resid.values
    assert abs(z.mean()) < 0.1
    assert abs(z.std() - 1.0) < 0.1, f"std(z) = {z.std()}"
    print("  PASS  test_fit_garch_window_std_resid_unit_variance")


# ---------------------------------------------------------------------------
# rolling_garch_fhs
# ---------------------------------------------------------------------------

def test_rolling_garch_fhs_shape_and_index():
    """Output has correct shape and the index matches the forecast convention."""
    rng = np.random.default_rng(5)
    n, w = 1500, 1000
    s = pd.Series(rng.standard_normal(n) * 0.01,
                  index=pd.date_range('2010-01-01', periods=n, freq='B'))
    fc = rolling_garch_fhs(s, window=w, alpha=0.99, refit_freq=100)
    # Same index convention as Phase 1: first forecast at index W
    assert fc.index[0] == s.index[w], f"first index = {fc.index[0]}, expected {s.index[w]}"
    assert list(fc.columns) == ['VaR', 'ES', 'sigma']
    assert len(fc) == n - w
    # No NaN values
    assert not fc.isna().any().any()
    print("  PASS  test_rolling_garch_fhs_shape_and_index")


def test_rolling_garch_fhs_var_positive():
    """At 99% confidence, VaR forecasts should be strictly positive (losses)."""
    rng = np.random.default_rng(6)
    n, w = 1300, 1000
    s = pd.Series(rng.standard_normal(n) * 0.01,
                  index=pd.date_range('2010-01-01', periods=n, freq='B'))
    fc = rolling_garch_fhs(s, window=w, alpha=0.99, refit_freq=50)
    assert (fc['VaR'] > 0).all()
    assert (fc['ES'] >= fc['VaR']).all()
    print("  PASS  test_rolling_garch_fhs_var_positive")


def test_rolling_garch_fhs_responds_to_volatility_change():
    """sigma forecast should rise after a volatility shock."""
    rng = np.random.default_rng(7)
    n_quiet = 1100
    n_loud = 200
    quiet = rng.standard_normal(n_quiet) * 0.005   # 0.5% vol
    loud = rng.standard_normal(n_loud) * 0.03      # 3.0% vol
    s = pd.Series(np.concatenate([quiet, loud]),
                  index=pd.date_range('2010-01-01', periods=n_quiet + n_loud, freq='B'))
    fc = rolling_garch_fhs(s, window=1000, alpha=0.99, refit_freq=20)
    # sigma forecast for the FIRST forecast date (within quiet regime)
    sigma_first = fc['sigma'].iloc[0]
    # sigma forecast at the END (after the loud regime is well-established)
    sigma_last = fc['sigma'].iloc[-1]
    # Allow generous bounds; sigma_last should be substantially higher.
    assert sigma_last > 2 * sigma_first, (
        f"sigma_first={sigma_first:.4f}, sigma_last={sigma_last:.4f}"
    )
    print(f"  PASS  test_rolling_garch_fhs_responds_to_volatility_change  "
          f"(sigma_first={sigma_first*100:.3f}%, sigma_last={sigma_last*100:.3f}%)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_step_var_es_gaussian_residuals,
        test_step_var_es_scales_with_sigma,
        test_step_var_es_heavier_tails_give_larger_es,
        test_step_var_es_empty_pool_returns_nan,
        test_fit_garch_window_recovers_persistence,
        test_fit_garch_window_std_resid_unit_variance,
        test_rolling_garch_fhs_shape_and_index,
        test_rolling_garch_fhs_var_positive,
        test_rolling_garch_fhs_responds_to_volatility_change,
    ]
    print(f"Running {len(tests)} tests for src/garch_fhs.py...\n")
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            failed.append((t.__name__, e))
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print()
    if not failed:
        print(f"All {len(tests)} tests passed.")
    else:
        print(f"{len(failed)} of {len(tests)} tests FAILED.")
    return len(failed)


if __name__ == '__main__':
    sys.exit(0 if run_all() == 0 else 1)
