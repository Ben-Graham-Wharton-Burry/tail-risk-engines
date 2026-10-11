"""Unit tests for src/extra_metrics.py - Phase 3 CED, downside risk, LASSO."""
import os, sys, traceback
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
from extra_metrics import (conditional_expected_drawdown, semivariance,
                           downside_deviation, sortino_ratio,
                           lasso_exceedance_probabilities)


def test_ced_is_tail_mean_above_quantile():
    mdd = np.linspace(0.05, 0.40, 1000)
    ced = conditional_expected_drawdown(mdd, alpha=0.95)
    q95 = float(np.quantile(mdd, 0.95))
    assert q95 < ced <= mdd.max(), (q95, ced)


def test_ced_handles_empty():
    assert np.isnan(conditional_expected_drawdown([]))


def test_semivariance_nonnegative_and_zero_when_no_downside():
    assert semivariance([0.01, 0.02, 0.03], target=0.0) == 0.0
    assert semivariance([-0.01, 0.02, -0.03], target=0.0) > 0.0


def test_downside_deviation_below_total_std_for_symmetric():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, 20000)
    dd = downside_deviation(r, target=float(r.mean()))
    assert dd < r.std(), (dd, r.std())


def test_downside_deviation_increases_with_target():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 10000)
    assert downside_deviation(r, target=0.02) > downside_deviation(r, target=0.0)


def test_sortino_decreases_as_target_rises():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, 10000)
    assert sortino_ratio(r, target=0.0) > sortino_ratio(r, target=0.004)


def test_lasso_probabilities_bounds_and_length():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (400, 4))
    y = (X[:, 0] + 0.3 * rng.normal(0, 1, 400) > 0).astype(int)
    p = lasso_exceedance_probabilities(X[:300], y[:300], X[300:])
    assert p.shape == (100,)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_lasso_separates_signal():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (600, 3))
    y = (X[:, 0] > 0).astype(int)                 # class driven by feature 0
    p = lasso_exceedance_probabilities(X[:450], y[:450], X[450:])
    y_test = y[450:]
    assert p[y_test == 1].mean() > p[y_test == 0].mean() + 0.2


def run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print('  PASS  ' + t.__name__); passed += 1
        except Exception as exc:
            print('  FAIL  ' + t.__name__ + ': ' + repr(exc)); traceback.print_exc()
    print('\nAll %d tests passed.' % passed if passed == len(tests)
          else '\n%d/%d tests passed.' % (passed, len(tests)))
    return passed == len(tests)


if __name__ == '__main__':
    sys.exit(0 if run_all() else 1)
