"""Unit tests for src/dcc_garch.py - Phase 3 DCC correlation and t-copula."""
import os, sys, traceback
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
from dcc_garch import estimate_dcc, t_copula_tail_dependence


def _bivnorm(rho, n, seed):
    rng = np.random.default_rng(seed)
    cov = [[1.0, rho], [rho, 1.0]]
    x = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    return x[:, 0], x[:, 1]


def test_dcc_constant_correlation_recovered():
    e1, e2 = _bivnorm(0.5, 2500, seed=0)
    res = estimate_dcc(e1, e2)
    assert 0.40 < res['rho_t'].mean() < 0.60, res['rho_t'].mean()
    assert abs(res['rho_bar'] - 0.5) < 0.08


def test_dcc_params_in_valid_region():
    e1, e2 = _bivnorm(0.3, 2000, seed=1)
    res = estimate_dcc(e1, e2)
    assert res['a'] >= 0.0 and res['b'] >= 0.0
    assert res['persistence'] < 1.0


def test_dcc_tracks_regime_change():
    a1, a2 = _bivnorm(-0.3, 2000, seed=2)
    b1, b2 = _bivnorm(0.7, 2000, seed=3)
    e1 = np.concatenate([a1, b1])
    e2 = np.concatenate([a2, b2])
    rho = estimate_dcc(e1, e2)['rho_t']
    first = rho[:500].mean()
    last = rho[-500:].mean()
    assert last > first + 0.3, (first, last)


def test_dcc_rejects_too_short():
    try:
        estimate_dcc(np.zeros(20), np.zeros(20))
        assert False, 'expected ValueError'
    except ValueError:
        pass


def test_t_copula_tail_dependence_bounds():
    for rho in (-0.5, 0.0, 0.3, 0.9):
        lam = t_copula_tail_dependence(rho, nu=6.0)
        assert 0.0 <= lam <= 1.0, (rho, lam)


def test_t_copula_tail_dependence_increases_with_rho():
    lo = t_copula_tail_dependence(0.2, nu=6.0)
    hi = t_copula_tail_dependence(0.8, nu=6.0)
    assert hi > lo, (lo, hi)


def test_t_copula_more_tail_dependence_with_heavier_tails():
    heavy = t_copula_tail_dependence(0.4, nu=3.0)
    light = t_copula_tail_dependence(0.4, nu=30.0)
    assert heavy > light, (heavy, light)


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
