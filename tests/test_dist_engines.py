"""Unit tests for src/dist_engines.py - Phase 3 MLE-nu and skew-t engines."""
import os, sys, traceback
import numpy as np, pandas as pd
from scipy import stats
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
from dist_engines import (estimate_t_nu, rolling_t_mle_var_es,
                          fit_skewt_shape, skewt_var_es, rolling_skewt_var_es)


def _real_returns(n=520):
    df = pd.read_csv(os.path.join(_HERE, '..', 'data', 'returns.csv'),
                     index_col=0, parse_dates=True)
    return df['Port_FO'].dropna().iloc[-n:]


def test_estimate_t_nu_recovers_df():
    rng = np.random.default_rng(0)
    x = stats.t.rvs(df=5, size=5000, random_state=rng) * 0.01
    nu = estimate_t_nu(x)
    assert 3.5 < nu < 8.0, nu


def test_estimate_t_nu_clamps_near_normal():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.01, 5000)
    nu = estimate_t_nu(x, nu_min=2.5, nu_max=100.0)
    assert 2.5 <= nu <= 100.0, nu


def test_rolling_t_mle_shape_and_positive():
    df = rolling_t_mle_var_es(_real_returns(700), window=400, alpha=0.99)
    assert list(df.columns) == ['VaR', 'ES', 'nu']
    assert len(df) > 0
    assert (df['VaR'] > 0).all()
    assert (df['ES'] >= df['VaR'] - 1e-12).all()
    assert (df['nu'] >= 2.5).all() and (df['nu'] <= 100.0).all()


def test_skewt_var_es_positive_and_ordered():
    var, es = skewt_var_es(0.0, 0.01, eta=6.0, lam=-0.2, alpha=0.99)
    assert var > 0 and es > var


def test_skewt_negative_skew_raises_var():
    var_sym, _ = skewt_var_es(0.0, 0.01, eta=6.0, lam=0.0, alpha=0.99)
    var_left, _ = skewt_var_es(0.0, 0.01, eta=6.0, lam=-0.4, alpha=0.99)
    assert var_left > var_sym, (var_left, var_sym)


def test_fit_skewt_shape_on_real_data():
    eta, lam = fit_skewt_shape(_real_returns(1000))
    assert eta > 2.0
    assert -1.0 < lam < 1.0


def test_rolling_skewt_shape_and_positive():
    df = rolling_skewt_var_es(_real_returns(320), window=250, alpha=0.99,
                              refit_freq=20)
    assert list(df.columns) == ['VaR', 'ES', 'eta', 'lam']
    assert len(df) > 0
    assert (df['VaR'] > 0).all()
    assert (df['ES'] >= df['VaR'] - 1e-12).all()


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
