"""Unit tests for src/asym_garch.py - Phase 3 asymmetric GARCH (GJR / EGARCH)."""
import os, sys, traceback
import numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
from asym_garch import (fit_asymmetric_window, gjr_step_sigma2,
                        egarch_step_lnsigma2, rolling_asymmetric_fhs)


def _vol_change_series(n_low=400, n_high=400, seed=0):
    rng = np.random.default_rng(seed)
    r = np.concatenate([rng.normal(0.0, 0.005, n_low),
                        rng.normal(0.0, 0.025, n_high)])
    return pd.Series(r, index=pd.date_range('2000-01-03', periods=len(r), freq='B'))


def _real_returns(n=520):
    path = os.path.join(_HERE, '..', 'data', 'returns.csv')
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df['Port_FO'].dropna().iloc[-n:]


def test_gjr_step_leverage_asymmetry():
    s_neg = gjr_step_sigma2(0.1, 0.05, 0.10, 0.85, r_prev=-2.0, sigma2_prev=1.0)
    s_pos = gjr_step_sigma2(0.1, 0.05, 0.10, 0.85, r_prev=2.0, sigma2_prev=1.0)
    assert s_neg > s_pos, (s_neg, s_pos)


def test_gjr_step_reduces_to_garch_when_gamma_zero():
    a = gjr_step_sigma2(0.1, 0.05, 0.0, 0.85, r_prev=-2.0, sigma2_prev=1.0)
    b = gjr_step_sigma2(0.1, 0.05, 0.0, 0.85, r_prev=2.0, sigma2_prev=1.0)
    assert abs(a - b) < 1e-12


def test_egarch_step_leverage_asymmetry():
    ln_neg = egarch_step_lnsigma2(0.0, 0.1, -0.10, 0.95, z_prev=-2.0, lnsigma2_prev=0.0)
    ln_pos = egarch_step_lnsigma2(0.0, 0.1, -0.10, 0.95, z_prev=2.0, lnsigma2_prev=0.0)
    assert ln_neg > ln_pos, (ln_neg, ln_pos)


def test_fit_gjr_returns_params():
    res = fit_asymmetric_window(_vol_change_series(seed=1), model='gjr')
    for k in ('omega', 'alpha[1]', 'gamma[1]', 'beta[1]'):
        assert k in res.params.index and np.isfinite(res.params[k])


def test_fit_egarch_returns_sensible_params_on_real_data():
    res = fit_asymmetric_window(_real_returns(600), model='egarch')
    for k in ('omega', 'alpha[1]', 'gamma[1]', 'beta[1]'):
        assert k in res.params.index and np.isfinite(res.params[k])
    assert abs(res.params['beta[1]']) < 1.0


def test_fit_rejects_unknown_model():
    try:
        fit_asymmetric_window(_vol_change_series(), model='aparch')
        assert False
    except ValueError:
        pass


def test_rolling_gjr_shape_and_positive():
    df = rolling_asymmetric_fhs(_vol_change_series(seed=3), window=250,
                                alpha=0.99, model='gjr', refit_freq=25)
    assert list(df.columns) == ['VaR', 'ES', 'sigma']
    assert len(df) > 0 and (df['VaR'] > 0).all()
    assert (df['ES'] >= df['VaR'] - 1e-12).all()


def test_rolling_gjr_responds_to_volatility_change():
    df = rolling_asymmetric_fhs(_vol_change_series(400, 400, seed=5),
                                window=250, alpha=0.99, model='gjr', refit_freq=25)
    assert df['sigma'].iloc[-50:].mean() > 2.0 * df['sigma'].iloc[:50].mean()


def test_rolling_egarch_shape_and_positive_on_real_data():
    df = rolling_asymmetric_fhs(_real_returns(180), window=140, alpha=0.99,
                                model='egarch')
    assert list(df.columns) == ['VaR', 'ES', 'sigma']
    assert len(df) > 0 and np.isfinite(df['VaR']).all() and (df['VaR'] > 0).all()
    assert (df['ES'] >= df['VaR'] - 1e-12).all()


def test_rolling_rejects_unknown_model():
    try:
        rolling_asymmetric_fhs(_vol_change_series(), 250, 0.99, model='tarch')
        assert False
    except ValueError:
        pass


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
