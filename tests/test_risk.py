"""Unit tests for src/risk.py.

Run with:

    python tests/test_risk.py

The tests check:
- Closed-form Gaussian VaR/ES against textbook values at 95% and 99%.
- Definitional properties: P(L > VaR) = 1 - alpha; ES = E[L | L > VaR]
  (cross-checked via numerical integration with scipy.integrate.quad).
- Scaling/shifting invariances.
- Monotonicity of t-ES in nu (heavier tails -> larger ES).
- Convergence of the t-engine to the Gaussian engine as nu -> infinity.
- Rolling driver: correct shape, correct index, exact match against direct
  per-window application of the closed-form engines.
- Backtest helper: correct counts in edge cases and well-calibrated rate
  on a large simulated Gaussian sample.

No external test framework needed (uses bare asserts).
"""

import os
import sys

# Make ``import risk`` find ../src/risk.py regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad

from risk import (
    gaussian_var_es,
    student_t_var_es,
    rolling_var_es,
    backtest_var_es,
    kupiec_pof_test,
    christoffersen_cc_test,
    mcneil_frey_es_test,
    acerbi_szekely_z2,
    backtest_es,
)


# ---------------------------------------------------------------------------
# Gaussian closed-form tests
# ---------------------------------------------------------------------------

def test_gaussian_99_standard():
    var, es = gaussian_var_es(mu=0.0, sigma=1.0, alpha=0.99)
    # z_99 = 2.32634787; ES = phi(z_99) / 0.01 = 0.02665214 / 0.01
    assert abs(var - 2.32634787) < 1e-6, f"VaR={var}"
    assert abs(es - 2.66521422) < 1e-6, f"ES={es}"
    print("  PASS  test_gaussian_99_standard")


def test_gaussian_95_standard():
    var, es = gaussian_var_es(mu=0.0, sigma=1.0, alpha=0.95)
    assert abs(var - 1.64485363) < 1e-6
    assert abs(es - 2.06271309) < 1e-6
    print("  PASS  test_gaussian_95_standard")


def test_gaussian_mean_shift_invariance():
    v0, e0 = gaussian_var_es(0.0, 1.0, 0.99)
    v1, e1 = gaussian_var_es(0.01, 1.0, 0.99)
    assert abs((v1 - v0) - (-0.01)) < 1e-12
    assert abs((e1 - e0) - (-0.01)) < 1e-12
    print("  PASS  test_gaussian_mean_shift_invariance")


def test_gaussian_scale_linearity():
    v0, e0 = gaussian_var_es(0.0, 1.0, 0.99)
    v1, e1 = gaussian_var_es(0.0, 2.5, 0.99)
    assert abs(v1 - 2.5 * v0) < 1e-12
    assert abs(e1 - 2.5 * e0) < 1e-12
    print("  PASS  test_gaussian_scale_linearity")


def test_gaussian_es_strictly_greater_than_var():
    var, es = gaussian_var_es(0.0, 1.0, 0.99)
    assert es > var
    print("  PASS  test_gaussian_es_strictly_greater_than_var")


def test_gaussian_quantile_definition():
    """P(L > VaR) = 1 - alpha by definition."""
    mu, sigma, alpha = 0.0005, 0.012, 0.99
    var, _ = gaussian_var_es(mu, sigma, alpha)
    # P(r < -VaR) = Phi((-VaR - mu) / sigma)
    p = stats.norm.cdf(-var, loc=mu, scale=sigma)
    assert abs(p - (1 - alpha)) < 1e-10
    print("  PASS  test_gaussian_quantile_definition")


def test_gaussian_es_numerical():
    """ES = E[L | L > VaR] via numerical integration."""
    mu, sigma, alpha = 0.0005, 0.012, 0.99
    var, es = gaussian_var_es(mu, sigma, alpha)
    # E[L | L > VaR] = -E[r | r < -VaR]
    f = lambda r: stats.norm.pdf(r, loc=mu, scale=sigma)
    num, _ = quad(lambda r: r * f(r), -np.inf, -var)
    den, _ = quad(f, -np.inf, -var)
    expected_es = -num / den
    assert abs(es - expected_es) < 1e-8, f"closed-form={es}, numerical={expected_es}"
    print("  PASS  test_gaussian_es_numerical")


# ---------------------------------------------------------------------------
# Student-t closed-form tests
# ---------------------------------------------------------------------------

def test_student_t_quantile_definition():
    """P(L > VaR) = 1 - alpha under the standardized t parameterization."""
    mu, sigma, nu, alpha = 0.0, 0.01, 6, 0.99
    var, _ = student_t_var_es(mu, sigma, nu, alpha)
    sigma_eff = sigma * np.sqrt((nu - 2) / nu)
    p = stats.t.cdf((-var - mu) / sigma_eff, df=nu)
    assert abs(p - (1 - alpha)) < 1e-10
    print("  PASS  test_student_t_quantile_definition")


def test_student_t_es_numerical():
    """ES = E[L | L > VaR] via numerical integration."""
    mu, sigma, nu, alpha = 0.0005, 0.012, 6, 0.99
    var, es = student_t_var_es(mu, sigma, nu, alpha)
    sigma_eff = sigma * np.sqrt((nu - 2) / nu)
    f = lambda r: stats.t.pdf((r - mu) / sigma_eff, df=nu) / sigma_eff
    num, _ = quad(lambda r: r * f(r), -np.inf, -var)
    den, _ = quad(f, -np.inf, -var)
    expected_es = -num / den
    assert abs(es - expected_es) < 1e-6, f"closed-form={es}, numerical={expected_es}"
    print("  PASS  test_student_t_es_numerical")


def test_student_t_es_gt_var():
    var, es = student_t_var_es(0.0, 0.01, nu=6, alpha=0.99)
    assert es > var
    print("  PASS  test_student_t_es_gt_var")


def test_student_t_var_decreases_with_nu():
    """Lower nu (heavier tails) -> larger 99% VaR at fixed total sigma."""
    vars_ = [student_t_var_es(0.0, 0.01, nu, 0.99)[0] for nu in [4, 6, 10, 30]]
    assert vars_[0] > vars_[1] > vars_[2] > vars_[3], vars_
    print("  PASS  test_student_t_var_decreases_with_nu")


def test_student_t_es_decreases_with_nu():
    """Lower nu -> larger ES at fixed total sigma."""
    es_ = [student_t_var_es(0.0, 0.01, nu, 0.99)[1] for nu in [4, 6, 10, 30]]
    assert es_[0] > es_[1] > es_[2] > es_[3], es_
    print("  PASS  test_student_t_es_decreases_with_nu")


def test_student_t_approaches_gaussian_high_nu():
    v_g, e_g = gaussian_var_es(0.0, 1.0, 0.99)
    v_t, e_t = student_t_var_es(0.0, 1.0, nu=10_000, alpha=0.99)
    assert abs(v_t - v_g) < 1e-2, f"VaR diff = {v_t - v_g}"
    assert abs(e_t - e_g) < 1e-2, f"ES diff = {e_t - e_g}"
    print("  PASS  test_student_t_approaches_gaussian_high_nu")


def test_student_t_nu_too_small_raises():
    try:
        student_t_var_es(0.0, 1.0, nu=2, alpha=0.99)
    except ValueError:
        print("  PASS  test_student_t_nu_too_small_raises")
        return
    raise AssertionError("Expected ValueError for nu <= 2")


# ---------------------------------------------------------------------------
# Rolling-engine tests
# ---------------------------------------------------------------------------

def test_rolling_engine_shape_and_index():
    rng = np.random.default_rng(0)
    n, w = 2000, 250
    s = pd.Series(rng.normal(0, 0.01, size=n),
                  index=pd.date_range('2010-01-01', periods=n, freq='B'))
    # zero_mean=False: this test checks the exact first-window algebra including the
    # sample mean. The engine's default is zero_mean=True (a later design choice).
    fc = rolling_var_es(s, window=w, engine='gaussian', alpha=0.99, zero_mean=False)
    assert list(fc.columns) == ['VaR', 'ES']
    # First valid forecast is at index W (uses returns at positions 0..W-1).
    assert fc.index[0] == s.index[w], f"first index = {fc.index[0]}, expected {s.index[w]}"
    # Length: total - W
    assert len(fc) == n - w
    print("  PASS  test_rolling_engine_shape_and_index")


def test_rolling_engine_first_window_exact():
    rng = np.random.default_rng(42)
    n, w = 600, 250
    s = pd.Series(rng.normal(0, 0.01, size=n),
                  index=pd.date_range('2010-01-01', periods=n, freq='B'))

    # zero_mean=False: this test checks the exact first-window algebra including the
    # sample mean. The engine's default is zero_mean=True (a later design choice).
    fc = rolling_var_es(s, window=w, engine='gaussian', alpha=0.99, zero_mean=False)
    expected_mu = s.iloc[:w].mean()
    expected_sigma = s.iloc[:w].std(ddof=1)
    expected_var, expected_es = gaussian_var_es(expected_mu, expected_sigma, 0.99)

    assert abs(fc['VaR'].iloc[0] - expected_var) < 1e-12
    assert abs(fc['ES'].iloc[0] - expected_es) < 1e-12
    print("  PASS  test_rolling_engine_first_window_exact")


def test_rolling_engine_t_first_window_exact():
    rng = np.random.default_rng(1)
    n, w, nu = 500, 250, 6
    s = pd.Series(rng.normal(0, 0.01, size=n),
                  index=pd.date_range('2010-01-01', periods=n, freq='B'))
    fc = rolling_var_es(s, window=w, engine='t', alpha=0.99, nu=nu, zero_mean=False)
    expected_mu = s.iloc[:w].mean()
    expected_sigma = s.iloc[:w].std(ddof=1)
    expected_var, expected_es = student_t_var_es(expected_mu, expected_sigma, nu, 0.99)
    assert abs(fc['VaR'].iloc[0] - expected_var) < 1e-12
    assert abs(fc['ES'].iloc[0] - expected_es) < 1e-12
    print("  PASS  test_rolling_engine_t_first_window_exact")


def test_rolling_engine_unknown_raises():
    s = pd.Series(np.zeros(300),
                  index=pd.date_range('2010-01-01', periods=300, freq='B'))
    try:
        rolling_var_es(s, window=100, engine='laplace', alpha=0.99)
    except ValueError:
        print("  PASS  test_rolling_engine_unknown_raises")
        return
    raise AssertionError("Expected ValueError for unknown engine")


# ---------------------------------------------------------------------------
# Backtest tests
# ---------------------------------------------------------------------------

def test_backtest_no_exceptions():
    n = 1000
    idx = pd.date_range('2010-01-01', periods=n, freq='B')
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, size=n), index=idx)
    fc = pd.DataFrame({'VaR': [1.0] * n, 'ES': [1.5] * n}, index=idx)
    result = backtest_var_es(r, fc, alpha=0.99)
    assert result['n_exceptions'] == 0
    assert result['exception_rate'] == 0.0
    print("  PASS  test_backtest_no_exceptions")


def test_backtest_all_exceptions():
    n = 100
    idx = pd.date_range('2010-01-01', periods=n, freq='B')
    r = pd.Series([-0.01] * n, index=idx)
    fc = pd.DataFrame({'VaR': [0.0] * n, 'ES': [0.005] * n}, index=idx)
    result = backtest_var_es(r, fc, alpha=0.99)
    assert result['n_exceptions'] == n
    assert result['exception_rate'] == 1.0
    assert abs(result['mean_loss_given_exception'] - 0.01) < 1e-12
    # severity = L - ES = 0.01 - 0.005 = 0.005
    assert abs(result['mean_es_severity'] - 0.005) < 1e-12
    print("  PASS  test_backtest_all_exceptions")


def test_backtest_well_calibrated_gaussian():
    """For iid Gaussian sample with true VaR, exception rate ~ 1 - alpha."""
    rng = np.random.default_rng(0)
    n = 50_000
    sigma_true = 0.01
    idx = pd.date_range('2010-01-01', periods=n, freq='B')
    r = pd.Series(rng.normal(0, sigma_true, size=n), index=idx)
    var, es = gaussian_var_es(0.0, sigma_true, 0.99)
    fc = pd.DataFrame({'VaR': [var] * n, 'ES': [es] * n}, index=idx)
    result = backtest_var_es(r, fc, alpha=0.99)
    # SE of exception rate ~ sqrt(0.01 * 0.99 / 50000) ~ 0.00044
    # Allow generous +/- 0.002 (about 4.5 SE)
    assert abs(result['exception_rate'] - 0.01) < 0.002, result
    print("  PASS  test_backtest_well_calibrated_gaussian")


# ---------------------------------------------------------------------------
# Kupiec POF tests
# ---------------------------------------------------------------------------

def test_kupiec_perfect_calibration_returns_zero():
    """If pi_hat == p exactly, LR_uc = 0 and p-value = 1."""
    n, x, alpha = 1000, 10, 0.99   # pi_hat = 0.01 = 1 - alpha
    e = [1] * x + [0] * (n - x)
    r = kupiec_pof_test(e, alpha)
    assert abs(r['lr_uc']) < 1e-10, f"LR_uc = {r['lr_uc']}"
    assert abs(r['p_value'] - 1.0) < 1e-10
    assert r['reject_5pct'] is False
    print("  PASS  test_kupiec_perfect_calibration_returns_zero")


def test_kupiec_severe_over_rejection_rejects():
    """Many more exceptions than expected should reject H0."""
    n, x, alpha = 1000, 50, 0.99   # pi_hat = 0.05 vs target 0.01
    e = [1] * x + [0] * (n - x)
    r = kupiec_pof_test(e, alpha)
    assert r['lr_uc'] > 30, r['lr_uc']           # well into the rejection region
    assert r['p_value'] < 1e-6
    assert r['reject_5pct'] is True
    print("  PASS  test_kupiec_severe_over_rejection_rejects")


def test_kupiec_zero_exceptions_handled():
    """All-zero exception series should not raise and should reject (over-conservative VaR)."""
    n, alpha = 10_000, 0.99
    e = [0] * n
    r = kupiec_pof_test(e, alpha)
    # LR = -2 * n * log(1 - p) = -2 * 10000 * log(0.99) ~ 200.9
    assert abs(r['lr_uc'] - (-2 * n * np.log(1 - (1 - alpha)))) < 1e-8
    assert r['p_value'] < 1e-10
    print("  PASS  test_kupiec_zero_exceptions_handled")


def test_kupiec_all_exceptions_handled():
    n, alpha = 100, 0.99
    e = [1] * n
    r = kupiec_pof_test(e, alpha)
    expected_lr = -2 * n * np.log(1 - alpha)
    assert abs(r['lr_uc'] - expected_lr) < 1e-8
    assert r['p_value'] < 1e-10
    print("  PASS  test_kupiec_all_exceptions_handled")


def test_kupiec_textbook_example():
    """n=250, x=15, alpha=0.95 -> LR_uc ~ 0.50 (Jorion-style worked example)."""
    n, x, alpha = 250, 15, 0.95   # pi_hat = 0.06, p = 0.05
    e = [1] * x + [0] * (n - x)
    r = kupiec_pof_test(e, alpha)
    assert 0.4 < r['lr_uc'] < 0.6, f"LR_uc = {r['lr_uc']}"
    assert r['p_value'] > 0.4
    assert r['reject_5pct'] is False
    print("  PASS  test_kupiec_textbook_example")


# ---------------------------------------------------------------------------
# Christoffersen CC tests
# ---------------------------------------------------------------------------

def test_christoffersen_clustered_rejects_independence():
    """A single big cluster of exceptions strongly rejects independence."""
    e = [1] * 20 + [0] * 180
    r = christoffersen_cc_test(e, alpha=0.95)
    assert r['lr_ind'] > 30, r['lr_ind']
    assert r['p_ind'] < 1e-6
    assert r['reject_ind_5pct'] is True
    print("  PASS  test_christoffersen_clustered_rejects_independence")


def test_christoffersen_spread_does_not_reject_independence():
    """Evenly spaced exceptions should not strongly reject independence."""
    n = 5000
    e = np.zeros(n, dtype=int)
    e[::100] = 1                  # exactly 1% rate, perfectly spread
    r = christoffersen_cc_test(e, alpha=0.99)
    # No n11 transitions, so pi_11 = 0; pi_01 = 50/(50*99) tiny.
    # Independence statistic should still be modest (won't reject for spread-out exceptions).
    # The most important property: p_ind should be > 0.05 (not significant).
    assert r['p_ind'] > 0.05, f"p_ind = {r['p_ind']}"
    print("  PASS  test_christoffersen_spread_does_not_reject_independence")


def test_christoffersen_iid_simulation():
    """For long iid Bernoulli sample, independence test should not reject."""
    rng = np.random.default_rng(0)
    n, p = 50_000, 0.01
    e = (rng.random(n) < p).astype(int)
    r = christoffersen_cc_test(e, alpha=0.99)
    # With n=50k and 5% level, p_ind > 0.05 with about 95% probability;
    # seed 0 is chosen for reproducibility.
    assert r['p_ind'] > 0.05, f"p_ind = {r['p_ind']}"
    print("  PASS  test_christoffersen_iid_simulation")


def test_christoffersen_lr_cc_equals_uc_plus_ind():
    """The joint LR_cc must equal LR_uc + LR_ind."""
    rng = np.random.default_rng(1)
    e = (rng.random(1000) < 0.05).astype(int)
    r = christoffersen_cc_test(e, alpha=0.95)
    assert abs(r['lr_cc'] - (r['lr_uc'] + r['lr_ind'])) < 1e-10
    print("  PASS  test_christoffersen_lr_cc_equals_uc_plus_ind")


def test_christoffersen_transition_counts():
    """Hand-checked transition counts."""
    # e = [0,0,1,1,0,1,0]
    # transitions: (0,0),(0,1),(1,1),(1,0),(0,1),(1,0)
    # n00=1, n01=2, n10=2, n11=1
    e = [0, 0, 1, 1, 0, 1, 0]
    r = christoffersen_cc_test(e, alpha=0.95)
    assert r['n00'] == 1
    assert r['n01'] == 2
    assert r['n10'] == 2
    assert r['n11'] == 1
    print("  PASS  test_christoffersen_transition_counts")


# ---------------------------------------------------------------------------
# backtest_var_es regression — new fields exist, old fields unchanged
# ---------------------------------------------------------------------------

def test_backtest_includes_formal_tests():
    n = 1000
    idx = pd.date_range('2010-01-01', periods=n, freq='B')
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, size=n), index=idx)
    fc = pd.DataFrame({'VaR': [0.02] * n, 'ES': [0.025] * n}, index=idx)
    result = backtest_var_es(r, fc, alpha=0.99)
    # Old contract
    for k in ['n_obs', 'n_exceptions', 'exception_rate',
              'expected_exception_rate', 'mean_loss_given_exception',
              'mean_es_severity']:
        assert k in result
    # New contract
    for k in ['kupiec', 'christoffersen', 'exception_series']:
        assert k in result, f"missing key: {k}"
    assert 'lr_uc' in result['kupiec']
    assert 'lr_cc' in result['christoffersen']
    assert len(result['exception_series']) == n
    print("  PASS  test_backtest_includes_formal_tests")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Phase 3 cleanup (Code V5): tests for the formal ES backtest functions
# (mcneil_frey_es_test, acerbi_szekely_z2, backtest_es) now consolidated
# in src/risk.py. These were previously duplicated in src/es_tests.py and
# inline in notebook 04; both copies have been deleted in V5.
# ---------------------------------------------------------------------------

def _make_es(n, exc_positions, loss_on_exc, var=0.02, es=0.03):
    """Returns + forecasts with controlled exception positions and loss sizes."""
    idx = pd.RangeIndex(n)
    VaR = np.full(n, var, dtype=float)
    ES = np.full(n, es, dtype=float)
    ret = np.full(n, 0.001, dtype=float)
    pos = np.atleast_1d(np.asarray(exc_positions, dtype=int))
    losses = np.broadcast_to(np.asarray(loss_on_exc, dtype=float), pos.shape)
    for p, l in zip(pos, losses):
        ret[p] = -l
    return (pd.Series(ret, index=idx),
            pd.DataFrame({'VaR': VaR, 'ES': ES}, index=idx))


def test_mcneil_frey_returns_correct_keys():
    returns, fc = _make_es(500, [10, 50, 100, 200, 400], 0.03)
    r = mcneil_frey_es_test(returns, fc, alpha=0.99, n_bootstrap=500)
    for k in ('test', 'n_exc', 'mean_residual', 'p_value', 'reject_5pct',
              'ci_lower', 'ci_upper', 'n_bootstrap'):
        assert k in r, f'missing key {k!r}'
    assert r['n_exc'] == 5


def test_mcneil_frey_calibrated_not_rejected():
    rng = np.random.default_rng(10)
    exc = sorted(rng.choice(4000, size=120, replace=False))
    losses = 0.03 + rng.normal(0.0, 0.003, 120)
    returns, fc = _make_es(4000, exc, losses, var=0.012)
    r = mcneil_frey_es_test(returns, fc, alpha=0.97, n_bootstrap=4000, seed=10)
    assert r['p_value'] > 0.05, r
    assert not r['reject_5pct']


def test_mcneil_frey_rejects_understated_es():
    rng = np.random.default_rng(11)
    exc = sorted(rng.choice(4000, size=120, replace=False))
    losses = 0.05 + rng.normal(0.0, 0.003, 120)
    returns, fc = _make_es(4000, exc, losses, var=0.012)
    r = mcneil_frey_es_test(returns, fc, alpha=0.97, n_bootstrap=4000, seed=11)
    assert r['mean_residual'] > 0.01
    assert r['p_value'] < 0.05 and r['reject_5pct'] is True, r


def test_mcneil_frey_too_few_exceptions_handled():
    returns, fc = _make_es(500, [100], 0.05)
    r = mcneil_frey_es_test(returns, fc, alpha=0.99)
    assert r['n_exc'] == 1
    assert not np.isfinite(r['mean_residual'])
    assert not r['reject_5pct']


def test_acerbi_szekely_z2_returns_correct_keys():
    returns, fc = _make_es(500, [10, 50, 100, 200, 400], 0.03)
    r = acerbi_szekely_z2(returns, fc, alpha=0.99, n_bootstrap=500)
    for k in ('test', 'n_obs', 'n_exc', 'Z2', 'ci_lower', 'ci_upper',
              'reject_5pct', 'n_bootstrap'):
        assert k in r, f'missing key {k!r}'


def test_acerbi_szekely_z2_calibrated_near_zero():
    exc = np.linspace(0, 3999, 100, dtype=int).tolist()
    returns, fc = _make_es(4000, exc, 0.03, var=0.02, es=0.03)
    r = acerbi_szekely_z2(returns, fc, alpha=0.975, n_bootstrap=500)
    assert abs(r['Z2']) < 0.05, r['Z2']


def test_acerbi_szekely_z2_negative_for_understated_es():
    exc = np.linspace(0, 3999, 100, dtype=int).tolist()
    returns, fc = _make_es(4000, exc, 0.06, var=0.02, es=0.03)
    r = acerbi_szekely_z2(returns, fc, alpha=0.975, n_bootstrap=500)
    assert r['Z2'] < -0.5, r['Z2']


def test_backtest_es_combines_both():
    exc = np.linspace(0, 3999, 100, dtype=int).tolist()
    returns, fc = _make_es(4000, exc, 0.05, var=0.012, es=0.03)
    r = backtest_es(returns, fc, alpha=0.97,
                    n_bootstrap_mf=500, n_bootstrap_z2=500)
    assert 'mcneil_frey' in r and 'acerbi_szekely' in r
    # mean_residual lives in the McNeil-Frey sub-result, as backtest_es documents
    assert 'n_exc' in r and 'mean_residual' in r['mcneil_frey']
    assert r['n_exc'] == r['mcneil_frey']['n_exc']


def _all_tests():
    return [
        test_gaussian_99_standard,
        test_gaussian_95_standard,
        test_gaussian_mean_shift_invariance,
        test_gaussian_scale_linearity,
        test_gaussian_es_strictly_greater_than_var,
        test_gaussian_quantile_definition,
        test_gaussian_es_numerical,
        test_student_t_quantile_definition,
        test_student_t_es_numerical,
        test_student_t_es_gt_var,
        test_student_t_var_decreases_with_nu,
        test_student_t_es_decreases_with_nu,
        test_student_t_approaches_gaussian_high_nu,
        test_student_t_nu_too_small_raises,
        test_rolling_engine_shape_and_index,
        test_rolling_engine_first_window_exact,
        test_rolling_engine_t_first_window_exact,
        test_rolling_engine_unknown_raises,
        test_backtest_no_exceptions,
        test_backtest_all_exceptions,
        test_backtest_well_calibrated_gaussian,
        # Kupiec
        test_kupiec_perfect_calibration_returns_zero,
        test_kupiec_severe_over_rejection_rejects,
        test_kupiec_zero_exceptions_handled,
        test_kupiec_all_exceptions_handled,
        test_kupiec_textbook_example,
        # Christoffersen
        test_christoffersen_clustered_rejects_independence,
        test_christoffersen_spread_does_not_reject_independence,
        test_christoffersen_iid_simulation,
        test_christoffersen_lr_cc_equals_uc_plus_ind,
        test_christoffersen_transition_counts,
        # backtest_var_es extension
        test_backtest_includes_formal_tests,
        # Phase 3 ES backtests (formerly in es_tests.py)
        test_mcneil_frey_returns_correct_keys,
        test_mcneil_frey_calibrated_not_rejected,
        test_mcneil_frey_rejects_understated_es,
        test_mcneil_frey_too_few_exceptions_handled,
        test_acerbi_szekely_z2_returns_correct_keys,
        test_acerbi_szekely_z2_calibrated_near_zero,
        test_acerbi_szekely_z2_negative_for_understated_es,
        test_backtest_es_combines_both,
    ]


def run_all():
    tests = _all_tests()
    print(f"Running {len(tests)} tests for src/risk.py...\n")
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
        for name, e in failed:
            print(f"   - {name}: {e}")
    return len(failed)


if __name__ == '__main__':
    sys.exit(0 if run_all() == 0 else 1)
