"""Unit tests for src/drawdowns.py.

Run with:

    python tests/test_drawdowns.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from drawdowns import (
    compute_wealth_drawdowns,
    max_drawdown,
    drawdown_episodes,
    time_under_water_stats,
    cdar,
    simulate_mdd_distribution,
)


# ---------------------------------------------------------------------------
# compute_wealth_drawdowns & max_drawdown
# ---------------------------------------------------------------------------

def test_no_drawdown_for_positive_returns():
    """Strictly positive returns -> drawdown is identically zero."""
    r = pd.Series([0.01, 0.02, 0.005, 0.01, 0.03],
                  index=pd.date_range('2020-01-01', periods=5, freq='B'))
    d = compute_wealth_drawdowns(r)
    assert (d['drawdown'] == 0.0).all()
    assert max_drawdown(r) == 0.0
    print("  PASS  test_no_drawdown_for_positive_returns")


def test_drawdown_handcomputed():
    """Hand-computed example: -50% drawdown after a single big loss."""
    # Returns: +100%, -50%, +20%
    # Wealth:  1.0 -> 2.0 -> 1.0 -> 1.2
    # Peak:    1.0 -> 2.0 -> 2.0 -> 2.0
    # DD:      0.0 -> 0.0 -> -0.50 -> -0.40
    r = pd.Series([1.0, -0.5, 0.20],
                  index=pd.date_range('2020-01-01', periods=3, freq='B'))
    d = compute_wealth_drawdowns(r)
    assert abs(d['wealth'].iloc[-1] - 1.2) < 1e-12
    assert abs(d['drawdown'].iloc[1] - (-0.50)) < 1e-12
    assert abs(d['drawdown'].iloc[2] - (-0.40)) < 1e-12
    assert abs(max_drawdown(r) - 0.50) < 1e-12
    print("  PASS  test_drawdown_handcomputed")


# ---------------------------------------------------------------------------
# drawdown_episodes
# ---------------------------------------------------------------------------

def test_episodes_single_full_recovery():
    """One drawdown, fully recovered."""
    # Wealth path: 1, 1.1, 1.0, 0.9, 1.0, 1.2
    # peaks at 1.1, troughs at 0.9, recovers at 1.2 day 5? Actually need = peak (1.1)
    # We need to construct returns so the path is exactly this.
    wealth = np.array([1.0, 1.1, 1.0, 0.9, 1.0, 1.2])
    rets = wealth[1:] / wealth[:-1] - 1.0
    r = pd.Series(rets, index=pd.date_range('2020-01-01', periods=len(rets), freq='B'))
    eps = drawdown_episodes(r)
    assert len(eps) == 1
    e = eps.iloc[0]
    assert abs(e['depth'] - (1.0 - 0.9 / 1.1)) < 1e-12, f"depth={e['depth']}"
    # Path recovers at the day wealth >= 1.1; that's day index 5 (wealth=1.2).
    assert e['recovery_date'] is not pd.NaT
    print(f"  PASS  test_episodes_single_full_recovery  (depth={e['depth']:.4f})")


def test_episodes_unrecovered():
    """Drawdown that does NOT recover within the sample -> recovery_date is NaT."""
    wealth = np.array([1.0, 1.2, 1.0, 0.95, 0.90, 0.85])
    rets = wealth[1:] / wealth[:-1] - 1.0
    r = pd.Series(rets, index=pd.date_range('2020-01-01', periods=len(rets), freq='B'))
    eps = drawdown_episodes(r)
    assert len(eps) == 1
    e = eps.iloc[0]
    assert pd.isna(e['recovery_date'])
    assert pd.isna(e['days_to_recovery'])
    assert e['total_underwater_days'] == 4   # 4 days strictly underwater
    print("  PASS  test_episodes_unrecovered")


def test_episodes_min_depth_filter():
    """min_depth filters out small drawdowns."""
    # Two drawdowns: one 2%, one 30%.
    wealth = np.array([1.0, 1.0, 0.98, 1.0, 1.0, 0.70, 1.0])
    rets = wealth[1:] / wealth[:-1] - 1.0
    r = pd.Series(rets, index=pd.date_range('2020-01-01', periods=len(rets), freq='B'))
    eps_all = drawdown_episodes(r)
    eps_big = drawdown_episodes(r, min_depth=0.10)
    assert len(eps_all) == 2
    assert len(eps_big) == 1
    assert eps_big['depth'].iloc[0] > 0.10
    print("  PASS  test_episodes_min_depth_filter")


# ---------------------------------------------------------------------------
# time_under_water_stats
# ---------------------------------------------------------------------------

def test_tuw_stats_basic():
    """Sanity check: all-positive returns -> 0% time underwater."""
    r = pd.Series([0.01] * 10, index=pd.date_range('2020-01-01', periods=10, freq='B'))
    s = time_under_water_stats(r)
    assert s['n_underwater_days'] == 0
    assert s['pct_time_underwater'] == 0.0
    assert s['n_episodes'] == 0
    print("  PASS  test_tuw_stats_basic")


def test_tuw_stats_typical_portfolio():
    """For a noisy series with realistic vol, pct_time_underwater should be substantial."""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0003, 0.012, size=5000),  # ~7.5% annual, ~19% vol
                  index=pd.date_range('2010-01-01', periods=5000, freq='B'))
    s = time_under_water_stats(r)
    # A diffusion path with these parameters is underwater most of the time
    # (typically 90-99% — fraction grows with sample length even for positive drift)
    assert 0.3 < s['pct_time_underwater'] < 0.999
    assert s['n_episodes'] >= 1
    print(f"  PASS  test_tuw_stats_typical_portfolio  "
          f"(pct={s['pct_time_underwater']:.1%}, episodes={s['n_episodes']})")


# ---------------------------------------------------------------------------
# CDaR
# ---------------------------------------------------------------------------

def test_cdar_no_drawdown():
    r = pd.Series([0.01] * 100, index=pd.date_range('2020-01-01', periods=100, freq='B'))
    assert cdar(r, alpha=0.95) == 0.0
    print("  PASS  test_cdar_no_drawdown")


def test_cdar_geq_mdd_for_extreme_alpha():
    """For alpha very close to 1, CDaR should approach MDD."""
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.02, size=3000),
                  index=pd.date_range('2010-01-01', periods=3000, freq='B'))
    mdd = max_drawdown(r)
    cdar_high = cdar(r, alpha=0.999)
    # CDaR_{alpha} <= MDD always (it's averaging over the worst (1-alpha) of drawdowns)
    assert cdar_high <= mdd + 1e-12
    # And for very high alpha, it should be close
    assert cdar_high > 0.5 * mdd, f"cdar={cdar_high}, mdd={mdd}"
    print(f"  PASS  test_cdar_geq_mdd_for_extreme_alpha  (mdd={mdd:.4f}, cdar_999={cdar_high:.4f})")


# ---------------------------------------------------------------------------
# simulate_mdd_distribution
# ---------------------------------------------------------------------------

def test_simulate_mdd_gaussian_shape():
    mdd = simulate_mdd_distribution('gaussian', sigma=0.01, horizon=252,
                                    n_sims=2000, seed=0)
    assert mdd.shape == (2000,)
    assert (mdd >= 0).all()
    assert (mdd < 1).all()
    print(f"  PASS  test_simulate_mdd_gaussian_shape  (mean MDD over 1y = {mdd.mean():.3f})")


def test_simulate_mdd_t_heavier_than_gaussian():
    """For the same sigma, simulated MDD distribution from t(4) should be heavier-tailed."""
    mdd_g = simulate_mdd_distribution('gaussian', sigma=0.01, n_sims=5000, seed=0)
    mdd_t = simulate_mdd_distribution('t', sigma=0.01, nu=4, n_sims=5000, seed=0)
    # 99th percentile of MDD should be substantially higher under t(4)
    q99_g = np.quantile(mdd_g, 0.99)
    q99_t = np.quantile(mdd_t, 0.99)
    assert q99_t > q99_g, f"q99_t={q99_t}, q99_g={q99_g}"
    print(f"  PASS  test_simulate_mdd_t_heavier_than_gaussian  "
          f"(q99 Gaussian={q99_g:.3f}, q99 t(4)={q99_t:.3f})")


def test_simulate_mdd_fhs_with_pool():
    rng = np.random.default_rng(0)
    # Use a larger pool so the bootstrap doesn't over-represent rare outliers.
    # With only 1000 samples from N(0,1), a single -3.9σ draw in the pool gets
    # bootstrapped repeatedly and biases simulated MDD upward; at 10k the
    # bootstrap and direct Gaussian sampling agree within ~0.3 percentage points.
    pool = rng.standard_normal(10_000)
    mdd = simulate_mdd_distribution('fhs', sigma=0.01, residual_pool=pool,
                                    n_sims=2000, seed=0)
    assert mdd.shape == (2000,)
    assert (mdd >= 0).all()
    mdd_g = simulate_mdd_distribution('gaussian', sigma=0.01, n_sims=2000, seed=0)
    assert abs(mdd.mean() - mdd_g.mean()) < 0.01, (
        f"fhs mean={mdd.mean()}, gauss mean={mdd_g.mean()}"
    )
    print("  PASS  test_simulate_mdd_fhs_with_pool")


def test_simulate_mdd_fhs_requires_pool():
    try:
        simulate_mdd_distribution('fhs', sigma=0.01, n_sims=100)
    except ValueError:
        print("  PASS  test_simulate_mdd_fhs_requires_pool")
        return
    raise AssertionError("Expected ValueError when residual_pool missing for FHS")


# ---------------------------------------------------------------------------
# simulate_mdd_distribution -- bootstrap engines
# ---------------------------------------------------------------------------

def test_simulate_mdd_iid_bootstrap_basic():
    """i.i.d. bootstrap resamples a raw-return pool; valid MDD array."""
    rng = np.random.default_rng(0)
    pool = rng.normal(0.0003, 0.012, size=3000)
    mdd = simulate_mdd_distribution('iid_bootstrap', return_pool=pool,
                                    horizon=252, n_sims=2000, seed=0)
    assert mdd.shape == (2000,)
    assert (mdd >= 0).all()
    assert (mdd < 1).all()
    print(f"  PASS  test_simulate_mdd_iid_bootstrap_basic  (mean MDD = {mdd.mean():.3f})")


def test_simulate_mdd_iid_bootstrap_requires_pool():
    try:
        simulate_mdd_distribution('iid_bootstrap', n_sims=100)
    except ValueError:
        print("  PASS  test_simulate_mdd_iid_bootstrap_requires_pool")
        return
    raise AssertionError("Expected ValueError when return_pool missing for iid_bootstrap")


def test_simulate_mdd_block_bootstrap_basic():
    """Block bootstrap produces a valid MDD array."""
    rng = np.random.default_rng(1)
    pool = rng.normal(0.0003, 0.012, size=3000)
    mdd = simulate_mdd_distribution('block_bootstrap', return_pool=pool,
                                    block_length=20, horizon=252,
                                    n_sims=2000, seed=1)
    assert mdd.shape == (2000,)
    assert (mdd >= 0).all()
    assert (mdd < 1).all()
    print(f"  PASS  test_simulate_mdd_block_bootstrap_basic  (mean MDD = {mdd.mean():.3f})")


def test_simulate_mdd_block_bootstrap_requires_pool():
    try:
        simulate_mdd_distribution('block_bootstrap', n_sims=100)
    except ValueError:
        print("  PASS  test_simulate_mdd_block_bootstrap_requires_pool")
        return
    raise AssertionError("Expected ValueError when return_pool missing for block_bootstrap")


def test_block_length_one_approximates_iid():
    """block_length=1 reduces the block bootstrap to the i.i.d. bootstrap.

    The two engines use different RNG call patterns, so the simulated MDD
    arrays are not identical, but their distributions should coincide.
    """
    rng = np.random.default_rng(2)
    pool = rng.normal(0.0003, 0.012, size=4000)
    mdd_iid = simulate_mdd_distribution('iid_bootstrap', return_pool=pool,
                                        n_sims=5000, seed=7)
    mdd_blk1 = simulate_mdd_distribution('block_bootstrap', return_pool=pool,
                                         block_length=1, n_sims=5000, seed=7)
    assert abs(mdd_iid.mean() - mdd_blk1.mean()) < 0.01, (
        f"iid mean={mdd_iid.mean()}, block(L=1) mean={mdd_blk1.mean()}"
    )
    print("  PASS  test_block_length_one_approximates_iid")


def test_block_bootstrap_preserves_clustering():
    """Block resampling retains volatility clustering -> deeper simulated MDD.

    Build a pool with a contiguous 60-day 'crash' embedded in calm returns.
    The i.i.d. bootstrap scatters those crash days across each path; the
    block bootstrap (block_length=30) frequently draws the crash run intact,
    so its simulated MDD distribution is materially deeper in the tail.
    """
    pool = np.full(1000, 0.0008)
    pool[300:360] = -0.025          # 60 consecutive crash days
    mdd_iid = simulate_mdd_distribution('iid_bootstrap', return_pool=pool,
                                        horizon=252, n_sims=4000, seed=3)
    mdd_blk = simulate_mdd_distribution('block_bootstrap', return_pool=pool,
                                        block_length=30, horizon=252,
                                        n_sims=4000, seed=3)
    q95_iid = np.quantile(mdd_iid, 0.95)
    q95_blk = np.quantile(mdd_blk, 0.95)
    assert q95_blk > q95_iid, f"q95 block={q95_blk}, q95 iid={q95_iid}"
    assert mdd_blk.mean() > mdd_iid.mean()
    print(f"  PASS  test_block_bootstrap_preserves_clustering  "
          f"(q95 iid={q95_iid:.3f}, q95 block={q95_blk:.3f})")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        test_no_drawdown_for_positive_returns,
        test_drawdown_handcomputed,
        test_episodes_single_full_recovery,
        test_episodes_unrecovered,
        test_episodes_min_depth_filter,
        test_tuw_stats_basic,
        test_tuw_stats_typical_portfolio,
        test_cdar_no_drawdown,
        test_cdar_geq_mdd_for_extreme_alpha,
        test_simulate_mdd_gaussian_shape,
        test_simulate_mdd_t_heavier_than_gaussian,
        test_simulate_mdd_fhs_with_pool,
        test_simulate_mdd_fhs_requires_pool,
        test_simulate_mdd_iid_bootstrap_basic,
        test_simulate_mdd_iid_bootstrap_requires_pool,
        test_simulate_mdd_block_bootstrap_basic,
        test_simulate_mdd_block_bootstrap_requires_pool,
        test_block_length_one_approximates_iid,
        test_block_bootstrap_preserves_clustering,
    ]
    print(f"Running {len(tests)} tests for src/drawdowns.py...\n")
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
