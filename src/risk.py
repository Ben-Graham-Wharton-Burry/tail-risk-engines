"""Risk-metric module for Phase 1 of the tail-risk study.

Two analytic, closed-form distributional engines for 1-day-ahead VaR and ES:

    gaussian_var_es(mu, sigma, alpha)
        Normal returns: VaR = sigma * Phi^-1(alpha) - mu,
                        ES  = sigma * phi(Phi^-1(alpha)) / (1 - alpha) - mu.

    student_t_var_es(mu, sigma, nu, alpha)
        Student-t returns with location mu and TOTAL standard deviation sigma
        (so the scale parameter sigma_eff = sigma * sqrt((nu-2)/nu)):
                        VaR = sigma_eff * t_nu^-1(alpha) - mu,
                        ES  = sigma_eff * f_t(t_nu^-1(alpha); nu)
                              * (nu + t_nu^-1(alpha)^2)
                              / ((1 - alpha) * (nu - 1)) - mu.

Plus a rolling-window driver:

    rolling_var_es(returns, window, engine, alpha, nu=None)
        Vectorized rolling 1-day-ahead forecasts. The DataFrame index is the
        date the forecast applies to (i.e., a value at date d is the forecast
        made using returns through d-1 inclusive, to be tested against the
        realized return on d).

And a backtest helper:

    backtest_var_es(returns, forecasts, alpha)
        Exception rate, mean realized loss conditional on exception, and
        mean ES severity (realized loss - ES) on exception days.

Convention: losses are positive. L = -r. VaR is the loss exceeded with
probability 1 - alpha; ES is the conditional expected loss given VaR breach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Closed-form VaR and ES
# ---------------------------------------------------------------------------

def gaussian_var_es(mu, sigma, alpha):
    """Analytic VaR and ES for a Gaussian return distribution.

    Parameters
    ----------
    mu : float or array-like
        Mean of the return distribution.
    sigma : float or array-like
        Standard deviation of the return distribution.
    alpha : float
        Confidence level, e.g. 0.99. Loss exceeds VaR with probability 1 - alpha.

    Returns
    -------
    (VaR, ES) : tuple of float or array

    Notes
    -----
    For r ~ N(mu, sigma^2), loss L = -r:
        VaR_alpha = sigma * Phi^-1(alpha) - mu
        ES_alpha  = sigma * phi(Phi^-1(alpha)) / (1 - alpha) - mu
    where Phi and phi are the standard-normal CDF and PDF.
    """
    z = stats.norm.ppf(alpha)
    var = sigma * z - mu
    es = sigma * stats.norm.pdf(z) / (1.0 - alpha) - mu
    return var, es


def student_t_var_es(mu, sigma, nu, alpha):
    """Analytic VaR and ES for a Student-t return distribution.

    The model is r = mu + sigma_eff * X with X ~ standard t(nu), where
    sigma_eff is chosen so that Var(r) = sigma^2, i.e.
    sigma_eff = sigma * sqrt((nu - 2) / nu). This makes sigma directly
    interpretable as the total return standard deviation - matching the
    Gaussian engine and allowing apples-to-apples comparison.

    Parameters
    ----------
    mu : float or array-like
        Mean of the return distribution.
    sigma : float or array-like
        Total standard deviation of the return distribution.
    nu : int or float
        Degrees of freedom. Must satisfy nu > 2 (variance existence).
    alpha : float
        Confidence level, e.g. 0.99.

    Returns
    -------
    (VaR, ES) : tuple of float or array

    Notes
    -----
    Closed-form expressions (with sigma_eff = sigma * sqrt((nu-2)/nu) and
    q = t_nu^-1(alpha)):
        VaR_alpha = sigma_eff * q - mu
        ES_alpha  = sigma_eff * f_t(q; nu) * (nu + q^2)
                    / ((1 - alpha) * (nu - 1)) - mu
    The standardized-t ES formula appears in McNeil, Frey, & Embrechts (2015),
    *Quantitative Risk Management*, eq. (2.13)-style derivations.
    """
    if nu <= 2:
        raise ValueError(
            f"Variance of Student-t exists only for nu > 2, got nu={nu}."
        )

    q = stats.t.ppf(alpha, df=nu)
    f_q = stats.t.pdf(q, df=nu)
    sigma_eff = sigma * np.sqrt((nu - 2.0) / nu)

    var = sigma_eff * q - mu
    es_std = f_q * (nu + q ** 2) / ((1.0 - alpha) * (nu - 1.0))
    es = sigma_eff * es_std - mu
    return var, es


# ---------------------------------------------------------------------------
# Rolling-window driver
# ---------------------------------------------------------------------------

def rolling_var_es(returns, window, engine, alpha, nu=None, zero_mean=True):
    """Compute rolling 1-day-ahead VaR and ES forecasts.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns (simple returns), date-indexed.
    window : int
        Number of trailing return observations used to estimate the
        distribution parameters at each forecast date (e.g., 1250 for
        approximately 5 trading years).
    engine : {'gaussian', 't'}
        Distribution family used for the analytic VaR/ES.
    alpha : float
        Confidence level (e.g., 0.99).
    nu : int, optional
        Degrees of freedom for the Student-t engine. Ignored for Gaussian
        but required for the t engine.
    zero_mean : bool, default True
        If True (the standard for 1-day-ahead VaR/ES), the conditional mean is
        set to zero rather than the rolling sample mean. At a 1-day horizon the
        drift is negligible and hard to estimate, and a stale multi-year rolling
        mean can bias the VaR/ES level; zero-mean removes that bias. Set False to
        recover the previous rolling-mean behaviour.

    Returns
    -------
    pd.DataFrame
        Columns ``['VaR', 'ES']`` indexed by the **forecast date** - i.e., the
        value at date d is the forecast made at the end of day d-1 using the W
        returns immediately preceding d, evaluated against the realized return
        on d.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    if zero_mean:
        mu_hat = pd.Series(0.0, index=returns.index)
    else:
        mu_hat = returns.rolling(window).mean()
    sigma_hat = returns.rolling(window).std()  # ddof=1 by default

    if engine == 'gaussian':
        var, es = gaussian_var_es(mu_hat, sigma_hat, alpha)
    elif engine == 't':
        if nu is None:
            raise ValueError("nu must be provided for engine='t'.")
        var, es = student_t_var_es(mu_hat, sigma_hat, nu, alpha)
    else:
        raise ValueError(f"Unknown engine: {engine!r}. Use 'gaussian' or 't'.")

    # Rolling computations at date d use returns through d inclusive.
    # The forecast built from those parameters applies to d+1.
    # Shift by one so the index is the forecast date.
    forecasts = pd.DataFrame({'VaR': var, 'ES': es}).shift(1).dropna()
    return forecasts


# ---------------------------------------------------------------------------
# Formal VaR backtests (Kupiec & Christoffersen)
# ---------------------------------------------------------------------------

def _safe_log(x):
    """log(x) returning 0 when x == 0 (so 0 * log(0) terms vanish)."""
    return np.log(x) if x > 0 else 0.0


def kupiec_pof_test(exceptions, alpha):
    """Kupiec Proportion-of-Failures (POF) unconditional-coverage test.

    Tests H0: probability of a VaR exception equals 1 - alpha.

    Parameters
    ----------
    exceptions : array-like of int/bool
        Daily indicator series in chronological order. 1 = exception
        (realized loss exceeded VaR), 0 = no exception.
    alpha : float
        Confidence level used for the VaR forecasts (e.g., 0.99).

    Returns
    -------
    dict with keys ``n``, ``x``, ``pi_hat``, ``p_target``, ``lr_uc``,
    ``p_value``, ``reject_5pct``.

    Notes
    -----
    Reference: Kupiec, P.H. (1995). "Techniques for Verifying the Accuracy
    of Risk Measurement Models." *Journal of Derivatives* 3(2), 73-84.

    Under H0, LR_uc = -2 * log(L0/L1) ~ chi-square(1), where
        L0 = (1-p)^(n-x) * p^x,   p = 1 - alpha
        L1 = (1-pi_hat)^(n-x) * pi_hat^x,   pi_hat = x / n.
    """
    e = np.asarray(exceptions).astype(int)
    n = int(len(e))
    x = int(e.sum())
    p = 1.0 - alpha
    pi_hat = (x / n) if n > 0 else float('nan')

    # LR statistic with explicit handling of degenerate cases
    if n == 0:
        lr_uc = float('nan')
    elif x == 0:
        # pi_hat = 0; L1 = (1-0)^n = 1; LR = -2 * log(L0)
        lr_uc = -2.0 * n * np.log(1.0 - p)
    elif x == n:
        lr_uc = -2.0 * n * np.log(p)
    else:
        log_l0 = x * np.log(p) + (n - x) * np.log(1.0 - p)
        log_l1 = x * np.log(pi_hat) + (n - x) * np.log(1.0 - pi_hat)
        lr_uc = -2.0 * (log_l0 - log_l1)

    p_value = float(1.0 - stats.chi2.cdf(lr_uc, df=1)) if np.isfinite(lr_uc) else float('nan')

    return {
        'test': 'Kupiec POF (unconditional coverage)',
        'n': n,
        'x': x,
        'pi_hat': pi_hat,
        'p_target': p,
        'lr_uc': float(lr_uc),
        'p_value': p_value,
        'reject_5pct': bool(p_value < 0.05) if np.isfinite(p_value) else False,
    }


def christoffersen_cc_test(exceptions, alpha):
    """Christoffersen conditional-coverage test (joint of Kupiec + independence).

    Tests three nested hypotheses on the exception indicator series:

    * Unconditional coverage (Kupiec)     - LR_uc  ~ chi-square(1)
    * Independence (no clustering)        - LR_ind ~ chi-square(1)
    * Joint conditional coverage          - LR_cc  ~ chi-square(2)

    The independence test compares H0: probability of an exception is
    independent of yesterday's outcome (single transition probability pi)
    against H1: separate transition probabilities pi_01 and pi_11.

    Parameters
    ----------
    exceptions : array-like of int/bool
        Daily indicator series, chronological order.
    alpha : float
        Confidence level used for the VaR forecasts.

    Returns
    -------
    dict with keys ``n``, ``x``, ``n00``, ``n01``, ``n10``, ``n11``,
    ``pi_01``, ``pi_11``, ``lr_uc``, ``p_uc``, ``lr_ind``, ``p_ind``,
    ``lr_cc``, ``p_cc``, ``reject_uc_5pct``, ``reject_ind_5pct``,
    ``reject_cc_5pct``.

    Notes
    -----
    Reference: Christoffersen, P.F. (1998). "Evaluating Interval Forecasts."
    *International Economic Review* 39(4), 841-862.
    """
    e = np.asarray(exceptions).astype(int)
    n = int(len(e))

    # Transition counts (n-1 transitions for n observations)
    if n >= 2:
        prev = e[:-1]
        nxt = e[1:]
        n00 = int(((prev == 0) & (nxt == 0)).sum())
        n01 = int(((prev == 0) & (nxt == 1)).sum())
        n10 = int(((prev == 1) & (nxt == 0)).sum())
        n11 = int(((prev == 1) & (nxt == 1)).sum())
    else:
        n00 = n01 = n10 = n11 = 0

    n0 = n00 + n01
    n1 = n10 + n11
    tot = n00 + n01 + n10 + n11

    pi = ((n01 + n11) / tot) if tot > 0 else 0.0
    pi_01 = (n01 / n0) if n0 > 0 else 0.0
    pi_11 = (n11 / n1) if n1 > 0 else 0.0

    # Log-likelihoods
    log_l0 = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    log_l1 = (n00 * _safe_log(1 - pi_01)
              + n01 * _safe_log(pi_01)
              + n10 * _safe_log(1 - pi_11)
              + n11 * _safe_log(pi_11))

    lr_ind = -2.0 * (log_l0 - log_l1)
    # Clip floating-point noise: lr_ind should be >= 0
    if -1e-10 < lr_ind < 0:
        lr_ind = 0.0
    p_ind = float(1.0 - stats.chi2.cdf(lr_ind, df=1)) if np.isfinite(lr_ind) else float('nan')

    # Pull Kupiec (unconditional)
    uc = kupiec_pof_test(e, alpha)
    lr_uc = uc['lr_uc']
    p_uc = uc['p_value']

    # Joint LR_cc = LR_uc + LR_ind under chi-square(2)
    lr_cc = lr_uc + lr_ind
    p_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2)) if np.isfinite(lr_cc) else float('nan')

    return {
        'test': 'Christoffersen conditional coverage',
        'n': n,
        'x': int(e.sum()),
        'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11,
        'pi_01': pi_01, 'pi_11': pi_11,
        'lr_uc': float(lr_uc), 'p_uc': float(p_uc) if p_uc is not None else float('nan'),
        'lr_ind': float(lr_ind), 'p_ind': p_ind,
        'lr_cc': float(lr_cc), 'p_cc': p_cc,
        'reject_uc_5pct': bool(p_uc < 0.05) if np.isfinite(p_uc) else False,
        'reject_ind_5pct': bool(p_ind < 0.05) if np.isfinite(p_ind) else False,
        'reject_cc_5pct': bool(p_cc < 0.05) if np.isfinite(p_cc) else False,
    }


# ---------------------------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------------------------

def backtest_var_es(returns, forecasts, alpha):
    """Backtest 1-day-ahead VaR and ES forecasts.

    Parameters
    ----------
    returns : pd.Series
        Realized returns, date-indexed.
    forecasts : pd.DataFrame
        Must have columns ``'VaR'`` and ``'ES'``, indexed by the date the
        forecast applies to (as produced by :func:`rolling_var_es`).
    alpha : float
        Confidence level used for the forecasts.

    Returns
    -------
    dict
        ``n_obs`` : number of forecast/realized pairs evaluated.
        ``n_exceptions`` : days where realized loss exceeded forecast VaR.
        ``exception_rate`` : n_exceptions / n_obs.
        ``expected_exception_rate`` : 1 - alpha.
        ``mean_loss_given_exception`` : mean realized loss on exception days.
        ``mean_es_severity`` : mean of (realized_loss - ES) on exception days.
            A positive value indicates ES understates the realized tail loss.
        ``kupiec`` : full Kupiec POF test result dict.
        ``christoffersen`` : full Christoffersen CC test result dict.
        ``exception_series`` : pd.Series of 0/1 exception indicators indexed by date.
    """
    df = forecasts.copy()
    df['return'] = returns.reindex(df.index)
    df = df.dropna(subset=['VaR', 'ES', 'return'])
    df['loss'] = -df['return']
    df['exception'] = df['loss'] > df['VaR']

    n_obs = len(df)
    n_exc = int(df['exception'].sum())

    if n_exc > 0:
        exc = df[df['exception']]
        mean_loss = float(exc['loss'].mean())
        mean_severity = float((exc['loss'] - exc['ES']).mean())
    else:
        mean_loss = float('nan')
        mean_severity = float('nan')

    exc_indicator = df['exception'].astype(int)
    kupiec = kupiec_pof_test(exc_indicator.values, alpha)
    christ = christoffersen_cc_test(exc_indicator.values, alpha)

    return {
        'n_obs': n_obs,
        'n_exceptions': n_exc,
        'exception_rate': n_exc / n_obs if n_obs else float('nan'),
        'expected_exception_rate': 1.0 - alpha,
        'mean_loss_given_exception': mean_loss,
        'mean_es_severity': mean_severity,
        'kupiec': kupiec,
        'christoffersen': christ,
        'exception_series': exc_indicator,
    }


# ---------------------------------------------------------------------------
# Formal ES backtests (McNeil-Frey & Acerbi-Szekely)
# ---------------------------------------------------------------------------

def mcneil_frey_es_test(returns, forecasts, alpha=0.99,
                        n_bootstrap=10_000, seed=42):
    """McNeil-Frey (2000) one-sided bootstrap test on exceedance residuals.

    Tests whether ES under-estimates losses on breach days. Operates on the
    exceedance residual e_t = L_t - ES_t restricted to t such that L_t > VaR_t.

    Parameters
    ----------
    returns : pd.Series
        Realised returns, date-indexed.
    forecasts : pd.DataFrame
        Must have columns 'VaR' and 'ES', indexed by forecast date.
    alpha : float
        Confidence level used for the VaR forecasts (only stored for output
        labelling; not used in the bootstrap math).
    n_bootstrap : int
        Number of bootstrap replications B.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys ``test``, ``n_exc``, ``mean_residual``, ``p_value``,
    ``reject_5pct``, ``ci_lower``, ``ci_upper``, ``n_bootstrap``.

    Notes
    -----
    H0: E[L_t - ES_t | L_t > VaR_t] = 0  (ES correctly calibrated on severity).
    H1: E[L_t - ES_t | L_t > VaR_t] > 0  (one-sided: ES under-estimates).

    Bootstrap procedure:
      1. Compute observed sample mean bar_eps of the n_e exceedance residuals.
      2. Centre residuals to mean zero (mimicking H0).
      3. Resample n_e values with replacement from the centred series; compute
         the bootstrap mean. Repeat B times.
      4. p-value = fraction of bootstrap means >= bar_eps.

    Also reports a non-parametric 95% bootstrap CI on the raw (uncentred)
    residual mean - useful for magnitude interpretation alongside the p-value.

    Reference: McNeil, A.J., & Frey, R. (2000). "Estimation of Tail-Related
    Risk Measures for Heteroscedastic Financial Time Series: An Extreme Value
    Approach." *Journal of Empirical Finance*, 7(3-4), 271-300.
    """
    df = forecasts.copy()
    df['return'] = returns.reindex(df.index)
    df = df.dropna(subset=['VaR', 'ES', 'return'])
    df['loss'] = -df['return']
    df['exception'] = df['loss'] > df['VaR']

    exc = df[df['exception']]
    n_exc = int(len(exc))

    if n_exc < 2:
        return {
            'test': 'McNeil-Frey ES bootstrap',
            'n_exc': n_exc,
            'mean_residual': float('nan'),
            'p_value': float('nan'),
            'reject_5pct': False,
            'ci_lower': float('nan'),
            'ci_upper': float('nan'),
            'n_bootstrap': n_bootstrap,
        }

    residuals = (exc['loss'] - exc['ES']).values
    obs_mean = float(residuals.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_exc, size=(n_bootstrap, n_exc))

    centered = residuals - obs_mean
    boot_means_h0 = centered[idx].mean(axis=1)
    p_value = float((boot_means_h0 >= obs_mean).mean())

    boot_means_raw = residuals[idx].mean(axis=1)
    ci_lower = float(np.percentile(boot_means_raw, 2.5))
    ci_upper = float(np.percentile(boot_means_raw, 97.5))

    return {
        'test': 'McNeil-Frey ES bootstrap',
        'n_exc': n_exc,
        'mean_residual': obs_mean,
        'p_value': p_value,
        'reject_5pct': bool(p_value < 0.05),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_bootstrap': n_bootstrap,
    }


def acerbi_szekely_z2(returns, forecasts, alpha=0.99,
                      n_bootstrap=5_000, seed=42):
    """Acerbi-Szekely (2014) Z2 statistic for ES backtesting.

    Single normalised summary statistic with non-parametric bootstrap CI.

    Z2 = -1/(T*(1-alpha)) * sum_t [L_t * I_t / ES_t] + 1

    Under H0 (ES correctly calibrated), E[Z2] = 0.
        Z2 < 0 -> ES under-estimates breach losses (regulatory concern).
        Z2 > 0 -> ES over-estimates breach losses (conservative model).

    Parameters
    ----------
    returns : pd.Series
        Realised returns, date-indexed.
    forecasts : pd.DataFrame
        Must have columns 'VaR' and 'ES'.
    alpha : float
        Confidence level used for the forecasts.
    n_bootstrap : int
        Number of bootstrap replications (default 5_000 to keep memory modest
        when T is large; the loop is O(n_bootstrap * T)).
    seed : int

    Returns
    -------
    dict with keys ``test``, ``n_obs``, ``n_exc``, ``Z2``, ``ci_lower``,
    ``ci_upper``, ``reject_5pct`` (True if 95% bootstrap CI excludes 0),
    ``n_bootstrap``.

    Notes
    -----
    Reference: Acerbi, C., & Szekely, B. (2014). "Back-Testing Expected
    Shortfall." *Risk*, 27(11), 76-81.

    The original paper derives critical values by simulating the assumed
    predictive distribution F_t under H0; for a model-free comparison across
    engines we instead report Z2 with a non-parametric bootstrap CI built by
    resampling the per-day contributions L_t * I_t / ES_t with replacement.
    For one-sided formal hypothesis testing, see :func:`mcneil_frey_es_test`.
    """
    df = forecasts.copy()
    df['return'] = returns.reindex(df.index)
    df = df.dropna(subset=['VaR', 'ES', 'return'])
    df['loss'] = -df['return']
    df['exception'] = (df['loss'] > df['VaR']).astype(int)

    T = int(len(df))
    p = 1.0 - alpha
    n_exc = int(df['exception'].sum())

    if T < 2 or n_exc == 0:
        return {
            'test': 'Acerbi-Szekely Z2',
            'n_obs': T,
            'n_exc': n_exc,
            'Z2': float('nan'),
            'ci_lower': float('nan'),
            'ci_upper': float('nan'),
            'reject_5pct': False,
            'n_bootstrap': n_bootstrap,
        }

    df['contrib'] = df['loss'] * df['exception'] / df['ES']
    contribs = df['contrib'].values
    z2_obs = float(-contribs.sum() / (T * p) + 1.0)

    rng = np.random.default_rng(seed)
    boot_z2 = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(contribs, size=T, replace=True)
        boot_z2[i] = -sample.sum() / (T * p) + 1.0

    ci_lower = float(np.percentile(boot_z2, 2.5))
    ci_upper = float(np.percentile(boot_z2, 97.5))
    reject = (ci_upper < 0.0) or (ci_lower > 0.0)

    return {
        'test': 'Acerbi-Szekely Z2',
        'n_obs': T,
        'n_exc': n_exc,
        'Z2': z2_obs,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'reject_5pct': bool(reject),
        'n_bootstrap': n_bootstrap,
    }


def backtest_es(returns, forecasts, alpha,
                n_bootstrap_mf=10_000, n_bootstrap_z2=5_000, seed=42):
    """Run both ES backtests in one call.

    Parallel to :func:`backtest_var_es`, but for Expected Shortfall.

    Parameters
    ----------
    returns : pd.Series
    forecasts : pd.DataFrame with columns 'VaR', 'ES'
    alpha : float
    n_bootstrap_mf, n_bootstrap_z2 : int
    seed : int

    Returns
    -------
    dict
        ``mcneil_frey``    : full McNeil-Frey result dict.
        ``acerbi_szekely`` : full Acerbi-Szekely Z2 result dict.
        ``n_exc``          : number of breach days (consistent across both).
    """
    mf = mcneil_frey_es_test(returns, forecasts, alpha=alpha,
                             n_bootstrap=n_bootstrap_mf, seed=seed)
    z2 = acerbi_szekely_z2(returns, forecasts, alpha=alpha,
                           n_bootstrap=n_bootstrap_z2, seed=seed)
    return {
        'mcneil_frey': mf,
        'acerbi_szekely': z2,
        'n_exc': mf.get('n_exc'),
    }
