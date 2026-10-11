"""Phase 3 dynamic-dependence engine: DCC correlation and t-copula tail risk.

The Phase 1-2 analysis is univariate - each portfolio is modelled as a single
return series, so the equity-bond dependence is fixed inside the portfolio
weights. The headline 2022 finding (equities and bonds falling together) is a
*dependence* event that a univariate model can only describe after the fact.
Phase 3 adds a dependence layer:

    estimate_dcc(resid1, resid2)
        Engle (2002) Dynamic Conditional Correlation, DCC(1,1). Given the
        standardised residuals of two univariate GARCH fits it estimates the
        correlation dynamics (a, b) by quasi-maximum-likelihood and returns
        the time-varying correlation rho_t. rho_t rising toward +1 in a
        crisis is the quantitative signature of a diversification breakdown.

    t_copula_tail_dependence(rho, nu)
        Lower/upper tail-dependence coefficient of a bivariate Student-t
        copula - the probability that one leg is in its tail given the other
        is. Unlike the Gaussian copula (zero tail dependence), the t-copula
        gives a positive, regime-relevant joint-crash probability.

References
----------
Engle, R. (2002). "Dynamic Conditional Correlation." Journal of Business &
    Economic Statistics 20(3), 339-350.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Engle DCC(1,1) for two series
# ---------------------------------------------------------------------------

def _dcc_recursion(a, b, eps, Qbar):
    """Run the DCC(1,1) recursion; return (0.5 * negloglik, rho_t array)."""
    n = len(eps)
    Q = Qbar.copy()
    ll = 0.0
    rho = np.empty(n)
    for t in range(n):
        d0 = np.sqrt(Q[0, 0])
        d1 = np.sqrt(Q[1, 1])
        r01 = Q[0, 1] / (d0 * d1)
        rho[t] = r01
        det_r = 1.0 - r01 * r01
        if det_r <= 1e-12:
            return 1e10, rho
        e0, e1 = eps[t, 0], eps[t, 1]
        quad = (e0 * e0 - 2.0 * r01 * e0 * e1 + e1 * e1) / det_r
        ll += np.log(det_r) + quad - (e0 * e0 + e1 * e1)
        outer = np.array([[e0 * e0, e0 * e1], [e0 * e1, e1 * e1]])
        Q = (1.0 - a - b) * Qbar + a * outer + b * Q
    return 0.5 * ll, rho


def _dcc_negloglik(params, eps, Qbar):
    a, b = params
    if a < 0.0 or b < 0.0 or a + b >= 0.9999:
        return 1e10
    return _dcc_recursion(a, b, eps, Qbar)[0]


def estimate_dcc(resid1, resid2):
    """Estimate an Engle DCC(1,1) model for two standardised-residual series.

    Parameters
    ----------
    resid1, resid2 : array-like
        Standardised residuals (each approximately mean 0, variance 1) from
        univariate GARCH fits of the two assets.

    Returns
    -------
    dict with keys ``a, b, persistence, rho_bar, rho_t, n``.
        ``rho_t`` is the time-varying conditional-correlation array;
        ``rho_bar`` the unconditional correlation; ``persistence`` = a + b.
    """
    e1 = np.asarray(resid1, dtype=float)
    e2 = np.asarray(resid2, dtype=float)
    mask = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[mask], e2[mask]
    if len(e1) < 50:
        raise ValueError('need at least 50 paired observations for DCC')

    eps = np.column_stack([e1, e2])
    rho_bar = float(np.corrcoef(e1, e2)[0, 1])
    Qbar = np.array([[1.0, rho_bar], [rho_bar, 1.0]])

    res = minimize(_dcc_negloglik, x0=np.array([0.03, 0.95]),
                   args=(eps, Qbar), method='L-BFGS-B',
                   bounds=[(0.0, 0.5), (0.0, 0.999)])
    a, b = float(res.x[0]), float(res.x[1])
    _, rho_t = _dcc_recursion(a, b, eps, Qbar)

    return {'a': a, 'b': b, 'persistence': a + b, 'rho_bar': rho_bar,
            'rho_t': rho_t, 'n': int(len(eps))}


# ---------------------------------------------------------------------------
# Student-t copula tail dependence
# ---------------------------------------------------------------------------

def t_copula_tail_dependence(rho, nu):
    """Tail-dependence coefficient of a bivariate Student-t copula.

    Returns lambda = 2 * T_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ), the
    (symmetric) lower = upper tail-dependence coefficient: the limiting
    probability that one margin is in its tail given the other is. A Gaussian
    copula has lambda = 0 for any rho < 1; the t-copula does not, which is why
    it is the right tool for joint-crash risk.
    """
    if rho >= 1.0:
        return 1.0
    if rho <= -1.0:
        return 0.0
    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * stats.t.cdf(arg, df=nu + 1.0))
