"""Truncated-power-law estimation for avalanche (yield-event) size statistics.

Model:  P(s) = s^(-kappa) * exp(-(s/s_star)^beta) / Z   for s >= xmin.

The normalisation Z is computed numerically on a log grid, so kappa is not
restricted to the range where the incomplete-gamma closed form is convenient,
and the stretched cutoff (beta != 1) costs nothing extra.

MLE notes: the log-likelihood per event is
    -kappa*ln s - (s/s_star)^beta - ln Z(kappa, s_star, beta)
Fits are exposed at three levels: joint (kappa, s_star) for pooled bands,
scalar s_star at fixed kappa for per-voxel fields, and a (kappa, s_star, beta)
variant to test the cutoff shape itself.
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar


def _logZ(kappa, s_star, xmin, beta=1.0, npts=3000):
    hi = max(50.0 * s_star, 5.0, 100.0 * xmin)
    s = np.logspace(np.log10(xmin), np.log10(hi), npts)
    ln_f = -kappa * np.log(s) - (s / s_star) ** beta
    m = ln_f.max()
    return m + np.log(np.trapezoid(np.exp(ln_f - m), s))


def nll(s, kappa, s_star, xmin, beta=1.0):
    """Mean negative log-likelihood of sizes s >= xmin."""
    s = s[s >= xmin]
    if len(s) == 0 or s_star <= 0 or kappa < -1 or kappa > 4:
        return np.inf
    return (kappa * np.mean(np.log(s)) + np.mean((s / s_star) ** beta)
            + _logZ(kappa, s_star, xmin, beta))


def fit_joint(s, xmin, kappa0=1.3, s0=0.1):
    """Joint MLE of (kappa, s_star) with beta = 1."""
    s = s[s >= xmin]

    def f(theta):
        return nll(s, theta[0], np.exp(theta[1]), xmin)

    r = minimize(f, [kappa0, np.log(s0)], method='Nelder-Mead',
                 options=dict(xatol=1e-4, fatol=1e-7, maxiter=800))
    return dict(kappa=r.x[0], s_star=float(np.exp(r.x[1])),
                nll=r.fun, n=len(s), ok=r.success)


def fit_joint_beta(s, xmin, kappa0=1.3, s0=0.1):
    """Joint MLE of (kappa, s_star, beta) — tests the cutoff shape."""
    s = s[s >= xmin]

    def f(theta):
        k, ls, lb = theta
        b = np.exp(lb)
        if b < 0.2 or b > 5:
            return np.inf
        return nll(s, k, np.exp(ls), xmin, beta=b)

    r = minimize(f, [kappa0, np.log(s0), 0.0], method='Nelder-Mead',
                 options=dict(xatol=1e-4, fatol=1e-7, maxiter=1500))
    return dict(kappa=r.x[0], s_star=float(np.exp(r.x[1])),
                beta=float(np.exp(r.x[2])), nll=r.fun, n=len(s), ok=r.success)


def fit_sstar(s, kappa, xmin, lo=1e-3, hi=20.0):
    """Scalar MLE of s_star at fixed kappa (per-voxel workhorse)."""
    s = s[s >= xmin]
    if len(s) < 30:
        return np.nan

    def f(ls):
        return nll(s, kappa, np.exp(ls), xmin)

    r = minimize_scalar(f, bounds=(np.log(lo), np.log(hi)), method='bounded',
                        options=dict(xatol=1e-4))
    return float(np.exp(r.x))


def fit_sstar_boot(s, kappa, xmin, n_boot=80, seed=0):
    """s_star with a bootstrap standard error on ln s_star."""
    s = s[s >= xmin]
    est = fit_sstar(s, kappa, xmin)
    if not np.isfinite(est):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sb = s[rng.integers(0, len(s), len(s))]
        v = fit_sstar(sb, kappa, xmin)
        if np.isfinite(v):
            vals.append(np.log(v))
    return est, (np.std(vals) if len(vals) >= 20 else np.nan)


def nll_lognormal(s, xmin):
    """Best-fit mean NLL of a truncated lognormal — the standard rival model.
    Returns (nll, mu, sigma)."""
    from scipy.stats import norm
    s = s[s >= xmin]
    ln = np.log(s)

    def f(theta):
        mu, lsg = theta
        sg = np.exp(lsg)
        z = (np.log(xmin) - mu) / sg
        tail = max(1.0 - norm.cdf(z), 1e-300)
        return (np.mean((ln - mu) ** 2) / (2 * sg ** 2) + np.log(sg)
                + 0.5 * np.log(2 * np.pi) + np.mean(ln) + np.log(tail))

    r = minimize(f, [np.mean(ln), np.log(np.std(ln))], method='Nelder-Mead')
    return r.fun, r.x[0], float(np.exp(r.x[1]))
