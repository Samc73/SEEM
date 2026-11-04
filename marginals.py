"""Truncated power law (TPL) marginal for drop sizes, with a free lower cutoff.

    P(s) = s^-kappa exp(-s / s_star) / Z      for s >= xmin

This is the same law as size_law.ln_density_tpl and notebook cell 36, but with
xmin as a parameter: size_law hardcodes XMIN = 1e-6 (tau units), and PE drops
live on a different scale.  Z and the CDF are numerical integrals on a log
grid, done in ln s (the integrand f(s) s varies smoothly there).

The grid only depends on (kappa, s_star, xmin), never on the data, so the
same CDF can be evaluated later on new events (the copula's PIT in step 7).

Used by 6_fit_marginals.py and 7_fit_copulas.py.  numpy + scipy.
"""
import numpy as np
from scipy.optimize import minimize

N_GRID = 4000
TAIL_SPAN = 60.0      # grid runs to xmin + 60 s_star; exp(-60) is negligible
KAPPA_RANGE = (-1.0, 4.0)
IMPOSSIBLE = 1e12     # finite stand-in for -inf, so Nelder-Mead can compare


def _grid(kappa, s_star, xmin):
    """Log grid on [xmin, upper], the ln integrand on it, and its cumulative
    integral (unnormalised, in units of exp(peak))."""
    upper = max(xmin + TAIL_SPAN * s_star, 10.0 * xmin)
    ln_s = np.linspace(np.log(xmin), np.log(upper), N_GRID)
    s = np.exp(ln_s)
    ln_integrand = (1.0 - kappa) * ln_s - s / s_star      # ln[f(s) s]
    peak = ln_integrand.max()
    w = np.exp(ln_integrand - peak)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(ln_s))])
    return ln_s, cumulative, peak


def ln_normalisation(kappa, s_star, xmin):
    _, cumulative, peak = _grid(kappa, s_star, xmin)
    return peak + np.log(cumulative[-1])


def ln_density(s, kappa, s_star, xmin):
    """Normalised ln P(s); -inf below xmin."""
    s = np.asarray(s, float)
    out = np.full(s.shape, -np.inf)
    ok = s >= xmin
    out[ok] = -kappa * np.log(s[ok]) - s[ok] / s_star - ln_normalisation(kappa, s_star, xmin)
    return out


def cdf(s, kappa, s_star, xmin):
    """P(S <= s); 0 below xmin, 1 beyond the grid."""
    ln_s, cumulative, _ = _grid(kappa, s_star, xmin)
    s = np.asarray(s, float)
    with np.errstate(divide='ignore'):
        x = np.log(np.maximum(s, 1e-300))
    return np.interp(x, ln_s, cumulative / cumulative[-1], left=0.0, right=1.0)


def ks_distance(s, kappa, s_star, xmin):
    """Kolmogorov-Smirnov distance between the sample (all >= xmin) and the law."""
    x = np.sort(s)
    model = cdf(x, kappa, s_star, xmin)
    n = len(x)
    return float(max(np.max(np.arange(1, n + 1) / n - model), np.max(model - np.arange(n) / n)))


def fit(s, xmin):
    """Maximum-likelihood (kappa, s_star) for the sizes s >= xmin.

    The likelihood only needs mean(ln s) and mean(s), so cost does not grow
    with the number of events.  Two starts (kappa 1.0 and 0.5) guard against
    the simplex stalling; the better one is kept.
    """
    s = np.asarray(s, float)
    s = s[s >= xmin]
    n = len(s)
    mean_ln_s = np.log(s).mean()
    mean_s = s.mean()

    def neg_ll(theta):
        kappa, ln_s_star = theta
        if not KAPPA_RANGE[0] < kappa < KAPPA_RANGE[1]:
            return IMPOSSIBLE
        value = (kappa * mean_ln_s + mean_s / np.exp(ln_s_star)
                 + ln_normalisation(kappa, np.exp(ln_s_star), xmin))
        return value if np.isfinite(value) else IMPOSSIBLE

    best = None
    for kappa0 in (1.0, 0.5):
        r = minimize(neg_ll, [kappa0, np.log(np.median(s))], method='Nelder-Mead',
                     options=dict(xatol=1e-5, fatol=1e-10, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    kappa, s_star = float(best.x[0]), float(np.exp(best.x[1]))
    return dict(kappa=kappa, s_star=s_star, n=n, ll=float(-best.fun * n),
                ks=ks_distance(s, kappa, s_star, xmin), ok=bool(best.success))
