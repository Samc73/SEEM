"""Normalized maximum-likelihood comparison of candidate size distributions.

Each family supplies an unnormalized log-density ln f(s; theta) and a support
upper bound; the normalization over the fit window [xmin, hi] is computed
numerically on a log grid, so any shape symbolic regression proposes can be
promoted to a proper conditional density P(s | s >= xmin) and compared to the
canonical rivals on equal likelihood footing.

Families (theta layout):
  tpl        (kappa, ln s*)          s^-kappa exp(-s/s*)
  tpl_beta   (kappa, ln s*, ln beta) s^-kappa exp(-(s/s*)^beta)
  lognormal  (mu, ln sigma)
  powerlaw   (kappa,)                bounded at the observed maximum
  sqrtlog    (a, ln(c/smax - 1))     exp(-a*sqrt(ln(c/s))), support s < c
  invpow     (a, ln eps, ln(c/smax - 1))
             (c - s)^a / (s + eps)^a on (0, c) -- the SR-discovered form:
             power-law decay with small-size rounding eps and a hard ceiling
             c, the SAME exponent at both ends (SR's tied-exponent constraint)
  invpow4    (k, m, ln eps, ln(c/smax - 1))
             (c - s)^m / (s + eps)^k -- decoupled exponents, to test whether
             the tie m = k that SR found is real or an artifact of its
             complexity budget
"""
import numpy as np
from scipy.optimize import minimize


def _lnf(name, s, th, smax):
    if name == 'tpl':
        k, lss = th
        return -k * np.log(s) - s / np.exp(lss)
    if name == 'tpl_beta':
        k, lss, lb = th
        return -k * np.log(s) - (s / np.exp(lss)) ** np.exp(lb)
    if name == 'lognormal':
        mu, lsg = th
        sg = np.exp(lsg)
        return -np.log(s) - (np.log(s) - mu) ** 2 / (2 * sg * sg)
    if name == 'powerlaw':
        return -th[0] * np.log(s)
    if name == 'sqrtlog':
        a, lc = th
        c = smax * (1 + np.exp(lc))
        z = np.log(c / s)
        out = np.full_like(s, -np.inf)
        m = z > 0
        out[m] = -abs(a) * np.sqrt(z[m])
        return out
    if name == 'invpow':
        a, le, lc = th
        eps, c = np.exp(le), smax * (1 + np.exp(lc))
        out = np.full_like(s, -np.inf)
        m = s < c
        out[m] = a * (np.log(c - s[m]) - np.log(s[m] + eps))
        return out
    if name == 'invpow4':
        k, mm, le, lc = th
        eps, c = np.exp(le), smax * (1 + np.exp(lc))
        out = np.full_like(s, -np.inf)
        msk = s < c
        out[msk] = mm * np.log(c - s[msk]) - k * np.log(s[msk] + eps)
        return out
    raise ValueError(name)


def _upper(name, th, smax):
    if name in ('powerlaw',):
        return smax
    if name == 'sqrtlog':
        return smax * (1 + np.exp(th[1]))
    if name in ('invpow', 'invpow4'):
        return smax * (1 + np.exp(th[-1]))
    return None  # unbounded


NPAR = dict(tpl=2, tpl_beta=3, lognormal=2, powerlaw=1, sqrtlog=2,
            invpow=3, invpow4=4)
X0 = dict(tpl=[1.0, np.log(0.05)], tpl_beta=[1.0, np.log(0.05), 0.0],
          lognormal=[-5.0, np.log(2.0)], powerlaw=[1.2],
          sqrtlog=[3.0, np.log(0.5)],
          invpow=[1.0, np.log(3e-4), np.log(0.15)],
          invpow4=[1.0, 1.0, np.log(3e-4), np.log(0.15)])


def _grid(xmin, hi, bounded, npts=4000):
    """Log grid, refined toward a finite support ceiling when bounded."""
    if not bounded:
        return np.geomspace(xmin, hi, npts)
    g1 = np.geomspace(xmin, hi * 0.9, npts // 2)
    g2 = hi - np.geomspace(hi * 1e-7, hi * 0.1, npts // 2)[::-1]
    return np.unique(np.concatenate([g1, g2]))


def nll(name, s, th, xmin, smax=None, npts=4000):
    """Mean negative log-likelihood of the conditional density on [xmin, hi]."""
    smax = smax if smax is not None else s.max()
    hi = _upper(name, th, smax)
    hi_num = hi if hi is not None else max(100 * smax, 1.0)
    if hi is not None and hi <= xmin:
        return np.inf
    grid = _grid(xmin, hi_num, hi is not None, npts)
    with np.errstate(all='ignore'):
        lf_g = _lnf(name, grid, th, smax)
        lf_s = _lnf(name, s, th, smax)
    if not np.all(np.isfinite(lf_s)):
        return np.inf
    m = lf_g.max()
    lZ = m + np.log(np.trapezoid(np.exp(lf_g - m), grid))
    if not np.isfinite(lZ):
        return np.inf
    return float(lZ - np.mean(lf_s))


def fit(name, s, xmin, x0=None, smax=None):
    """MLE for one family on sizes s >= xmin. Returns dict with params/NLL."""
    s = np.asarray(s, float)
    s = s[s >= xmin]
    smax = smax if smax is not None else s.max()

    def f(th):
        return nll(name, s, th, xmin, smax)

    best = None
    starts = [np.asarray(x0 if x0 is not None else X0[name], float)]
    rng = np.random.default_rng(1)
    for _ in range(3):
        starts.append(starts[0] * rng.uniform(0.5, 1.8, len(starts[0])))
    for p0 in starts:
        r = minimize(f, p0, method='Nelder-Mead',
                     options=dict(xatol=1e-4, fatol=1e-8, maxiter=2000))
        if best is None or r.fun < best.fun:
            best = r
    return dict(name=name, theta=best.x, nll=best.fun, n=len(s),
                total_nll=best.fun * len(s),
                aic=2 * NPAR[name] + 2 * best.fun * len(s), ok=best.success)


def cv_nll(name, s, tid, xmin, smax=None):
    """Trajectory-blocked 2-fold CV: fit on even-parity trajectories, score
    held-out mean NLL on odd, and vice versa. Returns mean held-out NLL."""
    s = np.asarray(s, float)
    out, ns = 0.0, 0
    for par in (0, 1):
        tr, te = s[(tid % 2) == par], s[(tid % 2) != par]
        tr, te = tr[tr >= xmin], te[te >= xmin]
        if len(tr) < 50 or len(te) < 50:
            return np.nan
        r = fit(name, tr, xmin, smax=smax)
        out += nll(name, te, r['theta'], xmin,
                   smax if smax is not None else max(tr.max(), te.max())) * len(te)
        ns += len(te)
    return out / ns


def mean_size(name, th, xmin, smax, npts=4000):
    """E[s | s >= xmin] under the fitted conditional density."""
    hi = _upper(name, th, smax)
    hi_num = hi if hi is not None else max(100 * smax, 1.0)
    grid = _grid(xmin, hi_num, hi is not None, npts)
    with np.errstate(all='ignore'):
        lf = _lnf(name, grid, th, smax)
    m = lf.max()
    w = np.exp(lf - m)
    return float(np.trapezoid(grid * w, grid) / np.trapezoid(w, grid))


def sample(name, th, xmin, smax, size, rng, npts=4000):
    """Inverse-CDF sampling from the fitted conditional density."""
    hi = _upper(name, th, smax)
    hi_num = hi if hi is not None else max(100 * smax, 1.0)
    grid = _grid(xmin, hi_num, hi is not None, npts)
    with np.errstate(all='ignore'):
        lf = _lnf(name, grid, th, smax)
    w = np.exp(lf - lf.max())
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(grid))))
    cdf /= cdf[-1]
    return np.interp(rng.random(size), cdf, grid)
