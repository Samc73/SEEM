"""The stress-drop size law, its normalisation, and the rival truncated power law.

The law that the analysis is built on is, at a fixed state (u, tau),

    P(s | u, tau) = (s_c - s)^m / (s + eps)^k / Z(s_c)      for XMIN <= s < s_c,

i.e. a power law of exponent k that is rounded off below the scale eps and cut
off by a hard ceiling s_c, approached as a power m.  Only s_c depends on the
state; (k, m, eps) turn out to be global.  Nothing here is normalised
analytically: Z is a numerical integral on a log grid, which costs little and
keeps the code honest for any (k, m, eps).

The functions come in two flavours:

  * one-ceiling-at-a-time (`mean_neg_log_likelihood`) for the free per-cell fits
    of step 2, where (k, m, eps) change from call to call;
  * many-ceilings-at-once (`tabulate_ln_normalisation` + `ln_density_with_leak`)
    for step 3, where the shape is frozen so ln Z can be tabulated on a grid of
    ceilings once and interpolated per event.

The truncated power law s^-kappa exp(-s/s*) is the standard avalanche form and
is kept here only as the rival that step 2 measures the size law against.

Used by 2_fit_size_law.py, 3_fit_ceiling_law.py, 4_figures.py.  numpy + scipy.
"""
import numpy as np

# --- numerical conventions, all inherited from the original analysis ---------
XMIN = 1e-6              # smallest size kept; smaller drops are numerical noise
LEAK_FRACTION = 1e-3     # weight of the background component (see below)
BACKGROUND_MAX = 5.0     # background is log-uniform on [XMIN, BACKGROUND_MAX]
N_INTEGRATION_FREE = 4000   # log-grid points for ln Z when the shape is free
N_INTEGRATION_TABLE = 1500  # ... and when it is frozen and ln Z is tabulated
CEILING_GRID_POINTS = 240
CEILING_GRID_LO = 1e-4   # ln Z is tabulated for ceilings in this range only,
CEILING_GRID_HI = 50.0   # which comfortably brackets every fitted ceiling
IMPOSSIBLE = 1e12        # finite stand-in for -inf likelihood, so that the
                         # simplex optimiser can still compare two bad points


def integration_grid(ceiling, n_points):
    """Log grid on [XMIN, ceiling), refined towards the ceiling.

    Half the points are geometric from XMIN up to 0.9 * ceiling; the other half
    crowd into the last decade below the ceiling, where the factor
    (s_c - s)^m does all its varying and plain log spacing would miss it.
    """
    lower = np.geomspace(XMIN, ceiling * 0.9, n_points // 2)
    upper = ceiling - np.geomspace(ceiling * 1e-7, ceiling * 0.1, n_points // 2)[::-1]
    return np.unique(np.concatenate([lower, upper]))


def ln_unnormalised(s, k, m, eps, ceiling):
    """ln[(s_c - s)^m / (s + eps)^k], and the mask of sizes below the ceiling."""
    s = np.asarray(s, float)
    inside = s < ceiling
    out = np.full(s.shape, -np.inf)
    out[inside] = m * np.log(ceiling - s[inside]) - k * np.log(s[inside] + eps)
    return out, inside


def ln_normalisation(k, m, eps, ceiling, n_points):
    """ln Z = ln of the integral of the unnormalised density over [XMIN, s_c)."""
    grid = integration_grid(ceiling, n_points)
    ln_f, _ = ln_unnormalised(grid, k, m, eps, ceiling)
    peak = ln_f.max()
    return peak + np.log(np.trapezoid(np.exp(ln_f - peak), grid))


def mean_neg_log_likelihood(s, k, m, eps, ceiling, n_points=N_INTEGRATION_FREE):
    """Mean -ln P per event, for one cell with one ceiling.

    Returns a large finite number if any event sits at or above the ceiling:
    step 2 uses no leak, so such a parameter set is simply impossible.
    """
    if ceiling <= XMIN:
        return IMPOSSIBLE
    ln_f, inside = ln_unnormalised(s, k, m, eps, ceiling)
    if not inside.all():
        return IMPOSSIBLE
    ln_z = ln_normalisation(k, m, eps, ceiling, n_points)
    if not np.isfinite(ln_z):
        return IMPOSSIBLE
    return float(ln_z - ln_f.mean())


def tabulate_ln_normalisation(k, m, eps):
    """ln Z on a fixed grid of ceilings, for a frozen shape.

    Step 3 evaluates the ceiling at every event's own state, so ln Z would
    otherwise be recomputed millions of times.  Tabulating it on 240 log-spaced
    ceilings and interpolating in ln s_c is accurate to far better than the
    likelihood differences being measured, and makes the fits tractable.
    """
    ceilings = np.geomspace(CEILING_GRID_LO, CEILING_GRID_HI, CEILING_GRID_POINTS)
    table = np.array([ln_normalisation(k, m, eps, c, N_INTEGRATION_TABLE)
                      for c in ceilings])
    return np.log(ceilings), table


def ln_density_with_leak(s, ln_sc, k, m, eps, ln_ceiling_grid, ln_norm_table):
    """Per-event ln P(s) for a per-event ceiling, with the background leak.

    A smooth ceiling surface s_c(u, tau) cannot possibly sit above every single
    one of ~187,000 events; one event above it would send the likelihood to
    -inf and the fit would be driven entirely by the worst outlier.  So every
    model here -- including the per-cell reference table, so that the
    comparison is fair -- is mixed with the same fixed background:

        P = (1 - PI) * P_size_law + PI * (log-uniform on [XMIN, 5])

    with PI = 1e-3 held fixed, never fitted.  It is an outlier allowance, not a
    free parameter.

    Returns (ln P per event, mask of events that are below their own ceiling).
    """
    ln_sc = np.clip(ln_sc, ln_ceiling_grid[0], ln_ceiling_grid[-1])
    ceiling = np.exp(ln_sc)
    inside = s < ceiling
    ln_p = np.full(len(s), -np.inf)
    ln_p[inside] = (m * np.log(ceiling[inside] - s[inside])
                    - k * np.log(s[inside] + eps)
                    - np.interp(ln_sc[inside], ln_ceiling_grid, ln_norm_table))
    ln_background = -np.log(s) - np.log(np.log(BACKGROUND_MAX / XMIN))
    ln_mixture = np.logaddexp(np.log1p(-LEAK_FRACTION) + ln_p,
                              np.log(LEAK_FRACTION) + ln_background)
    return ln_mixture, inside


def ln_density_tpl(s, kappa, ln_s_star, s_max):
    """Normalised ln density of the truncated power law s^-kappa exp(-s/s*).

    It has no upper bound, so the normalising integral is taken out to
    100 * s_max, far beyond where the exponential has killed the density.
    """
    s = np.asarray(s, float)
    upper = max(100.0 * s_max, 1.0)
    grid = np.geomspace(XMIN, upper, N_INTEGRATION_FREE)
    s_star = np.exp(ln_s_star)
    ln_f_grid = -kappa * np.log(grid) - grid / s_star
    peak = ln_f_grid.max()
    ln_z = peak + np.log(np.trapezoid(np.exp(ln_f_grid - peak), grid))
    return -kappa * np.log(s) - s / s_star - ln_z


def mean_neg_log_likelihood_tpl(s, kappa, ln_s_star):
    """Mean -ln P per event for the truncated power law."""
    ln_p = ln_density_tpl(s, kappa, ln_s_star, s.max())
    if not np.all(np.isfinite(ln_p)):
        return IMPOSSIBLE
    return float(-ln_p.mean())
