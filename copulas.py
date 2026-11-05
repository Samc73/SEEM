"""Bivariate copula families, their fitting, and a held-out contest between them.

Families:
    independence      c = 1 (baseline, ln L = 0)
    gaussian (rho)    no tail dependence
    t (rho, nu)       equal tail dependence in both corners
    clayton (theta)   lower-tail dependence only
    gumbel (theta)    upper-tail dependence only
    frank (theta)     no tail dependence
    bb1 (theta, delta)  Clayton-Gumbel: lambda_L = 2^(-1/(theta delta)),
                        lambda_U = 2 - 2^(1/delta).  delta = 1 is Clayton(theta),
                        theta -> 0 is Gumbel(delta).
    bb7 (theta, delta)  Joe-Clayton:    lambda_L = 2^(-1/delta),
                        lambda_U = 2 - 2^(1/theta).  theta = 1 is Clayton(delta),
                        delta -> 0 is Joe(theta).
    survival_<name>   the 180-degree rotation: c(1-u, 1-v); swaps the two tails.
BB1 and BB7 have both tails already; the rotation swaps which parameter sets
which tail, so both orientations are fitted.  Clayton, Gumbel, BB1 and BB7
only model positive dependence; run_contest stops if Kendall's tau <= 0.

Densities are closed form.  BB1 and BB7 are Archimedean, C = psi(phi(u) + phi(v)),
so c = psi''(s) phi'(u) phi'(v) with s = phi(u) + phi(v); everything is done
in logs because phi blows up in the corners.  test_copulas.py checks every
density against statsmodels (where it has the family), against the mixed
derivative of our own CDF, for normalisation, and against its limits.

The contest (run_contest) fits every family two ways:
    'cml' ranks (pseudo-observations): needs no marginal model -- the reference
    'ifm' PIT through fitted parametric marginals (here the TPLs)
and scores each on held-out data with step 3's 2-fold split by trajectory
(even ids vs odd).  Error bars come from resampling TRAJECTORIES with
replacement and re-summing the held-out per-event scores; this measures how
much the score depends on which trajectories were sampled, not refitting noise.
"""
import itertools
import numpy as np
from scipy import integrate, stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, ndtri, stdtrit

CLIP = 1e-10
FAMILIES = ('independence', 'gaussian', 't', 'clayton', 'gumbel', 'frank',
            'survival_clayton', 'survival_gumbel',
            'bb1', 'survival_bb1', 'bb7', 'survival_bb7')
BASE_PARAMETERS = dict(independence=(), gaussian=('rho',), t=('rho', 'nu'),
                       clayton=('theta',), gumbel=('theta',), frank=('theta',),
                       bb1=('theta', 'delta'), bb7=('theta', 'delta'))
# box bounds of every parameter; a fit that ends on one is flagged
BOUNDS = dict(gaussian=[(-0.995, 0.995)], t=[(-0.995, 0.995), (1.0, 300.0)],
              clayton=[(1e-4, 60.0)], gumbel=[(1.0, 60.0)], frank=[(-60.0, 60.0)],
              bb1=[(1e-3, 30.0), (1.0, 30.0)], bb7=[(1.0, 30.0), (1e-3, 30.0)])
# starting grids for the two-parameter fits (every combination is tried)
STARTS = dict(t=[None, (3.0, 8.0, 30.0)],           # rho starts from the gaussian fit
              bb1=[(0.1, 0.5, 2.0), (1.2, 2.0, 4.0)],
              bb7=[(1.2, 2.0, 4.0), (0.3, 1.0, 3.0)])
BOUND_TOLERANCE = 1e-3


def base_family(family):
    return family[len('survival_'):] if family.startswith('survival_') else family


def n_parameters(family):
    return len(BASE_PARAMETERS[base_family(family)])


def _clip(u):
    return np.clip(u, CLIP, 1.0 - CLIP)


def _log_expm1(y):
    """log(e^y - 1) for y > 0, without overflow for large y."""
    y = np.asarray(y, float)
    return np.where(y > 30, y + np.log1p(-np.exp(-np.minimum(y, 700))), np.log(np.expm1(np.minimum(y, 30))))


# --- log densities ------------------------------------------------------------
def _ln_gaussian(u, v, rho):
    x, y = ndtri(u), ndtri(v)
    r2 = 1.0 - rho * rho
    return -0.5 * np.log(r2) - (rho * rho * (x * x + y * y) - 2.0 * rho * x * y) / (2.0 * r2)


def _ln_t(u, v, rho, nu):
    x, y = stdtrit(nu, u), stdtrit(nu, v)
    r2 = 1.0 - rho * rho
    return (gammaln((nu + 2) / 2) + gammaln(nu / 2) - 2 * gammaln((nu + 1) / 2)
            - 0.5 * np.log(r2)
            - (nu + 2) / 2 * np.log1p((x * x + y * y - 2 * rho * x * y) / (nu * r2))
            + (nu + 1) / 2 * (np.log1p(x * x / nu) + np.log1p(y * y / nu)))


def _ln_clayton(u, v, theta):
    lu, lv = np.log(u), np.log(v)
    return (np.log1p(theta) - (1 + theta) * (lu + lv)
            - (2 + 1 / theta) * np.log(np.exp(-theta * lu) + np.exp(-theta * lv) - 1))


def _ln_gumbel(u, v, theta):
    a, b = -np.log(u), -np.log(v)
    s = a ** theta + b ** theta
    big_a = s ** (1 / theta)
    return (-big_a + a + b + (theta - 1) * (np.log(a) + np.log(b))
            + (2 / theta - 2) * np.log(s) + np.log1p((theta - 1) / big_a))


def _ln_frank(u, v, theta):
    if abs(theta) < 1e-8:
        return np.zeros_like(u)
    a = -np.expm1(-theta)                         # 1 - e^-theta
    denominator = a - np.expm1(-theta * u) * np.expm1(-theta * v)
    return np.log(theta * a) - theta * (u + v) - 2 * np.log(np.abs(denominator))


def _ln_bb1(u, v, theta, delta):
    """phi(t) = (t^-theta - 1)^delta,  psi(s) = (1 + s^(1/delta))^(-1/theta)."""
    def ln_x(t):                                  # ln(t^-theta - 1)
        return _log_expm1(-theta * np.log(t))

    def ln_abs_phi_prime(t):                      # ln[delta theta (t^-theta-1)^(delta-1) t^(-theta-1)]
        return np.log(delta * theta) + (delta - 1) * ln_x(t) - (theta + 1) * np.log(t)

    ln_s = np.logaddexp(delta * ln_x(u), delta * ln_x(v))
    ln_w = ln_s / delta                           # w = s^(1/delta)
    ln_1pw = np.logaddexp(0.0, ln_w)
    # psi''(s) = s^(1/delta-2) (1+w)^(-1/theta-2) / (theta delta)
    #            * [ ((1/theta+1)/delta + 1 - 1/delta) w + (1 - 1/delta) ]
    coef_w = (1 / theta + 1) / delta + 1 - 1 / delta
    with np.errstate(divide='ignore'):
        ln_bracket = np.logaddexp(np.log(coef_w) + ln_w, np.log(1 - 1 / delta))
    ln_psi2 = (-np.log(theta * delta) + (1 / delta - 2) * ln_s
               + (-1 / theta - 2) * ln_1pw + ln_bracket)
    return ln_psi2 + ln_abs_phi_prime(u) + ln_abs_phi_prime(v)


def _ln_bb7(u, v, theta, delta):
    """phi(t) = (1 - (1-t)^theta)^-delta - 1,  psi(s) = 1 - (1 - (1+s)^(-1/delta))^(1/theta)."""
    def ln_x(t):                                  # ln(1 - (1-t)^theta)
        ln_y = theta * np.log1p(-t)               # ln (1-t)^theta
        # near t = 1, (1-t)^theta is tiny and 1 - it rounds to 1: use log1p;
        # near t = 0 it is close to 1: use expm1
        return np.where(ln_y < np.log(0.5), np.log1p(-np.exp(ln_y)),
                        np.log(-np.expm1(np.minimum(ln_y, -1e-300))))

    def ln_abs_phi_prime(t):                      # ln[delta theta (1-t)^(theta-1) x^(-delta-1)]
        return np.log(delta * theta) + (theta - 1) * np.log1p(-t) - (delta + 1) * ln_x(t)

    ln_s = np.logaddexp(_log_expm1(-delta * ln_x(u)), _log_expm1(-delta * ln_x(v)))
    ln_1ps = np.logaddexp(0.0, ln_s)              # L = ln(1 + s)
    ln_z = -ln_1ps / delta                        # z = (1+s)^(-1/delta)
    ln_1mz = np.log(-np.expm1(ln_z))
    a = 1 / theta - 1
    # psi''(s) = z (1-z)^(a-1) / (theta delta (1+s)^2) * [ (1-z)(1 + 1/delta) - a z / delta ]
    bracket = np.exp(ln_1mz) * (1 + 1 / delta) - a * np.exp(ln_z) / delta
    ln_psi2 = (-np.log(theta * delta) + ln_z + (a - 1) * ln_1mz - 2 * ln_1ps + np.log(bracket))
    return ln_psi2 + ln_abs_phi_prime(u) + ln_abs_phi_prime(v)


_LOG_DENSITY = dict(gaussian=_ln_gaussian, t=_ln_t, clayton=_ln_clayton, gumbel=_ln_gumbel,
                    frank=_ln_frank, bb1=_ln_bb1, bb7=_ln_bb7)


def log_density(family, params, u, v):
    u, v = _clip(np.asarray(u, float)), _clip(np.asarray(v, float))
    if family == 'independence':
        return np.zeros_like(u)
    if family.startswith('survival_'):
        return log_density(base_family(family), params, 1 - u, 1 - v)
    return _LOG_DENSITY[family](u, v, *params)


# --- copula CDF (finite-q tail checks and tests) ------------------------------
def cdf(family, params, u, v):
    if family == 'independence':
        return u * v
    if family.startswith('survival_'):
        return u + v - 1 + cdf(base_family(family), params, 1 - u, 1 - v)
    if family == 'gaussian':
        rho = params[0]
        return stats.multivariate_normal(cov=[[1, rho], [rho, 1]]).cdf(
            np.stack([ndtri(u), ndtri(v)], axis=-1))
    if family == 't':
        rho, nu = params
        return stats.multivariate_t(shape=[[1, rho], [rho, 1]], df=nu).cdf(
            np.stack([stdtrit(nu, u), stdtrit(nu, v)], axis=-1))
    if family == 'bb1':
        theta, delta = params
        s = np.expm1(-theta * np.log(u)) ** delta + np.expm1(-theta * np.log(v)) ** delta
        return (1 + s ** (1 / delta)) ** (-1 / theta)
    if family == 'bb7':
        theta, delta = params
        x_u = -np.expm1(theta * np.log1p(-u))
        x_v = -np.expm1(theta * np.log1p(-v))
        z = (x_u ** -delta + x_v ** -delta - 1) ** (-1 / delta)
        return 1 - (1 - z) ** (1 / theta)
    theta = params[0]
    if family == 'clayton':
        return (u ** -theta + v ** -theta - 1) ** (-1 / theta)
    if family == 'gumbel':
        return np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta))
    if family == 'frank':
        return -np.log1p(np.expm1(-theta * u) * np.expm1(-theta * v) / np.expm1(-theta)) / theta
    raise ValueError(family)


# --- implied dependence measures ----------------------------------------------
def kendall_tau(family, params):
    if family == 'independence':
        return 0.0
    if family.startswith('survival_'):
        return kendall_tau(base_family(family), params)          # rotation keeps tau
    if family in ('gaussian', 't'):
        return 2 / np.pi * np.arcsin(params[0])
    if family == 'bb1':
        theta, delta = params
        return 1 - 2 / (delta * (theta + 2))
    if family == 'bb7':
        # Archimedean: tau = 1 + 4 int_0^1 phi(t) / phi'(t) dt
        theta, delta = params

        def ratio(t):
            x = 1 - (1 - t) ** theta
            return -(x - x ** (delta + 1)) / (delta * theta * (1 - t) ** (theta - 1))
        return 1 + 4 * integrate.quad(ratio, 0, 1, limit=200)[0]
    theta = params[0]
    if family == 'clayton':
        return theta / (theta + 2)
    if family == 'gumbel':
        return 1 - 1 / theta
    if family == 'frank':
        debye = integrate.quad(lambda t: t / np.expm1(t) if t != 0 else 1.0, 0, theta)[0] / theta
        return 1 - 4 / theta * (1 - debye)
    raise ValueError(family)


def tail_dependence(family, params):
    """Asymptotic (lambda_lower, lambda_upper)."""
    if family.startswith('survival_'):
        lower, upper = tail_dependence(base_family(family), params)
        return upper, lower
    if family == 't':
        rho, nu = params
        lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
        return float(lam), float(lam)
    if family == 'clayton':
        return float(2 ** (-1 / params[0])), 0.0
    if family == 'gumbel':
        return 0.0, float(2 - 2 ** (1 / params[0]))
    if family == 'bb1':
        theta, delta = params
        return float(2 ** (-1 / (theta * delta))), float(2 - 2 ** (1 / delta))
    if family == 'bb7':
        theta, delta = params
        return float(2 ** (-1 / delta)), float(2 - 2 ** (1 / theta))
    return 0.0, 0.0


def joint_tail(family, params, q):
    """Model's finite-q tail measures: P(V <= q | U <= q) and P(V > 1-q | U > 1-q)."""
    lower = cdf(family, params, q, q) / q
    p = 1 - q
    upper = (1 - 2 * p + cdf(family, params, p, p)) / q
    return float(lower), float(upper)


def empirical_joint_tail(u, v, q):
    lower = np.mean((u <= q) & (v <= q)) / q
    upper = np.mean((u > 1 - q) & (v > 1 - q)) / q
    return float(lower), float(upper)


# --- fitting --------------------------------------------------------------------
def _at_bound(base, params):
    """Names of parameters that ended on (within tolerance of) a box bound."""
    flags = []
    for name, value, (lo, hi) in zip(BASE_PARAMETERS[base], params, BOUNDS[base]):
        tolerance = BOUND_TOLERANCE * max(1.0, abs(lo), abs(hi))
        if value - lo < tolerance:
            flags.append('%s at lower bound %g' % (name, lo))
        elif hi - value < tolerance:
            flags.append('%s at upper bound %g' % (name, hi))
    return flags


def fit_family(family, u, v):
    """Maximum likelihood.  Returns dict(params, ll, at_bound, n_starts).

    One-parameter families: bounded scalar search (the likelihood is unimodal).
    Two-parameter families: L-BFGS-B from every point of a starting grid; the
    best end point is kept.  Any parameter ending on its box bound is flagged.
    """
    u, v = _clip(np.asarray(u, float)), _clip(np.asarray(v, float))
    if family == 'independence':
        return dict(params=np.array([]), ll=0.0, at_bound=[], n_starts=0)
    base = base_family(family)
    bounds = BOUNDS[base]

    def neg(params):
        value = -np.sum(log_density(family, params, u, v))
        return value if np.isfinite(value) else 1e300

    if len(bounds) == 1:
        r = minimize_scalar(lambda p: neg([p]), bounds=bounds[0], method='bounded',
                            options=dict(xatol=1e-6))
        params, ll, n_starts = np.array([r.x]), float(-r.fun), 1
    else:
        grids = list(STARTS[base])
        if base == 't':
            grids[0] = (fit_family('gaussian', u, v)['params'][0],)
        best = None
        for start in itertools.product(*grids):
            start = [min(max(s, lo + 1e-6), hi - 1e-6) for s, (lo, hi) in zip(start, bounds)]
            r = minimize(neg, start, method='L-BFGS-B', bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        params, ll = np.array(best.x), float(-best.fun)
        n_starts = int(np.prod([len(g) for g in grids]))
    flags = _at_bound(base, params)
    if ll <= -1e299:
        flags.append('FIT FAILED: non-finite likelihood from every start')
    return dict(params=params, ll=ll, at_bound=flags, n_starts=n_starts)


def pseudo_observations(x):
    """Ranks / (n + 1), ties averaged."""
    return stats.rankdata(x) / (len(x) + 1.0)


def ecdf_transform(train, test):
    """Apply the TRAINING sample's empirical CDF to held-out points, scaled
    like pseudo-observations and kept strictly inside (0, 1)."""
    sorted_train = np.sort(train)
    return (np.searchsorted(sorted_train, test, side='right') + 0.5) / (len(train) + 1.0)


# --- the contest -------------------------------------------------------------------
def run_contest(x, y, traj, ifm_pit, families=FAMILIES, n_boot=2000, q_tail=0.05, seed=0):
    """Fit and score every family, by ranks ('cml') and by marginals ('ifm').

    x, y     paired sizes (any monotone scale; only ranks matter for cml)
    traj     trajectory id of each pair, for the split and the bootstrap
    ifm_pit  function(train_mask or None) -> (u, v) for ALL pairs, using
             marginals fitted on the trajectories in train_mask (None = all)
    """
    x, y, traj = np.asarray(x), np.asarray(y), np.asarray(traj)
    n = len(x)
    tau_empirical = stats.kendalltau(x, y)[0]
    if tau_empirical <= 0:
        raise ValueError('Kendall tau %.3f <= 0: the one-sided families need rotating' % tau_empirical)

    even = (traj % 2) == 0
    folds = (even, ~even)                 # train on one, score the other
    trajectories, traj_index = np.unique(traj, return_inverse=True)
    rng = np.random.default_rng(seed)
    # one shared set of bootstrap resamples, so family differences are paired
    boot_weights = np.stack([np.bincount(rng.integers(0, len(trajectories), len(trajectories)),
                                         minlength=len(trajectories))
                             for _ in range(n_boot)])

    def transforms(method):
        """(u, v) for in-sample fitting, and per-fold (train u, v, test u, v)."""
        if method == 'cml':
            full = (pseudo_observations(x), pseudo_observations(y))
            per_fold = [(pseudo_observations(x[train]), pseudo_observations(y[train]),
                         ecdf_transform(x[train], x[~train]), ecdf_transform(y[train], y[~train]))
                        for train in folds]
            return full, per_fold
        per_fold = []
        for train in folds:
            uf, vf = ifm_pit(train)
            per_fold.append((uf[train], vf[train], uf[~train], vf[~train]))
        return ifm_pit(None), per_fold

    results = {}
    for method in ('cml', 'ifm'):
        (u, v), per_fold = transforms(method)
        held_out_events = np.zeros((len(families), n))
        rows = {}
        for i, family in enumerate(families):
            fit = fit_family(family, u, v)
            params, ll, k = fit['params'], fit['ll'], n_parameters(family)
            fold_flags = []
            for f, (train, (u_tr, v_tr, u_te, v_te)) in enumerate(zip(folds, per_fold)):
                fold_fit = fit_family(family, u_tr, v_tr)
                fold_flags += ['fold %d: %s' % (f, flag) for flag in fold_fit['at_bound']]
                held_out_events[i, ~train] = log_density(family, fold_fit['params'], u_te, v_te)
            lam_lower, lam_upper = tail_dependence(family, params) if k else (0.0, 0.0)
            fin_lower, fin_upper = joint_tail(family, params, q_tail)
            rows[family] = dict(n_parameters=k, params=[float(p) for p in params],
                                param_names=list(BASE_PARAMETERS[base_family(family)]),
                                ll=ll, aic=2 * k - 2 * ll, bic=k * np.log(n) - 2 * ll,
                                at_bound=fit['at_bound'], fold_at_bound=fold_flags,
                                n_starts=fit['n_starts'],
                                held_out_ll=float(held_out_events[i].sum()),
                                tau=float(kendall_tau(family, params)),
                                lambda_lower=lam_lower, lambda_upper=lam_upper,
                                tail_lower_q=fin_lower, tail_upper_q=fin_upper)

        # bootstrap by trajectory over the held-out per-event scores
        per_traj = np.stack([np.bincount(traj_index, weights=held_out_events[i],
                                         minlength=len(trajectories))
                             for i in range(len(families))])
        boot = boot_weights @ per_traj.T                    # (n_boot, families)
        best = int(np.argmax(held_out_events.sum(axis=1)))
        for i, family in enumerate(families):
            diff = boot[:, i] - boot[:, best]
            rows[family].update(
                held_out_ci=[float(np.percentile(boot[:, i], 2.5)),
                             float(np.percentile(boot[:, i], 97.5))],
                delta_best=float(held_out_events[i].sum() - held_out_events[best].sum()),
                delta_best_ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                p_beats_best=float(np.mean(diff > 0)))
        results[method] = dict(best=families[best], families=rows,
                               boot_scores=boot, family_order=list(families))
    lower_emp, upper_emp = empirical_joint_tail(pseudo_observations(x), pseudo_observations(y), q_tail)
    results['empirical'] = dict(n=n, n_trajectories=int(len(trajectories)), kendall_tau=float(tau_empirical),
                                spearman_rho=float(stats.spearmanr(x, y)[0]),
                                tail_lower_q=lower_emp, tail_upper_q=upper_emp, q_tail=q_tail,
                                n_boot=n_boot)
    return results


def paired_difference(results, method, family_a, family_b):
    """Held-out (a - b) with its trajectory-bootstrap 95% CI and P(a > b)."""
    r = results[method]
    i, j = r['family_order'].index(family_a), r['family_order'].index(family_b)
    diff = r['boot_scores'][:, i] - r['boot_scores'][:, j]
    point = r['families'][family_a]['held_out_ll'] - r['families'][family_b]['held_out_ll']
    return float(point), [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], \
        float(np.mean(diff > 0))


def print_contest(results):
    e = results['empirical']
    print('n = %d pairs in %d trajectories   Kendall tau %.3f   Spearman rho %.3f'
          % (e['n'], e['n_trajectories'], e['kendall_tau'], e['spearman_rho']))
    print('empirical tail at q = %.2f: lower %.3f  upper %.3f  (independence: %.3f)'
          % (e['q_tail'], e['tail_lower_q'], e['tail_upper_q'], e['q_tail']))
    for method, label in (('cml', 'RANKS (reference)'), ('ifm', 'TPL MARGINALS (IFM)')):
        rows = results[method]['families']
        order = sorted(rows, key=lambda f: -rows[f]['held_out_ll'])
        print('\n%s  -- ranked by held-out ln L gain over independence' % label)
        print('  %-17s %3s %9s %20s %9s %20s %6s  %6s %6s %6s  %s'
              % ('family', 'par', 'held-out', '95% CI', 'vs best', '95% CI', 'P>best',
                 'tau', 'tailL', 'tailU', 'params'))
        for f in order:
            r = rows[f]
            flag = '  <-- AT BOUND: ' + '; '.join(r['at_bound']) if r['at_bound'] else ''
            print('  %-17s %3d %9.1f [%8.1f, %8.1f] %9.1f [%8.1f, %8.1f] %6.3f  %6.3f %6.3f %6.3f  %s%s'
                  % (f, r['n_parameters'], r['held_out_ll'], *r['held_out_ci'], r['delta_best'],
                     *r['delta_best_ci'], r['p_beats_best'], r['tau'], r['tail_lower_q'],
                     r['tail_upper_q'],
                     ', '.join('%s %.4g' % (nm, p) for nm, p in zip(r['param_names'], r['params'])),
                     flag))
        fold_flags = [(f, rows[f]['fold_at_bound']) for f in order if rows[f]['fold_at_bound']]
        for f, flags in fold_flags:
            print('  (held-out fold fit of %s: %s)' % (f, '; '.join(flags)))
