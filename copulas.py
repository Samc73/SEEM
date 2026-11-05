"""Bivariate copula families, their fitting, and a held-out contest between them.

Families (all one-parameter except t):
    independence      c = 1 (baseline, ln L = 0)
    gaussian (rho)    no tail dependence
    t (rho, nu)       symmetric tail dependence in both corners
    clayton (theta)   lower-tail dependence: small drops of one go with small of the other
    gumbel (theta)    upper-tail dependence: large drops go together
    frank (theta)     no tail dependence, lighter centre than gaussian
    survival_clayton  clayton rotated 180 deg: UPPER-tail dependence
    survival_gumbel   gumbel rotated 180 deg: LOWER-tail dependence
Clayton and Gumbel only model positive dependence; run_contest stops if the
data's Kendall tau is negative.

Densities are written out in closed form (faster than statsmodels and easy to
check); test_copulas.py checks them against statsmodels.distributions.copula.

The contest (run_contest) fits every family two ways:
    'cml' ranks (pseudo-observations): needs no marginal model -- the reference
    'ifm' PIT through fitted parametric marginals (here the TPLs)
and scores each on held-out data with step 3's 2-fold split by trajectory
(even ids vs odd).  Error bars come from resampling TRAJECTORIES with
replacement and re-summing the held-out per-event scores; this measures how
much the score depends on which trajectories were sampled, not refitting noise.
"""
import numpy as np
from scipy import integrate, stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, ndtri, stdtrit

CLIP = 1e-10
FAMILIES = ('independence', 'gaussian', 't', 'clayton', 'gumbel', 'frank',
            'survival_clayton', 'survival_gumbel')
N_PARAMETERS = dict(independence=0, gaussian=1, t=2, clayton=1, gumbel=1, frank=1,
                    survival_clayton=1, survival_gumbel=1)
BOUNDS = dict(gaussian=(-0.995, 0.995), clayton=(1e-4, 60.0), gumbel=(1.0 + 1e-6, 60.0),
              frank=(-60.0, 60.0))
NU_RANGE = (1.0, 300.0)


# --- log densities ------------------------------------------------------------
def _clip(u):
    return np.clip(u, CLIP, 1.0 - CLIP)


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


def log_density(family, params, u, v):
    u, v = _clip(np.asarray(u, float)), _clip(np.asarray(v, float))
    if family == 'independence':
        return np.zeros_like(u)
    if family.startswith('survival_'):
        return log_density(family[len('survival_'):], params, 1 - u, 1 - v)
    if family == 't':
        return _ln_t(u, v, params[0], params[1])
    return dict(gaussian=_ln_gaussian, clayton=_ln_clayton, gumbel=_ln_gumbel,
                frank=_ln_frank)[family](u, v, params[0])


# --- copula CDF (only needed for finite-q tail checks) ------------------------
def cdf(family, params, u, v):
    if family == 'independence':
        return u * v
    if family.startswith('survival_'):
        return u + v - 1 + cdf(family[len('survival_'):], params, 1 - u, 1 - v)
    if family == 'gaussian':
        rho = params[0]
        return stats.multivariate_normal(cov=[[1, rho], [rho, 1]]).cdf(
            np.stack([ndtri(u), ndtri(v)], axis=-1))
    if family == 't':
        rho, nu = params
        return stats.multivariate_t(shape=[[1, rho], [rho, 1]], df=nu).cdf(
            np.stack([stdtrit(nu, u), stdtrit(nu, v)], axis=-1))
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
        return kendall_tau(family[len('survival_'):], params)   # rotation keeps tau
    if family in ('gaussian', 't'):
        return 2 / np.pi * np.arcsin(params[0])
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
        lower, upper = tail_dependence(family[len('survival_'):], params)
        return upper, lower
    if family == 't':
        rho, nu = params
        lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
        return float(lam), float(lam)
    if family == 'clayton':
        return float(2 ** (-1 / params[0])), 0.0
    if family == 'gumbel':
        return 0.0, float(2 - 2 ** (1 / params[0]))
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
def fit_family(family, u, v):
    """Maximum likelihood; returns (params, total ln L)."""
    u, v = _clip(np.asarray(u, float)), _clip(np.asarray(v, float))
    if family == 'independence':
        return np.array([]), 0.0
    if family == 't':
        rho0, _ = fit_family('gaussian', u, v)

        def neg(theta):
            return -np.sum(_ln_t(u, v, np.tanh(theta[0]), np.exp(theta[1])))
        r = minimize(neg, [np.arctanh(rho0[0]), np.log(8.0)], method='L-BFGS-B',
                     bounds=[(-3.0, 3.0), tuple(np.log(NU_RANGE))])
        return np.array([np.tanh(r.x[0]), np.exp(r.x[1])]), float(-r.fun)
    base = family[len('survival_'):] if family.startswith('survival_') else family

    def neg(theta):
        value = -np.sum(log_density(family, [theta], u, v))
        return value if np.isfinite(value) else 1e300
    r = minimize_scalar(neg, bounds=BOUNDS[base], method='bounded', options=dict(xatol=1e-6))
    return np.array([r.x]), float(-r.fun)


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
        raise ValueError('Kendall tau %.3f <= 0: Clayton/Gumbel need rotating' % tau_empirical)

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
            per_fold = []
            for train in folds:
                per_fold.append((pseudo_observations(x[train]), pseudo_observations(y[train]),
                                 ecdf_transform(x[train], x[~train]),
                                 ecdf_transform(y[train], y[~train])))
            return full, per_fold
        u, v = ifm_pit(None)
        per_fold = []
        for train in folds:
            uf, vf = ifm_pit(train)
            per_fold.append((uf[train], vf[train], uf[~train], vf[~train]))
        return (u, v), per_fold

    results = {}
    for method in ('cml', 'ifm'):
        (u, v), per_fold = transforms(method)
        held_out_events = np.zeros((len(families), n))
        rows = {}
        for i, family in enumerate(families):
            params, ll = fit_family(family, u, v)
            k = N_PARAMETERS[family]
            for train, (u_tr, v_tr, u_te, v_te) in zip(folds, per_fold):
                params_fold, _ = fit_family(family, u_tr, v_tr)
                held_out_events[i, ~train] = log_density(family, params_fold, u_te, v_te)
            lam_lower, lam_upper = (tail_dependence(family, params)
                                    if family != 'independence' else (0.0, 0.0))
            fin_lower, fin_upper = joint_tail(family, params, q_tail)
            rows[family] = dict(n_parameters=k, params=[float(p) for p in params],
                                ll=ll, aic=2 * k - 2 * ll, bic=k * np.log(n) - 2 * ll,
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
        results[method] = dict(best=families[best], families=rows)
    lower_emp, upper_emp = empirical_joint_tail(pseudo_observations(x), pseudo_observations(y), q_tail)
    results['empirical'] = dict(n=n, n_trajectories=int(len(trajectories)), kendall_tau=float(tau_empirical),
                                spearman_rho=float(stats.spearmanr(x, y)[0]),
                                tail_lower_q=lower_emp, tail_upper_q=upper_emp, q_tail=q_tail,
                                n_boot=n_boot)
    return results


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
            print('  %-17s %3d %9.1f [%8.1f, %8.1f] %9.1f [%8.1f, %8.1f] %6.3f  %6.3f %6.3f %6.3f  %s'
                  % (f, r['n_parameters'], r['held_out_ll'], *r['held_out_ci'], r['delta_best'],
                     *r['delta_best_ci'], r['p_beats_best'], r['tau'], r['tail_lower_q'],
                     r['tail_upper_q'], ', '.join('%.4g' % p for p in r['params'])))
