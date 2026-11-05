"""Synthetic checks for copulas.py (and marginals.py), run before real data.

    python test_copulas.py

1. Each closed-form log density matches statsmodels' at random points (where
   statsmodels has the family), and (1b) the log of the mixed derivative of
   our own CDF -- this is the check for BB1, BB7 and frank with theta < 0.
2. Each density integrates to 1 (Monte Carlo mean of c under independence),
   and (2b) BB1 / BB7 reduce to their limits: BB1 -> Clayton (delta = 1) and
   Gumbel (theta -> 0); BB7 -> Clayton (theta = 1) and Joe (delta -> 0).
2c. BB1 / BB7 samplers -- built from the frailty (Laplace-transform) form, so
   they share no code with the densities -- reproduce the CDF and Kendall tau.
3. Recovery: draw pairs from a KNOWN copula and parameter, push them through
   TPL inverse CDFs with parameters like the most-visited cell's, and run the
   full run_contest (ranks and IFM, held-out split, trajectory bootstrap).
   The true family must win or be statistically tied with the winner (it can
   be tied with a family that nests it, e.g. Clayton inside BB1), and no
   parameter may end on a bound.
3b. Parameter accuracy: at n = 40000, both methods recover every parameter
   within 5%.  (At n = 4000 one estimate scatters by up to ~9% -- BB1's theta --
   while the mean over repeated samples is within 3%, so 5% is checked where
   it measures the code rather than the sample.)
4. On independent data, no family beats independence clearly.
Exits non-zero if any check fails.
"""
import sys
import warnings
import numpy as np
from scipy import stats
from scipy.special import gammaln
from statsmodels.distributions.copula.api import (ClaytonCopula, FrankCopula, GaussianCopula,
                                                  GumbelCopula, StudentTCopula)
import copulas
import marginals

warnings.filterwarnings('ignore')
N_PAIRS = 4000           # about the size of the real paired sample (3906)
N_TRAJECTORIES = 800
N_BOOT = 500
PARAMETER_TOLERANCE = 0.05
N_ACCURACY = 40000      # parameter accuracy is checked at this size (3b)
PE = dict(kappa=0.738, s_star=1.061e-3, xmin=1e-6)       # cell (18, 11), step 6
STRESS = dict(kappa=0.872, s_star=0.1761, xmin=1e-3)
failures = []


def check(ok, message):
    print('  %s  %s' % ('ok  ' if ok else 'FAIL', message))
    if not ok:
        failures.append(message)


# --- samplers -----------------------------------------------------------------
def statsmodels_copula(family, params):
    base = copulas.base_family(family)
    if base == 'gaussian':
        return GaussianCopula(corr=params[0])
    if base == 't':
        return StudentTCopula(corr=params[0], df=params[1])
    return dict(clayton=ClaytonCopula, gumbel=GumbelCopula, frank=FrankCopula)[base](theta=params[0])


def positive_stable(alpha, n, rng):
    """Laplace transform exp(-s^alpha), 0 < alpha <= 1 (Kanter / Chambers-Mallows-Stuck)."""
    if alpha == 1:
        return np.ones(n)
    w = rng.uniform(0, np.pi, n)
    e = rng.exponential(size=n)
    return (np.sin(alpha * w) / np.sin(w) ** (1 / alpha)
            * (np.sin((1 - alpha) * w) / e) ** ((1 - alpha) / alpha))


def sibuya(alpha, n, rng):
    """P(N > k) = Gamma(k + 1 - alpha) / (Gamma(k + 1) Gamma(1 - alpha)), by bisection."""
    if alpha == 1:
        return np.ones(n)
    target = np.log(rng.random(n))

    def ln_survival(k):
        # ln Gamma(z - alpha) - ln Gamma(z), z = k + 1.  Directly from gammaln it
        # cancels catastrophically for large k (both terms ~ k ln k), so beyond
        # 1e6 use the asymptotic series -alpha ln z + alpha (alpha + 1) / (2 z).
        z = k + 1.0
        direct = gammaln(np.minimum(z, 1e6) - alpha) - gammaln(np.minimum(z, 1e6))
        series = -alpha * np.log(z) + alpha * (alpha + 1) / (2 * z)
        return np.where(z < 1e6, direct, series) - gammaln(1 - alpha)
    lo, hi = np.zeros(n), np.full(n, 1e17)          # survival(lo) > target >= survival(hi)
    for _ in range(80):
        mid = np.floor((lo + hi) / 2)
        above = ln_survival(mid) > target
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return hi


def sample_bb(family, params, n, rng):
    """Marshall-Olkin: U_i = psi(E_i / M) with M the frailty whose Laplace transform is psi."""
    theta, delta = params
    e = rng.exponential(size=(n, 2))
    if family == 'bb1':
        # psi(s) = (1 + s^(1/delta))^(-1/theta):  M = S V^delta, S ~ stable(1/delta), V ~ Gamma(1/theta)
        m = positive_stable(1 / delta, n, rng) * rng.gamma(1 / theta, size=n) ** delta
        s = e / m[:, None]
        return (1 + s ** (1 / delta)) ** (-1 / theta)
    # bb7: psi(s) = 1 - (1 - (1+s)^(-1/delta))^(1/theta):  M ~ Gamma(N / delta), N ~ Sibuya(1/theta)
    m = rng.gamma(sibuya(1 / theta, n, rng) / delta)
    s = e / m[:, None]
    return 1 - (-np.expm1(-np.log1p(s) / delta)) ** (1 / theta)


def sample(family, params, n, rng):
    if family == 'independence':
        return rng.random(n), rng.random(n)
    base = copulas.base_family(family)
    if base in ('bb1', 'bb7'):
        uv = sample_bb(base, params, n, rng)
    else:
        uv = statsmodels_copula(family, params).rvs(n, random_state=rng)
    if family.startswith('survival_'):
        uv = 1 - uv
    return uv[:, 0], uv[:, 1]


def tpl_inverse(p, kappa, s_star, xmin):
    ln_s, cumulative, _ = marginals._grid(kappa, s_star, xmin)
    return np.exp(np.interp(p, cumulative / cumulative[-1], ln_s))


def ln_joe(u, v, theta):
    a, b = (1 - u) ** theta, (1 - v) ** theta
    return ((1 / theta - 2) * np.log(a + b - a * b) + (theta - 1) * (np.log1p(-u) + np.log1p(-v))
            + np.log(theta - 1 + a + b - a * b))


rng = np.random.default_rng(1)

print('1. log densities against statsmodels')
for family, params in [('gaussian', [0.6]), ('t', [0.5, 4.0]), ('clayton', [2.0]),
                       ('gumbel', [1.8]), ('frank', [5.0]),
                       ('survival_clayton', [1.5]), ('survival_gumbel', [1.4])]:
    u, v = rng.random(500), rng.random(500)
    ours = copulas.log_density(family, params, u, v)
    if family.startswith('survival_'):
        theirs = statsmodels_copula(family, params).logpdf(np.column_stack([1 - u, 1 - v]))
    else:
        theirs = statsmodels_copula(family, params).logpdf(np.column_stack([u, v]))
    error = np.max(np.abs(ours - theirs))
    check(error < 1e-6, '%-17s %-12s max |difference| %.1e' % (family, params, error))

print('\n1b. log density = log of the mixed derivative of our own CDF (finite differences)')
# (gaussian and t CDFs are scipy numerical integrals, ~1e-5 accurate: too coarse
# for finite differences; their densities are already checked against statsmodels)
h = 1e-4
u, v = rng.uniform(0.05, 0.95, 200), rng.uniform(0.05, 0.95, 200)
for family, params in [('clayton', [2.0]), ('gumbel', [1.8]), ('frank', [5.0]), ('frank', [-3.0]),
                       ('survival_clayton', [1.5]), ('survival_gumbel', [1.4]),
                       ('bb1', [0.8, 2.5]), ('bb1', [2.0, 1.2]), ('bb1', [0.05, 3.0]),
                       ('survival_bb1', [0.8, 2.5]),
                       ('bb7', [2.5, 1.5]), ('bb7', [1.2, 3.0]), ('bb7', [4.0, 0.2]),
                       ('survival_bb7', [2.5, 1.5])]:
    def c(a, b):
        return copulas.cdf(family, params, a, b)
    numeric = np.log((c(u + h, v + h) - c(u + h, v - h) - c(u - h, v + h) + c(u - h, v - h))
                     / (4 * h * h))
    error = np.max(np.abs(copulas.log_density(family, params, u, v) - numeric))
    check(error < 1e-3, '%-17s %-12s max |difference| %.1e' % (family, params, error))

print('\n2. densities integrate to 1')
u, v = rng.random(2_000_000), rng.random(2_000_000)
for family, params in [('gaussian', [0.6]), ('t', [0.5, 4.0]), ('clayton', [2.0]),
                       ('gumbel', [1.8]), ('frank', [5.0]), ('bb1', [0.8, 2.5]),
                       ('bb1', [2.0, 1.2]), ('bb7', [2.5, 1.5]), ('bb7', [1.2, 3.0])]:
    mean = np.mean(np.exp(copulas.log_density(family, params, u, v)))
    check(abs(mean - 1) < 0.02, '%-9s %-11s integral %.4f' % (family, params, mean))

print('\n2b. BB1 and BB7 limits')
u, v = rng.uniform(0.001, 0.999, 2000), rng.uniform(0.001, 0.999, 2000)
for label, ours, reference in [
        ('bb1(theta 2, delta 1)    = clayton(2)', copulas.log_density('bb1', [2.0, 1.0], u, v),
         copulas.log_density('clayton', [2.0], u, v)),
        ('bb1(theta 1e-6, delta 1.8) = gumbel(1.8)', copulas.log_density('bb1', [1e-6, 1.8], u, v),
         copulas.log_density('gumbel', [1.8], u, v)),
        ('bb7(theta 1, delta 2)    = clayton(2)', copulas.log_density('bb7', [1.0, 2.0], u, v),
         copulas.log_density('clayton', [2.0], u, v)),
        ('bb7(theta 1.8, delta 1e-6) = joe(1.8)', copulas.log_density('bb7', [1.8, 1e-6], u, v),
         ln_joe(u, v, 1.8))]:
    error = np.max(np.abs(ours - reference))
    check(error < 1e-3, '%-42s max |difference| %.1e' % (label, error))
for label, got, expected in [
        ('bb1 tau at delta 1 = clayton tau', copulas.kendall_tau('bb1', [2.0, 1.0]),
         copulas.kendall_tau('clayton', [2.0])),
        ('bb1 tau at theta -> 0 = gumbel tau', copulas.kendall_tau('bb1', [1e-9, 1.8]),
         copulas.kendall_tau('gumbel', [1.8])),
        ('bb7 tau at theta 1 = clayton tau', copulas.kendall_tau('bb7', [1.0, 2.0]),
         copulas.kendall_tau('clayton', [2.0]))]:
    check(abs(got - expected) < 1e-6, '%-42s %.6f vs %.6f' % (label, got, expected))

print('\n2c. BB samplers (frailty form) reproduce the CDF and Kendall tau (n = 20000)')
for family, params in [('bb1', [0.8, 2.5]), ('bb1', [2.0, 1.2]), ('bb7', [2.5, 1.5]),
                       ('bb7', [1.2, 3.0])]:
    su, sv = sample(family, params, 20000, rng)
    tau_sample = stats.kendalltau(su, sv)[0]
    tau_model = copulas.kendall_tau(family, params)
    check(abs(tau_sample - tau_model) < 0.015,
          '%-4s %-11s tau sample %.4f, model %.4f' % (family, params, tau_sample, tau_model))
    points = [(0.1, 0.1), (0.3, 0.6), (0.5, 0.5), (0.9, 0.9)]
    errors = [abs(np.mean((su <= a) & (sv <= b)) - copulas.cdf(family, params, a, b)) for a, b in points]
    check(max(errors) < 0.01, '%-4s %-11s max |empirical C - model C| at 4 points %.4f'
          % (family, params, max(errors)))

print('\n3. recovery through the full contest (n = %d, %d trajectories)' % (N_PAIRS, N_TRAJECTORIES))
cases = [('clayton', [2.0]), ('gumbel', [1.8]), ('survival_clayton', [1.5]),
         ('survival_gumbel', [1.5]), ('gaussian', [0.6]), ('t', [0.5, 4.0]), ('frank', [5.0]),
         ('bb1', [0.8, 2.5]), ('survival_bb1', [0.8, 2.5]),
         ('bb7', [2.5, 1.5]), ('survival_bb7', [2.5, 1.5])]
for true_family, true_params in cases:
    u, v = sample(true_family, true_params, N_PAIRS, rng)
    x = tpl_inverse(u, **PE)
    y = tpl_inverse(v, **STRESS)
    traj = rng.integers(0, N_TRAJECTORIES, N_PAIRS)

    def ifm_pit(train):
        mask = np.ones(N_PAIRS, bool) if train is None else train
        fx = marginals.fit(x[mask], PE['xmin'])
        fy = marginals.fit(y[mask], STRESS['xmin'])
        return (marginals.cdf(x, fx['kappa'], fx['s_star'], PE['xmin']),
                marginals.cdf(y, fy['kappa'], fy['s_star'], STRESS['xmin']))

    results = copulas.run_contest(x, y, traj, ifm_pit, n_boot=N_BOOT)
    print('\n  true %s %s' % (true_family, true_params))
    for method in ('cml', 'ifm'):
        rows = results[method]['families']
        row = rows[true_family]
        ranked = sorted(rows, key=lambda f: -rows[f]['held_out_ll'])
        tied = row['delta_best_ci'][1] >= 0          # CI of (true - best) reaches 0
        check(ranked[0] == true_family or tied,
              '%s: best %-17s true ranked %d, true - best %.1f [%.1f, %.1f]'
              % (method, ranked[0], ranked.index(true_family) + 1, row['delta_best'],
                 *row['delta_best_ci']))
        # at n = 4000 a single estimate scatters by up to ~9% (BB1's theta), so
        # parameters are only reported here; accuracy is checked in 3b
        check(not row['at_bound'],
              '%s: %s (true %s), no parameter at a bound%s'
              % (method, ', '.join('%s %.3f' % (nm, p) for nm, p in zip(row['param_names'], row['params'])),
                 true_params, '  AT BOUND ' + '; '.join(row['at_bound']) if row['at_bound'] else ''))

print('\n3b. parameter accuracy: in-sample fits at n = %d, every parameter within %.0f%%'
      % (N_ACCURACY, 100 * PARAMETER_TOLERANCE))
for true_family, true_params in cases:
    u, v = sample(true_family, true_params, N_ACCURACY, rng)
    x, y = tpl_inverse(u, **PE), tpl_inverse(v, **STRESS)
    fx, fy = marginals.fit(x, PE['xmin']), marginals.fit(y, STRESS['xmin'])
    transforms = dict(cml=(copulas.pseudo_observations(x), copulas.pseudo_observations(y)),
                      ifm=(marginals.cdf(x, fx['kappa'], fx['s_star'], PE['xmin']),
                           marginals.cdf(y, fy['kappa'], fy['s_star'], STRESS['xmin'])))
    for method, (pu, pv) in transforms.items():
        fit = copulas.fit_family(true_family, pu, pv)
        errors = [abs(e - t) / abs(t) for e, t in zip(fit['params'], true_params)]
        names = copulas.BASE_PARAMETERS[copulas.base_family(true_family)]
        check(max(errors) < PARAMETER_TOLERANCE and not fit['at_bound'],
              '%-17s %s: %s (true %s)  worst relative error %.1f%%%s'
              % (true_family, method, ', '.join('%s %.3f' % (nm, p) for nm, p in zip(names, fit['params'])),
                 true_params, 100 * max(errors),
                 '  AT BOUND ' + '; '.join(fit['at_bound']) if fit['at_bound'] else ''))

print('\n4. independent data: no family should beat independence clearly')
# run_contest refuses tau <= 0, so draw until the sample tau is (slightly) positive
while True:
    u, v = rng.random(N_PAIRS), rng.random(N_PAIRS)
    x, y = tpl_inverse(u, **PE), tpl_inverse(v, **STRESS)
    if stats.kendalltau(x, y)[0] > 0:
        break
traj = rng.integers(0, N_TRAJECTORIES, N_PAIRS)
results = copulas.run_contest(
    x, y, traj, lambda train: (marginals.cdf(x, PE['kappa'], PE['s_star'], PE['xmin']),
                               marginals.cdf(y, STRESS['kappa'], STRESS['s_star'], STRESS['xmin'])),
    n_boot=N_BOOT)
for method in ('cml', 'ifm'):
    rows = results[method]['families']
    best = max(rows, key=lambda f: rows[f]['held_out_ll'])
    check(rows[best]['held_out_ci'][0] <= 0,
          '%s: best %s held-out gain %.1f, CI [%.1f, %.1f] includes 0'
          % (method, best, rows[best]['held_out_ll'], *rows[best]['held_out_ci']))

print('\n%d failure(s)' % len(failures))
sys.exit(1 if failures else 0)
