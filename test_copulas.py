"""Synthetic checks for copulas.py (and marginals.py), run before real data.

    python test_copulas.py

1. Each closed-form log density matches statsmodels' at random points, and
   (1b) the log of the mixed derivative of our own CDF.
2. Each density integrates to 1 (Monte Carlo mean of c under independence).
3. Recovery: draw pairs from a KNOWN copula and parameter (statsmodels'
   sampler, so not our own code), push them through TPL inverse CDFs with
   parameters like the most-visited cell's, and run the full run_contest --
   both rank-based and IFM, held-out split, trajectory bootstrap.  The true
   family should win or be statistically tied with the winner, and its
   parameter should come back close to the truth.
Exits non-zero if any check fails.
"""
import sys
import warnings
import numpy as np
from statsmodels.distributions.copula.api import (ClaytonCopula, FrankCopula, GaussianCopula,
                                                  GumbelCopula, StudentTCopula)
import copulas
import marginals

warnings.filterwarnings('ignore')
N_PAIRS = 4000           # about the size of the real paired sample (3906)
N_TRAJECTORIES = 800
N_BOOT = 500
PE = dict(kappa=0.738, s_star=1.061e-3, xmin=1e-6)       # cell (18, 11), step 6
STRESS = dict(kappa=0.872, s_star=0.1761, xmin=1e-3)
failures = []


def check(ok, message):
    print('  %s  %s' % ('ok  ' if ok else 'FAIL', message))
    if not ok:
        failures.append(message)


def statsmodels_copula(family, params):
    base = family.replace('survival_', '')
    if base == 'gaussian':
        return GaussianCopula(corr=params[0])
    if base == 't':
        return StudentTCopula(corr=params[0], df=params[1])
    return dict(clayton=ClaytonCopula, gumbel=GumbelCopula, frank=FrankCopula)[base](theta=params[0])


def sample(family, params, n, rng):
    if family == 'independence':
        return rng.random(n), rng.random(n)
    uv = statsmodels_copula(family, params).rvs(n, random_state=rng)
    if family.startswith('survival_'):
        uv = 1 - uv
    return uv[:, 0], uv[:, 1]


def tpl_inverse(p, kappa, s_star, xmin):
    ln_s, cumulative, _ = marginals._grid(kappa, s_star, xmin)
    return np.exp(np.interp(p, cumulative / cumulative[-1], ln_s))


rng = np.random.default_rng(1)

print('1. log densities against statsmodels')
for family, params in [('gaussian', [0.6]), ('t', [0.5, 4.0]), ('clayton', [2.0]),
                       ('gumbel', [1.8]), ('frank', [5.0]), ('frank', [-3.0]),
                       ('survival_clayton', [1.5]), ('survival_gumbel', [1.4])]:
    u, v = rng.random(500), rng.random(500)
    ours = copulas.log_density(family, params, u, v)
    if family.startswith('survival_'):
        theirs = statsmodels_copula(family, params).logpdf(np.column_stack([1 - u, 1 - v]))
    else:
        theirs = statsmodels_copula(family, params).logpdf(np.column_stack([u, v]))
    if np.all(np.isnan(theirs)):          # statsmodels has no frank with theta < 0
        check(True, '%-17s %-12s not defined in statsmodels; covered by 1b' % (family, params))
        continue
    error = np.max(np.abs(ours - theirs))
    check(error < 1e-6, '%-17s %-12s max |difference| %.1e' % (family, params, error))

print('\n1b. log density = log of the mixed derivative of our own CDF (finite differences)')
h = 1e-4
u, v = rng.uniform(0.05, 0.95, 200), rng.uniform(0.05, 0.95, 200)
# (gaussian and t CDFs are scipy numerical integrals, ~1e-5 accurate: too coarse
# for finite differences; their densities are already checked against statsmodels)
for family, params in [('clayton', [2.0]), ('gumbel', [1.8]),
                       ('frank', [5.0]), ('frank', [-3.0]), ('survival_clayton', [1.5]),
                       ('survival_gumbel', [1.4])]:
    def c(a, b):
        return copulas.cdf(family, params, a, b)
    numeric = np.log((c(u + h, v + h) - c(u + h, v - h) - c(u - h, v + h) + c(u - h, v - h))
                     / (4 * h * h))
    error = np.max(np.abs(copulas.log_density(family, params, u, v) - numeric))
    check(error < 1e-3, '%-17s %-12s max |difference| %.1e' % (family, params, error))

print('\n2. densities integrate to 1')
u, v = rng.random(2_000_000), rng.random(2_000_000)
for family, params in [('gaussian', [0.6]), ('t', [0.5, 4.0]), ('clayton', [2.0]),
                       ('gumbel', [1.8]), ('frank', [5.0])]:
    mean = np.mean(np.exp(copulas.log_density(family, params, u, v)))
    check(abs(mean - 1) < 0.02, '%-9s integral %.4f' % (family, mean))

print('\n3. recovery through the full contest (n = %d, %d trajectories)' % (N_PAIRS, N_TRAJECTORIES))
cases = [('clayton', [2.0]), ('gumbel', [1.8]), ('survival_clayton', [1.5]),
         ('survival_gumbel', [1.5]), ('gaussian', [0.6]), ('t', [0.5, 4.0]), ('frank', [5.0])]
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
        estimate = row['params'][0]
        check(abs(estimate - true_params[0]) < 0.1 * abs(true_params[0]),
              '%s: parameter %.3f (true %.3f)%s'
              % (method, estimate, true_params[0],
                 ', nu %.2f (true %.1f)' % (row['params'][1], true_params[1]) if true_family == 't' else ''))

print('\n4. independent data: no family should beat independence clearly')
# run_contest refuses tau <= 0, so draw until the sample tau is (slightly) positive
while True:
    u, v = rng.random(N_PAIRS), rng.random(N_PAIRS)
    x, y = tpl_inverse(u, **PE), tpl_inverse(v, **STRESS)
    if copulas.stats.kendalltau(x, y)[0] > 0:
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
