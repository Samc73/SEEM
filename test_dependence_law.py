"""Synthetic checks for dependence_law.py, run before step 9 touches real sizes.

    python test_dependence_law.py

The real paired events' STATES (u, tau), voxels and trajectories are kept
(so the design, folds and voxel table look exactly like the real fit's), but
their sizes are replaced by Gumbel draws with a KNOWN theta(u, tau):
  A. product form   theta = 1 + exp(-0.5 + 0.4 * 100 u tau)
  B. constant       theta = 3.36 everywhere (step 7's value in cell (18, 11))
  C. linear form    eta = 0.8 + 0.25 z_u + 0.3 z_tau
  D. interaction + rate   eta = 0.8 + 0.25 z_u + 0.3 z_tau + 0.1 z_u z_tau
                                - 0.1 ln(rate/rate_mid), at the real events' rates;
     scored with every +rate form and voxel_table+rate
Checks, with the same 2-fold trajectory split and trajectory bootstrap as step 9:
  * the true form is best on held-out data or tied with the best (a richer
    form that nests it can tie);
  * when theta varies, the true form beats the constant clearly;
  * when theta is constant, the linear and quadratic forms do not beat it
    beyond noise;
  * the voxel table does not beat the true form beyond noise;
  * cases C and D (theta varies WITHIN voxels): the true form beats the
    voxel table (D: voxel_table+rate) clearly on held-out data;
  * case E: with constant theta but sizes that shrink with rate inside voxels,
    pooled ranks show a spurious positive rate effect and voxel x rate ranks
    (step 9's main ranking) do not;
  * case D: the true form beats its rate-free version clearly,
    voxel_table+rate beats voxel_table clearly, and its shared rate
    coefficient is within 30% of the truth in each fold;
  * a form that contains the truth (e.g. interaction, quadratic in case C) is not clearly
    worse than it, and no form is flagged in either fold (catches a failed fit);
  * in-sample coefficients of the true form are within 5% of the truth, on
    N_STACK independent samples of the layout pooled (at the real n the
    product form's intercept alone scatters by ~4.6%, so 5% is checked where
    it measures the code rather than the sample).
Also checks that the rank grid is symmetric (training ranks: mean 0.5 and
min = 1 - max in every voxel; held-out ranks midway between training ranks),
that every training fold passes the nesting check, and the per-event Gumbel sampler against Gumbel's tau = 1 - 1/theta.
Exits non-zero if any check fails.
"""
import json
import sys
import time
import warnings
import numpy as np
from scipy import stats
import dependence_law as dl

warnings.filterwarnings('ignore', category=RuntimeWarning)
TOLERANCE = 0.05
N_BOOT = 1000
N_STACK = 4      # coefficient accuracy is checked on 4 independent samples of the layout
                 # (~337k events): at the real n the product intercept, an extrapolation to
                 # 100 u tau = 0, scatters by ~4.6% (mean bias over 8 samples: -1.3%)
failures = []


def check(ok, message):
    print('  %s  %s' % ('ok  ' if ok else 'FAIL', message))
    if not ok:
        failures.append(message)


def sample_gumbel(theta, rng):
    """Marshall-Olkin with a positive-stable frailty, one theta per event:
    S has Laplace transform exp(-s^alpha), alpha = 1/theta, and U = exp(-(E/S)^alpha)."""
    alpha = 1.0 / np.asarray(theta)
    n = len(alpha)
    w = rng.uniform(0, np.pi, n)
    e = rng.exponential(size=n)
    s = (np.sin(alpha * w) / np.sin(w) ** (1 / alpha)
         * (np.sin((1 - alpha) * w) / e) ** ((1 - alpha) / alpha))
    pair = rng.exponential(size=(n, 2))
    return np.exp(-(pair / s[:, None]) ** alpha[:, None]).T


# --- the real layout ---------------------------------------------------------------
events = np.load('out/joint_events.npz')
with open('out/marginals.json') as handle:
    xmin = json.load(handle)['xmin']
du, dtau, cell_all = events['du'], events['dtau'], events['cell']
paired = (du < 0) & (dtau < 0) & (-du >= xmin['pe']) & (-dtau >= xmin['stress'])
counts = np.bincount(cell_all[paired], minlength=int(events['n_bins']) ** 2)
keep = paired & (counts[cell_all] >= 100)
u_state, tau_state = events['u'][keep], events['tau'][keep]
cell, traj = cell_all[keep], events['traj'][keep]
rates = np.unique(events['rate'])
ln_rate = np.log(events['rate'][keep] / rates[len(rates) // 2])      # as step 3 and step 9
rate_index = np.searchsorted(rates, events['rate'][keep])
RANK_LEVELS = [cell * 9 + rate_index, cell * 3 + rate_index // 3, cell]   # as step 9's main ranks
fold_id = traj % 2
print('layout: %d events in %d voxels, %d trajectories' % (keep.sum(), len(np.unique(cell)),
                                                          len(np.unique(traj))))
rng = np.random.default_rng(3)

print('\n0. per-event Gumbel sampler')
for theta in (1.5, 3.36, 5.0):
    a, b = sample_gumbel(np.full(20000, theta), rng)
    tau = stats.kendalltau(a, b)[0]
    check(abs(tau - (1 - 1 / theta)) < 0.01, 'theta %.2f: sample tau %.4f, Gumbel tau %.4f'
          % (theta, tau, 1 - 1 / theta))

print('\n0b. rank grid (dl.RANK_RULE) is symmetric for training AND held-out events, every voxel, both folds')
sizes = rng.random(len(cell))
worst_mean = worst_sym = worst_grid = 0.0
for k in (0, 1):
    train = fold_id != k
    u_tr, u_te, _ = dl.held_out_uniforms(sizes, cell, train)
    for c in np.unique(cell[train]):
        m = cell[train] == c
        n_c = m.sum()
        worst_mean = max(worst_mean, abs(u_tr[m].mean() - 0.5))
        worst_sym = max(worst_sym, abs(u_tr[m].min() + u_tr[m].max() - 1))
        mt = cell[~train] == c                 # held-out: on the half-grid (k + 0.5)/(n + 1), k = 0..n
        steps = u_te[mt] * (n_c + 1) - 0.5
        if mt.any():
            worst_grid = max(worst_grid, np.abs(steps - np.round(steps)).max(),
                             0 if (steps.min() > -1e-9 and steps.max() < n_c + 1e-9) else 1)
check(worst_mean < 1e-12, 'training ranks: mean 0.5 in every voxel (worst deviation %.1e)' % worst_mean)
check(worst_sym < 1e-12, 'training ranks: min = 1 - max in every voxel (worst deviation %.1e)' % worst_sym)
check(worst_grid < 1e-9, 'held-out ranks on (k + 0.5)/(n + 1), k = 0..n, midway between training ranks '
      '(worst deviation %.1e)' % worst_grid)

scale = dl.scaling(u_state, tau_state)
# D: theta varies within voxels AND with cooling rate (a rate effect of about the size the
#    data suggest); scored with every +rate form and the voxel_table+rate reference
cases = [('product', np.array([-0.5, 0.4])),
         ('constant', np.array([np.log(3.36 - 1)])),
         ('linear', np.array([0.8, 0.25, 0.3])),
         ('interaction+rate', np.array([0.8, 0.25, 0.3, 0.1, -0.1]))]
CONTAINS_TRUTH = dict(product=(), constant=(), linear=('interaction', 'quadratic'),
                      **{'interaction+rate': ('quadratic+rate',)})
for true_form, true_beta in cases:
    start = time.time()
    with_rate = true_form.endswith(dl.RATE)
    lr = ln_rate if with_rate else None
    models = dl.MODELS_WITH_RATE if with_rate else dl.MODELS
    theta = dl.theta_of(true_beta, dl.design(true_form, u_state, tau_state, scale, lr))
    x_size, y_size = sample_gumbel(theta, rng)
    print('\n%s truth %s  (theta %.2f to %.2f)' % (true_form, np.round(true_beta, 3),
                                                   theta.min(), theta.max()))
    held, folds = dl.cross_validate(x_size, y_size, u_state, tau_state, cell, fold_id, rng, ln_rate=lr)
    boot = dl.bootstrap_scores(held, traj, N_BOOT, rng)
    totals = {m: held[m].sum() for m in models}
    best = max(totals, key=totals.get)
    for m in models:
        print('     %-18s held-out %10.1f   vs best %+8.1f' % (m, totals[m], totals[m] - totals[best]))
    vs_best = dl.compare(held, boot, true_form, best)
    check(best == true_form or vs_best['ci'][1] >= 0,
          'true form best or tied: best %s, true - best %+.1f [%+.1f, %+.1f]'
          % (best, vs_best['difference'], *vs_best['ci']))
    if true_form == 'constant':
        for richer in ('linear', 'interaction', 'quadratic'):
            c = dl.compare(held, boot, richer, 'constant')
            check(c['ci'][0] <= 0, '%s does not beat constant beyond noise: %+.1f [%+.1f, %+.1f]'
                  % (richer, c['difference'], *c['ci']))
    else:
        c = dl.compare(held, boot, true_form, 'constant')
        check(c['ci'][0] > 0, 'true form beats constant clearly: %+.1f [%+.1f, %+.1f]'
              % (c['difference'], *c['ci']))
    for table in ((dl.TABLE, dl.TABLE_RATE) if with_rate else (dl.TABLE,)):
        c = dl.compare(held, boot, table, true_form)
        check(c['ci'][0] <= 0, '%s does not beat the true form beyond noise: %+.1f [%+.1f, %+.1f]'
              % (table, c['difference'], *c['ci']))
    if true_form in ('linear', 'interaction+rate'):
        # theta varies WITHIN voxels here: one theta per voxel cannot follow it, and pays
        # for 119 noisy estimates, so the closed form should win clearly
        reference = dl.TABLE_RATE if with_rate else dl.TABLE
        c = dl.compare(held, boot, true_form, reference)
        check(c['ci'][0] > 0, 'theta varies within voxels: %s beats %s clearly: %+.1f [%+.1f, %+.1f]'
              % (true_form, reference, c['difference'], *c['ci']))
    if with_rate:
        base = true_form[:-len(dl.RATE)]
        c = dl.compare(held, boot, true_form, base)
        check(c['ci'][0] > 0, 'rate effect detected: %s beats %s clearly: %+.1f [%+.1f, %+.1f]'
              % (true_form, base, c['difference'], *c['ci']))
        c = dl.compare(held, boot, dl.TABLE_RATE, dl.TABLE)
        check(c['ci'][0] > 0, 'voxel_table+rate beats voxel_table clearly: %+.1f [%+.1f, %+.1f]'
              % (c['difference'], *c['ci']))
        g = [fd['forms'][dl.TABLE_RATE]['g'] for fd in folds]
        check(all(abs(gi - true_beta[-1]) < 0.3 * abs(true_beta[-1]) for gi in g),
              'voxel_table+rate rate coefficient per fold %s (truth %.2f, within 30%%)'
              % (np.round(g, 3), true_beta[-1]))
    for richer in CONTAINS_TRUTH[true_form]:
        # a form that contains the truth must not score clearly below it (catches a failed fit)
        c = dl.compare(held, boot, richer, true_form)
        check(c['ci'][1] >= 0, '%s (contains the truth) not clearly worse: %+.1f [%+.1f, %+.1f]'
              % (richer, c['difference'], *c['ci']))
    nesting = ['fold %d: %s' % (fold['fold'], v) for fold in folds for v in fold['nesting_violations']]
    check(not nesting, 'nesting check on each training fold (every model >= the models it contains)%s'
          % (': ' + '; '.join(nesting) if nesting else ''))
    flags = [f for fold in folds for m, rec in fold['forms'].items() for f in rec.get('flags', [])]
    check(not flags, 'no flags on any model in either fold%s' % (': ' + '; '.join(flags) if flags else ''))

    stack = lambda a: np.tile(a, N_STACK)
    x_design = dl.design(true_form, stack(u_state), stack(tau_state), scale,
                         stack(ln_rate) if with_rate else None)
    x_big, y_big = sample_gumbel(dl.theta_of(true_beta, x_design), rng)
    cell_big = stack(cell)
    fit = dl.fit_form(x_design, dl.within_voxel_ranks(x_big, cell_big),
                      dl.within_voxel_ranks(y_big, cell_big), [np.zeros(len(true_beta))])
    errors = np.abs(fit['beta'] - true_beta) / np.abs(true_beta)
    check(errors.max() < TOLERANCE, 'coefficients at n = %d: %s (truth %s), worst error %.1f%%'
          % (len(x_big), np.round(fit['beta'], 4), np.round(true_beta, 3), 100 * errors.max()))
    print('     (%.0f s)' % (time.time() - start))

print('\nE. constant theta, but sizes shrink with cooling rate inside voxels (as in the data):'
      ' pooled ranks fake a rate effect; voxel x rate ranks remove it')
theta = np.full(len(cell), 3.36)
a, b = sample_gumbel(theta, rng)
x_size, y_size = a * np.exp(-0.3 * ln_rate), b * np.exp(-0.15 * ln_rate)   # monotone within each rate
for name, levels in (('pooled over rates', None), ('voxel x rate', RANK_LEVELS)):
    held, folds = dl.cross_validate(x_size, y_size, u_state, tau_state, cell, fold_id, rng,
                                    ln_rate=ln_rate, rank_levels=levels)
    boot = dl.bootstrap_scores(held, traj, N_BOOT, rng)
    g = [fd['forms'][dl.TABLE_RATE]['g'] for fd in folds]
    c = dl.compare(held, boot, dl.TABLE_RATE, dl.TABLE)
    k = dl.compare(held, boot, 'constant' + dl.RATE, 'constant')
    print('     %-18s voxel_table+rate g per fold %s;  table+rate - table %+.1f [%+.1f, %+.1f];'
          '  constant+rate - constant %+.1f [%+.1f, %+.1f]'
          % (name, np.round(g, 4), c['difference'], *c['ci'], k['difference'], *k['ci']))
    if levels is None:
        check(c['ci'][0] > 0 and min(g) > 0.02,
              'pooled ranks: spurious positive rate effect appears (the artifact this case is built to show)')
    else:
        check(c['ci'][0] <= 0 and k['ci'][0] <= 0 and max(abs(gi) for gi in g) < 0.02,
              'voxel x rate ranks: no rate effect beyond noise, |g| < 0.02 in every fold')
        print('     rank level per fold (events at voxel x rate, voxel x 3 rates, voxel): %s'
              % [fd['rank_level_events'] for fd in folds])

print('\n%d failure(s)' % len(failures))
sys.exit(1 if failures else 0)
