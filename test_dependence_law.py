"""Synthetic checks for dependence_law.py, run before step 9 touches real sizes.

    python test_dependence_law.py

The real paired events' STATES (u, tau), voxels and trajectories are kept
(so the design, folds and voxel table look exactly like the real fit's), but
their sizes are replaced by Gumbel draws with a KNOWN theta(u, tau):
  A. product form   theta = 1 + exp(-0.5 + 0.4 * 100 u tau)
  B. constant       theta = 3.36 everywhere (step 7's value in cell (18, 11))
  C. linear form    eta = 0.8 + 0.25 z_u + 0.3 z_tau
Checks, with the same 2-fold trajectory split and trajectory bootstrap as step 9:
  * the true form is best on held-out data or tied with the best (a richer
    form that nests it can tie);
  * when theta varies, the true form beats the constant clearly;
  * when theta is constant, the linear and quadratic forms do not beat it
    beyond noise;
  * the voxel table does not beat the true form beyond noise;
  * in-sample coefficients of the true form are within 5% of the truth, on
    N_STACK independent samples of the layout pooled (at the real n the
    product form's intercept alone scatters by ~4.6%, so 5% is checked where
    it measures the code rather than the sample).
Also checks the per-event Gumbel sampler against Gumbel's tau = 1 - 1/theta.
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

scale = dl.scaling(u_state, tau_state)
cases = [('product', np.array([-0.5, 0.4])),
         ('constant', np.array([np.log(3.36 - 1)])),
         ('linear', np.array([0.8, 0.25, 0.3]))]
for true_form, true_beta in cases:
    start = time.time()
    theta = dl.theta_of(true_beta, dl.design(true_form, u_state, tau_state, scale))
    x_size, y_size = sample_gumbel(theta, rng)
    print('\n%s truth %s  (theta %.2f to %.2f)' % (true_form, np.round(true_beta, 3),
                                                   theta.min(), theta.max()))
    held, folds = dl.cross_validate(x_size, y_size, u_state, tau_state, cell, fold_id, rng)
    boot = dl.bootstrap_scores(held, traj, N_BOOT, rng)
    totals = {m: held[m].sum() for m in dl.MODELS}
    best = max(totals, key=totals.get)
    for m in dl.MODELS:
        print('     %-12s held-out %10.1f   vs best %+8.1f' % (m, totals[m], totals[m] - totals[best]))
    vs_best = dl.compare(held, boot, true_form, best)
    check(best == true_form or vs_best['ci'][1] >= 0,
          'true form best or tied: best %s, true - best %+.1f [%+.1f, %+.1f]'
          % (best, vs_best['difference'], *vs_best['ci']))
    if true_form == 'constant':
        for richer in ('linear', 'quadratic'):
            c = dl.compare(held, boot, richer, 'constant')
            check(c['ci'][0] <= 0, '%s does not beat constant beyond noise: %+.1f [%+.1f, %+.1f]'
                  % (richer, c['difference'], *c['ci']))
    else:
        c = dl.compare(held, boot, true_form, 'constant')
        check(c['ci'][0] > 0, 'true form beats constant clearly: %+.1f [%+.1f, %+.1f]'
              % (c['difference'], *c['ci']))
    c = dl.compare(held, boot, 'voxel_table', true_form)
    check(c['ci'][0] <= 0, 'voxel table does not beat the true form beyond noise: %+.1f [%+.1f, %+.1f]'
          % (c['difference'], *c['ci']))
    flags = [f for fold in folds for f in fold['forms'][true_form]['flags']]
    check(not flags, 'no flags on the true form in either fold%s' % (': ' + '; '.join(flags) if flags else ''))

    stack = lambda a: np.tile(a, N_STACK)
    theta_big = dl.theta_of(true_beta, dl.design(true_form, stack(u_state), stack(tau_state), scale))
    x_big, y_big = sample_gumbel(theta_big, rng)
    cell_big = stack(cell)
    x_design = dl.design(true_form, stack(u_state), stack(tau_state), scale)
    fit = dl.fit_form(x_design, dl.within_voxel_ranks(x_big, cell_big),
                      dl.within_voxel_ranks(y_big, cell_big), [np.zeros(len(true_beta))])
    errors = np.abs(fit['beta'] - true_beta) / np.abs(true_beta)
    check(errors.max() < TOLERANCE, 'coefficients at n = %d: %s (truth %s), worst error %.1f%%'
          % (len(x_big), np.round(fit['beta'], 4), np.round(true_beta, 3), 100 * errors.max()))
    print('     (%.0f s)' % (time.time() - start))

print('\n%d failure(s)' % len(failures))
sys.exit(1 if failures else 0)
