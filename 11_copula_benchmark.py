"""Step 11: how much better could ANY copula do than Gumbel in cell (18, 11)?

Input : out/joint_events.npz (step 5), out/marginals.json (step 6)
Output: out/copula_benchmark.json

The sample, uniforms and split are step 7's: paired events in the most-visited
cell, ranks (pseudo-observations), even trajectory ids against odd, with
held-out events ranked against the training ECDF.  The benchmark is a
transformation kernel copula (copulas.kernel_log_density), which assumes no
family shape.  Its bandwidth is chosen inside each training half by
leave-one-group-out ln L over blocks of trajectories, so the test half is
never used to tune it.

Reading the result: on true Gumbel data the kernel scores slightly BELOW
Gumbel (the price of having no shape; test_copulas.py section 5).  So
    kernel - gumbel <= ~0   Gumbel is about as good as a copula gets here
    kernel - gumbel >> 0    Gumbel's shape is wrong, by about that much
The difference is also split by region of the unit square, to show where
any gain comes from.  Nothing else is fitted.
"""
import json
import time
import warnings
import numpy as np
from scipy import stats
import copulas
import dependence_law as dl

warnings.filterwarnings('ignore', category=RuntimeWarning)
N_BOOT = 2000
SEED = 0
Q_TAIL = 0.05
INNER_GROUPS = 5            # trajectory blocks for choosing the bandwidth
PARAMETRIC = ('gumbel', 'survival_clayton', 't')
CORNER = 0.9                # region split: both uniforms above / below these
N_RESAMPLE = 400000         # draws from the kernel copula for its corner statistic

start_time = time.time()
rng = np.random.default_rng(SEED)
events = np.load('out/joint_events.npz')
with open('out/marginals.json') as handle:
    settings = json.load(handle)
n_bins = int(events['n_bins'])
c = settings['most_visited_cell']
xmin = settings['xmin']
du, dtau, traj_all = events['du'], events['dtau'], events['traj']
paired = ((events['cell'] == c) & (du < 0) & (dtau < 0)
          & (-du >= xmin['pe']) & (-dtau >= xmin['stress']))
x, y, traj = -du[paired], -dtau[paired], traj_all[paired]
n = len(x)
print('cell %d = (%d, %d): %d paired events in %d trajectories'
      % (c, c // n_bins, c % n_bins, n, len(np.unique(traj))))

# --- held-out scores, step 7's split ---------------------------------------------
even = (traj % 2) == 0
models = ('kernel',) + PARAMETRIC
held = {m: np.zeros(n) for m in models}
u_test_all, v_test_all = np.zeros(n), np.zeros(n)
folds = []
for train in (even, ~even):
    test = ~train
    u_tr, v_tr = copulas.pseudo_observations(x[train]), copulas.pseudo_observations(y[train])
    u_te, v_te = copulas.ecdf_transform(x[train], x[test]), copulas.ecdf_transform(y[train], y[test])
    u_test_all[test], v_test_all[test] = u_te, v_te
    h, scores, flags = copulas.choose_kernel_bandwidth(u_tr, v_tr, (traj[train] // 2) % INNER_GROUPS)
    held['kernel'][test] = copulas.kernel_log_density(u_tr, v_tr, u_te, v_te, h)
    fold = dict(n_train=int(train.sum()), n_test=int(test.sum()), bandwidth=h, bandwidth_flags=flags,
                bandwidth_scores=dict(zip([float(b) for b in copulas.KERNEL_BANDWIDTHS], scores.tolist())),
                params={})
    for family in PARAMETRIC:
        fit = copulas.fit_family(family, u_tr, v_tr)
        held[family][test] = copulas.log_density(family, fit['params'], u_te, v_te)
        fold['params'][family] = [float(p) for p in fit['params']]
        fold['bandwidth_flags'] += ['%s: %s' % (family, f) for f in fit['at_bound']]
    folds.append(fold)
    print('train on %s ids: bandwidth %.3f %s' % ('even' if train[0] == even[0] else 'odd', h,
                                                 '; '.join(fold['bandwidth_flags'])))

boot = dl.bootstrap_scores(held, traj, N_BOOT, rng)
totals = {m: float(held[m].sum()) for m in models}
print('\nheld-out ln L gain over independence (step 7 reported gumbel 3165.8 on this split)')
for m in models:
    print('  %-17s %8.1f' % (m, totals[m]))

comparisons = {}
gumbel_gain = totals['gumbel']
print('\nkernel minus each parametric copula, 95%% CI from %d trajectory resamples (sampling only)' % N_BOOT)
for family in PARAMETRIC:
    r = dl.compare(held, boot, 'kernel', family)
    r['per_1000_events'] = 1000 * r['difference'] / n
    comparisons['kernel - %s' % family] = r
    print('  kernel - %-17s %+8.1f [%+7.1f, %+7.1f]   P(kernel better) %.3f   %+.2f per 1,000 events'
          % (family, r['difference'], *r['ci'], r['p_positive'], r['per_1000_events']))
k_g = comparisons['kernel - gumbel']
print('  kernel - gumbel is %.1f%% of Gumbel\'s gain over independence' % (100 * k_g['difference'] / gumbel_gain))

# --- where does the difference come from? ----------------------------------------
upper = (u_test_all > CORNER) & (v_test_all > CORNER)
lower = (u_test_all < 1 - CORNER) & (v_test_all < 1 - CORNER)
regions = dict(upper_corner=upper, lower_corner=lower, rest=~(upper | lower))
by_region = {}
print('\nkernel - gumbel by region of the held-out uniforms (corners: both > %.1f / both < %.1f)'
      % (CORNER, 1 - CORNER))
for name, mask in regions.items():
    masked = {m: held[m] * mask for m in ('kernel', 'gumbel')}
    r = dl.compare(masked, dl.bootstrap_scores(masked, traj, N_BOOT, np.random.default_rng(SEED)),
                   'kernel', 'gumbel')
    r['n_events'] = int(mask.sum())
    by_region[name] = r
    print('  %-13s %5d events   %+8.1f [%+7.1f, %+7.1f]' % (name, mask.sum(), r['difference'], *r['ci']))

# --- does the kernel copula reproduce the corner statistics? ------------------------
u_all, v_all = copulas.pseudo_observations(x), copulas.pseudo_observations(y)
h_all, _, flags_all = copulas.choose_kernel_bandwidth(u_all, v_all, (traj // 2) % INNER_GROUPS)
z = np.vstack([stats.norm.ppf(copulas._clip(u_all)), stats.norm.ppf(copulas._clip(v_all))])
draws = stats.gaussian_kde(z, bw_method=h_all).resample(N_RESAMPLE, seed=SEED)
u_k, v_k = copulas.pseudo_observations(draws[0]), copulas.pseudo_observations(draws[1])
kernel_corners = copulas.empirical_joint_tail(u_k, v_k, Q_TAIL)
empirical_corners = copulas.empirical_joint_tail(u_all, v_all, Q_TAIL)
gumbel_all = copulas.fit_family('gumbel', u_all, v_all)['params']
gumbel_corners = copulas.joint_tail('gumbel', gumbel_all, Q_TAIL)
kernel_tau = float(stats.kendalltau(u_k[:20000], v_k[:20000])[0])
print('\ncorner statistics at q = %.2f (lower, upper) and Kendall tau, all events'
      % Q_TAIL)
print('  data            %.3f  %.3f   tau %.3f' % (*empirical_corners, stats.kendalltau(x, y)[0]))
print('  kernel (h %.3f) %.3f  %.3f   tau %.3f   %s' % (h_all, *kernel_corners, kernel_tau,
                                                       '; '.join(flags_all)))
print('  gumbel          %.3f  %.3f   tau %.3f' % (*gumbel_corners,
                                                   copulas.kendall_tau('gumbel', gumbel_all)))

with open('out/copula_benchmark.json', 'w') as handle:
    json.dump(dict(cell=int(c), n=n, n_boot=N_BOOT, seed=SEED, held_out_totals=totals,
                   comparisons=comparisons, by_region=by_region, corner_threshold=CORNER, folds=folds,
                   corners=dict(q=Q_TAIL, data=list(map(float, empirical_corners)),
                                kernel=list(map(float, kernel_corners)),
                                gumbel=list(map(float, gumbel_corners)),
                                kernel_bandwidth_all=h_all, kernel_tau=kernel_tau),
                   note='kernel = transformation kernel copula (no family shape); '
                        'intervals are sampling-only trajectory bootstraps'), handle, indent=1)
print('\nwrote out/copula_benchmark.json  (%.0f s)' % (time.time() - start_time))
