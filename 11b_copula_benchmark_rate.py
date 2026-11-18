"""Step 11b: step 11's kernel-vs-Gumbel benchmark with step 9's ranks (voxel x rate).

Input : out/joint_events.npz (step 5), out/marginals.json (step 6),
        out/dependence_law.json (step 9c: which voxels the family check picked)
Output: out/copula_benchmark_rate.json   (step 11's out/copula_benchmark.json is left alone)
        out/copula_benchmark_rate_events.npz   per-event held-out uniforms and log densities

Voxels: cell (18, 11), the three highest and the lowest measured tau_K among the
voxels with >= 300 paired events (step 9c's family check).  In each: paired
events of that voxel, ranks within voxel x rate with step 9's fallbacks
(dependence_law.grouped_uniforms, MIN_RANK_TRAIN), 2 folds by trajectory parity,
kernel bandwidth chosen inside each training half as in step 11, every family
in copulas.FAMILIES fitted on the training half.  Cell (18, 11) is also run with
step 11's pooled ranks as a reproduction check (should give +314).

Reported per voxel, per 1,000 events: kernel - gumbel and kernel - best
parametric family (best = highest held-out total, chosen on the same held-out
data, which flatters the family slightly), with paired trajectory-bootstrap
intervals (sampling only), and kernel - gumbel split by region of the square.
"""
import json
import time
import warnings
import numpy as np
import copulas
import dependence_law as dl

warnings.filterwarnings('ignore', category=RuntimeWarning)
N_BOOT = 2000
SEED = 0
INNER_GROUPS = 5            # as step 11
CORNER = 0.9                # as step 11
OUT = 'out/copula_benchmark_rate.json'
EVENTS_OUT = 'out/copula_benchmark_rate_events.npz'   # per event: held-out uniforms and ln c per model (for the figure)

t0 = time.time()
events = np.load('out/joint_events.npz')
with open('out/marginals.json') as handle:
    xmin = json.load(handle)['xmin']
with open('out/dependence_law.json') as handle:
    picked = json.load(handle)['diagnostics']['family']
n_bins = int(events['n_bins'])
du, dtau, cell_all, traj_all = events['du'], events['dtau'], events['cell'], events['traj']
rate_index_all = np.searchsorted(np.unique(events['rate']), events['rate'])
paired = (du < 0) & (dtau < 0) & (-du >= xmin['pe']) & (-dtau >= xmin['stress'])
PARAMETRIC = tuple(f for f in copulas.FAMILIES if f != 'independence')

RUNS = [(k, picked[k]['cell'], 'voxel_rate') for k in
        ('cell_18_11', 'highest_tau', 'second_highest_tau', 'third_highest_tau', 'lowest_tau')]
RUNS.append(('cell_18_11_pooled', picked['cell_18_11']['cell'], 'voxel'))


def benchmark(c, ranks):
    m = paired & (cell_all == c)
    x, y, traj, ri = -du[m], -dtau[m], traj_all[m], rate_index_all[m]
    n = len(x)
    levels = [ri, ri // 3, np.zeros(n, int)] if ranks == 'voxel_rate' else [np.zeros(n, int)]
    even = traj % 2 == 0
    held = {f: np.zeros(n) for f in ('kernel',) + PARAMETRIC}
    u_te_all, v_te_all = np.zeros(n), np.zeros(n)
    folds = []
    for train in (even, ~even):
        test = ~train
        u_tr, u_te, lev_u, fb_u = dl.grouped_uniforms(x, levels, train)
        v_tr, v_te, lev_v, fb_v = dl.grouped_uniforms(y, levels, train)
        u_te_all[test], v_te_all[test] = u_te, v_te
        h, _, flags = copulas.choose_kernel_bandwidth(u_tr, v_tr, (traj[train] // 2) % INNER_GROUPS)
        held['kernel'][test] = copulas.kernel_log_density(u_tr, v_tr, u_te, v_te, h)
        for f in PARAMETRIC:
            fit = copulas.fit_family(f, u_tr, v_tr)
            held[f][test] = copulas.log_density(f, fit['params'], u_te, v_te)
            flags += ['%s: %s' % (f, fl) for fl in fit['at_bound']]
        folds.append(dict(n_train=int(train.sum()), n_test=int(test.sum()), bandwidth=float(h), flags=flags,
                          rank_level_counts=np.bincount(lev_u, minlength=len(levels)).tolist(),
                          no_training_reference=int(fb_u + fb_v)))
    boot = dl.bootstrap_scores(held, traj, N_BOOT, np.random.default_rng(SEED))
    totals = {f: float(held[f].sum()) for f in held}
    best = max(PARAMETRIC, key=totals.get)

    def per1000(r):
        return dict(r, per_1000_events=1000 * r['difference'] / n,
                    ci_per_1000=[1000 * v / n for v in r['ci']])

    regions = {}
    upper = (u_te_all > CORNER) & (v_te_all > CORNER)
    lower = (u_te_all < 1 - CORNER) & (v_te_all < 1 - CORNER)
    for name, mask in dict(upper_corner=upper, lower_corner=lower, bulk=~(upper | lower)).items():
        masked = {f: held[f] * mask for f in ('kernel', 'gumbel')}
        r = dl.compare(masked, dl.bootstrap_scores(masked, traj, N_BOOT, np.random.default_rng(SEED)),
                       'kernel', 'gumbel')
        regions[name] = dict(per1000(r), n_events=int(mask.sum()))
    return dict(cell=int(c), cell_uv=[int(c // n_bins), int(c % n_bins)], n=n, ranks=ranks,
                n_trajectories=int(len(np.unique(traj))), held_out_totals=totals, best_parametric=best,
                kernel_minus_gumbel=per1000(dl.compare(held, boot, 'kernel', 'gumbel')),
                kernel_minus_best=per1000(dl.compare(held, boot, 'kernel', best)),
                by_region=regions, folds=folds,
                _events=dict(u=u_te_all, v=v_te_all, **{'ll_' + f: held[f] for f in held}))


results, per_event = {}, {}
for label, c, ranks in RUNS:
    r = benchmark(c, ranks)
    per_event.update({'%s/%s' % (label, k): v for k, v in r.pop('_events').items()})
    results[label] = r
    kg, kb = r['kernel_minus_gumbel'], r['kernel_minus_best']
    print('%-19s (%2d, %2d) n %4d %-10s  kernel-gumbel %+7.1f [%+6.1f, %+6.1f] = %+6.1f/1000   '
          'best %-16s kernel-best %+7.1f [%+6.1f, %+6.1f] = %+6.1f/1000   h %s  (%.0f s)'
          % (label, *r['cell_uv'], r['n'], ranks, kg['difference'], *kg['ci'], kg['per_1000_events'],
             r['best_parametric'], kb['difference'], *kb['ci'], kb['per_1000_events'],
             '/'.join('%.3f' % f['bandwidth'] for f in r['folds']), time.time() - t0))
    for name, g in r['by_region'].items():
        print('      %-13s %5d events  kernel-gumbel %+7.1f [%+6.1f, %+6.1f]'
              % (name, g['n_events'], g['difference'], *g['ci']))

# reproduction check: pooled ranks in cell (18, 11) must give step 11's number
with open('out/copula_benchmark.json') as handle:
    step11 = json.load(handle)['comparisons']['kernel - gumbel']['difference']
repro = results['cell_18_11_pooled']['kernel_minus_gumbel']['difference']
print('\nreproduction: pooled ranks %.2f vs step 11 %.2f' % (repro, step11))
assert abs(repro - step11) < 0.5, 'pooled-rank run does not reproduce step 11'

with open(OUT, 'w') as handle:
    json.dump(dict(step='11b', n_boot=N_BOOT, seed=SEED, corner_threshold=CORNER, min_rank_train=dl.MIN_RANK_TRAIN,
                   rank_rule=dl.RANK_RULE, results=results,
                   note='kernel = transformation kernel copula; best parametric chosen on the held-out data; '
                        'intervals are sampling-only paired trajectory bootstraps'), handle, indent=1)
np.savez_compressed(EVENTS_OUT, **per_event)
print('wrote %s and %s  (%.0f s)' % (OUT, EVENTS_OUT, time.time() - t0))
