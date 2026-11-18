"""Step 9: does a closed formula theta(u, tau, rate) describe the Gumbel dependence
as well as one theta per voxel?  See PLAN_step9.md.

Input : out/joint_events.npz (step 5), out/marginals.json (step 6),
        out/voxel_dependence.json (step 8: the 119 voxels, compared with the unpooled tau)
Output: out/dependence_law_scoreboard<suffix>.json   (suffix '' for the main ranks)
        out/dependence_law_checks.json               (with `checks`: the pre-fit checks only)
The final out/dependence_law.json is assembled by 9c_dependence_diagnostics.py
from this file, step 9b's replicates and step 9c's diagnostics.

    python 9_fit_dependence_law.py [ranks=voxel_rate | voxel_rate3 | voxel]   (~9 min each)
    python 9_fit_dependence_law.py checks

Ranks (uniforms) are taken within rank groups, finest first, falling back to a
coarser level when a group has < dependence_law.MIN_RANK_TRAIN training events:
  voxel_rate   voxel x cooling rate -> voxel x 3 rate groups -> voxel   (MAIN)
  voxel_rate3  voxel x 3 rate groups (slow / mid / fast) -> voxel       (sensitivity)
  voxel        voxel only, pooled over rates (the earlier runs)         (comparison)
Pooling over rates adds spurious concordance: fast-cooled events are smaller
on both marginals (pre-fit check 3).

Pre-fit checks (run first, before any closed form is fitted):
  fold balance  trajectories and paired events per fold and cooling rate, for
                both splits
  cooling rate  Kendall tau per (voxel, rate); does tau move with ln(rate) at
                fixed voxel?  Pooled within-voxel slope, and slow-vs-fast
                difference per voxel, with intervals from resampling
                trajectories within each rate.  Rate as in step 3:
                ln(rate / rate_mid), rate_mid the middle of the nine rates.
  marginals     two-sample KS (fast vs slow) of -du and -dtau inside each
                voxel, p-values by permuting whole trajectories; a shift is
                a caveat on the ranks pooled over rates within each voxel.
Every form is also fitted with + g ln(rate/rate_mid); the like-for-like
reference for those is voxel_table+rate (one eta per voxel + a shared g).

Main set: paired events (cutoffs of step 6) in voxels with >= 100 paired events.
Models: Gumbel with theta = 1 + exp(eta), eta constant / product (100 u tau) /
linear / interaction / quadratic in standardised (u, tau), each also with
+ g ln(rate/rate_mid); references: one theta per voxel (voxel_table) and
voxel_table+rate.  In-sample fits, then held-out scores by whole trajectories:
2 folds (even vs odd ids, as in steps 3 and 7) and 5 folds (id mod 5), with
sampling-only intervals (resample trajectories, re-sum held-out per-event
scores).  Refitting noise: step 9b (repeated splits).
Run test_dependence_law.py first.
"""
import json
import sys
import time
import warnings
import numpy as np
import copulas
import dependence_law as dl

warnings.filterwarnings('ignore', category=RuntimeWarning)
MIN_PAIRED = 100
N_BOOT = 2000
CHECKS_ONLY = 'checks' in sys.argv[1:]
RANKS = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('ranks=')), 'voxel_rate')
assert RANKS in ('voxel_rate', 'voxel_rate3', 'voxel'), RANKS
SUFFIX = dict(voxel_rate='', voxel_rate3='_voxel_rate3', voxel='_pooled')[RANKS]
MIN_TAU_GROUP = 10         # a (voxel, rate) group needs this many events for its tau in the unpooled voxel tau
N_KS_PERMUTATIONS = 200   # trajectory permutations for the marginal-vs-rate KS check
N_RATE_BOOT = 200          # trajectory resamples for the cooling-rate check
MIN_RATE_EVENTS = 30       # a (voxel, rate) group needs this many paired events for its tau
SEED = 0
CELL_OF_STEP_7 = None      # taken from out/marginals.json

start_time = time.time()
rng = np.random.default_rng(SEED)
events = np.load('out/joint_events.npz')
with open('out/marginals.json') as handle:
    msettings = json.load(handle)
with open('out/voxel_dependence.json') as handle:
    voxel_dependence = json.load(handle)
CELL_OF_STEP_7 = msettings['most_visited_cell']
xmin = msettings['xmin']
n_bins = int(events['n_bins'])
du, dtau, cell_all = events['du'], events['dtau'], events['cell']
paired = (du < 0) & (dtau < 0) & (-du >= xmin['pe']) & (-dtau >= xmin['stress'])
counts = np.bincount(cell_all[paired], minlength=n_bins * n_bins)


def select(mask):
    return dict(x=-du[mask], y=-dtau[mask], u=events['u'][mask], tau=events['tau'][mask],
                cell=cell_all[mask], traj=events['traj'][mask], rate=events['rate'][mask],
                rate_index=np.searchsorted(np.unique(events['rate']), events['rate'][mask]))


def rank_levels(data):
    c, r = data['cell'], data['rate_index']           # 9 rates; 3 groups = 3 slowest / middle / fastest
    return dict(voxel_rate=[c * 9 + r, c * 3 + r // 3, c], voxel_rate3=[c * 3 + r // 3, c], voxel=[c])[RANKS]


main = select(paired & (counts[cell_all] >= MIN_PAIRED))
everything = select(paired)
print('main set: %d paired events in %d voxels (>= %d each), %d trajectories;  all paired: %d'
      % (len(main['x']), len(np.unique(main['cell'])), MIN_PAIRED, len(np.unique(main['traj'])),
         len(everything['x'])))
n_params = dict(dl.N_COEFFICIENTS, voxel_table=len(np.unique(main['cell'])))


# --- pre-fit check 1: fold balance ------------------------------------------------------
traj_ids, first = np.unique(events['traj'], return_index=True)
traj_rate = events['rate'][first]
distinct_rates = np.unique(traj_rate)
RATE_MID = distinct_rates[len(distinct_rates) // 2]          # as step 3


def fold_balance(k):
    """Trajectories and main-set paired events per (fold, rate) for the split id mod k."""
    traj_fold = traj_ids % k
    event_fold = main['traj'] % k
    table = dict(trajectories=[[int(((traj_fold == f) & (traj_rate == r)).sum()) for r in distinct_rates]
                               for f in range(k)],
                 paired_events=[[int(((event_fold == f) & (main['rate'] == r)).sum()) for r in distinct_rates]
                                for f in range(k)])
    ev = np.array(table['paired_events'], float)
    share = ev / ev.sum(axis=0)                          # each fold's share of each rate's events
    table['worst_share_deviation'] = float(np.abs(share * k - 1).max())   # relative to 1/k
    table['fold_totals'] = ev.sum(axis=1).astype(int).tolist()
    table['worst_total_deviation'] = float(np.abs(ev.sum(axis=1) * k / ev.sum() - 1).max())
    print('\nFOLD BALANCE, trajectory id mod %d   (columns: cooling rates %s)'
          % (k, ' '.join('%.3g' % r for r in distinct_rates)))
    for f in range(k):
        print('  fold %d  trajectories  %s' % (f, ' '.join('%5d' % c for c in table['trajectories'][f])))
    for f in range(k):
        print('  fold %d  paired events %s  total %6d'
              % (f, ' '.join('%5d' % c for c in table['paired_events'][f]), table['fold_totals'][f]))
    print('  worst deviation from an equal share: %.1f%% within a rate, %.1f%% in fold totals'
          % (100 * table['worst_share_deviation'], 100 * table['worst_total_deviation']))
    return table


balance = {'2_folds': fold_balance(2), '5_folds': fold_balance(5)}


# --- pre-fit check 2: does Kendall tau move with cooling rate at fixed voxel? ----------------
def rate_check():
    from scipy import stats
    ln_rate = np.log(main['rate'] / RATE_MID)
    cells = np.unique(main['cell'])
    groups = [(c, r) for c in cells for r in distinct_rates
              if ((main['cell'] == c) & (main['rate'] == r)).sum() >= MIN_RATE_EVENTS]
    gid = np.full(len(main['x']), -1)
    for i, (c, r) in enumerate(groups):
        gid[(main['cell'] == c) & (main['rate'] == r)] = i
    members = [np.flatnonzero(gid == i) for i in range(len(groups))]
    g_cell = np.array([c for c, _ in groups])
    g_lnr = np.log(np.array([r for _, r in groups]) / RATE_MID)
    fast_idx = {c: np.flatnonzero((main['cell'] == c) & (main['rate'] > RATE_MID)) for c in cells}
    slow_idx = {c: np.flatnonzero((main['cell'] == c) & (main['rate'] < RATE_MID)) for c in cells}

    def tau_of(idx, weights):
        rep_idx = np.repeat(idx, weights[idx])
        if len(rep_idx) < 3:
            return np.nan, 0
        return stats.kendalltau(main['x'][rep_idx], main['y'][rep_idx])[0], len(rep_idx)

    def statistics(weights):
        """weights = multiplicity of each event (trajectory resample).  Returns the pooled
        within-voxel slope of tau on ln rate (weights = group sizes), per-voxel
        tau(fast) - tau(slow), and the group taus."""
        tau_g, n_g = np.full(len(groups), np.nan), np.zeros(len(groups))
        for i, idx in enumerate(members):
            tau_g[i], n_g[i] = tau_of(idx, weights)
        num = den = 0.0
        for c in cells:
            m = (g_cell == c) & np.isfinite(tau_g)
            if m.sum() < 2:
                continue
            w = n_g[m]
            x = g_lnr[m] - np.average(g_lnr[m], weights=w)
            num += np.sum(w * x * (tau_g[m] - np.average(tau_g[m], weights=w)))
            den += np.sum(w * x * x)
        diff = {}
        for c in cells:
            if len(fast_idx[c]) >= MIN_RATE_EVENTS and len(slow_idx[c]) >= MIN_RATE_EVENTS:
                diff[int(c)] = tau_of(fast_idx[c], weights)[0] - tau_of(slow_idx[c], weights)[0]
        return (num / den if den > 0 else np.nan), diff, tau_g

    ones = np.ones(len(main['x']), int)
    slope, diff, tau_g = statistics(ones)
    # within-voxel drift of the state with ln rate (pooled within-voxel slope, on events)
    drift = {}
    for name, v in (('z_u', main['u']), ('z_tau', main['tau'])):
        z = (v - v.mean()) / v.std()
        num = den = 0.0
        for c in cells:
            m = main['cell'] == c
            x = ln_rate[m] - ln_rate[m].mean()
            num += np.sum(x * (z[m] - z[m].mean()))
            den += np.sum(x * x)
        drift[name] = float(num / den)
    # trajectory bootstrap, stratified by rate (each rate keeps its number of trajectories)
    brng = np.random.default_rng(SEED + 1)
    order = np.argsort(main['traj'], kind='stable')
    t_sorted = main['traj'][order]
    tr_index = {t: order[a:b] for t, a, b in zip(*np.unique(t_sorted, return_index=True),
                                                  np.append(np.unique(t_sorted, return_index=True)[1][1:],
                                                            len(t_sorted)))}
    by_rate = {r: traj_ids[traj_rate == r] for r in distinct_rates}
    boot_slope, boot_diff = [], {c: [] for c in diff}
    for b in range(N_RATE_BOOT):
        weights = np.zeros(len(main['x']), int)
        for r, ids in by_rate.items():
            for t in brng.choice(ids, len(ids)):
                if t in tr_index:
                    weights[tr_index[t]] += 1
        s_b, d_b, _ = statistics(weights)
        boot_slope.append(s_b)
        for c in diff:
            boot_diff[c].append(d_b.get(c, np.nan))
    boot_slope = np.array(boot_slope)
    ci = [float(np.nanpercentile(boot_slope, 2.5)), float(np.nanpercentile(boot_slope, 97.5))]
    per_voxel = {}
    for c, d in diff.items():
        bd = np.array(boot_diff[c])
        per_voxel[c] = dict(fast_minus_slow=float(d), ci=[float(np.nanpercentile(bd, 2.5)),
                                                          float(np.nanpercentile(bd, 97.5))])
    up = sum(v['ci'][0] > 0 for v in per_voxel.values())
    down = sum(v['ci'][1] < 0 for v in per_voxel.values())
    d_all = np.array([v['fast_minus_slow'] for v in per_voxel.values()])
    span = np.log(distinct_rates[-1] / distinct_rates[0])
    out = dict(rate_mid=float(RATE_MID), min_events_per_group=MIN_RATE_EVENTS, n_boot=N_RATE_BOOT,
               n_groups=len(groups), n_voxels_with_groups=int(len(np.unique(g_cell))),
               slope_tau_per_ln_rate=float(slope), slope_ci=ci,
               p_slope_positive=float(np.nanmean(boot_slope > 0)),
               tau_change_over_rate_range=float(slope * span), tau_change_ci=[ci[0] * span, ci[1] * span],
               state_drift_per_ln_rate=drift,
               fast_minus_slow=dict(n_voxels=len(per_voxel), ci_above_zero=int(up), ci_below_zero=int(down),
                                    median=float(np.median(d_all)), mean=float(d_all.mean()),
                                    per_voxel=per_voxel),
               group_tau=[dict(cell=int(c), rate=float(r), tau=float(t)) for (c, r), t in zip(groups, tau_g)])
    print('\nCOOLING RATE at fixed voxel (Kendall tau; %d (voxel, rate) groups with >= %d events in %d voxels)'
          % (len(groups), MIN_RATE_EVENTS, out['n_voxels_with_groups']))
    print('  pooled within-voxel slope d tau / d ln(rate): %+.4f [%+.4f, %+.4f]  (P > 0: %.2f; %d resamples of '
          'trajectories within each rate)' % (slope, *ci, out['p_slope_positive'], N_RATE_BOOT))
    print('  implied tau change from slowest to fastest rate (x%.0f): %+.3f [%+.3f, %+.3f]'
          % (np.exp(span), out['tau_change_over_rate_range'], *out['tau_change_ci']))
    print('  fast (> rate_mid) minus slow (< rate_mid), per voxel: %d voxels; median %+.3f, mean %+.3f; '
          'CI above 0 in %d, below 0 in %d (about %.0f each expected by chance)'
          % (len(per_voxel), np.median(d_all), d_all.mean(), up, down, 0.025 * len(per_voxel)))
    print('  state drift inside voxels per unit ln(rate): z_u %+.4f, z_tau %+.4f'
          % (drift['z_u'], drift['z_tau']))
    return out


rate_dependence = rate_check()


# --- pre-fit check 3: do the size marginals shift with rate inside voxels? --------------------
def marginal_rate_check():
    """Per voxel with >= MIN_RATE_EVENTS fast (> rate_mid) and slow (< rate_mid) paired
    events: two-sample KS on -du and on -dtau.  Two p-values: scipy's (treats events as
    independent, so too small: events within a trajectory are correlated) and a
    permutation p-value that swaps fast/slow labels between whole TRAJECTORIES of the
    voxel (rate is a trajectory property, so this keeps the correlation)."""
    from scipy import stats
    prng = np.random.default_rng(SEED + 2)
    rows = {}
    for c in np.unique(main['cell']):
        m = main['cell'] == c
        fast, slow = m & (main['rate'] > RATE_MID), m & (main['rate'] < RATE_MID)
        if fast.sum() < MIN_RATE_EVENTS or slow.sum() < MIN_RATE_EVENTS:
            continue
        keep = fast | slow
        trajs, t_index = np.unique(main['traj'][keep], return_inverse=True)
        t_fast = np.array([main['rate'][keep][t_index == i][0] > RATE_MID for i in range(len(trajs))])
        row = {}
        for name, values in (('pe', main['x']), ('stress', main['y'])):
            vals = values[keep]
            d = stats.ks_2samp(vals[t_fast[t_index]], vals[~t_fast[t_index]])
            exceed = 0
            for _ in range(N_KS_PERMUTATIONS):
                lab = prng.permutation(t_fast)[t_index]
                if lab.all() or not lab.any():
                    continue
                exceed += stats.ks_2samp(vals[lab], vals[~lab]).statistic >= d.statistic
            row[name] = dict(ks=float(d.statistic), p_events=float(d.pvalue),
                             p_trajectory_permutation=float((exceed + 1) / (N_KS_PERMUTATIONS + 1)),
                             log_median_ratio_fast_slow=float(np.log(np.median(vals[t_fast[t_index]])
                                                                    / np.median(vals[~t_fast[t_index]]))))
        rows[int(c)] = row
    out = dict(n_voxels=len(rows), n_permutations=N_KS_PERMUTATIONS, alpha=0.05, per_voxel=rows)
    print('\nSIZE MARGINALS vs COOLING RATE inside voxels (fast > rate_mid vs slow < rate_mid; %d voxels)'
          % len(rows))
    for name, label_ in (('pe', '-du'), ('stress', '-dtau')):
        p_ev = np.array([r[name]['p_events'] for r in rows.values()])
        p_perm = np.array([r[name]['p_trajectory_permutation'] for r in rows.values()])
        lmr = np.array([r[name]['log_median_ratio_fast_slow'] for r in rows.values()])
        out[name] = dict(significant_events=int((p_ev < 0.05).sum()),
                         significant_trajectory_permutation=int((p_perm < 0.05).sum()),
                         expected_by_chance=0.05 * len(rows),
                         median_log_median_ratio=float(np.median(lmr)),
                         fraction_fast_larger=float(np.mean(lmr > 0)))
        o = out[name]
        print('  %-6s KS p < 0.05 in %d / %d voxels by trajectory permutation (%d by the event-level test, '
              'which ignores within-trajectory correlation); %.1f expected by chance.  Median size fast/slow: '
              'x%.3f (median over voxels); fast larger in %.0f%% of voxels'
              % (label_, o['significant_trajectory_permutation'], len(rows), o['significant_events'],
                 o['expected_by_chance'], np.exp(o['median_log_median_ratio']), 100 * o['fraction_fast_larger']))
    shifted = any(out[q]['significant_trajectory_permutation'] > 2 * out[q]['expected_by_chance'] + 2
                  for q in ('pe', 'stress'))
    if not shifted:
        out['caveat'] = 'no clear shift of the size marginals with rate inside voxels'
    elif RANKS == 'voxel':
        out['caveat'] = ('CAVEAT: the size marginals shift with cooling rate inside voxels, and this run pools ranks '
                         'over rates within each voxel: part of what the copula sees comes from the rate mixing '
                         'marginals, not from dependence (the main run ranks within voxel x rate).')
    else:
        out['caveat'] = ('the size marginals shift with cooling rate inside voxels; this run ranks within %s, so the '
                         'shift does not enter the uniforms except for events that fall back to a coarser rank '
                         'group (counted per fold)' % dict(voxel_rate='voxel x rate', voxel_rate3='voxel x 3 rate groups')[RANKS])
    print('  ' + out['caveat'])
    return out


marginal_rate = marginal_rate_check()


# --- pre-fit check 4: per-voxel Kendall tau WITHOUT pooling over rates; group regression -------------
def unpooled_tau():
    """Per voxel: Kendall tau = event-weighted mean of the tau of its (voxel, rate) groups
    with >= MIN_TAU_GROUP events, so the marginal shift between rates cannot add
    concordance.  Intervals: trajectories resampled within each rate.  Compared with
    step 8's pooled tau.  Group regression: tau of each (voxel, rate) group with
    >= MIN_RATE_EVENTS events on its median z_u, z_tau and ln(rate/rate_mid),
    weighted by group size (model-free direction of each effect)."""
    from scipy import stats
    zu_all = (main['u'] - main['u'].mean()) / main['u'].std()
    zt_all = (main['tau'] - main['tau'].mean()) / main['tau'].std()
    cells = np.unique(main['cell'])
    groups, members = [], []
    for c in cells:
        for r in distinct_rates:
            idx = np.flatnonzero((main['cell'] == c) & (main['rate'] == r))
            if len(idx) >= MIN_TAU_GROUP:
                groups.append((int(c), float(r)))
                members.append(idx)
    n_g = np.array([len(m) for m in members], float)
    g_cell = np.array([c for c, _ in groups])
    covariates = np.column_stack([np.ones(len(groups)),
                                  [np.median(zu_all[m]) for m in members],
                                  [np.median(zt_all[m]) for m in members],
                                  np.log(np.array([r for _, r in groups]) / RATE_MID)])
    regress = n_g >= MIN_RATE_EVENTS
    voxel_cov = np.column_stack([np.ones(len(cells)),
                                 [np.median(zu_all[main['cell'] == c]) for c in cells],
                                 [np.median(zt_all[main['cell'] == c]) for c in cells]])
    voxel_n = np.array([(main['cell'] == c).sum() for c in cells], float)

    def wls(design, y, w):
        ok = np.isfinite(y)
        sw = np.sqrt(w[ok])
        return np.linalg.lstsq(design[ok] * sw[:, None], y[ok] * sw, rcond=None)[0]

    def statistics(weights):
        tau_g, w_g = np.full(len(groups), np.nan), np.zeros(len(groups))
        for i, idx in enumerate(members):
            rep_idx = np.repeat(idx, weights[idx])
            if len(rep_idx) >= 3:
                tau_g[i] = stats.kendalltau(main['x'][rep_idx], main['y'][rep_idx])[0]
                w_g[i] = len(rep_idx)
        tau_v = np.array([np.nansum((w_g * tau_g)[g_cell == c]) / max(w_g[(g_cell == c) & np.isfinite(tau_g)].sum(), 1)
                          for c in cells])
        coef_groups = wls(covariates[regress], tau_g[regress], n_g[regress])
        coef_voxels = wls(voxel_cov, tau_v, voxel_n)
        return tau_g, tau_v, coef_groups, coef_voxels

    ones = np.ones(len(main['x']), int)
    tau_g, tau_v, coef_groups, coef_voxels = statistics(ones)
    brng = np.random.default_rng(SEED + 3)
    order = np.argsort(main['traj'], kind='stable')
    t_sorted = main['traj'][order]
    starts = np.unique(t_sorted, return_index=True)
    bounds = np.append(starts[1][1:], len(t_sorted))
    tr_index = {t: order[a:b] for t, a, b in zip(starts[0], starts[1], bounds)}
    by_rate = {r: traj_ids[traj_rate == r] for r in distinct_rates}
    boot_v, boot_cg, boot_cv = [], [], []
    for _ in range(N_RATE_BOOT):
        weights = np.zeros(len(main['x']), int)
        for r, ids in by_rate.items():
            for t in brng.choice(ids, len(ids)):
                if t in tr_index:
                    weights[tr_index[t]] += 1
        _, tv, cg, cv = statistics(weights)
        boot_v.append(tv)
        boot_cg.append(cg)
        boot_cv.append(cv)
    pct = lambda a: [np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0)]
    v_lo, v_hi = pct(np.array(boot_v))
    cg_lo, cg_hi = pct(np.array(boot_cg))
    cv_lo, cv_hi = pct(np.array(boot_cv))

    old = {v['cell']: v for v in voxel_dependence['voxels']}
    old_tau = np.array([old[int(c)]['kendall_tau'] for c in cells])
    old_ci = np.array([old[int(c)]['tau_ci'] for c in cells])
    coef_old = wls(voxel_cov, old_tau, voxel_n)
    diff = tau_v - old_tau
    names = ['intercept', 'z_u', 'z_tau', 'ln(rate/rate_mid)']
    tau_typ = float(np.average(tau_v, weights=voxel_n))
    eta_per_tau = 1.0 / (tau_typ * (1 - tau_typ))          # eta = logit(tau) for Gumbel
    out = dict(min_events_per_group=MIN_TAU_GROUP, n_groups=len(groups), n_boot=N_RATE_BOOT,
               voxels=[dict(cell=int(c), n=int(voxel_n[i]), kendall_tau=float(tau_v[i]),
                            tau_ci=[float(v_lo[i]), float(v_hi[i])], step8_tau=float(old_tau[i]),
                            step8_ci=[float(old_ci[i, 0]), float(old_ci[i, 1])])
                       for i, c in enumerate(cells)],
               change_from_step8=dict(mean=float(diff.mean()), sd=float(diff.std(ddof=1)),
                                      min=float(diff.min()), max=float(diff.max()),
                                      correlation=float(np.corrcoef(tau_v, old_tau)[0, 1]),
                                      new_outside_step8_ci=int(((tau_v < old_ci[:, 0]) | (tau_v > old_ci[:, 1])).sum()),
                                      range_new=[float(tau_v.min()), float(tau_v.max())],
                                      range_step8=[float(old_tau.min()), float(old_tau.max())]),
               voxel_trend=dict(names=names[:3], unpooled=coef_voxels.tolist(),
                                unpooled_ci=[[float(a), float(b)] for a, b in zip(cv_lo, cv_hi)],
                                step8_pooled=coef_old.tolist()),
               group_regression=dict(names=names, n_groups=int(regress.sum()), coef=coef_groups.tolist(),
                                     ci=[[float(a), float(b)] for a, b in zip(cg_lo, cg_hi)],
                                     typical_tau=tau_typ, eta_per_unit_tau=eta_per_tau,
                                     rate_coef_in_eta=float(coef_groups[3] * eta_per_tau),
                                     rate_coef_in_eta_ci=[float(cg_lo[3] * eta_per_tau), float(cg_hi[3] * eta_per_tau)]))
    print('\nPER-VOXEL KENDALL TAU WITHOUT POOLING OVER RATES (event-weighted mean over (voxel, rate) groups with'
          ' >= %d events; %d resamples of trajectories within each rate)' % (MIN_TAU_GROUP, N_RATE_BOOT))
    ch = out['change_from_step8']
    print('  unpooled minus step 8 (pooled): mean %+.3f, sd %.3f, range [%+.3f, %+.3f]; correlation %.3f;'
          ' %d / %d voxels outside step 8\'s interval' % (ch['mean'], ch['sd'], ch['min'], ch['max'],
                                                          ch['correlation'], ch['new_outside_step8_ci'], len(cells)))
    print('  tau range across voxels: unpooled %.2f - %.2f, step 8 %.2f - %.2f'
          % (*ch['range_new'], *ch['range_step8']))
    print('  voxel tau on voxel median (z_u, z_tau), weighted by events:')
    for i, nm in enumerate(names[:3]):
        print('    %-9s unpooled %+.4f [%+.4f, %+.4f]   step 8 pooled %+.4f'
              % (nm, coef_voxels[i], cv_lo[i], cv_hi[i], coef_old[i]))
    print('  GROUP REGRESSION: tau of %d (voxel, rate) groups (>= %d events) on median z_u, z_tau and ln rate,'
          ' weighted by group size:' % (regress.sum(), MIN_RATE_EVENTS))
    for i, nm in enumerate(names):
        print('    %-18s %+.4f [%+.4f, %+.4f]' % (nm, coef_groups[i], cg_lo[i], cg_hi[i]))
    gr = out['group_regression']
    print('    rate coefficient in eta = logit(tau) units at tau %.2f: %+.3f [%+.3f, %+.3f]'
          '  (target from checkpoint 2\'s within-voxel slope: about -0.065)'
          % (tau_typ, gr['rate_coef_in_eta'], *gr['rate_coef_in_eta_ci']))
    return out


voxel_tau_unpooled = unpooled_tau()
if CHECKS_ONLY:
    with open('out/dependence_law_checks.json', 'w') as handle:
        json.dump(dict(fold_balance=balance, cooling_rate=rate_dependence, marginal_rate=marginal_rate,
                       voxel_tau_unpooled=voxel_tau_unpooled),
                  handle, indent=1)
    print('\nwrote out/dependence_law_checks.json  (%.0f s)' % (time.time() - start_time))
    sys.exit(0)

main['ln_rate'] = np.log(main['rate'] / RATE_MID)
everything['ln_rate'] = np.log(everything['rate'] / RATE_MID)
n_params[dl.TABLE_RATE] = n_params[dl.TABLE] + 1
MODELS = dl.MODELS_WITH_RATE
PRODUCT_NOTE = ('the product form a + b 100 u tau is kept for comparability with step 3.  It depends on '
                'U_REFERENCE (u = pe - U_REFERENCE): the reference sets where 100 u tau crosses zero and '
                'forces one shared sign on the u and tau effects.  A loss to the linear form is not physics.')


def label(m):
    return '%s (%d)' % (m, n_params[m])


# --- in-sample ---------------------------------------------------------------------
fit = dl.fit_in_sample(main['x'], main['y'], main['u'], main['tau'], main['cell'], rng,
                       ln_rate=main['ln_rate'], rank_levels=rank_levels(main))
n = len(main['x'])
in_sample = {}
print('\nIN-SAMPLE (ranks: %s, min %d training events per rank group; events per level %s; rank rule: %s)'
      % (RANKS, dl.MIN_RANK_TRAIN, fit['rank_level_events'], dl.RANK_RULE))
print('  %-22s %10s %10s %10s  %-13s %s' % ('model', 'ln L', 'AIC', 'BIC', 'theta range', 'coefficients / flags'))
for form in dl.FORMS + dl.RATE_FORMS:
    f = fit['fits'][form]
    k = dl.N_COEFFICIENTS[form]
    in_sample[form] = dict(beta=f['beta'].tolist(), names=dl.COEFFICIENT_NAMES[form], ll=f['ll'],
                           aic=2 * k - 2 * f['ll'], bic=k * np.log(n) - 2 * f['ll'], flags=f['flags'],
                           theta_range=f['theta_range'], frac_theta_near_one=f['frac_theta_near_one'])
    r = in_sample[form]
    print('  %-22s %10.1f %10.1f %10.1f  %5.2f - %5.2f  %s%s'
          % (label(form), r['ll'], r['aic'], r['bic'], *r['theta_range'],
             ', '.join('%s %+.4f' % (nm, b) for nm, b in zip(r['names'], r['beta'])),
             ('   FLAGS: ' + '; '.join(r['flags'])) if r['flags'] else ''))
for m, ll, flags, extra in ((dl.TABLE, fit['ll_table'], fit['table_flags'], dict(pooled_theta=fit['pooled'])),
                            (dl.TABLE_RATE, fit['table_rate']['ll'], fit['table_rate']['flags'],
                             dict(g=fit['table_rate']['g']))):
    k = n_params[m]
    in_sample[m] = dict(ll=ll, aic=2 * k - 2 * ll, bic=k * np.log(n) - 2 * ll, flags=flags, **extra)
    print('  %-22s %10.1f %10.1f %10.1f  %s%s' % (label(m), ll, in_sample[m]['aic'], in_sample[m]['bic'],
                                                 'shared g(ln rate) %+.4f  ' % extra['g'] if 'g' in extra else '',
                                                 ('FLAGS: ' + '; '.join(flags)) if flags else ''))
print('  nesting check (every model >= the models it contains): %s' % ('; '.join(fit['nesting_violations']) or 'ok'))
in_sample['nesting_violations'] = fit['nesting_violations']

print('\n  how the state coefficients move when + g ln(rate/rate_mid) is added (in-sample):')
coefficient_shift = {}
for form in dl.FORMS:
    a, b = in_sample[form]['beta'], in_sample[form + dl.RATE]['beta']
    coefficient_shift[form] = {nm: dict(without_rate=a[i], with_rate=b[i]) for i, nm in enumerate(dl.TERMS[form])}
    coefficient_shift[form]['g'] = b[-1]
    print('    %-12s %s;  g %+.4f' % (form, ', '.join('%s %+.4f -> %+.4f' % (nm, a[i], b[i])
                                                      for i, nm in enumerate(dl.TERMS[form])), b[-1]))
print('  correlation of ln(rate) with the state across events: z_u %+.3f, z_tau %+.3f, 100 u tau %+.3f'
      % (np.corrcoef(main['ln_rate'], main['u'])[0, 1], np.corrcoef(main['ln_rate'], main['tau'])[0, 1],
         np.corrcoef(main['ln_rate'], main['u'] * main['tau'])[0, 1]))
print('  note: ' + PRODUCT_NOTE)
print('  rate coefficient g (eta units) against the target of about -0.065 (checkpoint 2) and the group'
      ' regression\'s %+.3f: %s' % (voxel_tau_unpooled['group_regression']['rate_coef_in_eta'],
                                   ', '.join('%s %+.4f' % (f, in_sample[f + dl.RATE]['beta'][-1]) for f in dl.FORMS)
                                   + ', voxel_table %+.4f' % in_sample[dl.TABLE_RATE]['g']))


# --- held-out: 2 folds and 5 folds, sampling-only intervals ----------------------------
def scoreboard(data, fold_id, title):
    held, folds = dl.cross_validate(data['x'], data['y'], data['u'], data['tau'], data['cell'],
                                    fold_id, rng, ln_rate=data['ln_rate'], rank_levels=rank_levels(data))
    boot = dl.bootstrap_scores(held, data['traj'], N_BOOT, rng)
    totals = {m: float(held[m].sum()) for m in MODELS}
    best = max(totals, key=totals.get)
    rows = {}
    for m in MODELS:
        rows[m] = dict(n_parameters=n_params[m], held_out=totals[m],
                       vs_constant=dl.compare(held, boot, m, 'constant'),
                       vs_voxel_table=dl.compare(held, boot, m, dl.TABLE),
                       vs_voxel_table_rate=dl.compare(held, boot, m, dl.TABLE_RATE),
                       vs_best=dl.compare(held, boot, m, best))
        if m.endswith(dl.RATE):
            rows[m]['vs_without_rate'] = dl.compare(held, boot, m, m[:-len(dl.RATE)])
    tied = [m for m in MODELS if m == best or rows[m]['vs_best']['ci'][1] >= 0]
    simplest = min(tied, key=lambda m: n_params[m])
    fmt = lambda c: '%+8.1f [%+7.1f, %+7.1f]' % (c['difference'], *c['ci'])
    print('\n%s' % title)
    print('  %-22s %11s   %-28s %-28s %-28s' % ('model', 'held-out', 'minus constant [95% CI]',
                                                'like-for-like table [95% CI]', 'minus best [95% CI]'))
    for m in sorted(MODELS, key=lambda m: -totals[m]):
        r = rows[m]
        like = r['vs_voxel_table_rate'] if m.endswith(dl.RATE) else r['vs_voxel_table']
        print('  %-22s %11.1f   %-28s %-28s %-28s' % (label(m), r['held_out'], fmt(r['vs_constant']),
                                                       '--' if m.startswith('voxel_table') else fmt(like),
                                                       fmt(r['vs_best'])))
    print('  (like-for-like: a rate-free form minus voxel_table; a +rate form minus voxel_table+rate)')
    print('  gain from the rate term:  %s' % ';  '.join(
        '%s %s' % (m[:-len(dl.RATE)], fmt(rows[m]['vs_without_rate'])) for m in MODELS if m.endswith(dl.RATE)))
    flags = [(fd['fold'], m, fl) for fd in folds for m, rec in fd['forms'].items() for fl in rec.get('flags', [])]
    flags += [(fd['fold'], dl.TABLE, fl) for fd in folds for fl in fd['table_flags']]
    fallback = sum(fd['rank_fallback_events'] for fd in folds)
    nesting = ['fold %d: %s' % (fd['fold'], v) for fd in folds for v in fd['nesting_violations']]
    print('  nesting check on each training fold (every model >= the models it contains): %s'
          % ('; '.join(nesting) or 'ok'))
    print('  shared rate coefficient of voxel_table+rate per fold: %s   (sanity target about -0.065)'
          % ', '.join('%+.4f' % fd['forms'][dl.TABLE_RATE]['g'] for fd in folds))
    print('  rank level per fold, events (train | test) at %s: %s'
          % (' / '.join(dict(voxel_rate=['voxel x rate', 'voxel x 3 rates', 'voxel'], voxel_rate3=['voxel x 3 rates', 'voxel'],
                             voxel=['voxel'])[RANKS]),
             '; '.join('%s | %s' % (fd['rank_level_events']['train'], fd['rank_level_events']['test']) for fd in folds)))
    print('  best: %s.  Tied with the best (sampling-only CI reaches 0): %s.  Simplest of those: %s.'
          % (best, ', '.join(tied), simplest))
    print('  fold flags: %s.  Held-out events ranked without training events in their voxel: %d'
          % ('; '.join('fold %d %s: %s' % f for f in flags) if flags else 'none', fallback))
    return held, boot, dict(best=best, tied_with_best=tied, simplest_tied=simplest, models=rows,
                            folds=folds, rank_fallback_events=fallback, nesting_violations=nesting,
                            interval='sampling-only (trajectory bootstrap of held-out per-event scores)')


held2, boot2, split2 = scoreboard(main, main['traj'] % 2, 'HELD OUT, 2 folds (even vs odd trajectory id) -- sampling-only intervals')
held5, boot5, split5 = scoreboard(main, main['traj'] % 5, 'HELD OUT, 5 folds (trajectory id mod 5) -- sampling-only intervals')
with open('out/dependence_law_scoreboard%s.json' % SUFFIX, 'w') as handle:
    json.dump(dict(n_events=n, ranks=RANKS, min_rank_train=dl.MIN_RANK_TRAIN, rank_rule=dl.RANK_RULE,
                   rank_level_events_in_sample=fit['rank_level_events'],
                   product_note=PRODUCT_NOTE, fold_balance=balance, cooling_rate=rate_dependence,
                   marginal_rate=marginal_rate, voxel_tau_unpooled=voxel_tau_unpooled, in_sample=in_sample,
                   coefficient_shift=coefficient_shift, held_out_2fold=split2, held_out_5fold=split5),
              handle, indent=1)
print('\nwrote out/dependence_law_scoreboard%s.json  (%.0f s)' % (SUFFIX, time.time() - start_time))
