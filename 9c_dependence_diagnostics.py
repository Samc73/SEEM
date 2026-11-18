"""Step 9c: post-fit diagnostics of theta(u, tau, rate), with ranks within voxel x rate.

Input : out/joint_events.npz, out/marginals.json (the main set of step 9),
        out/dependence_law_scoreboard.json (step 9, ranks=voxel_rate: the unpooled
        per-voxel Kendall tau and its intervals, and the in-sample coefficients
        used as a cross-check)
        out/dependence_law_scoreboard_voxel_rate3.json, _pooled.json (step 9 sensitivity runs),
        out/dependence_law_replicates.json (step 9b merge)
Output: out/dependence_law.json   -- the FINAL step 9 result: everything above
        plus these diagnostics, assembled in one file (step 10 reads it)

    python 9c_dependence_diagnostics.py      (~4 min)

All models are refitted in-sample on the main set (as step 9's in-sample fit),
with voxel x rate ranks (main) and with ranks pooled over rates (comparison).
  calibration  per voxel, model-implied Kendall tau (Gumbel: 1 - 1/theta,
               averaged over the voxel's events) against the measured tau
               WITHOUT pooling over rates (step 9's pre-fit check 4) and its
               interval.  Count outside, signed mean bias, trend of the bias
               against measured tau.
  upper tail   per voxel, P(V > 0.95 | U > 0.95): model (Gumbel, averaged over
               the voxel's events) against the empirical value from the same
               uniforms, with an interval from resampling trajectories within
               each rate (uniforms held fixed).  Count of voxels where the model
               is above the interval; correlation of model - empirical with the
               voxel's u, tau, rate mix (mean ln rate) and measured tau.
  families     step 7's comparison (every family, ranks, held out by
               trajectory parity, paired trajectory bootstrap) in the voxels
               with the lowest, median and three highest measured tau among
               those with >= 300 paired events, and in cell (18, 11); voxel x
               rate ranks, and pooled ranks in cell (18, 11) to compare with
               step 7.
  coefficients every form's in-sample coefficients with intervals from the
               spread across the 50/50 and 80/20 splits (delete-d jackknife);
               the refitting bootstrap's percentiles as secondary
  effect size  held-out gain over a constant theta per 1,000 held-out events
               (50/50 and 80/20 splits, 5 folds) and the share of
               voxel_table+rate's gain each formula recovers.
"""
import json
import time
import warnings
import numpy as np
from scipy import stats
import copulas
import dependence_law as dl

warnings.filterwarnings('ignore', category=RuntimeWarning)
MIN_PAIRED = 100
MIN_FAMILY_VOXEL = 300
Q_TAIL = 0.05
N_BOOT = 200              # trajectory resamples for the tail intervals
N_BOOT_FAMILY = 2000      # as step 7
SEED = 0
MAIN_MODELS = ('interaction+rate', 'quadratic+rate', 'voxel_table+rate')
COMPARISON_MODELS = ('interaction', 'quadratic', 'voxel_table')

t0 = time.time()
rng = np.random.default_rng(SEED)
events = np.load('out/joint_events.npz')
with open('out/marginals.json') as handle:
    msettings = json.load(handle)
with open('out/dependence_law_scoreboard.json') as handle:
    board = json.load(handle)
assert board['ranks'] == 'voxel_rate'
xmin = msettings['xmin']
n_bins = int(events['n_bins'])
du, dtau, cell_all = events['du'], events['dtau'], events['cell']
paired = (du < 0) & (dtau < 0) & (-du >= xmin['pe']) & (-dtau >= xmin['stress'])
counts = np.bincount(cell_all[paired], minlength=n_bins * n_bins)
keep = paired & (counts[cell_all] >= MIN_PAIRED)
rates = np.unique(events['rate'])
x, y = -du[keep], -dtau[keep]
u_state, tau_state, cell, traj = events['u'][keep], events['tau'][keep], cell_all[keep], events['traj'][keep]
rate = events['rate'][keep]
rate_index = np.searchsorted(rates, rate)
ln_rate = np.log(rate / rates[len(rates) // 2])
LEVELS = dict(voxel_rate=[cell * 9 + rate_index, cell * 3 + rate_index // 3, cell], voxel=[cell])
cells = np.unique(cell)
n = len(x)
print('main set: %d paired events in %d voxels' % (n, len(cells)))

# --- refit in-sample, both rankings ----------------------------------------------------------------
fits, theta, uniforms = {}, {}, {}
for ranks, levels in LEVELS.items():
    f = dl.fit_in_sample(x, y, u_state, tau_state, cell, rng, ln_rate=ln_rate, rank_levels=levels)
    fits[ranks] = f
    uniforms[ranks] = (f['u'], f['v'])
    for form in dl.FORMS + dl.RATE_FORMS:
        theta[(ranks, form)] = dl.theta_of(f['fits'][form]['beta'],
                                           dl.design(form, u_state, tau_state, f['scale'], ln_rate))
    theta[(ranks, dl.TABLE)] = dl.table_theta(f['table'], f['pooled'], cell)
    theta[(ranks, dl.TABLE_RATE)] = 1 + np.exp(np.clip(dl.table_rate_eta(f['table_rate'], cell, ln_rate),
                                                       *dl.ETA_RANGE))
    print('in-sample fit, ranks %s: nesting %s  (%.0f s)'
          % (ranks, '; '.join(f['nesting_violations']) or 'ok', time.time() - t0))
# cross-check against step 9's in-sample coefficients (same data, same ranks)
worst = max(np.abs(np.array(fits['voxel_rate']['fits'][m]['beta']) - np.array(board['in_sample'][m]['beta'])).max()
            for m in dl.FORMS + dl.RATE_FORMS)
print('largest coefficient difference from step 9 in-sample fit (voxel x rate): %.1e' % worst)

# --- calibration -------------------------------------------------------------------------------------
measured = {v['cell']: v for v in board['voxel_tau_unpooled']['voxels']}
m_tau = np.array([measured[int(c)]['kendall_tau'] for c in cells])
m_ci = np.array([measured[int(c)]['tau_ci'] for c in cells])
index = np.searchsorted(cells, cell)
voxel_n = np.bincount(index, minlength=len(cells)).astype(float)
u_mid = np.array([np.median(u_state[cell == c]) for c in cells])
t_mid = np.array([np.median(tau_state[cell == c]) for c in cells])
rate_mix = np.bincount(index, weights=ln_rate, minlength=len(cells)) / voxel_n
shown = [('voxel_rate', m) for m in MAIN_MODELS + COMPARISON_MODELS] + \
        [('voxel', m) for m in ('interaction+rate', 'voxel_table+rate', 'voxel_table')]
calibration, calibration_rows = {}, []
print('\nCALIBRATION: implied Kendall tau (Gumbel, mean over each voxel\'s events) against the measured tau'
      ' without pooling over rates;\n  bias = implied - measured; trend = OLS slope of the bias on measured tau')
print('  trend on measured tau is pulled negative by the noise in the measured tau (see the null check below);'
      ' the trend on implied tau is not')
print('  %-34s %8s %10s %10s %24s %24s' % ('model (ranks)', 'outside', 'mean bias', 'mean |bias|',
                                          'trend on measured [95%]', 'trend on implied [95%]'))
for ranks, model in shown:
    implied = np.bincount(index, weights=dl.gumbel_tau(theta[(ranks, model)]), minlength=len(cells)) / voxel_n
    bias = implied - m_tau
    outside = (implied < m_ci[:, 0]) | (implied > m_ci[:, 1])
    slope = stats.linregress(m_tau, bias)
    slope_i = stats.linregress(implied, bias)
    key = '%s (%s)' % (model, ranks)
    calibration[key] = dict(ranks=ranks, model=model, n_voxels=len(cells), voxels_outside_tau_ci=int(outside.sum()),
                            mean_bias=float(bias.mean()), mean_abs_bias=float(np.abs(bias).mean()),
                            trend_slope=float(slope.slope),
                            trend_ci=[float(slope.slope - 1.96 * slope.stderr), float(slope.slope + 1.96 * slope.stderr)],
                            trend_r=float(slope.rvalue), trend_on_implied=float(slope_i.slope),
                            trend_on_implied_ci=[float(slope_i.slope - 1.96 * slope_i.stderr),
                                                 float(slope_i.slope + 1.96 * slope_i.stderr)])
    for i, c in enumerate(cells):
        calibration_rows.append(dict(ranks=ranks, model=model, cell=int(c), implied_tau=float(implied[i]),
                                     measured_tau=float(m_tau[i]), tau_ci=m_ci[i].tolist(), outside=bool(outside[i])))
    r = calibration[key]
    print('  %-34s %4d/%d %+10.4f %10.4f   %+.3f [%+.3f, %+.3f]   %+.3f [%+.3f, %+.3f]'
          % (key, r['voxels_outside_tau_ci'], len(cells), r['mean_bias'], r['mean_abs_bias'], r['trend_slope'],
             *r['trend_ci'], r['trend_on_implied'], *r['trend_on_implied_ci']))

# --- upper tail ----------------------------------------------------------------------------------------
p = 1 - Q_TAIL
ids, first = np.unique(traj, return_index=True)
by_rate = {r: ids[rate[first] == r] for r in rates}
order = np.argsort(traj, kind='stable')
starts = np.unique(traj[order], return_index=True)
members = dict(zip(starts[0], np.split(order, starts[1][1:])))
brng = np.random.default_rng(SEED + 1)
boot_weights = []
for _ in range(N_BOOT):
    w = np.zeros(n)
    for r, tr in by_rate.items():
        for t in brng.choice(tr, len(tr)):
            if t in members:
                w[members[t]] += 1
    boot_weights.append(w)
boot_weights = np.array(boot_weights)


def empirical_upper(u, v, weights):
    """P(V > p | U > p) per voxel, from the weighted events."""
    above_u = (u > p).astype(float)
    both = above_u * (v > p)
    return (np.bincount(index, weights=weights * both, minlength=len(cells))
            / np.maximum(np.bincount(index, weights=weights * above_u, minlength=len(cells)), 1e-12))


tail, tail_rows = {}, []
empirical = {}
for ranks in LEVELS:
    u, v = uniforms[ranks]
    point = empirical_upper(u, v, np.ones(n))
    boot = np.array([empirical_upper(u, v, w) for w in boot_weights])
    empirical[ranks] = (point, np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0))
print('\nUPPER TAIL: P(V > %.2f | U > %.2f) per voxel, model (Gumbel, mean over events) vs empirical (same'
      ' uniforms), %d trajectory resamples within rate' % (p, p, N_BOOT))
e_main = empirical['voxel_rate'][0]
print('  empirical (voxel x rate ranks): mean over voxels %.3f, range %.3f - %.3f'
      % (e_main.mean(), e_main.min(), e_main.max()))
print('  %-34s %10s %11s %11s   correlation of (model - empirical) with: %s'
      % ('model (ranks)', 'overshoot', 'undershoot', 'mean diff', 'u, tau, rate mix, measured tau'))
for ranks, model in shown:
    point, lo, hi = empirical[ranks]
    th = theta[(ranks, model)]
    joint = 1 - 2 * p + copulas.cdf('gumbel', [th], np.full(n, p), np.full(n, p))
    model_tail = np.bincount(index, weights=joint / (1 - p), minlength=len(cells)) / voxel_n
    diff = model_tail - point
    over, under = model_tail > hi, model_tail < lo
    corr = {name: stats.pearsonr(var, diff) for name, var in
            (('u', u_mid), ('tau', t_mid), ('rate_mix', rate_mix), ('measured_tau', m_tau))}
    key = '%s (%s)' % (model, ranks)
    tail[key] = dict(ranks=ranks, model=model, n_voxels=len(cells), overshoot=int(over.sum()),
                     undershoot=int(under.sum()), mean_difference=float(diff.mean()),
                     mean_model=float(model_tail.mean()), mean_empirical=float(point.mean()),
                     correlations={k: dict(r=float(c.statistic), p=float(c.pvalue)) for k, c in corr.items()})
    for i, c in enumerate(cells):
        tail_rows.append(dict(ranks=ranks, model=model, cell=int(c), model_tail=float(model_tail[i]),
                              empirical_tail=float(point[i]), ci=[float(lo[i]), float(hi[i])],
                              overshoot=bool(over[i]), undershoot=bool(under[i])))
    print('  %-34s %6d/%d %7d/%d %+11.3f   %s'
          % (key, over.sum(), len(cells), under.sum(), len(cells), diff.mean(),
             ', '.join('%s %+.2f (p %.2g)' % (k, c.statistic, c.pvalue) for k, c in corr.items())))


# --- null check: what does a CORRECT model score on these measurements? ------------------------------------
N_NULL = 5


def sample_gumbel(theta_i, srng):
    """Marshall-Olkin with a positive-stable frailty, one theta per event (as test_dependence_law.py)."""
    alpha = 1.0 / np.asarray(theta_i)
    w = srng.uniform(0, np.pi, len(alpha))
    e = srng.exponential(size=len(alpha))
    sv = (np.sin(alpha * w) / np.sin(w) ** (1 / alpha) * (np.sin((1 - alpha) * w) / e) ** ((1 - alpha) / alpha))
    pair = srng.exponential(size=(len(alpha), 2))
    return np.exp(-(pair / sv[:, None]) ** alpha[:, None]).T


true_theta = theta[('voxel_rate', 'interaction+rate')]
true_tail = np.bincount(index, weights=(1 - 2 * p + copulas.cdf('gumbel', [true_theta], np.full(n, p), np.full(n, p)))
                        / (1 - p), minlength=len(cells)) / voxel_n
true_tau = np.bincount(index, weights=dl.gumbel_tau(true_theta), minlength=len(cells)) / voxel_n
null_rows = []
for k in range(N_NULL):
    a, b = sample_gumbel(true_theta, np.random.default_rng(1000 + k))
    ua, _, _, _ = dl.grouped_uniforms(a, LEVELS['voxel_rate'], np.ones(n, bool))
    vb, _, _, _ = dl.grouped_uniforms(b, LEVELS['voxel_rate'], np.ones(n, bool))
    meas = np.zeros(len(cells))
    for i, c in enumerate(cells):
        num = den = 0.0
        for r in rates:
            m = (cell == c) & (rate == r)
            if m.sum() >= board['voxel_tau_unpooled']['min_events_per_group']:
                num += m.sum() * stats.kendalltau(a[m], b[m])[0]
                den += m.sum()
        meas[i] = num / den
    bias = true_tau - meas
    emp = empirical_upper(ua, vb, np.ones(n))
    boot = np.array([empirical_upper(ua, vb, w) for w in boot_weights])
    hi = np.percentile(boot, 97.5, axis=0)
    lo = np.percentile(boot, 2.5, axis=0)
    null_rows.append(dict(mean_bias=float(bias.mean()), trend_on_measured=float(stats.linregress(meas, bias).slope),
                          trend_on_implied=float(stats.linregress(true_tau, bias).slope),
                          tail_mean_difference=float((true_tail - emp).mean()),
                          tail_overshoot=int((true_tail > hi).sum()), tail_undershoot=int((true_tail < lo).sum())))
null = {key: dict(mean=float(np.mean([r[key] for r in null_rows])), values=[r[key] for r in null_rows])
        for key in null_rows[0]}
print('\nNULL CHECK: sizes drawn from Gumbel with the fitted interaction+rate theta at every real event (%d draws),'
      ' measured exactly as above with the TRUE theta as the model' % N_NULL)
print('  calibration: mean bias %+.4f, trend on measured tau %+.3f, trend on implied tau %+.3f'
      % (null['mean_bias']['mean'], null['trend_on_measured']['mean'], null['trend_on_implied']['mean']))
print('  upper tail: model - empirical %+.3f (the small rank groups bias the empirical tail low); voxels above the'
      ' interval %.1f, below %.1f (of %d)' % (null['tail_mean_difference']['mean'], null['tail_overshoot']['mean'],
                                             null['tail_undershoot']['mean'], len(cells)))


# --- family check -------------------------------------------------------------------------------------
def family_contest(c, ranks):
    m = cell == c
    levels = [lv[m] for lv in LEVELS[ranks]]
    t = traj[m]
    even = t % 2 == 0
    held = {f: np.zeros(m.sum()) for f in copulas.FAMILIES}
    flags = []
    for train in (even, ~even):
        u_tr, u_te, _, _ = dl.grouped_uniforms(x[m], levels, train)
        v_tr, v_te, _, _ = dl.grouped_uniforms(y[m], levels, train)
        for f in copulas.FAMILIES:
            fit = copulas.fit_family(f, u_tr, v_tr)
            flags += ['%s: %s' % (f, fl) for fl in fit['at_bound']]
            held[f][~train] = copulas.log_density(f, fit['params'], u_te, v_te) if copulas.n_parameters(f) else 0.0
    boot = dl.bootstrap_scores(held, t, N_BOOT_FAMILY, np.random.default_rng(SEED))
    totals = {f: float(held[f].sum()) for f in held}
    best = max(totals, key=totals.get)
    rows = {f: dict(held_out=totals[f], vs_best=dl.compare(held, boot, f, best)) for f in held}
    tied = [f for f in held if f == best or rows[f]['vs_best']['ci'][1] >= 0]
    return dict(cell=int(c), cell_uv=[int(c // n_bins), int(c % n_bins)], n=int(m.sum()), ranks=ranks,
                measured_tau=float(measured[int(c)]['kendall_tau']), best=best, tied_with_best=tied,
                gumbel_minus_best=rows['gumbel']['vs_best'], families=rows, flags=flags)


eligible = [c for c in cells if (cell == c).sum() >= MIN_FAMILY_VOXEL]
by_tau = sorted(eligible, key=lambda c: measured[int(c)]['kendall_tau'])
picks = dict(lowest_tau=by_tau[0], median_tau=by_tau[len(by_tau) // 2], highest_tau=by_tau[-1],
             second_highest_tau=by_tau[-2], third_highest_tau=by_tau[-3])
CELL_7 = msettings['most_visited_cell']
family = {}
print('\nFAMILY CHECK (step 7\'s families, ranks, 2 folds by trajectory parity, %d paired trajectory resamples);'
      ' %d voxels have >= %d paired events' % (N_BOOT_FAMILY, len(eligible), MIN_FAMILY_VOXEL))
for label, c, ranks in [(k, v, 'voxel_rate') for k, v in picks.items()] + \
                       [('cell_18_11', CELL_7, 'voxel_rate'), ('cell_18_11_pooled', CELL_7, 'voxel')]:
    r = family_contest(c, ranks)
    family[label] = r
    g = r['gumbel_minus_best']
    print('  %-18s cell %3d (%2d, %2d)  n %5d  measured tau %.3f  ranks %-10s best %-16s gumbel - best %+7.1f '
          '[%+6.1f, %+6.1f]  tied: %s'
          % (label, c, *r['cell_uv'], r['n'], r['measured_tau'], ranks, r['best'], g['difference'], *g['ci'],
             ', '.join(r['tied_with_best'])))
    top = sorted(r['families'], key=lambda f: -r['families'][f]['held_out'])[:5]
    print('      top 5 held-out gains over independence: %s'
          % ', '.join('%s %.1f' % (f, r['families'][f]['held_out']) for f in top))

diagnostics = dict(q_tail=Q_TAIL, n_boot_tail=N_BOOT, measured_tau='unpooled (step 9 pre-fit check 4)',
                   coefficient_check_max_abs_difference=float(worst),
                   calibration=calibration, calibration_rows=calibration_rows, tail=tail, tail_rows=tail_rows,
                   family=family, null_check=dict(n_draws=N_NULL, model='interaction+rate (voxel_rate), true theta',
                                                 summary=null, draws=null_rows))

# --- assemble the final out/dependence_law.json ---------------------------------------------------------
with open('out/dependence_law_replicates.json') as handle:
    replicates = json.load(handle)['results']
sensitivity = {}
for ranks, suffix in (('voxel_rate3', '_voxel_rate3'), ('voxel', '_pooled')):
    with open('out/dependence_law_scoreboard%s.json' % suffix) as handle:
        sb = json.load(handle)
    sensitivity[ranks] = dict(in_sample=sb['in_sample'], held_out_2fold=sb['held_out_2fold'],
                              held_out_5fold=sb['held_out_5fold'], rank_level_events_in_sample=sb['rank_level_events_in_sample'])

# effect size: held-out gain over a constant theta per 1,000 held-out events, and the share of
# voxel_table+rate's gain each formula recovers (per replicate, then mean +- sd)
EFFECT_MODELS = ('voxel_table+rate', 'interaction+rate', 'quadratic+rate', 'product+rate', 'voxel_table',
                 'interaction', 'quadratic')
effect = {}
print('\nEFFECT SIZE: held-out gain over a constant theta per 1,000 held-out events (mean +- sd over replicates),'
      ' and the share of voxel_table+rate\'s gain recovered')
for scheme in ('halving', 'split80'):
    rows = replicates['%s/voxel_rate' % scheme]['replicates']
    n_test = np.array([r['n_test'] for r in rows], float)
    gain = {m: 1000 * np.array([r['totals'][m] - r['totals']['constant'] for r in rows]) / n_test
            for m in EFFECT_MODELS}
    effect[scheme] = {}
    for m in EFFECT_MODELS:
        share = gain[m] / gain['voxel_table+rate']
        effect[scheme][m] = dict(gain_per_1000=float(gain[m].mean()), sd=float(gain[m].std(ddof=1)),
                                 share_of_table_rate=float(share.mean()), share_sd=float(share.std(ddof=1)))
        print('  %-8s %-18s %6.2f +- %.2f per 1,000   share %.3f +- %.3f'
              % (scheme, m, gain[m].mean(), gain[m].std(ddof=1), share.mean(), share.std(ddof=1)))
five = board['held_out_5fold']['models']
effect['5fold'] = {m: dict(gain_per_1000=1000 * five[m]['vs_constant']['difference'] / n,
                           ci=[1000 * c / n for c in five[m]['vs_constant']['ci']],
                           share_of_table_rate=five[m]['vs_constant']['difference']
                           / five['voxel_table+rate']['vs_constant']['difference']) for m in EFFECT_MODELS}
for m in ('interaction+rate', 'quadratic+rate'):
    print('  5 folds  %-18s %6.2f per 1,000   share %.3f' % (m, effect['5fold'][m]['gain_per_1000'],
                                                            effect['5fold'][m]['share_of_table_rate']))

# coefficients with intervals for every form.  MAIN interval: delete-d jackknife over the random
# splits (no trajectory is duplicated).  A split trains on m of the n trajectories of each rate and
# leaves d = n - m out; for subsamples drawn without replacement the full-sample variance is
#     var_full = (m / d) * var_across_splits          (delete-d jackknife; Shao & Wu 1989)
# so sd_full = sd_splits * sqrt(m / d): x 1 for the 50/50 splits (m = d) and x 2 for the 80/20 splits
# (m / d = 4).  (sqrt(m / n) would ignore that the splits are drawn from one fixed sample.)  Main
# interval: estimate +- 1.96 sd_full from the 50/50 splits; the 80/20 splits give an independent check
# of the same sd.  The mean over splits minus the estimate is reported as a training-size effect.
# SECONDARY: the refitting bootstrap's 2.5-97.5% (biased by tied ranks from duplicated trajectories).
JACKKNIFE_FACTOR = dict(halving=1.0, split80=2.0)       # sqrt(m / d), trajectories counted per rate
coefficients = {}
boot_coef = replicates['bootstrap/voxel_rate']['coefficients']


def split_betas(scheme, form):
    rows = replicates['%s/voxel_rate' % scheme]['replicates']
    if form == dl.TABLE_RATE:
        return np.array([[r['g_table_rate']] for r in rows])
    return np.array([r['betas'][form] for r in rows])


def coefficient_rows(form, names, estimates):
    out = []
    betas = {s: split_betas(s, form) for s in JACKKNIFE_FACTOR}
    for i, nm in enumerate(names):
        sd = {s: float(betas[s][:, i].std(ddof=1) * JACKKNIFE_FACTOR[s]) for s in betas}
        est = float(estimates[i])
        out.append(dict(name=nm, estimate=est, sd=sd['halving'],
                        ci=[est - 1.96 * sd['halving'], est + 1.96 * sd['halving']],
                        sd_check_split80=sd['split80'],
                        ci_check_split80=[est - 1.96 * sd['split80'], est + 1.96 * sd['split80']],
                        split_mean_minus_estimate=dict(halving=float(betas['halving'][:, i].mean() - est),
                                                       split80=float(betas['split80'][:, i].mean() - est)),
                        bootstrap_ci_secondary=boot_coef[form][i]['percentiles'],
                        bootstrap_median_secondary=boot_coef[form][i]['median']))
    return out


for f in dl.FORMS + dl.RATE_FORMS:
    coefficients[f] = coefficient_rows(f, dl.TERMS[f], board['in_sample'][f]['beta'])
coefficients[dl.TABLE_RATE] = coefficient_rows(dl.TABLE_RATE, ['g (shared rate coefficient)'],
                                               [board['in_sample'][dl.TABLE_RATE]['g']])
coefficient_method = ('main interval = estimate +- 1.96 sd, sd = sd over the 50 random 50/50 splits x sqrt(m/d) '
                      '(delete-d jackknife; m = d for halves, so x 1); check: the 80/20 splits x sqrt(4) = x 2. '
                      'Secondary: refitting bootstrap 2.5-97.5%, biased by tied ranks from duplicated trajectories.')
print('\nCOEFFICIENTS (main interval: delete-d jackknife over the 50/50 splits; check: 80/20 splits; '
      'secondary: bootstrap)')
for f in ('interaction+rate', 'quadratic+rate', 'product+rate', dl.TABLE_RATE):
    for c in coefficients[f]:
        print('  %-16s %-26s %+.4f [%+.4f, %+.4f]  sd %.4f (80/20 check %.4f)  split mean - est %+.4f  '
              'bootstrap %s' % (f, c['name'], c['estimate'], *c['ci'], c['sd'], c['sd_check_split80'],
                                c['split_mean_minus_estimate']['halving'],
                                '[%+.4f, %+.4f]' % tuple(c['bootstrap_ci_secondary'])))
ratio = np.array([c['sd_check_split80'] / c['sd'] for cs in coefficients.values() for c in cs])
print('  80/20 sd / 50/50 sd over all %d coefficients: median %.2f, range %.2f - %.2f'
      % (len(ratio), np.median(ratio), ratio.min(), ratio.max()))


def strip(result):
    """Replicate summaries without the per-replicate rows (kept in out/dependence_law_replicates.json)."""
    return {k: v for k, v in result.items() if k not in ('replicates', 'rank_level_events')}


final = dict(
    step=9, ranks='voxel_rate (main); voxel_rate3 and voxel (pooled) as sensitivity',
    n_events=n, n_voxels=len(cells), min_rank_train=dl.MIN_RANK_TRAIN, rank_rule=dl.RANK_RULE,
    product_note=board['product_note'],
    pre_fit_checks=dict(fold_balance=board['fold_balance'], cooling_rate=board['cooling_rate'],
                        marginal_rate=board['marginal_rate'], voxel_tau_unpooled=board['voxel_tau_unpooled']),
    in_sample=board['in_sample'], coefficient_shift_with_rate=board['coefficient_shift'],
    rank_level_events_in_sample=board['rank_level_events_in_sample'],
    coefficients=coefficients, coefficient_method=coefficient_method,
    scoreboards=dict(held_out_2fold=board['held_out_2fold'], held_out_5fold=board['held_out_5fold']),
    replicates={k: strip(v) for k, v in replicates.items()},
    effect_size=effect,
    sensitivity=sensitivity,
    diagnostics=diagnostics,
    notes=['main analysis ranks within voxel x cooling rate (dl.MIN_RANK_TRAIN training events per group, falling '
           'back to voxel x 3 rate groups, then voxel): the size marginals shift with rate inside voxels',
           'sampling-only intervals resample trajectories and re-sum held-out per-event scores; they are about 1.4x '
           'too narrow for near-ties (test coverage study)',
           'halving / split80: 50 random splits of the trajectories of each rate (50/50 and 80/20); deciding '
           'evidence; the spread is a stability guide, not a confidence interval',
           'bootstrap: refitting bootstrap, trajectories resampled within rate; duplicated trajectories handicap the '
           'voxel tables, so its model comparisons lean toward small models; used for coefficient intervals only',
           'coefficient intervals: main = estimate +- 1.96 sd with sd from the 50 random 50/50 splits (delete-d '
           'jackknife, sd x sqrt(m/d) = x 1), checked against the 80/20 splits (x 2); the refitting bootstrap '
           'percentiles are kept as secondary only: tied ranks from duplicated trajectories bias them (its intercept '
           'intervals exclude the estimates, and its slope medians are shifted too)',
           'calibration and tail: in-sample fits; the null check gives what a correct model scores on the same '
           'measurements (small rank groups bias the empirical upper tail low by about the null value)'])
with open('out/dependence_law.json', 'w') as handle:
    json.dump(final, handle, indent=1)
print('\nwrote out/dependence_law.json  (%.0f s)' % (time.time() - t0))
