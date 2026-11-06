"""Step 9: does a closed formula theta(u, tau) describe the Gumbel dependence
as well as one theta per voxel?  See PLAN_step9.md.

Input : out/joint_events.npz (step 5), out/marginals.json (step 6),
        out/voxel_dependence.json (step 8: the 119 voxels and their tau intervals)
Output: out/dependence_law.json

    python 9_fit_dependence_law.py [N_REFIT]     (default N_REFIT = 100)

Main set: paired events (cutoffs of step 6) in voxels with >= 100 paired
events.  Secondary: all paired events.  Uniforms are ranks within each voxel.
Models: Gumbel with theta = 1 + exp(eta), eta constant / product (100 u tau) /
linear / quadratic in standardised (u, tau), against one theta per voxel.

Scores are held out by whole trajectories: 2 folds (even vs odd ids, as in
steps 3 and 7) and 5 folds (id mod 5).  Two kinds of interval:
  sampling   resample trajectories, re-sum held-out per-event scores (as step 7)
  refitting  resample trajectories, then REFIT everything (ranks, coefficients,
             voxel table) inside the 2-fold split -- N_REFIT rounds
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
N_REFIT = int(sys.argv[1]) if len(sys.argv) > 1 else 100
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
                cell=cell_all[mask], traj=events['traj'][mask])


main = select(paired & (counts[cell_all] >= MIN_PAIRED))
everything = select(paired)
print('main set: %d paired events in %d voxels (>= %d each), %d trajectories;  all paired: %d'
      % (len(main['x']), len(np.unique(main['cell'])), MIN_PAIRED, len(np.unique(main['traj'])),
         len(everything['x'])))
n_params = dict(dl.N_COEFFICIENTS, voxel_table=len(np.unique(main['cell'])))


def label(m):
    return '%s (%d)' % (m, n_params[m])


# --- in-sample ---------------------------------------------------------------------
fit = dl.fit_in_sample(main['x'], main['y'], main['u'], main['tau'], main['cell'], rng)
n = len(main['x'])
in_sample = {}
print('\nIN-SAMPLE (ranks within voxel)')
print('  %-16s %10s %10s %10s  %-15s %s' % ('model', 'ln L', 'AIC', 'BIC', 'theta range', 'coefficients / flags'))
for form in dl.FORMS:
    f = fit['fits'][form]
    k = dl.N_COEFFICIENTS[form]
    in_sample[form] = dict(beta=f['beta'].tolist(), names=dl.COEFFICIENT_NAMES[form], ll=f['ll'],
                           aic=2 * k - 2 * f['ll'], bic=k * np.log(n) - 2 * f['ll'], flags=f['flags'],
                           theta_range=f['theta_range'], frac_theta_near_one=f['frac_theta_near_one'])
    r = in_sample[form]
    print('  %-16s %10.1f %10.1f %10.1f  %5.2f - %5.2f  %s%s'
          % (label(form), r['ll'], r['aic'], r['bic'], *r['theta_range'],
             ', '.join('%s %+.4f' % (nm, b) for nm, b in zip(r['names'], r['beta'])),
             ('   FLAGS: ' + '; '.join(r['flags'])) if r['flags'] else ''))
k = n_params['voxel_table']
in_sample['voxel_table'] = dict(ll=fit['ll_table'], aic=2 * k - 2 * fit['ll_table'],
                                bic=k * np.log(n) - 2 * fit['ll_table'], flags=fit['table_flags'],
                                pooled_theta=fit['pooled'])
print('  %-16s %10.1f %10.1f %10.1f  %s' % (label('voxel_table'), fit['ll_table'],
                                           in_sample['voxel_table']['aic'], in_sample['voxel_table']['bic'],
                                           ('FLAGS: ' + '; '.join(fit['table_flags'])) if fit['table_flags'] else ''))


# --- held-out: 2 folds and 5 folds, sampling-only intervals ----------------------------
def scoreboard(data, fold_id, title):
    held, folds = dl.cross_validate(data['x'], data['y'], data['u'], data['tau'], data['cell'],
                                    fold_id, rng)
    boot = dl.bootstrap_scores(held, data['traj'], N_BOOT, rng)
    totals = {m: float(held[m].sum()) for m in dl.MODELS}
    best = max(totals, key=totals.get)
    rows = {}
    for m in dl.MODELS:
        rows[m] = dict(n_parameters=n_params[m], held_out=totals[m],
                       vs_constant=dl.compare(held, boot, m, 'constant'),
                       vs_voxel_table=dl.compare(held, boot, m, 'voxel_table'),
                       vs_best=dl.compare(held, boot, m, best))
    tied = [m for m in dl.MODELS if m == best or rows[m]['vs_best']['ci'][1] >= 0]
    simplest = min(tied, key=lambda m: n_params[m])
    print('\n%s' % title)
    print('  %-16s %11s   %-30s %-30s %-28s' % ('model', 'held-out', 'minus constant [95% CI]',
                                                'minus voxel table [95% CI]', 'minus best [95% CI]'))
    for m in sorted(dl.MODELS, key=lambda m: -totals[m]):
        r = rows[m]
        fmt = lambda c: '%+8.1f [%+7.1f, %+7.1f]' % (c['difference'], *c['ci'])
        print('  %-16s %11.1f   %-30s %-30s %-28s' % (label(m), r['held_out'], fmt(r['vs_constant']),
                                                       fmt(r['vs_voxel_table']), fmt(r['vs_best'])))
    flags = [(fd['fold'], form, fl) for fd in folds for form in dl.FORMS for fl in fd['forms'][form]['flags']]
    flags += [(fd['fold'], 'voxel_table', fl) for fd in folds for fl in fd['table_flags']]
    fallback = sum(fd['rank_fallback_events'] for fd in folds)
    print('  best: %s.  Tied with the best (CI reaches 0): %s.  Simplest of those: %s.'
          % (best, ', '.join(tied), simplest))
    print('  fold flags: %s.  Held-out events ranked without training events in their voxel: %d'
          % ('; '.join('fold %d %s: %s' % f for f in flags) if flags else 'none', fallback))
    return held, boot, dict(best=best, tied_with_best=tied, simplest_tied=simplest, models=rows,
                            folds=folds, rank_fallback_events=fallback)


held2, boot2, split2 = scoreboard(main, main['traj'] % 2, 'HELD OUT, 2 folds (even vs odd trajectory id) -- sampling-only intervals')
held5, boot5, split5 = scoreboard(main, main['traj'] % 5, 'HELD OUT, 5 folds (trajectory id mod 5) -- sampling-only intervals')

# --- cell (18, 11): the formula against step 7's single-voxel Gumbel -------------------------
in_cell = main['cell'] == CELL_OF_STEP_7
held_cell = {m: held2[m][in_cell] for m in dl.MODELS}
boot_cell = dl.bootstrap_scores(held_cell, main['traj'][in_cell], N_BOOT, rng)
cell_check = {m: dl.compare(held_cell, boot_cell, m, 'voxel_table') for m in dl.FORMS}
print('\nCELL (18, 11), %d events, 2-fold held-out: formula minus its own voxel theta (step 7\'s Gumbel)'
      % in_cell.sum())
for m in dl.FORMS:
    c = cell_check[m]
    print('  %-16s %+8.1f [%+7.1f, %+7.1f]' % (label(m), c['difference'], *c['ci']))

# --- calibration: implied Kendall tau per voxel vs measured ----------------------------------
measured = {v['cell']: v for v in voxel_dependence['voxels']}
scale = fit['scale']
calibration = {}
residual_rows = []
for m in dl.MODELS:
    if m == 'voxel_table':
        theta = dl.table_theta(fit['table'], fit['pooled'], main['cell'])
    else:
        theta = dl.theta_of(fit['fits'][m]['beta'], dl.design(m, main['u'], main['tau'], scale))
    outside, residuals = 0, []
    for c in np.unique(main['cell']):
        implied = float(np.mean(dl.gumbel_tau(theta[main['cell'] == c])))
        lo, hi = measured[int(c)]['tau_ci']
        outside += int(implied < lo or implied > hi)
        residuals.append(implied - measured[int(c)]['kendall_tau'])
        residual_rows.append(dict(model=m, cell=int(c), implied_tau=implied,
                                  measured_tau=measured[int(c)]['kendall_tau'], tau_ci=[lo, hi]))
    residuals = np.array(residuals)
    calibration[m] = dict(voxels_outside_tau_ci=outside, n_voxels=len(residuals),
                          mean_residual=float(residuals.mean()),
                          mean_abs_residual=float(np.abs(residuals).mean()))
print('\nCALIBRATION: Gumbel-implied Kendall tau per voxel (in-sample) vs measured tau and its interval')
print('  %-16s %18s %15s %18s' % ('model', 'voxels outside CI', 'mean residual', 'mean |residual|'))
for m in dl.MODELS:
    c = calibration[m]
    print('  %-16s %11d / %3d %+15.4f %18.4f' % (label(m), c['voxels_outside_tau_ci'], c['n_voxels'],
                                                 c['mean_residual'], c['mean_abs_residual']))
print('  (about 6 of 119 expected by chance if the model were exact; the voxel table is the benchmark)')

# --- secondary: all paired events, 2 folds ------------------------------------------------------
n_params_main = dict(n_params)
n_params['voxel_table'] = len(np.unique(everything['cell']))
_, _, secondary = scoreboard(everything, everything['traj'] % 2,
                             'SECONDARY: all %d paired events, 2 folds -- sampling-only intervals'
                             % len(everything['x']))
n_params = n_params_main

# --- refitting bootstrap ------------------------------------------------------------------------
refit = None
if N_REFIT > 0:
    print('\nREFITTING BOOTSTRAP: %d rounds (resample trajectories, refit everything, 2 folds)' % N_REFIT)
    ids = np.unique(main['traj'])
    members = {t: np.nonzero(main['traj'] == t)[0] for t in ids}
    warm = {form: fit['fits'][form]['beta'] for form in dl.FORMS}
    totals_rounds, betas_rounds = [], {form: [] for form in dl.FORMS}
    round_start = time.time()
    for b in range(N_REFIT):
        pick = rng.integers(0, len(ids), len(ids))
        index = np.concatenate([members[ids[i]] for i in pick])
        data = {key: value[index] for key, value in main.items()}
        held_b, _ = dl.cross_validate(data['x'], data['y'], data['u'], data['tau'], data['cell'],
                                      data['traj'] % 2, rng, warm=warm)
        totals_rounds.append({m: float(held_b[m].sum()) for m in dl.MODELS})
        u_b = dl.within_voxel_ranks(data['x'], data['cell'])
        v_b = dl.within_voxel_ranks(data['y'], data['cell'])
        scale_b = dl.scaling(data['u'], data['tau'])
        for form in dl.FORMS:
            x_b = dl.design(form, data['u'], data['tau'], scale_b)
            betas_rounds[form].append(dl.fit_form(x_b, u_b, v_b, [warm[form]])['beta'].tolist())
        if b == 0:
            print('  first round took %.0f s' % (time.time() - round_start))
    totals_rounds = {m: np.array([t[m] for t in totals_rounds]) for m in dl.MODELS}

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    refit = dict(n_rounds=N_REFIT, differences={}, coefficients={})
    print('  held-out differences with refitting noise included (2 folds):')
    for m in dl.FORMS:
        for ref in ('constant', 'voxel_table'):
            if m == ref:
                continue
            d = totals_rounds[m] - totals_rounds[ref]
            refit['differences']['%s - %s' % (m, ref)] = dict(median=float(np.median(d)), ci=interval(d),
                                                              p_positive=float(np.mean(d > 0)))
            print('    %-12s - %-12s median %+8.1f  [%+8.1f, %+8.1f]  P(> 0) %.2f'
                  % (m, ref, np.median(d), *interval(d), np.mean(d > 0)))
    print('  coefficient intervals (in-sample refits):')
    for form in dl.FORMS:
        betas = np.array(betas_rounds[form])
        refit['coefficients'][form] = [dict(name=nm, estimate=float(est), ci=interval(betas[:, i]))
                                       for i, (nm, est) in enumerate(zip(dl.COEFFICIENT_NAMES[form],
                                                                         fit['fits'][form]['beta']))]
        print('    %-10s %s' % (form, ',  '.join('%s %+.4f [%+.4f, %+.4f]' % (r['name'], r['estimate'], *r['ci'])
                                                for r in refit['coefficients'][form])))

with open('out/dependence_law.json', 'w') as handle:
    json.dump(dict(min_paired=MIN_PAIRED, n_events=n, n_voxels=n_params['voxel_table'],
                   scaling=scale, in_sample=in_sample, voxel_table=fit['table'],
                   held_out_2fold=split2, held_out_5fold=split5, cell_18_11=cell_check,
                   calibration=calibration, calibration_rows=residual_rows,
                   secondary_all_paired=secondary, refit_bootstrap=refit,
                   notes=['uniforms are ranks within each voxel',
                          'sampling intervals resample trajectories and re-sum held-out scores; '
                          'refit intervals redo the whole 2-fold fit on each resample']),
              handle, indent=1)
print('\nwrote out/dependence_law.json  (%.0f s)' % (time.time() - start_time))
