"""Which copula describes the joint law of PE drop and stress drop?

Input : out/joint_events.npz (step 5), out/marginals.json and
        out/marginal_table.npz (step 6)
Output: out/copulas.json

Sample: PAIRED events in the most-visited cell (18, 11) -- du < 0 and
dtau < 0, with -du >= 1e-6 and -dtau >= 1e-3 (step 6's cutoffs).

Families: independence (baseline), gaussian, t, clayton, gumbel, frank,
survival clayton, survival gumbel.  Each is fitted two ways (copulas.py):
  ranks (pseudo-observations) -- the REFERENCE.  It needs no marginal model.
  IFM -- PIT through this cell's TPL marginals.  Step 6 found the TPL only
         approximate here, and the marginals are fitted on all of each
         quantity's drops (not only paired ones), so the PIT need not be
         uniform; its KS test is reported first.  For held-out scoring the
         marginals are refitted on the training trajectories.
Scored on held-out data with step 3's split (even trajectory ids against
odd), with 95% intervals from resampling trajectories.

Run test_copulas.py first: it checks this code on synthetic data.
"""
import json
import time
import warnings
import numpy as np
from scipy import stats
import copulas
import marginals

warnings.filterwarnings('ignore', category=RuntimeWarning)
N_BOOT = 2000
Q_TAIL = 0.05
SEED = 0

start_time = time.time()
events = np.load('out/joint_events.npz')
with open('out/marginals.json') as handle:
    settings = json.load(handle)
table = np.load('out/marginal_table.npz')
n_bins = int(events['n_bins'])
c = settings['most_visited_cell']
XMIN_PE = settings['xmin']['pe']
XMIN_STRESS = settings['xmin']['stress']

du, dtau, traj = events['du'], events['dtau'], events['traj']
in_cell = events['cell'] == c
pe_drop = in_cell & (du < 0) & (-du >= XMIN_PE)           # the PE marginal's sample
stress_drop = in_cell & (dtau < 0) & (-dtau >= XMIN_STRESS)
paired = pe_drop & stress_drop                             # the copula's sample
x = -du[paired]          # PE drop size
y = -dtau[paired]        # stress drop size
traj_pair = traj[paired]
print('cell %d = (%d, %d): %d PE drops, %d stress drops, %d paired'
      % (c, c // n_bins, c % n_bins, pe_drop.sum(), stress_drop.sum(), paired.sum()))


def ifm_pit(train):
    """PIT of every pair through TPL marginals.  train = None: step 6's fits on
    all trajectories; otherwise refit on the training parity's trajectories,
    using ALL of their drops in this cell (the same sample definition as step 6)."""
    if train is None:
        pe = (table['pe_kappa'][c], table['pe_s_star'][c])
        st = (table['stress_kappa'][c], table['stress_s_star'][c])
    else:
        parity = np.unique(traj_pair[train] % 2)
        assert len(parity) == 1, 'folds must be split by trajectory parity'
        same_parity = (traj % 2) == parity[0]
        fit_pe = marginals.fit(-du[pe_drop & same_parity], XMIN_PE)
        fit_st = marginals.fit(-dtau[stress_drop & same_parity], XMIN_STRESS)
        pe = (fit_pe['kappa'], fit_pe['s_star'])
        st = (fit_st['kappa'], fit_st['s_star'])
    return (marginals.cdf(x, pe[0], pe[1], XMIN_PE),
            marginals.cdf(y, st[0], st[1], XMIN_STRESS))


# --- PIT uniformity of the paired data under the TPL marginals ---------------
u_pit, v_pit = ifm_pit(None)
pit = {}
print('\nPIT of paired events under the cell\'s TPL marginals (uniform if the')
print('marginals described the paired events exactly):')
for name, values, all_drops in (('PE', u_pit, -du[pe_drop]), ('stress', v_pit, -dtau[stress_drop])):
    ks = stats.kstest(values, 'uniform')
    pit[name] = dict(ks=float(ks.statistic), p=float(ks.pvalue), mean=float(values.mean()),
                     median_paired=float(np.median(x if name == 'PE' else y)),
                     median_all_drops=float(np.median(all_drops)))
    print('  %-6s KS %.4f  p %.2g   mean PIT %.3f (0.5 if uniform)   median size: paired %.3g, all drops %.3g'
          % (name, ks.statistic, ks.pvalue, values.mean(), pit[name]['median_paired'],
             pit[name]['median_all_drops']))

# --- the contest -----------------------------------------------------------
print()
results = copulas.run_contest(x, y, traj_pair, ifm_pit, n_boot=N_BOOT, q_tail=Q_TAIL, seed=SEED)
copulas.print_contest(results)

print('\nside by side (held-out ln L gain over independence, 95% CI by trajectory):')
print('  %4s   %-40s   %-40s' % ('rank', 'RANKS (reference)', 'TPL MARGINALS (IFM)'))
orders = {m: sorted(results[m]['families'], key=lambda f: -results[m]['families'][f]['held_out_ll'])
          for m in ('cml', 'ifm')}
for rank in range(len(copulas.FAMILIES)):
    cells = []
    for m in ('cml', 'ifm'):
        f = orders[m][rank]
        r = results[m]['families'][f]
        cells.append('%-17s %7.1f [%7.1f, %7.1f]' % (f, r['held_out_ll'], *r['held_out_ci']))
    print('  %4d   %-40s   %-40s' % (rank + 1, cells[0], cells[1]))

# --- does the second parameter help?  BB families against the one-parameter leaders
TWO_PARAMETER = ('bb1', 'survival_bb1', 'bb7', 'survival_bb7')
ONE_PARAMETER_LEADERS = ('gumbel', 'survival_clayton')
SAME_MODEL = 1.0   # held-out CI entirely within +-1 nat: the two fits are the same model
e = results['empirical']
comparisons = {}
for m, label in (('cml', 'RANKS (reference)'), ('ifm', 'TPL MARGINALS (IFM)')):
    rows = results[m]['families']
    print('\n%s: two-parameter fits against the data' % label)
    print('  %-14s %-26s %6s %8s %8s %8s %8s  %s'
          % ('family', 'params', 'tau', 'lam_L', 'lam_U', 'tailL.05', 'tailU.05', 'bound flags'))
    print('  %-14s %-26s %6.3f %8s %8s %8.3f %8.3f'
          % ('data', '', e['kendall_tau'], '', '', e['tail_lower_q'], e['tail_upper_q']))
    for f in TWO_PARAMETER + ONE_PARAMETER_LEADERS:
        r = rows[f]
        flags = '; '.join(r['at_bound'] + r['fold_at_bound']) or '-'
        print('  %-14s %-26s %6.3f %8.3f %8.3f %8.3f %8.3f  %s'
              % (f, ', '.join('%s %.3f' % (nm, p) for nm, p in zip(r['param_names'], r['params'])),
                 r['tau'], r['lambda_lower'], r['lambda_upper'], r['tail_lower_q'],
                 r['tail_upper_q'], flags))
    print('  held-out difference (two-parameter minus one-parameter), 95% CI by trajectory, P(> 0):')
    comparisons[m] = {}
    for f in TWO_PARAMETER:
        for g in ONE_PARAMETER_LEADERS:
            point, ci, p = copulas.paired_difference(results, m, f, g)
            # a two-parameter family that collapses onto its one-parameter limit
            # differs from it only by rounding; do not call that better or worse
            verdict = ('same model (collapsed to its limit)' if max(abs(ci[0]), abs(ci[1])) < SAME_MODEL
                       else 'better' if ci[0] > 0 else 'worse' if ci[1] < 0 else 'not distinguishable')
            comparisons[m]['%s - %s' % (f, g)] = dict(difference=point, ci=ci, p_positive=p,
                                                      verdict=verdict)
            print('    %-14s - %-17s %+8.1f [%+7.1f, %+7.1f]  P %.3f  -> %s'
                  % (f, g, point, *ci, p, verdict))

note = ('rank-based (cml) results are the reference: the TPL marginals are only '
        'approximate in this cell and were fitted on all drops, not only paired ones')
print('\nNOTE: ' + note)
print('NOTE: intervals resample trajectories with replacement and re-sum held-out scores; '
      'they do not include refitting noise')

with open('out/copulas.json', 'w') as handle:
    json.dump(dict(cell=c, cell_u=c // n_bins, cell_tau=c % n_bins,
                   xmin=dict(pe=XMIN_PE, stress=XMIN_STRESS),
                   n_pe_drops=int(pe_drop.sum()), n_stress_drops=int(stress_drop.sum()),
                   n_paired=int(paired.sum()), pit=pit, notes=[note],
                   results={m: ({k: v for k, v in r.items() if k != 'boot_scores'}
                                if m in ('cml', 'ifm') else r) for m, r in results.items()},
                   two_parameter_comparisons=comparisons),
              handle, indent=1)
print('\nwrote out/copulas.json  (%.0fs)' % (time.time() - start_time))
