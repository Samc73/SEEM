"""Per-voxel truncated-power-law marginals for PE drops and stress drops.

Input : out/joint_events.npz (step 5)
Output: out/marginals.json      settings, notes, pooled fits, summary
        out/marginal_table.npz  per-cell fits, one set of arrays per marginal

Marginals (each fitted on ITS OWN drops -- the "separate subsets" choice):
    pe            -du   over events with du < 0,    s >= 1e-6
    stress        -dtau over events with dtau < 0,  s >= 1e-3   (main)
    stress_1e-06  same, s >= 1e-6   } sensitivity check on the
    stress_1e-04  same, s >= 1e-4   } stress cutoff only

Why these cutoffs (see the step-5 review):
  * PE: below ~1e-7 there is a separate numerical-noise bump that holds almost
    all "PE only" events; 1e-6 sits clear of it and kappa is stable there.
  * stress: below ~1e-3 the drops are dominated by stress-only "aging" events
    (PE rises during the drop), a different population from the paired drops
    the copula uses.  1e-3 keeps 99% of paired events.  It therefore EXCLUDES
    most stress-only aging events from the stress marginal -- this is stated
    in the output and in marginals.json.

Cells with fewer than MIN_EVENTS drops above the cutoff get no fit (NaN in the
table, i.e. masked).  Every array in the table is n_bins * n_bins long and is
indexed by step 1's cell id, cell = bin_u * n_bins + bin_tau.
"""
import json
import time
import numpy as np
import marginals

XMIN_PE = 1e-6
XMIN_STRESS = 1e-3
XMIN_STRESS_SENSITIVITY = (1e-6, 1e-4)
MIN_EVENTS = 40          # same threshold as step 2's per-cell ceilings
MAP_MIN_EVENTS = 100     # stricter: a cell is only SHOWN on maps with this many;
                         # below ~80 events kappa can come out negative (noise)

start_time = time.time()
events = np.load('out/joint_events.npz')
du = events['du']
dtau = events['dtau']
cell = events['cell']
n_bins = int(events['n_bins'])
n_cells = n_bins * n_bins
most_visited = int(np.argmax(events['visits']))

pe_drop = du < 0
stress_drop = dtau < 0
stress_only = stress_drop & (du >= 0)

# name -> (sizes of that marginal's drops, cells of those drops, xmin)
configurations = {'pe': (-du[pe_drop], cell[pe_drop], XMIN_PE),
                  'stress': (-dtau[stress_drop], cell[stress_drop], XMIN_STRESS)}
for xmin in XMIN_STRESS_SENSITIVITY:
    configurations['stress_%.0e' % xmin] = (-dtau[stress_drop], cell[stress_drop], xmin)

FIELDS = ('kappa', 's_star', 'n', 'll', 'ks', 'ok')
table = {}
pooled = {}
for name, (size, size_cell, xmin) in configurations.items():
    keep = size >= xmin
    size, size_cell = size[keep], size_cell[keep]
    pooled[name] = dict(xmin=xmin, **marginals.fit(size, xmin))

    arrays = {f: np.full(n_cells, np.nan) for f in FIELDS}
    arrays['n'] = np.bincount(size_cell, minlength=n_cells).astype(float)
    for c in np.nonzero(arrays['n'] >= MIN_EVENTS)[0]:
        result = marginals.fit(size[size_cell == c], xmin)
        for f in FIELDS:
            if f != 'n':
                arrays[f][c] = result[f]
    for f in FIELDS:
        table['%s_%s' % (name, f)] = arrays[f]
    print('%-14s xmin %.0e  pooled kappa %.3f  s* %.3e  KS %.4f  (%d drops)  %.0fs'
          % (name, xmin, pooled[name]['kappa'], pooled[name]['s_star'],
             pooled[name]['ks'], pooled[name]['n'], time.time() - start_time))

# how many stress-only aging drops each cutoff throws away
size_stress_only = -dtau[stress_only]
excluded_aging = {('%.0e' % x): int((size_stress_only < x).sum())
                  for x in (XMIN_STRESS,) + XMIN_STRESS_SENSITIVITY}
table['stress_only_below_xmin'] = np.bincount(cell[stress_only & (-dtau < XMIN_STRESS)],
                                              minlength=n_cells).astype(float)
table['stress_only_total'] = np.bincount(cell[stress_only], minlength=n_cells).astype(float)

# --- report -------------------------------------------------------------------
note_aging = ('stress cutoff %.0e excludes %d of %d stress-only (aging) drops '
              '(%.0f%%): the main stress marginal describes mainly drops that '
              'come with a PE drop, not aging events'
              % (XMIN_STRESS, excluded_aging['%.0e' % XMIN_STRESS], len(size_stress_only),
                 100 * excluded_aging['%.0e' % XMIN_STRESS] / len(size_stress_only)))
note_mask = ('cells with fewer than %d drops above the cutoff are masked (NaN); '
             'maps show only cells with at least %d' % (MIN_EVENTS, MAP_MIN_EVENTS))
print('\nNOTE: ' + note_aging)
print('NOTE: ' + note_mask)

print('\n%-14s %6s %8s   %-17s %-17s %s'
      % ('marginal', 'cells', 'not ok', 'kappa med [range]', 's* median', 'KS median'))
summary = {}
for name in configurations:
    kappa = table[name + '_kappa']
    fitted = np.isfinite(kappa)
    summary[name] = dict(
        n_cells_fitted=int(fitted.sum()),
        n_cells_masked=int(((table[name + '_n'] > 0) & ~fitted).sum()),
        n_not_converged=int((table[name + '_ok'][fitted] == 0).sum()),
        kappa_median=float(np.median(kappa[fitted])),
        kappa_min=float(kappa[fitted].min()), kappa_max=float(kappa[fitted].max()),
        s_star_median=float(np.median(table[name + '_s_star'][fitted])),
        ks_median=float(np.median(table[name + '_ks'][fitted])))
    row = summary[name]
    print('%-14s %6d %8d   %.3f [%.2f, %.2f]  %.3e         %.4f'
          % (name, row['n_cells_fitted'], row['n_not_converged'], row['kappa_median'],
             row['kappa_min'], row['kappa_max'], row['s_star_median'], row['ks_median']))

print('\nsensitivity of the stress marginal to its cutoff (cells fitted at both):')
sensitivity = {}
for xmin in XMIN_STRESS_SENSITIVITY:
    other = 'stress_%.0e' % xmin
    both = np.isfinite(table['stress_kappa']) & np.isfinite(table[other + '_kappa'])
    d_kappa = table[other + '_kappa'][both] - table['stress_kappa'][both]
    ratio_s = table[other + '_s_star'][both] / table['stress_s_star'][both]
    sensitivity[other] = dict(n_cells=int(both.sum()),
                              d_kappa_median=float(np.median(d_kappa)),
                              d_kappa_abs_max=float(np.abs(d_kappa).max()),
                              s_star_ratio_median=float(np.median(ratio_s)))
    print('  %s vs 1e-03: %3d cells   kappa shift median %+.3f (max |%.3f|)   s* ratio median %.3f'
          % (other, both.sum(), np.median(d_kappa), np.abs(d_kappa).max(), np.median(ratio_s)))

c = most_visited
print('\nmost-visited cell %d = (%d, %d):' % (c, c // n_bins, c % n_bins))
for name in configurations:
    if np.isfinite(table[name + '_kappa'][c]):
        print('  %-14s n %5d  kappa %.3f  s* %.3e  KS %.4f'
              % (name, table[name + '_n'][c], table[name + '_kappa'][c],
                 table[name + '_s_star'][c], table[name + '_ks'][c]))
    else:
        print('  %-14s n %5d  masked' % (name, table[name + '_n'][c]))
print('  stress-only drops in this cell: %d, of which %d below %.0e'
      % (table['stress_only_total'][c], table['stress_only_below_xmin'][c], XMIN_STRESS))

np.savez_compressed('out/marginal_table.npz', n_bins=n_bins, **table)
with open('out/marginals.json', 'w') as handle:
    json.dump(dict(xmin=dict(pe=XMIN_PE, stress=XMIN_STRESS,
                             stress_sensitivity=list(XMIN_STRESS_SENSITIVITY)),
                   min_events=MIN_EVENTS, map_min_events=MAP_MIN_EVENTS,
                   most_visited_cell=most_visited,
                   notes=[note_aging, note_mask],
                   excluded_stress_only_drops=excluded_aging,
                   n_stress_only_drops=int(len(size_stress_only)),
                   pooled=pooled, summary=summary, sensitivity=sensitivity),
              handle, indent=1)
print('\nwrote out/marginals.json and out/marginal_table.npz  (%.0fs)' % (time.time() - start_time))
