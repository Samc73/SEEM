"""Figure for step 11b: where the kernel copula gains over Gumbel (and Frank).

Input : out/copula_benchmark_rate.json, out/copula_benchmark_rate_events.npz (step 11b),
        out/dependence_law.json (measured tau_K of the family-check voxels)
Output: figures/kernel_benchmark.png (+ figures/panels/)

(a-c) held-out ln c_kernel - ln c_model, summed over the held-out events in each
      cell of a 10 x 10 grid on the held-out uniforms, per 1,000 events of the
      voxel (so the cells add up to the voxel's total per 1,000 events).
(d)   kernel - Gumbel and kernel - best named family per 1,000 events, per voxel,
      with step 11b's sampling-only 95% intervals.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

N_GRID = 10
HIGH = 'second_highest_tau'          # (19, 14): clearest kernel - Gumbel gain, no bandwidth flag
NICE = dict(gumbel='Gumbel', frank='Frank', survival_clayton='flipped Clayton')

with open('out/copula_benchmark_rate.json') as handle:
    res = json.load(handle)['results']
ev = np.load('out/copula_benchmark_rate_events.npz')
with open('out/dependence_law.json') as handle:
    law = json.load(handle)
measured = {k: v['measured_tau'] for k, v in law['diagnostics']['family'].items()}
# for scale: how far the 5-parameter formula trails the voxel table (step 9, 80/20 splits)
FORMULA_GAP = -law['replicates']['split80/voxel_rate']['comparisons'][
    'interaction+rate - voxel_table+rate']['mean_per_1000']


def grid_map(label, model):
    u, v = ev[label + '/u'], ev[label + '/v']
    d = ev[label + '/ll_kernel'] - ev[label + '/ll_' + model]
    n = len(u)
    # totals must match step 11b's json
    assert abs(d.sum() - (res[label]['held_out_totals']['kernel'] - res[label]['held_out_totals'][model])) < 1e-6
    iu, iv = np.minimum((u * N_GRID).astype(int), N_GRID - 1), np.minimum((v * N_GRID).astype(int), N_GRID - 1)
    total = np.zeros((N_GRID, N_GRID))
    count = np.zeros((N_GRID, N_GRID))
    np.add.at(total, (iv, iu), d)
    np.add.at(count, (iv, iu), 1)
    out = np.where(count > 0, 1000 * total / n, np.nan)
    upper = (u > 0.9) & (v > 0.9)
    lower = (u < 0.1) & (v < 0.1)
    regions = {k: 1000 * d[m].sum() / n for k, m in
               (('upper corner', upper), ('lower corner', lower), ('bulk', ~(upper | lower)))}
    if model == 'gumbel':     # region split must match step 11b's json
        for k, j in (('upper corner', 'upper_corner'), ('lower corner', 'lower_corner'), ('bulk', 'bulk')):
            assert abs(regions[k] - res[label]['by_region'][j]['per_1000_events']) < 1e-6
    return out, regions, 1000 * d.sum() / n


maps = [('cell_18_11', 'gumbel'), (HIGH, 'gumbel'), (HIGH, 'frank')]
computed = [grid_map(*m) for m in maps]
vmax = max(np.nanmax(np.abs(c[0])) for c in computed)

fig = plt.figure(figsize=(13.5, 9.6))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.06], height_ratios=[1, 0.95], hspace=0.42, wspace=0.28)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax = fig.add_subplot(gs[0, 3])
for ax, (label, model), (m, regions, total), key in zip(axes, maps, computed, 'abc'):
    im = ax.imshow(m, origin='lower', extent=(0, 1, 0, 1), cmap=fs.DIVERGING, vmin=-vmax, vmax=vmax)
    r = res[label]
    ax.set_title('(%d, %d), τ_K %.2f, n %d\nkernel − %s: %+.0f per 1,000'
                 % (*r['cell_uv'], measured[label], r['n'], NICE[model], total), fontsize=11)
    ax.set_xlabel('PE-drop rank U (held out)')
    ax.set_ylabel('stress-drop rank V (held out)')
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    for corner, (x0, y0) in (('upper corner', (0.9, 0.9)), ('lower corner', (0, 0))):
        ax.add_patch(plt.Rectangle((x0, y0), 0.1, 0.1, fill=False, ec=fs.INK, lw=1.2))
    ax.text(0.03, 0.97, 'bulk %+.0f\nupper corner %+.0f\nlower corner %+.0f'
            % (regions['bulk'], regions['upper corner'], regions['lower corner']),
            transform=ax.transAxes, va='top', ha='left', fontsize=10, color=fs.INK,
            bbox=dict(fc=fs.SURFACE, ec='none', alpha=0.85, pad=1.5))
    fs.letter(ax, key)
cb = fig.colorbar(im, cax=cax)
cb.set_label('kernel − model, held-out ln L\nper 1,000 voxel events, per cell')

# (d) per voxel
ax = fig.add_subplot(gs[1, :3])
order = sorted(['lowest_tau', 'cell_18_11', 'third_highest_tau', 'second_highest_tau', 'highest_tau'],
               key=lambda k: measured[k])
xs = np.arange(len(order))
for off, key, colour, marker, name in ((-0.12, 'kernel_minus_gumbel', fs.SERIES[0], 'o', 'kernel − Gumbel'),
                                       (+0.12, 'kernel_minus_best', fs.SERIES[1], 's', 'kernel − best named family')):
    y = [res[k][key]['per_1000_events'] for k in order]
    lo = [y[i] - res[k][key]['ci_per_1000'][0] for i, k in enumerate(order)]
    hi = [res[k][key]['ci_per_1000'][1] - y[i] for i, k in enumerate(order)]
    ax.errorbar(xs + off, y, yerr=[lo, hi], fmt=marker, color=colour, ms=8, capsize=4, lw=1.6, label=name)
pooled = res['cell_18_11_pooled']['kernel_minus_gumbel']
i18 = order.index('cell_18_11')
ax.plot(i18 - 0.12, pooled['per_1000_events'], 'D', mfc='none', mec=fs.INK_2, ms=9, mew=1.5,
        label='kernel − Gumbel, ranks pooled over rates (step 11)')
ax.axhline(0, color=fs.INK_2, lw=1)
ax.axhline(FORMULA_GAP, color=fs.INK_2, lw=1.2, ls='--',
           label='for scale: voxel table − 5-parameter formula, all voxels (step 9, 80/20): %.1f' % FORMULA_GAP)
ax.set_xticks(xs)
ax.set_xticklabels(['(%d, %d)\nτ_K %.2f, n %d\nbest: %s' % (*res[k]['cell_uv'], measured[k], res[k]['n'],
                                                          NICE[res[k]['best_parametric']]) for k in order])
ax.set_xlim(-0.5, len(order) - 0.5)
ax.set_ylabel('held-out ln L per 1,000 events')
ax.yaxis.grid(True)
ax.legend(loc='upper left', ncol=1)
fs.letter(ax, 'd')
fs.save(fig, 'kernel_benchmark', panels=dict(a=[axes[0]], b=[axes[1]], c=[axes[2], cax], d=[ax]))
