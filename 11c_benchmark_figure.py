"""Figures for step 11b: where the kernel copula gains, and the copula shapes with clean ranks.

Input : out/copula_benchmark_rate.json, out/copula_benchmark_rate_events.npz (step 11b),
        out/dependence_law.json (measured tau_K of the family-check voxels),
        out/joint_events.npz, out/marginals.json (cell (18, 11)'s paired events, for the shapes)
Output: figures/kernel_benchmark.png (+ figures/panels/)
        figures/copula_shapes.png (+ figures/panels/)   data, Gaussian, Gumbel, kernel in cell (18, 11)
        figures/exploratory/copula_shapes_6family.png   backup: data + step 7's five families

kernel_benchmark
(a-c) held-out ln c_kernel - ln c_model, summed over the held-out events in each
      cell of a 10 x 10 grid on the held-out uniforms, per 1,000 events of the
      voxel (so the cells add up to the voxel's total per 1,000 events), with
      the four-region split of copulas.square_regions (q = 0.2).
(d)   kernel - Gumbel and kernel - best named family per 1,000 events, per voxel,
      with step 11b's sampling-only 95% intervals.
copula_shapes
      ranks within voxel x rate (step 9's convention) on all of the cell's paired
      events; every model fitted in-sample on those ranks (kernel bandwidth chosen
      by trajectory blocks, as in step 11b); titles give step 11b's held-out gain
      over independence per 1,000 events.  Replaces step 8's pooled-rank version.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.special import ndtr, ndtri
import copulas
import dependence_law as dl
import figstyle as fs

N_GRID = 10
Q_REGION = 0.2
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
    regions = {k: 1000 * d[m].sum() / n for k, m in copulas.square_regions(u, v, Q_REGION).items()}
    # the four regions must add up to the panel total and match step 11b's json
    assert abs(sum(regions.values()) - 1000 * d.sum() / n) < 1e-9
    for k, val in regions.items():
        assert abs(val - res[label]['by_region_q20'][model][k]['per_1000_events']) < 1e-6
    return out, regions, 1000 * d.sum() / n


maps = [('cell_18_11', 'gumbel'), (HIGH, 'gumbel'), (HIGH, 'frank')]
computed = [grid_map(*m) for m in maps]
vmax = max(np.nanmax(np.abs(c[0])) for c in computed)

fig = plt.figure(figsize=(13.5, 10.2))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.06], height_ratios=[1, 0.95], hspace=0.62, wspace=0.28)
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
    q = Q_REGION
    for x0, y0 in ((1 - q, 1 - q), (0, 0)):         # upper and lower tail
        ax.add_patch(plt.Rectangle((x0, y0), q, q, fill=False, ec=fs.INK, lw=1.4))
    ax.plot([q, q, 1], [1, q, q], color=fs.INK, lw=1.0, ls='--')   # inner border of the low-rank edges
    ax.text(0.5, -0.24, 'upper tail %+.0f, lower tail %+.0f\nlow-rank edges %+.0f, middle %+.0f'
            % (regions['upper_tail'], regions['lower_tail'], regions['low_rank_edges'], regions['middle']),
            transform=ax.transAxes, va='top', ha='center', fontsize=10.5, color=fs.INK)
    fs.letter(ax, key)
cb = fig.colorbar(im, cax=cax)
cb.set_label('kernel − model, held-out ln L\nper 1,000 voxel events, per cell')

# (d) per voxel
ax = fig.add_subplot(gs[1, :3])
order = sorted(['lowest_tau', 'cell_18_11', 'third_highest_tau', 'second_highest_tau', 'highest_tau'],
               key=lambda k: measured[k])
xs = np.arange(len(order))
for off, key, colour, marker, name in ((-0.12, 'kernel_minus_gumbel', fs.SERIES[0], 'o', 'kernel − Gumbel'),
                                       (+0.12, 'kernel_minus_best', fs.SERIES[1], 's',
                                        'kernel − best named family (chosen on the same held-out data)')):
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


# --- copula shapes in cell (18, 11), clean ranks -----------------------------------------------------
events = np.load('out/joint_events.npz')
with open('out/marginals.json') as handle:
    xmin = json.load(handle)['xmin']
top = res['cell_18_11']
c = top['cell']
du, dtau = events['du'], events['dtau']
m = (events['cell'] == c) & (du < 0) & (dtau < 0) & (-du >= xmin['pe']) & (-dtau >= xmin['stress'])
assert m.sum() == top['n']
x, y, traj = -du[m], -dtau[m], events['traj'][m]
ri = np.searchsorted(np.unique(events['rate']), events['rate'])[m]
levels = [ri, ri // 3, np.zeros(len(x), int)]
everyone = np.ones(len(x), bool)
u_all = dl.grouped_uniforms(x, levels, everyone)[0]
v_all = dl.grouped_uniforms(y, levels, everyone)[0]
zx, zy = ndtri(u_all), ndtri(v_all)
grid = np.linspace(-3.5, 3.5, 181)
gx, gy = np.meshgrid(grid, grid)
normal_pdf = stats.norm.pdf(gx) * stats.norm.pdf(gy)
h_all = copulas.choose_kernel_bandwidth(u_all, v_all, (traj // 2) % 5)[0]
data_hist, _, _ = np.histogram2d(zx, zy, bins=36, range=[[-3.5, 3.5], [-3.5, 3.5]], density=True)
LEVELS = [0.005, 0.02, 0.05, 0.1, 0.2]
FAMILY_LABEL = dict(gaussian='Gaussian', gumbel='Gumbel', kernel='kernel copula', survival_clayton='flipped Clayton',
                    bb7='BB7', survival_bb1='flipped BB1')


def model_density(name):
    if name == 'kernel':
        ln_c = copulas.kernel_log_density(u_all, v_all, ndtr(gx).ravel(), ndtr(gy).ravel(), h_all).reshape(gx.shape)
    else:
        ln_c = copulas.log_density(name, copulas.fit_family(name, u_all, v_all)['params'], ndtr(gx), ndtr(gy))
    return np.exp(ln_c) * normal_pdf


def shapes_figure(names, ncols, figsize, path, with_panels):
    nrows = int(np.ceil((len(names) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, sharex=True, sharey=True)
    panels = {}
    for key, ax, name in zip('abcdef', axes.ravel(), ('data',) + names):
        ax.pcolormesh(np.linspace(-3.5, 3.5, 37), np.linspace(-3.5, 3.5, 37),
                      np.where(data_hist.T > 0, data_hist.T, np.nan), cmap=fs.BLUE_RAMP,
                      norm=LogNorm(vmin=1e-3, vmax=0.5))
        if name == 'data':
            ax.set_title('data: %d paired events' % len(zx))
        else:
            ax.contour(gx, gy, model_density(name), levels=LEVELS, colors=fs.SERIES[1], linewidths=1.6)
            ax.set_title('%s: gain %.0f per 1,000' % (FAMILY_LABEL[name],
                                                      1000 * top['held_out_totals'][name] / top['n']))
        ax.set_aspect('equal')
        fs.letter(ax, key)
        panels[key] = [ax]
    for ax in axes[-1]:
        ax.set_xlabel('PE drop (normal score of rank)')
    for ax in axes[:, 0]:
        ax.set_ylabel('stress drop (normal score of rank)')
    fig.suptitle('Joint shape of the two drops in voxel (%d, %d), ranks within voxel × cooling rate.\n'
                 'Blue: data.  Orange: fitted model (same density levels in every panel).  Upper right = both drops '
                 'large.\nGain = held-out ln L over independence per 1,000 events (step 11b).' % tuple(top['cell_uv']),
                 fontsize=12)
    fs.save(fig, path, panels if with_panels else None)


shapes_figure(('gaussian', 'gumbel', 'kernel'), 2, (9.6, 10.4), 'copula_shapes', True)
shapes_figure(('gaussian', 'gumbel', 'survival_clayton', 'bb7', 'survival_bb1'), 3, (13, 9.6),
              'exploratory/copula_shapes_6family', False)
