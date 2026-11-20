"""Figures for step 11b: where the kernel copula gains, and the copula shapes with clean ranks.

Input : out/copula_benchmark_rate.json, out/copula_benchmark_rate_events.npz (step 11b),
        out/dependence_law.json (measured tau_K of the family-check voxels),
        out/joint_events.npz, out/marginals.json (cell (18, 11)'s paired events, for the shapes)
Output: figures/kernel_benchmark.png (+ figures/panels/)
        figures/copula_shapes_rank_{a,b}.png              Gumbel and kernel in cell (18, 11), rank scale
        figures/exploratory/copula_shapes_normal_score.png   the same two on normal scores
        figures/copula_shapes.png                          Gumbel and kernel in raw drop sizes (log axes)
        figures/exploratory/copula_shapes_6family.png   backup: data + step 7's five families
        figures/copula_marginals.png                       PE and stress drop CCDFs in cell (18, 11)

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
      Slide figures: no suptitle.  rank_a: step 11b's held-out uniforms as a
      20 x 20 histogram, model copula density as contours; rank_b: model density
      as a heat map, held-out uniforms as dots.  Same levels/scale in both panels.
copula_marginals
      CCDFs of all drops in the cell (step 6's fit sample) with step 6's TPL, and
      of the paired events in the slowest and fastest three cooling rates.  Units
      assume LAMMPS metal units (u in eV/atom; tau = stress[bar] / 1e4, in GPa).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from scipy.special import ndtr, ndtri
import copulas
import dependence_law as dl
import marginals
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


def gain(name):
    return 1000 * top['held_out_totals'][name] / top['n']


def shapes_figure(names, ncols, figsize, path, with_data=True):
    shown = (('data',) if with_data else ()) + names
    nrows = int(np.ceil(len(shown) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, sharex=True, sharey=True,
                             squeeze=False)
    for key, ax, name in zip('abcdef', axes.ravel(), shown):
        ax.pcolormesh(np.linspace(-3.5, 3.5, 37), np.linspace(-3.5, 3.5, 37),
                      np.where(data_hist.T > 0, data_hist.T, np.nan), cmap=fs.BLUE_RAMP,
                      norm=LogNorm(vmin=1e-3, vmax=0.5))
        if name == 'data':
            ax.set_title('data: %d paired events' % len(zx))
        else:
            ax.contour(gx, gy, model_density(name), levels=LEVELS, colors=fs.SERIES[1], linewidths=1.6)
            ax.set_title(('%s: gain %.0f per 1,000' if with_data else '%s (gain %.0f per 1,000)')
                         % (FAMILY_LABEL[name], gain(name)))
        ax.set_aspect('equal')
        fs.letter(ax, key)
    for ax in axes[-1]:
        ax.set_xlabel('PE drop (normal score of rank)')
    for ax in axes[:, 0]:
        ax.set_ylabel('stress drop (normal score of rank)')
    if with_data:
        fig.suptitle('Joint shape of the two drops in voxel (%d, %d), ranks within voxel × cooling rate.\n'
                     'Blue: data.  Orange: fitted model (same density levels in every panel).  Upper right = both '
                     'drops large.\nGain = held-out ln L over independence per 1,000 events (step 11b).'
                     % tuple(top['cell_uv']), fontsize=12)
    fs.save(fig, path)


# rank scale: step 11b's held-out uniforms (voxel x rate ranks) as data; models as copula densities c(U, V)
u_ho, v_ho = ev['cell_18_11/u'], ev['cell_18_11/v']
assert len(u_ho) == top['n']
N_RANK_BINS = 20
rank_hist = np.histogram2d(u_ho, v_ho, bins=N_RANK_BINS, range=[[0, 1], [0, 1]], density=True)[0]
rank_grid = (np.arange(200) + 0.5) / 200
gu, gv = np.meshgrid(rank_grid, rank_grid)
RANK_LEVELS = [0.5, 1, 2, 4, 8]                     # copula density, both panels; 1 = independence
RANK_NORM = LogNorm(vmin=0.1, vmax=20)              # one held-out event in a bin = 0.10
RANK_CMAP = fs.BLUE_RAMP.with_extremes(under=fs.SURFACE, bad=fs.SURFACE)


def copula_density(name):
    if name == 'kernel':
        return np.exp(copulas.kernel_log_density(u_all, v_all, gu.ravel(), gv.ravel(), h_all).reshape(gu.shape))
    return np.exp(copulas.log_density(name, copulas.fit_family(name, u_all, v_all)['params'], gu, gv))


def rank_figure(path, style):
    """style 'a': data histogram + model contours; 'b': model heat map + data as dots."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), constrained_layout=True, sharex=True, sharey=True)
    edges = np.linspace(0, 1, N_RANK_BINS + 1)
    for key, ax, name in zip('ab', axes, ('gumbel', 'kernel')):
        dens = copula_density(name)
        if style == 'a':
            im = ax.pcolormesh(edges, edges, np.where(rank_hist.T > 0, rank_hist.T, np.nan), cmap=RANK_CMAP,
                               norm=RANK_NORM)
            cs = ax.contour(gu, gv, dens, levels=RANK_LEVELS, colors=fs.SERIES[1], linewidths=1.6)
            ax.clabel(cs, fmt='%g', fontsize=10)
        else:
            im = ax.pcolormesh(gu, gv, dens, cmap=RANK_CMAP, norm=RANK_NORM, shading='nearest')
            ax.scatter(u_ho, v_ho, s=3, color=fs.INK, alpha=0.3, lw=0, rasterized=True)
        ax.set_title('%s (gain %.0f per 1,000)' % (FAMILY_LABEL[name], gain(name)))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel('PE-drop rank U')
        fs.letter(ax, key)
    axes[0].set_ylabel('stress-drop rank V')
    if style == 'a':
        label = 'held-out events per unit area (1 = independence)\norange contours: model copula density'
    else:
        label = 'model copula density (1 = independence)\ndots: %d held-out events' % len(u_ho)
    cb = fig.colorbar(im, ax=list(axes), shrink=0.8, extend='max')
    cb.set_label(label)
    fs.save(fig, path)


shapes_figure(('gumbel', 'kernel'), 2, (9.6, 5.0), 'exploratory/copula_shapes_normal_score', with_data=False)
rank_figure('copula_shapes_rank_a', 'a')
rank_figure('copula_shapes_rank_b', 'b')


# raw units: density of (log10 PE drop, log10 stress drop) = c(F(a), G(b)) f(a) g(b), with F, f (and G, g) the
# voxel's empirical marginal of the log drop, smoothed by a Gaussian KDE so the density is continuous.
# shortcut: the copula was fitted on voxel x rate ranks but is mapped through the voxel's pooled marginals.
def smoothed_marginal(sample, at):
    kde = stats.gaussian_kde(sample)
    h = np.sqrt(kde.covariance[0, 0])
    z = (at[:, None] - sample[None, :]) / h
    return np.clip(ndtr(z).mean(1), 1e-9, 1 - 1e-9), kde(at)


lx, ly = np.log10(x), np.log10(y)
a_grid = np.linspace(np.log10(xmin['pe']), lx.max() + 0.2, 220)
b_grid = np.linspace(np.log10(xmin['stress']), ly.max() + 0.2, 220)
F, f = smoothed_marginal(lx, a_grid)
G, g = smoothed_marginal(ly, b_grid)
FU, GV = np.meshgrid(F, G)
RAW_LEVELS = [0.03, 0.1, 0.3, 1]                     # per decade², both panels
RAW_NORM = LogNorm(vmin=0.02, vmax=3)


def raw_figure():
    a_edges = np.linspace(a_grid[0], a_grid[-1], 31)
    b_edges = np.linspace(b_grid[0], b_grid[-1], 31)
    hist = np.histogram2d(lx, ly, bins=[a_edges, b_edges], density=True)[0]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9), constrained_layout=True, sharex=True, sharey=True)
    for key, ax, name in zip('ab', axes, ('gumbel', 'kernel')):
        if name == 'kernel':
            ln_c = copulas.kernel_log_density(u_all, v_all, FU.ravel(), GV.ravel(), h_all).reshape(FU.shape)
        else:
            params = copulas.fit_family(name, u_all, v_all)['params']
            ln_c = copulas.log_density(name, params, FU, GV)
            theta = float(np.ravel(params)[0])
            ax.text(0.03, 0.97,'$C(u,v)=\\exp\\{-[(-\\ln u)^{\\theta}+(-\\ln v)^{\\theta}]^{1/\\theta}\\}$\n'
                    'fitted θ = %.2f  (Kendall τ$_K$ = 1 − 1/θ = %.2f)' % (theta, 1 - 1 / theta),
                    transform=ax.transAxes, ha='left', va='top', fontsize=10.5, color=fs.INK,
                    bbox=dict(boxstyle='round,pad=0.4', fc=fs.SURFACE, ec=fs.GRID))
        dens = np.exp(ln_c) * f[None, :] * g[:, None]
        print('%s: max raw density %.2f per decade²' % (name, dens.max()))
        im = ax.pcolormesh(10 ** a_edges, 10 ** b_edges, np.where(hist.T > 0, hist.T, np.nan), cmap=RANK_CMAP,
                           norm=RAW_NORM)
        cs = ax.contour(10 ** a_grid, 10 ** b_grid, dens, levels=RAW_LEVELS, colors=fs.SERIES[1], linewidths=1.6)
        ax.clabel(cs, fmt='%g', fontsize=10)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(FAMILY_LABEL[name] if name == 'kernel' else 'Gumbel')
        ax.set_xlabel('PE drop −Δu (eV/atom)')
        fs.letter(ax, key)
    axes[0].set_ylabel('stress drop −Δτ (GPa)')
    cb = fig.colorbar(im, ax=list(axes), shrink=0.8, extend='both')
    cb.set_label('%d paired events, density per decade²\norange contours: model, same units' % len(x))
    fs.save(fig, 'copula_shapes')


raw_figure()
shapes_figure(('gaussian', 'gumbel', 'survival_clayton', 'bb7', 'survival_bb1'), 3, (13, 9.6),
              'exploratory/copula_shapes_6family')


# --- marginals of cell (18, 11): CCDFs with step 6's TPL, slow vs fast cooling --------------------------
table = np.load('out/marginal_table.npz')
thirds = ri // 3                                      # 0 = slowest three of the nine rates, 2 = fastest
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
for ax, key, signed, paired, name, xlabel in (
        (axes[0], 'pe', du, x, 'PE', 'PE drop −Δu (eV/atom)'),
        (axes[1], 'stress', dtau, y, 'stress', 'stress drop −Δτ (GPa)')):
    cut = xmin[key]
    cut_text = r'$10^{%d}$' % round(np.log10(cut))
    own = -signed[(events['cell'] == c) & (signed < 0) & (-signed >= cut)]
    assert len(own) == table[key + '_n'][c]            # exactly step 6's fit sample
    kappa, s_star = table[key + '_kappa'][c], table[key + '_s_star'][c]
    for sample, colour, lw, label in (
            (own, fs.INK_2, 1.4, 'all %s drops in the cell (n %d)' % (name, len(own))),
            (paired[thirds == 0], fs.SERIES[0], 2, 'paired, slowest 3 cooling rates (n %d)' % (thirds == 0).sum()),
            (paired[thirds == 2], fs.SERIES[1], 2, 'paired, fastest 3 cooling rates (n %d)' % (thirds == 2).sum())):
        s = np.sort(sample)
        ax.step(s, 1 - np.arange(len(s)) / len(s), where='post', color=colour, lw=lw, label=label)
    s = np.logspace(np.log10(cut), np.log10(own.max()), 300)
    ax.plot(s, 1 - marginals.cdf(s, kappa, s_star, cut), color=fs.INK, ls='--', lw=1.6,
            label='step 6 TPL fit to all drops:\nκ %.2f, s* %.2g, xmin %s' % (kappa, s_star, cut_text))
    ax.axvline(cut, color=fs.INK_2, ls=':', lw=1.2)
    ax.text(cut * 1.15, 1.5, 'paired-event cutoff %s' % cut_text, fontsize=10, color=fs.INK_2, va='top')
    ratio = np.median(paired[thirds == 2]) / np.median(paired[thirds == 0])
    ax.set_title('%s drop: median fast / slow = %.2f' % (name, ratio))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1e-4, 2)
    ax.set_xlabel(xlabel)
    ax.legend(loc='lower left')
axes[0].set_ylabel('P(drop ≥ s), voxel (%d, %d)' % tuple(top['cell_uv']))
fs.letter(axes[0], 'a'); fs.letter(axes[1], 'b')
fs.save(fig, 'copula_marginals')
