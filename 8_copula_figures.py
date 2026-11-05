"""Figures for the copula analysis (steps 5-7).

Input : out/joint_events.npz, out/marginals.json, out/marginal_table.npz,
        out/copulas.json
Output: figures/marginal_maps.png      TPL kappa and s* per voxel, PE and stress
        figures/marginal_fits.png      TPL fits in the most-visited cell
        figures/pit_histograms.png     PIT of paired events under those fits
        figures/copula_shapes.png      data vs model copula densities (normal scores)
        figures/copula_scores.png      held-out gains and differences, ranks and IFM
        figures/tail_comparison.png    tau and 5% corners: data vs four families
        figures/voxel_dependence.png   Kendall tau and Gumbel theta per voxel
        out/voxel_dependence.json      the numbers behind the last figure

Analysis is NOT redone here: every marginal and copula parameter is read from
steps 6 and 7.  The only new quantities are, per voxel with >= 100 paired
events: Kendall tau (a statistic), a rank-based Gumbel fit (copulas.fit_family,
the same code as step 7), and trajectory-bootstrap intervals for tau and for
the most-visited cell's empirical tau and 5% corner values.
"""
import json
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.special import ndtr, ndtri
import copulas
import marginals

warnings.filterwarnings('ignore', category=RuntimeWarning)
N_BOOT = 1000
SEED = 0
DPI = 150

# --- style: reference palette (dataviz skill), light mode -------------------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'            # secondary text, data marks shown as "reference"
GRID = '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # categorical slots 1-4, fixed order
BLUE_RAMP = ListedColormap(['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
                            '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281',
                            '#0d366b'])
BLUE_RAMP.set_bad(SURFACE)
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': INK_2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.grid': False, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9, 'axes.titlesize': 10, 'legend.frameon': False, 'lines.linewidth': 2})
FAMILY_LABEL = dict(gumbel='Gumbel', survival_clayton='flipped Clayton', bb7='BB7',
                    survival_bb1='flipped BB1', bb1='BB1', survival_bb7='flipped BB7',
                    gaussian='Gaussian', t='t', frank='Frank', clayton='Clayton',
                    survival_gumbel='flipped Gumbel', independence='independence')
COMPARED = ('gumbel', 'survival_clayton', 'bb7', 'survival_bb1')   # one colour each, fixed

os.makedirs('figures', exist_ok=True)

# --- inputs -------------------------------------------------------------------
events = np.load('out/joint_events.npz')
table = np.load('out/marginal_table.npz')
with open('out/marginals.json') as handle:
    msettings = json.load(handle)
with open('out/copulas.json') as handle:
    cop = json.load(handle)
n_bins = int(events['n_bins'])
u_edges, tau_edges = events['u_edges'], events['tau_edges']
top = msettings['most_visited_cell']
XMIN_PE, XMIN_STRESS = msettings['xmin']['pe'], msettings['xmin']['stress']
MAP_MIN = msettings['map_min_events']
du, dtau, traj, cell = events['du'], events['dtau'], events['traj'], events['cell']
u_state_all, tau_state_all = events['u'], events['tau']     # the state at each event
paired_all = (du < 0) & (dtau < 0) & (-du >= XMIN_PE) & (-dtau >= XMIN_STRESS)


def display_edges(edges):
    """The outer ring of the grid spans the extremes; draw it one inner bin wide."""
    e = edges.copy()
    width = e[2] - e[1]
    e[0], e[-1] = e[1] - width, e[-2] + width
    return e


U_DISPLAY, TAU_DISPLAY = display_edges(u_edges), display_edges(tau_edges)


def to_grid(values):
    return values.reshape(n_bins, n_bins).T            # rows = tau bin, cols = u bin


def outline_top(ax):
    i, j = top // n_bins, top % n_bins
    ax.add_patch(Rectangle((U_DISPLAY[i], TAU_DISPLAY[j]), U_DISPLAY[i + 1] - U_DISPLAY[i],
                           TAU_DISPLAY[j + 1] - TAU_DISPLAY[j], fill=False, edgecolor=SERIES[1],
                           linewidth=2))


def state_axes(ax):
    ax.set_xlabel('u = pe − u₀')
    ax.set_ylabel('τ = stress / 10⁴')
    ax.set_xlim(U_DISPLAY[0], U_DISPLAY[-1])
    ax.set_ylim(TAU_DISPLAY[0], TAU_DISPLAY[-1])


def save(fig, name):
    fig.savefig('figures/' + name, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print('wrote figures/' + name)


# --- 1. marginal maps -----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
for row, (name, label) in enumerate((('pe', 'PE drop −Δu  (≥ 1e-6)'),
                                     ('stress', 'stress drop −Δτ  (≥ 1e-3)'))):
    shown = table[name + '_n'] >= MAP_MIN
    for col, (field, title, norm) in enumerate((('kappa', 'κ', Normalize()),
                                                ('s_star', 's*', LogNorm()))):
        values = np.where(shown, table['%s_%s' % (name, field)], np.nan)
        ax = axes[row, col]
        mesh = ax.pcolormesh(U_DISPLAY, TAU_DISPLAY, to_grid(values), cmap=BLUE_RAMP, norm=norm,
                             edgecolors=SURFACE, linewidth=0.5)
        fig.colorbar(mesh, ax=ax, label=title, shrink=0.9)
        outline_top(ax)
        state_axes(ax)
        ax.set_title('%s — %s   (%d voxels)' % (label, title, shown.sum()))
fig.suptitle('Truncated power law s^−κ exp(−s/s*) per voxel.  Blank: fewer than %d drops.  '
             'Orange box: most-visited voxel (18, 11).  Outer ring drawn one bin wide.' % MAP_MIN,
             fontsize=9, color=INK_2)
save(fig, 'marginal_maps.png')

# --- 2. TPL fits in the most-visited cell ----------------------------------------
in_top = cell == top
pe_drops = -du[in_top & (du < 0) & (-du >= XMIN_PE)]
st_drops = -dtau[in_top & (dtau < 0) & (-dtau >= XMIN_STRESS)]
paired_top = paired_all & in_top
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
for ax, name, drops, paired_sizes, xmin, xlabel in (
        (axes[0], 'pe', pe_drops, -du[paired_top], XMIN_PE, 'PE drop −Δu'),
        (axes[1], 'stress', st_drops, -dtau[paired_top], XMIN_STRESS, 'stress drop −Δτ')):
    kappa, s_star = table[name + '_kappa'][top], table[name + '_s_star'][top]
    edges = np.geomspace(xmin, drops.max() * 1.01, 31)
    centres = np.sqrt(edges[1:] * edges[:-1])
    for sizes, colour, marker, label in ((drops, INK_2, 'o', 'all drops (fitted), n = %d'),
                                         (paired_sizes, SERIES[1], 's', 'paired drops only, n = %d')):
        counts, _ = np.histogram(sizes, bins=edges)
        density = counts / (len(sizes) * np.diff(edges))
        ok = counts > 0
        ax.plot(centres[ok], density[ok], marker, color=colour, markersize=5, linestyle='none',
                label=label % len(sizes))
    curve = np.geomspace(xmin, drops.max(), 300)
    ax.plot(curve, np.exp(marginals.ln_density(curve, kappa, s_star, xmin)), color=SERIES[0],
            label='TPL fit: κ %.3f, s* %.3g' % (kappa, s_star))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('probability density')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, which='major')
fig.suptitle('Most-visited voxel (18, 11): TPL fitted to all drops of each kind; '
             'the copula sees only the paired ones', fontsize=10)
save(fig, 'marginal_fits.png')

# --- 3. PIT histograms ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True, sharey=True)
for ax, name, sizes, xmin, label in ((axes[0], 'pe', -du[paired_top], XMIN_PE, 'PE'),
                                     (axes[1], 'stress', -dtau[paired_top], XMIN_STRESS, 'stress')):
    pit = marginals.cdf(sizes, table[name + '_kappa'][top], table[name + '_s_star'][top], xmin)
    ax.hist(pit, bins=20, range=(0, 1), density=True, color=SERIES[0], edgecolor=SURFACE,
            linewidth=2)
    ax.axhline(1.0, color=INK_2, linestyle='--', linewidth=1.2, label='uniform')
    k = cop['pit']['PE' if name == 'pe' else 'stress']
    ax.set_title('%s: KS %.3f (p %.1g), mean PIT %.3f' % (label, k['ks'], k['p'], k['mean']))
    ax.set_xlabel('PIT = F_TPL(size) of paired events')
    ax.legend(loc='upper left', fontsize=8)
axes[0].set_ylabel('density')
fig.suptitle('PIT of the %d paired events under the voxel\'s TPL marginals '
             '(flat if the marginals described them)' % paired_top.sum(), fontsize=10)
save(fig, 'pit_histograms.png')

# --- 4. copula shapes in normal-score space ----------------------------------------
x_top, y_top = -du[paired_top], -dtau[paired_top]
zx = ndtri(copulas.pseudo_observations(x_top))
zy = ndtri(copulas.pseudo_observations(y_top))
grid = np.linspace(-3.5, 3.5, 181)
gx, gy = np.meshgrid(grid, grid)
normal_pdf = stats.norm.pdf(gx) * stats.norm.pdf(gy)
panels = ('gaussian', 'gumbel', 'survival_clayton', 'bb7', 'survival_bb1')
fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True)
data_hist, _, _ = np.histogram2d(zx, zy, bins=36, range=[[-3.5, 3.5], [-3.5, 3.5]], density=True)
levels = [0.005, 0.02, 0.05, 0.1, 0.2]
for ax, family in zip(axes.ravel(), ('data',) + panels):
    ax.pcolormesh(np.linspace(-3.5, 3.5, 37), np.linspace(-3.5, 3.5, 37),
                  np.where(data_hist.T > 0, data_hist.T, np.nan), cmap=BLUE_RAMP,
                  norm=LogNorm(vmin=1e-3, vmax=0.5))
    if family == 'data':
        ax.set_title('data: %d paired events (normal scores)' % len(zx))
    else:
        r = cop['results']['cml']['families'][family]
        density = np.exp(copulas.log_density(family, r['params'], ndtr(gx), ndtr(gy))) * normal_pdf
        contours = ax.contour(gx, gy, density, levels=levels, colors=SERIES[1], linewidths=1.4)
        ax.clabel(contours, fontsize=6, fmt='%g')
        ax.set_title('%s  (held-out gain %.0f)' % (FAMILY_LABEL[family], r['held_out_ll']))
    ax.set_aspect('equal')
for ax in axes[1]:
    ax.set_xlabel('Φ⁻¹(rank of PE drop)')
for ax in axes[:, 0]:
    ax.set_ylabel('Φ⁻¹(rank of stress drop)')
fig.suptitle('Copula shape, most-visited voxel. Blue: data density. Orange: rank-fitted model density '
             '(same contour levels in every panel). Upper right = both drops large.', fontsize=10)
save(fig, 'copula_shapes.png')

# --- 5. held-out scores ---------------------------------------------------------------
families = [f for f in cop['results']['cml']['families'] if f != 'independence']
families.sort(key=lambda f: -cop['results']['cml']['families'][f]['held_out_ll'])
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
y_pos = np.arange(len(families))
height = 0.38
for k, (method, label, hatch) in enumerate((('cml', 'ranks (reference)', None),
                                             ('ifm', 'TPL marginals (IFM)', '//'))):
    rows = cop['results'][method]['families']
    offset = (-0.5 + k) * (height + 0.04)
    for ax, value, ci_key in ((axes[0], 'held_out_ll', 'held_out_ci'),
                              (axes[1], 'delta_best', 'delta_best_ci')):
        values = np.array([rows[f][value] for f in families])
        ci = np.array([rows[f][ci_key] for f in families])
        ax.barh(y_pos + offset, values, height=height, color=SERIES[k], hatch=hatch,
                edgecolor=SURFACE, linewidth=0, label=label)
        ax.errorbar(values, y_pos + offset, xerr=[values - ci[:, 0], ci[:, 1] - values],
                    fmt='none', ecolor=INK, elinewidth=1, capsize=2)
for ax in axes:
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FAMILY_LABEL[f] for f in families])
    ax.invert_yaxis()
    ax.grid(True, axis='x')
    ax.axvline(0, color=INK_2, linewidth=0.8)
axes[0].set_xlabel('held-out ln L gain over independence')
axes[0].set_title('Held-out score, 95% CI from resampling trajectories')
axes[0].legend(loc='lower right')
axes[1].set_xlabel('held-out ln L minus the best family of the same method')
axes[1].set_title('Difference from the best (paired CI): bars reaching 0 are ties')
best_cml = cop['results']['cml']['best']
axes[1].set_xlim(min(-1300, axes[1].get_xlim()[0]), 60)
fig.suptitle('Families ordered by the rank-based score.  Best: %s (ranks), %s (IFM).  '
             'Intervals do not include refitting noise.'
             % (FAMILY_LABEL[best_cml], FAMILY_LABEL[cop['results']['ifm']['best']]), fontsize=10)
save(fig, 'copula_scores.png')


# --- 6. tau and 5% corners: data vs four families ----------------------------------------
def corner_stats(x, y):
    u, v = copulas.pseudo_observations(x), copulas.pseudo_observations(y)
    lower, upper = copulas.empirical_joint_tail(u, v, 0.05)
    return np.array([stats.kendalltau(x, y)[0], lower, upper])


rng = np.random.default_rng(SEED)
traj_top = traj[paired_top]
trajectories = np.unique(traj_top)
members = [np.nonzero(traj_top == t)[0] for t in trajectories]
boot = []
for _ in range(N_BOOT):
    pick = np.concatenate([members[i] for i in rng.integers(0, len(trajectories), len(trajectories))])
    boot.append(corner_stats(x_top[pick], y_top[pick]))
boot = np.array(boot)
data_stats = corner_stats(x_top, y_top)
data_ci = np.percentile(boot, [2.5, 97.5], axis=0)

measures = ('Kendall τ', 'lower 5% corner\nP(V ≤ .05 | U ≤ .05)', 'upper 5% corner\nP(V > .95 | U > .95)')
fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
width = 0.15
x_pos = np.arange(len(measures))
ax.bar(x_pos - 2 * width, data_stats, width=width - 0.02, color=INK_2, label='data')
ax.errorbar(x_pos - 2 * width, data_stats, yerr=[data_stats - data_ci[0], data_ci[1] - data_stats],
            fmt='none', ecolor=INK, elinewidth=1, capsize=3)
for i, (x, v) in enumerate(zip(x_pos - 2 * width, data_stats)):
    ax.text(x, data_ci[1][i] + 0.015, '%.2f' % v, ha='center', fontsize=7, color=INK)
for k, family in enumerate(COMPARED):
    r = cop['results']['cml']['families'][family]
    values = [r['tau'], r['tail_lower_q'], r['tail_upper_q']]
    xs = x_pos + (k - 1) * width
    ax.bar(xs, values, width=width - 0.02, color=SERIES[k], label=FAMILY_LABEL[family])
    for x, v in zip(xs, values):
        ax.text(x, v + 0.015, '%.2f' % v, ha='center', fontsize=7, color=INK)
ax.set_xticks(x_pos)
ax.set_xticklabels(measures)
ax.set_ylim(0, 1)
ax.set_ylabel('value')
ax.grid(True, axis='y')
ax.legend(ncol=5, loc='upper left', fontsize=8)
ax.set_title('Most-visited voxel: data (95% CI, trajectories resampled) vs rank-fitted models.  '
             'Independence gives τ = 0 and 0.05 in each corner.', fontsize=9)
save(fig, 'tail_comparison.png')

# --- 7. dependence across voxels -----------------------------------------------------------
paired_counts = np.bincount(cell[paired_all], minlength=n_bins * n_bins)
voxels = np.nonzero(paired_counts >= MAP_MIN)[0]
tau_map = np.full(n_bins * n_bins, np.nan)
theta_map = np.full(n_bins * n_bins, np.nan)
voxel_rows = []
for c in voxels:
    m = paired_all & (cell == c)
    x, y, t = -du[m], -dtau[m], traj[m]
    tau_c = stats.kendalltau(x, y)[0]
    fit = copulas.fit_family('gumbel', copulas.pseudo_observations(x), copulas.pseudo_observations(y))
    ids = np.unique(t)
    groups = [np.nonzero(t == i)[0] for i in ids]
    tau_boot = []
    for _ in range(200):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(ids), len(ids))])
        tau_boot.append(stats.kendalltau(x[pick], y[pick])[0])
    lo, hi = np.percentile(tau_boot, [2.5, 97.5])
    tau_map[c], theta_map[c] = tau_c, fit['params'][0]
    voxel_rows.append(dict(cell=int(c), cell_u=int(c // n_bins), cell_tau=int(c % n_bins),
                           n_paired=int(m.sum()), n_trajectories=int(len(ids)),
                           u_mid=float(np.median(u_state_all[m])),
                           tau_state_mid=float(np.median(tau_state_all[m])),
                           kendall_tau=float(tau_c), tau_ci=[float(lo), float(hi)],
                           gumbel_theta=float(fit['params'][0]), gumbel_at_bound=fit['at_bound']))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
for ax, values, label in ((axes[0], tau_map, 'Kendall τ (empirical)'),
                          (axes[1], theta_map, 'Gumbel θ (rank-based fit)')):
    mesh = ax.pcolormesh(U_DISPLAY, TAU_DISPLAY, to_grid(values), cmap=BLUE_RAMP,
                         edgecolors=SURFACE, linewidth=0.5)
    fig.colorbar(mesh, ax=ax, label=label, shrink=0.9)
    norm = mesh.norm
    for c in voxels:
        i, j = c // n_bins, c % n_bins
        shade = norm(values[c])
        ax.text(0.5 * (U_DISPLAY[i] + U_DISPLAY[i + 1]), 0.5 * (TAU_DISPLAY[j] + TAU_DISPLAY[j + 1]),
                '%d' % paired_counts[c], ha='center', va='center', fontsize=6,
                color='white' if shade > 0.55 else INK)
    outline_top(ax)
    state_axes(ax)
    ax.set_title('%s, %d voxels with ≥ %d paired events (number = events)'
                 % (label, len(voxels), MAP_MIN))
fig.suptitle('Dependence between PE drop and stress drop across the state space.  '
             'Orange box: most-visited voxel.', fontsize=10)
save(fig, 'voxel_dependence.png')

# summary of whether the dependence varies with state
taus = np.array([r['kendall_tau'] for r in voxel_rows])
cis = np.array([r['tau_ci'] for r in voxel_rows])
top_row = next(r for r in voxel_rows if r['cell'] == top)
outside_top = int(np.sum((cis[:, 1] < top_row['tau_ci'][0]) | (cis[:, 0] > top_row['tau_ci'][1])))
u_state = np.array([r['u_mid'] for r in voxel_rows])
t_state = np.array([r['tau_state_mid'] for r in voxel_rows])
summary = dict(n_voxels=len(voxel_rows), tau_min=float(taus.min()), tau_max=float(taus.max()),
               tau_median=float(np.median(taus)),
               theta_min=float(np.nanmin(theta_map)), theta_max=float(np.nanmax(theta_map)),
               voxels_ci_disjoint_from_top=outside_top,
               spearman_tau_vs_u=[float(v) for v in stats.spearmanr(u_state, taus)],
               spearman_tau_vs_stress=[float(v) for v in stats.spearmanr(t_state, taus)],
               corner_bootstrap=dict(data=data_stats.tolist(), ci_low=data_ci[0].tolist(),
                                     ci_high=data_ci[1].tolist()))
print('\nvoxels with >= %d paired events: %d' % (MAP_MIN, len(voxel_rows)))
print('Kendall tau across them: %.3f to %.3f (median %.3f); Gumbel theta %.2f to %.2f'
      % (taus.min(), taus.max(), np.median(taus), np.nanmin(theta_map), np.nanmax(theta_map)))
print('voxels whose tau CI does not overlap the most-visited voxel\'s [%.3f, %.3f]: %d'
      % (*top_row['tau_ci'], outside_top))
print('Spearman of voxel tau vs state: u %.2f (p %.2g), stress %.2f (p %.2g)'
      % (*summary['spearman_tau_vs_u'], *summary['spearman_tau_vs_stress']))
print('most-visited voxel corners: tau %.3f [%.3f, %.3f], lower %.3f [%.3f, %.3f], upper %.3f [%.3f, %.3f]'
      % tuple(np.ravel(np.column_stack([data_stats, data_ci[0], data_ci[1]]))))
for r in sorted(voxel_rows, key=lambda r: -r['n_paired']):
    print('  (%2d,%2d) n %5d  traj %4d  tau %.3f [%.3f, %.3f]  gumbel theta %.2f%s'
          % (r['cell_u'], r['cell_tau'], r['n_paired'], r['n_trajectories'], r['kendall_tau'],
             *r['tau_ci'], r['gumbel_theta'], '  AT BOUND' if r['gumbel_at_bound'] else ''))
with open('out/voxel_dependence.json', 'w') as handle:
    json.dump(dict(min_paired=MAP_MIN, summary=summary, voxels=voxel_rows), handle, indent=1)
print('wrote out/voxel_dependence.json')
