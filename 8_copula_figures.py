"""Figures for the copula analysis (steps 5-7).

Input : out/joint_events.npz, out/marginals.json, out/marginal_table.npz,
        out/copulas.json
Output: figures/marginal_maps.png      truncated-power-law kappa and s* per voxel, PE and stress
        figures/marginal_fits.png      the fits in the most-visited voxel
        figures/pit_histograms.png     PIT of paired events under those fits
        (figures/copula_shapes.png is drawn by 11c_benchmark_figure.py, with voxel x rate ranks)
        figures/copula_scores.png      held-out gains and differences, ranks and IFM
        figures/tail_comparison.png    Kendall's tau and 5% corners: data vs four families
        figures/panels/<name>_<letter>.png   each panel of the multi-panel figures, for slides
        out/voxel_dependence.json      per voxel with >= 100 paired events: Kendall tau (pooled
                                       over rates), its interval and a Gumbel fit (read by step 9;
                                       the map of dependence across voxels is drawn by step 10
                                       from step 9's unpooled tau)

Analysis is NOT redone here: every marginal and copula parameter is read from
steps 6 and 7.  The only new quantities are, per voxel with >= 100 paired
events: Kendall tau (a statistic), a rank-based Gumbel fit (copulas.fit_family,
the same code as step 7), and trajectory-bootstrap intervals for tau and for
the most-visited cell's empirical tau and 5% corner values.
After changing any figure, open the PNG and check it (CLAUDE.md).
"""
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.patches import Patch, Rectangle
from scipy import stats
import copulas
import marginals
from figstyle import (BLUE_RAMP, FEW_EVENTS, GRID, INK, INK_2, KTAU, SERIES, SURFACE, T_LABEL, U_LABEL,
                      display_edges, letter, save)

warnings.filterwarnings('ignore', category=RuntimeWarning)
N_BOOT = 1000
SEED = 0
FAMILY_LABEL = dict(gumbel='Gumbel', survival_clayton='flipped Clayton', bb7='BB7',
                    survival_bb1='flipped BB1', bb1='BB1', survival_bb7='flipped BB7',
                    gaussian='Gaussian', t='Student t', frank='Frank', clayton='Clayton',
                    survival_gumbel='flipped Gumbel', independence='independence')
COMPARED = ('gumbel', 'survival_clayton', 'bb7', 'survival_bb1')   # one colour each, fixed

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
U_DISPLAY, TAU_DISPLAY = display_edges(u_edges), display_edges(tau_edges)
TOP_LABEL = '(%d, %d)' % (top // n_bins, top % n_bins)


def to_grid(values):
    return values.reshape(n_bins, n_bins).T            # rows = tau bin, cols = u bin


def outline_top(ax):
    i, j = top // n_bins, top % n_bins
    ax.add_patch(Rectangle((U_DISPLAY[i], TAU_DISPLAY[j]), U_DISPLAY[i + 1] - U_DISPLAY[i],
                           TAU_DISPLAY[j + 1] - TAU_DISPLAY[j], fill=False, edgecolor=SERIES[1],
                           linewidth=2.2))


def state_axes(ax, cells):
    """Axis labels, and limits trimmed to the shown voxels plus one voxel of margin."""
    ii, jj = cells // n_bins, cells % n_bins
    ax.set_xlabel(U_LABEL)
    ax.set_ylabel(T_LABEL)
    ax.set_xlim(U_DISPLAY[max(ii.min() - 1, 0)], U_DISPLAY[min(ii.max() + 2, n_bins)])
    ax.set_ylim(TAU_DISPLAY[max(jj.min() - 1, 0)], TAU_DISPLAY[min(jj.max() + 2, n_bins)])


def few_events_layer(ax, counts):
    """Grey: voxels with some drops, but fewer than MAP_MIN (not shown)."""
    grey = np.where((counts > 0) & (counts < MAP_MIN), 1.0, np.nan)
    ax.pcolormesh(U_DISPLAY, TAU_DISPLAY, to_grid(grey), cmap=ListedColormap([FEW_EVENTS]),
                  edgecolors=SURFACE, linewidth=0.5)


# --- 1. marginal maps -----------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
occupied = np.nonzero((table['pe_n'] >= MAP_MIN) | (table['stress_n'] >= MAP_MIN))[0]
panels = {}
for row, (name, label) in enumerate((('pe', 'PE drop −Δu ≥ 10⁻⁶'), ('stress', 'stress drop −Δτ ≥ 10⁻³'))):
    counts = table[name + '_n']
    shown = counts >= MAP_MIN
    for col, (field, title, norm) in enumerate((('kappa', 'exponent κ', Normalize()),
                                                ('s_star', 'cutoff s* (units of the drop)', LogNorm()))):
        values = np.where(shown, table['%s_%s' % (name, field)], np.nan)
        ax = axes[row, col]
        mesh = ax.pcolormesh(U_DISPLAY, TAU_DISPLAY, to_grid(values), cmap=BLUE_RAMP, norm=norm,
                             edgecolors=SURFACE, linewidth=0.5)
        few_events_layer(ax, counts)          # drawn last: the data layer paints empty cells
        bar = fig.colorbar(mesh, ax=ax, label=title, shrink=0.9)
        outline_top(ax)
        state_axes(ax, occupied)
        ax.set_title('%s: %s\n%d voxels with ≥ %d drops' % (label, title.split(' (')[0], shown.sum(), MAP_MIN))
        key = 'abcd'[2 * row + col]
        letter(ax, key)
        panels[key] = [ax, bar.ax]
fig.legend(handles=[Patch(facecolor=FEW_EVENTS, edgecolor=INK_2, linewidth=0.5,
                           label='voxel with fewer than %d drops (not fitted)' % MAP_MIN),
                     Patch(facecolor='none', edgecolor=SERIES[1], linewidth=2,
                           label='most-visited voxel %s' % TOP_LABEL)], loc='outside lower center', ncol=2)
fig.suptitle(r'Size distribution of drops in each voxel: $p(s) \propto s^{-\kappa}\,e^{-s/s^*}$')
save(fig, 'marginal_maps', panels)

# --- 2. fits in the most-visited cell ----------------------------------------------
in_top = cell == top
pe_drops = -du[in_top & (du < 0) & (-du >= XMIN_PE)]
st_drops = -dtau[in_top & (dtau < 0) & (-dtau >= XMIN_STRESS)]
paired_top = paired_all & in_top
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
panels = {}
for key, ax, name, drops, paired_sizes, xmin, xlabel in (
        ('a', axes[0], 'pe', pe_drops, -du[paired_top], XMIN_PE, 'PE drop −Δu'),
        ('b', axes[1], 'stress', st_drops, -dtau[paired_top], XMIN_STRESS, 'stress drop −Δτ')):
    kappa, s_star = table[name + '_kappa'][top], table[name + '_s_star'][top]
    edges = np.geomspace(xmin, drops.max() * 1.01, 31)
    centres = np.sqrt(edges[1:] * edges[:-1])
    for sizes, colour, marker, label in ((drops, INK_2, 'o', 'all drops (fitted), n = %d'),
                                         (paired_sizes, SERIES[1], 's', 'paired drops only, n = %d')):
        counts, _ = np.histogram(sizes, bins=edges)
        density = counts / (len(sizes) * np.diff(edges))
        ok = counts > 0
        ax.plot(centres[ok], density[ok], marker, color=colour, markersize=6, linestyle='none',
                label=label % len(sizes))
    curve = np.geomspace(xmin, drops.max(), 300)
    ax.plot(curve, np.exp(marginals.ln_density(curve, kappa, s_star, xmin)), color=SERIES[0],
            label='fit: κ = %.3f, s* = %.3g' % (kappa, s_star))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('probability density')
    ax.legend(loc='lower left')
    ax.grid(True, which='major')
    ax.set_title('%s, most-visited voxel %s' % (xlabel.split(' −')[0], TOP_LABEL))
    letter(ax, key)
    panels[key] = [ax]
fig.suptitle('Truncated power law fitted to all drops of each kind; the copula uses only the paired drops')
save(fig, 'marginal_fits', panels)

# --- 3. PIT histograms ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, sharey=True)
panels = {}
for key, ax, name, sizes, xmin, label in (('a', axes[0], 'pe', -du[paired_top], XMIN_PE, 'PE drop'),
                                          ('b', axes[1], 'stress', -dtau[paired_top], XMIN_STRESS, 'stress drop')):
    pit = marginals.cdf(sizes, table[name + '_kappa'][top], table[name + '_s_star'][top], xmin)
    ax.hist(pit, bins=20, range=(0, 1), density=True, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax.axhline(1.0, color=INK_2, linestyle='--', linewidth=1.4, label='uniform (perfect fit)')
    k = cop['pit']['PE' if name == 'pe' else 'stress']
    ax.set_title('%s: KS distance %.3f, mean %.2f (uniform: 0.50)' % (label, k['ks'], k['mean']))
    ax.set_xlabel('fitted CDF value of each paired %s' % label)
    ax.legend(loc='upper left')
    letter(ax, key)
    panels[key] = [ax]
axes[0].set_ylabel('density')
fig.suptitle('Do the fitted size laws describe the %d paired events in voxel %s?  (flat = yes)'
             % (paired_top.sum(), TOP_LABEL))
save(fig, 'pit_histograms', panels)

# --- 4. copula shapes: drawn by 11c_benchmark_figure.py with ranks within voxel x rate ---------

# --- 5. held-out scores ---------------------------------------------------------------
families = [f for f in cop['results']['cml']['families'] if f != 'independence']
families.sort(key=lambda f: -cop['results']['cml']['families'][f]['held_out_ll'])
fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), constrained_layout=True)
y_pos = np.arange(len(families))
height = 0.38
for k, (method, label, hatch) in enumerate((('cml', 'ranks (reference)', None),
                                             ('ifm', 'fitted size laws (secondary)', '//'))):
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
axes[0].set_title('Held-out score (95% interval: trajectories resampled)')
axes[0].legend(loc='lower right')
axes[1].set_xlabel('held-out ln L minus the best family (same method)')
axes[1].set_title('Difference from the best: intervals reaching 0 are ties')
axes[1].set_xlim(min(-1300, axes[1].get_xlim()[0]), 60)
letter(axes[0], 'a')
letter(axes[1], 'b')
fig.suptitle('Which copula family fits voxel %s?  %d paired events; intervals do not include refitting noise'
             % (TOP_LABEL, paired_top.sum()))
save(fig, 'copula_scores', dict(a=[axes[0]], b=[axes[1]]))


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

measures = (KTAU, 'both drops small\nP(V ≤ 0.05 | U ≤ 0.05)', 'both drops large\nP(V > 0.95 | U > 0.95)')
fig, ax = plt.subplots(figsize=(12, 5.6), constrained_layout=True)
width = 0.15
x_pos = np.arange(len(measures))
ax.bar(x_pos - 2 * width, data_stats, width=width - 0.02, color=INK_2, label='data (95% interval)')
ax.errorbar(x_pos - 2 * width, data_stats, yerr=[data_stats - data_ci[0], data_ci[1] - data_stats],
            fmt='none', ecolor=INK, elinewidth=1.2, capsize=4)
for i, (x, v) in enumerate(zip(x_pos - 2 * width, data_stats)):
    ax.text(x, data_ci[1][i] + 0.015, '%.2f' % v, ha='center', fontsize=10, color=INK)
hatches = (None, '//', '..', 'xx')
for k, family in enumerate(COMPARED):
    r = cop['results']['cml']['families'][family]
    values = [r['tau'], r['tail_lower_q'], r['tail_upper_q']]
    xs = x_pos + (k - 1) * width
    ax.bar(xs, values, width=width - 0.02, color=SERIES[k], hatch=hatches[k], edgecolor=SURFACE,
           label=FAMILY_LABEL[family])
    for x, v in zip(xs, values):
        ax.text(x, v + 0.015, '%.2f' % v, ha='center', fontsize=10, color=INK)
for k, xc in enumerate(x_pos[1:]):       # the corners only: independence gives 0.05 there
    ax.plot([xc - 2.6 * width, xc + 2.6 * width], [0.05, 0.05], color=INK, linewidth=1.4, linestyle=':',
            label='independence (corners: 0.05)' if k == 0 else None)
ax.set_xticks(x_pos)
ax.set_xticklabels(measures)
ax.set_ylim(0, 1)
ax.set_ylabel('value')
ax.grid(True, axis='y')
ax.legend(ncol=3, loc='upper left')
ax.set_title(r'Voxel %s, %d paired events: data vs rank-fitted families (independence: $\tau_K$ = 0)'
             % (TOP_LABEL, paired_top.sum()))
save(fig, 'tail_comparison')

# --- 7. dependence across voxels (numbers only; the map is drawn by step 10) ------------------
paired_counts = np.bincount(cell[paired_all], minlength=n_bins * n_bins)
voxels = np.nonzero(paired_counts >= MAP_MIN)[0]
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
    theta_map[c] = fit['params'][0]
    voxel_rows.append(dict(cell=int(c), cell_u=int(c // n_bins), cell_tau=int(c % n_bins),
                           n_paired=int(m.sum()), n_trajectories=int(len(ids)),
                           u_mid=float(np.median(u_state_all[m])),
                           tau_state_mid=float(np.median(tau_state_all[m])),
                           kendall_tau=float(tau_c), tau_ci=[float(lo), float(hi)],
                           gumbel_theta=float(fit['params'][0]), gumbel_at_bound=fit['at_bound']))

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
with open('out/voxel_dependence.json', 'w') as handle:
    json.dump(dict(min_paired=MAP_MIN, summary=summary, voxels=voxel_rows), handle, indent=1)
print('wrote out/voxel_dependence.json')
