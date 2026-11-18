"""Figures for step 9: dependence across the state space, the theta(u, tau, rate)
scoreboard and the residual maps.  Ranks within voxel x cooling rate throughout.

Input : out/joint_events.npz (grid, paired-event counts), out/marginals.json (cutoffs),
        out/dependence_law.json (the final step 9 result, from 9c_dependence_diagnostics.py)
Output: figures/voxel_dependence.png             measured Kendall's tau per voxel and its precision
        figures/dependence_scoreboard.png        effect size, and each formula against its voxel table
        figures/dependence_tau_residuals.png     implied minus measured Kendall's tau, formula vs table
        figures/dependence_tail_residuals.png    upper-tail model minus empirical, formula vs table
        figures/panels/<name>_<letter>.png       each panel on its own, for slides

Nothing is fitted here; every number is read from out/dependence_law.json.
After changing any figure, open the PNG and check it (CLAUDE.md).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from figstyle import (BLUE_RAMP, DIVERGING, FEW_EVENTS, INK, INK_2, KTAU, SERIES, SURFACE, T_LABEL, U_LABEL,
                      display_edges, letter, save)

events = np.load('out/joint_events.npz')
with open('out/marginals.json') as handle:
    msettings = json.load(handle)
with open('out/dependence_law.json') as handle:
    law = json.load(handle)
n_bins = int(events['n_bins'])
n_events = law['n_events']
xmin = msettings['xmin']
TOP = msettings['most_visited_cell']
du, dtau, cell_all = events['du'], events['dtau'], events['cell']
paired = (du < 0) & (dtau < 0) & (-du >= xmin['pe']) & (-dtau >= xmin['stress'])
PAIRED_COUNTS = np.bincount(cell_all[paired], minlength=n_bins * n_bins)
U_DISPLAY, TAU_DISPLAY = display_edges(events['u_edges']), display_edges(events['tau_edges'])
diag = law['diagnostics']
null = diag['null_check']['summary']
NAME = {'interaction+rate': 'interaction + rate (5 parameters)', 'quadratic+rate': 'quadratic + rate (7)',
        'product+rate': 'product + rate (3)', 'voxel_table+rate': 'voxel table + rate (120)',
        'voxel_table': 'voxel table (119)', 'interaction': 'interaction (4)', 'quadratic': 'quadratic (6)'}
DEFINITION = ('θ = 1 + exp(η).  Interaction + rate: η = a + b·u′ + c·τ′ + e·u′τ′ + g·ln(rate / rate_mid), '
              'with u′, τ′ the standardised energy and stress.\nVoxel table + rate: one θ per voxel (119) plus one '
              'shared cooling-rate coefficient.')


# --- map helpers ----------------------------------------------------------------------------------
def map_axes(ax):
    shown = np.nonzero(PAIRED_COUNTS >= 100)[0]
    ii, jj = shown // n_bins, shown % n_bins
    ax.set_xlim(U_DISPLAY[max(ii.min() - 1, 0)], U_DISPLAY[min(ii.max() + 2, n_bins)])
    ax.set_ylim(TAU_DISPLAY[max(jj.min() - 1, 0)], TAU_DISPLAY[min(jj.max() + 2, n_bins)])
    ax.set_xlabel(U_LABEL)
    ax.set_ylabel(T_LABEL)


def draw_map(ax, values, cmap, norm, outlined=(), top_box=False):
    grid = np.full(n_bins * n_bins, np.nan)
    for c, v in values.items():
        grid[c] = v
    mesh = ax.pcolormesh(U_DISPLAY, TAU_DISPLAY, grid.reshape(n_bins, n_bins).T, cmap=cmap, norm=norm,
                         edgecolors=SURFACE, linewidth=0.5)
    grey = np.where((PAIRED_COUNTS > 0) & (PAIRED_COUNTS < 100), 1.0, np.nan)
    ax.pcolormesh(U_DISPLAY, TAU_DISPLAY, grey.reshape(n_bins, n_bins).T, cmap=ListedColormap([FEW_EVENTS]),
                  edgecolors=SURFACE, linewidth=0.5)
    for c in outlined:
        i, j = c // n_bins, c % n_bins
        ax.add_patch(Rectangle((U_DISPLAY[i], TAU_DISPLAY[j]), U_DISPLAY[i + 1] - U_DISPLAY[i],
                               TAU_DISPLAY[j + 1] - TAU_DISPLAY[j], fill=False, edgecolor=INK, linewidth=1.3))
    if top_box:
        i, j = TOP // n_bins, TOP % n_bins
        ax.add_patch(Rectangle((U_DISPLAY[i], TAU_DISPLAY[j]), U_DISPLAY[i + 1] - U_DISPLAY[i],
                               TAU_DISPLAY[j + 1] - TAU_DISPLAY[j], fill=False, edgecolor=SERIES[1], linewidth=2.2))
    map_axes(ax)
    return mesh


GREY_KEY = Patch(facecolor=FEW_EVENTS, edgecolor=INK_2, linewidth=0.5, label='fewer than 100 paired events')
OUTLINE_KEY = Patch(facecolor='none', edgecolor=INK, linewidth=1.3, label='outside the 95% interval')

# --- 1. measured dependence across voxels -------------------------------------------------------------
voxels = law['pre_fit_checks']['voxel_tau_unpooled']['voxels']
tau_values = {v['cell']: v['kendall_tau'] for v in voxels}
half_width = {v['cell']: 0.5 * (v['tau_ci'][1] - v['tau_ci'][0]) for v in voxels}
trend = law['pre_fit_checks']['voxel_tau_unpooled']['group_regression']
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)
mesh = draw_map(axes[0], tau_values, BLUE_RAMP, None, top_box=True)
bar_a = fig.colorbar(mesh, ax=axes[0], shrink=0.9, label=KTAU)
axes[0].set_title('%s per voxel (%d voxels, %.2f – %.2f)' % (KTAU, len(voxels), min(tau_values.values()),
                                                            max(tau_values.values())))
mesh = draw_map(axes[1], half_width, BLUE_RAMP, None, top_box=True)
bar_b = fig.colorbar(mesh, ax=axes[1], shrink=0.9, label='half-width of the 95% interval')
axes[1].set_title('Precision: half-width of the 95% interval')
letter(axes[0], 'a', x=-0.1)
letter(axes[1], 'b', x=-0.1)
fig.legend(handles=[GREY_KEY, Patch(facecolor='none', edgecolor=SERIES[1], linewidth=2,
                                    label='most-visited voxel (18, 11)')], loc='outside lower center', ncol=2)
fig.suptitle('Dependence between PE drop and stress drop, measured within each cooling rate and averaged.\n'
             'Regression over (voxel, rate) groups: %s rises with u (%+.3f per sd) and stress (%+.3f per sd) and '
             'falls with cooling rate (%+.4f per unit ln rate, at fixed state)'
             % (KTAU, trend['coef'][1], trend['coef'][2], trend['coef'][3]))
save(fig, 'voxel_dependence', dict(a=[axes[0], bar_a.ax], b=[axes[1], bar_b.ax]))

# --- 2. scoreboard ------------------------------------------------------------------------------------
rep = law['replicates']
five = law['scoreboards']['held_out_5fold']['models']
effect = law['effect_size']
fig, axes = plt.subplots(2, 1, figsize=(13, 13), constrained_layout=True, height_ratios=[0.8, 1.2])

# (a) effect size: gain over a constant theta per 1,000 held-out events
ax = axes[0]
order = ('voxel_table+rate', 'quadratic+rate', 'interaction+rate', 'product+rate', 'voxel_table', 'quadratic',
         'interaction')
y = np.arange(len(order))
for k, (scheme, label, colour, hatch) in enumerate((('halving', '50/50 splits', SERIES[0], None),
                                                    ('split80', '80/20 splits', SERIES[1], '//'))):
    means = np.array([effect[scheme][m]['gain_per_1000'] for m in order])
    sds = np.array([effect[scheme][m]['sd'] for m in order])
    ax.barh(y + (k - 0.5) * 0.38, means, height=0.36, color=colour, hatch=hatch, edgecolor=SURFACE,
            label='%s (mean ± sd over 50)' % label)
    ax.errorbar(means, y + (k - 0.5) * 0.38, xerr=sds, fmt='none', ecolor=INK, elinewidth=1.2, capsize=3)
    for yi, m in zip(y, order):
        if m != 'voxel_table+rate':
            ax.text(means[list(order).index(m)] + sds[list(order).index(m)] + 0.25, yi + (k - 0.5) * 0.38,
                    '%.0f%%' % (100 * effect[scheme][m]['share_of_table_rate']), va='center', fontsize=10,
                    color=INK_2)
ax.set_yticks(y)
ax.set_yticklabels([NAME[m] for m in order])
ax.invert_yaxis()
ax.set_xlim(0, 25)
ax.grid(True, axis='x')
ax.set_xlabel('held-out ln L gain over a single θ, per 1,000 held-out events')
ax.set_title('Gain over a single θ (% = share of the voxel table + rate’s gain)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=2)
letter(ax, 'a', x=-0.3)

# (b) each formula minus its like-for-like table
N_PARAMS = dict(product=2, linear=3, interaction=4, quadratic=6)


def formula_vs_table(ax, rows, legend=dict(loc='lower left')):
    """Each formula in rows minus its voxel table: every split, mean ± sd, and the 5-fold interval."""
    rng = np.random.default_rng(0)
    for i, form in enumerate(rows):
        table = 'voxel_table+rate' if form.endswith('+rate') else 'voxel_table'
        key = '%s - %s' % (form, table)
        wins = []
        for scheme, marker, colour, offset in (('halving', 'o', SERIES[0], -0.24), ('split80', 's', SERIES[1], 0.0)):
            e = rep['%s/voxel_rate' % scheme]['comparisons'][key]
            vals = np.array(e['per_1000'])
            ax.scatter(vals, i + offset + rng.uniform(-0.06, 0.06, len(vals)), s=12, color=colour, alpha=0.45,
                       linewidths=0)
            ax.errorbar(vals.mean(), i + offset, xerr=vals.std(ddof=1), fmt=marker, color=INK,
                        markerfacecolor=colour, markersize=9, capsize=3, elinewidth=1.5)
            wins.append(e['wins'])
        c = five[form]['vs_voxel_table_rate' if form.endswith('+rate') else 'vs_voxel_table']
        point, lo, hi = 1000 * np.array([c['difference'], *c['ci']]) / n_events
        ax.errorbar(point, i + 0.24, xerr=[[point - lo], [hi - point]], fmt='D', color=INK,
                    markerfacecolor=SERIES[2], markersize=8, capsize=3, elinewidth=1.5)
        ax.text(1.01, i, 'won %d (50/50)\nwon %d (80/20)' % tuple(wins), transform=ax.get_yaxis_transform(),
                va='center', fontsize=10, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(['%s (%d)' % (f.replace('+rate', ' + rate'), N_PARAMS[f.split('+')[0]] + ('+' in f))
                        for f in rows])
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.grid(True, axis='x')
    ax.text(0.02, -0.5, 'formula better →', fontsize=10, color=INK_2)
    ax.text(-0.02, -0.5, '← table better', fontsize=10, color=INK_2, ha='right')
    ax.legend(handles=[
        Line2D([], [], marker='o', color=INK, markerfacecolor=SERIES[0], linestyle='', markersize=9,
               label='50/50 splits: each (dots), mean ± sd'),
        Line2D([], [], marker='s', color=INK, markerfacecolor=SERIES[1], linestyle='', markersize=9,
               label='80/20 splits: each (dots), mean ± sd'),
        Line2D([], [], marker='D', color=INK, markerfacecolor=SERIES[2], linestyle='', markersize=8,
               label='5 folds, 95% sampling interval')], **legend)


ax = axes[1]
formula_vs_table(ax, [f + s for f in ('product', 'linear', 'interaction', 'quadratic') for s in ('', '+rate')])
ax.set_xlabel('held-out ln L per 1,000 held-out events: formula minus its voxel table '
              '(rate-free forms vs voxel table; + rate forms vs voxel table + rate)')
ax.set_title('Against one θ per voxel: a tie with half the data, a clear table lead with 80%')
letter(ax, 'b', x=-0.3)
fig.suptitle('Held-out comparison of the θ(u, τ, rate) formulas, %d paired events in 119 voxels; whole '
             'trajectories held out, stratified by rate\n%s' % (n_events, DEFINITION), fontsize=12)
save(fig, 'dependence_scoreboard', dict(a=[axes[0]], b=[axes[1]]))

# slide version of (b): the three + rate formulas only
fig, ax = plt.subplots(figsize=(11, 5.0), constrained_layout=True)
formula_vs_table(ax, ['product+rate', 'interaction+rate', 'quadratic+rate'],
                 legend=dict(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3))
ax.set_xlabel('held-out ln L per 1,000 held-out events: formula minus one θ per voxel (+ rate)')
ax.set_title('Formula vs one θ per voxel: a tie with half the data, the table ahead with 80%')
save(fig, 'panels/dependence_scoreboard_b_slide')


# --- 3 and 4. residual maps -------------------------------------------------------------------------
def residual_figure(name, rows_key, value, outside, limit_label, title, subtitle, legend_extra):
    shown = ('interaction+rate', 'voxel_table+rate')
    rows = [r for r in diag[rows_key] if r['ranks'] == 'voxel_rate']
    values = {m: {r['cell']: value(r) for r in rows if r['model'] == m} for m in shown}
    out = {m: [r['cell'] for r in rows if r['model'] == m and outside(r)] for m in shown}
    limit = max(abs(v) for m in shown for v in values[m].values())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), constrained_layout=True)
    for key, ax, m in zip('ab', axes, shown):
        mesh = draw_map(ax, values[m], DIVERGING, norm, outlined=out[m])
        ax.set_title('%s\n%s' % (NAME[m], subtitle(m)))
        letter(ax, key, x=-0.1)
    bar = fig.colorbar(mesh, ax=axes, shrink=0.85, label=limit_label)
    fig.legend(handles=[GREY_KEY, OUTLINE_KEY] + legend_extra, loc='outside lower center', ncol=3)
    fig.suptitle(title)
    save(fig, name, dict(a=[axes[0]], b=[axes[1], bar.ax]))


def tau_subtitle(m):
    c = diag['calibration']['%s (voxel_rate)' % m]
    return ('%d / %d voxels outside the interval · mean bias %+.3f\ntrend of the bias on implied %s: %+.2f'
            % (c['voxels_outside_tau_ci'], c['n_voxels'], c['mean_bias'], KTAU.split(' ')[-1], c['trend_on_implied']))


residual_figure('dependence_tau_residuals', 'calibration_rows',
                lambda r: r['implied_tau'] - r['measured_tau'], lambda r: r['outside'],
                'implied minus measured %s  (blue: model too weak, red: too strong)' % KTAU,
                'Model-implied minus measured %s per voxel (in-sample fits).\nReference: a correct model shows mean '
                'bias %+.3f and trend %+.2f on these measurements' % (KTAU, abs(null['mean_bias']['mean']),
                                                                       null['trend_on_implied']['mean']),
                tau_subtitle, [])


def tail_subtitle(m):
    t = diag['tail']['%s (voxel_rate)' % m]
    return ('%d / %d voxels above the interval, %d below · mean %+.3f'
            % (t['overshoot'], t['n_voxels'], t['undershoot'], t['mean_difference']))


residual_figure('dependence_tail_residuals', 'tail_rows',
                lambda r: r['model_tail'] - r['empirical_tail'], lambda r: r['overshoot'] or r['undershoot'],
                'model minus empirical P(V > 0.95 | U > 0.95)  (red: model tail too heavy)',
                'Both drops large: model minus empirical P(V > 0.95 | U > 0.95) per voxel.\nReference: a correct model '
                'scores %+.3f on average and lands above the interval in %.0f voxels (small rank groups bias the '
                'empirical value low)' % (null['tail_mean_difference']['mean'], null['tail_overshoot']['mean']),
                tail_subtitle, [])
