import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Figures 16-18: where preparation memory acts, the repair of the u-spread,
and the ceiling-vs-sample-maximum control."""
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRATCH = _os.path.join(_HERE, 'out')
FIG = _os.path.join(_HERE, '..', 'figures') + '/'
mw = json.load(open(SCRATCH + '/memory_where.json'))
sm = json.load(open(SCRATCH + '/sim_memory.json'))
pin = json.load(open(SCRATCH + '/ceiling_pinning.json'))
pins = json.load(open(SCRATCH + '/ceiling_pinning_synth.json'))
rates = np.array(sm['rates'])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG + name, dpi=300)
    plt.close(fig)
    print('saved', name)


# ---- 16: which channel carries the memory ----
order = ['hazard p_drop', 'aging hazard p_age', 'geo-mean size', 'mean size <s>', 'plastic rate q']
labels = {'hazard p_drop': 'event hazard $p(u,\\tau)$', 'aging hazard p_age': 'aging hazard',
          'geo-mean size': 'geometric-mean event size', 'mean size <s>': 'mean event size $\\langle s\\rangle$',
          'plastic rate q': 'plastic rate $q$'}
fig, A = plt.subplots(figsize=(7.6, 5.4))
y = np.arange(len(order))
for i, k in enumerate(order):
    c = mw['channels'][k]
    lo, hi = c['control_1684']
    A.barh(i, hi - lo, left=lo, height=0.55, color='0.85', ec='none')
    A.plot(c['slow_fast_1684'], [i, i], color='crimson', lw=2.2, solid_capstyle='round')
    A.plot(c['slow_fast'], i, 'o', color='crimson', ms=8, mec='k', zorder=5)
    A.text(3.4, i, '$\\beta$ = %+.3f ± %.3f' % (c['beta'], c['se']), va='center', fontsize=9)
A.axvline(1, color='k', lw=1)
A.set_xscale('log')
A.set_xlim(0.5, 7.5)
A.set_xticks([0.5, 0.7, 1, 1.5, 2, 3], ['0.5', '0.7', '1', '1.5', '2', '3'])
A.set_yticks(y, [labels[k] for k in order])
A.set_xlabel('slow-cooled / fast-cooled at the same $(u,\\tau)$ cell  (median; bar: 16–84% of cells)')
A.set_title('Preparation memory acts through event size, not event rate')
A.plot([], [], 's', color='0.85', ms=12, label='control: even vs odd trajectories, same preparation mix')
A.plot([], [], 'o-', color='crimson', mec='k', label='slowest 3 vs fastest 3 cooling rates (64× in rate)')
A.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), frameon=False)
A.invert_yaxis()
save(fig, 'fig16_memory_channels.png')

# ---- 17: repairing the u-spread with a one-parameter ceiling memory ----
fig, (L, Rr) = plt.subplots(1, 2, figsize=(12.5, 4.8))
g_data = np.array(sm['data']['g_ds'])
cu_data = np.array(sm['data']['curves_u'])
arms = [a for a in ['model4', 'mem_sc', 'mem_relax_pw', 'mem_sc_u', 'mem_sc_k', 'mem_empir'] if a in sm]
style = {'model4': (':', 'k', 'README model (preparation-blind)'),
         'mem_sc': ('--', 'crimson', 'ceiling × (rate)$^{\\beta}$, one global $\\beta$'),
         'mem_relax_pw': ('-', 'seagreen', 'ceiling × (rate)$^{\\beta(\\gamma)}$, relaxing (Fig. 19)'),
         'mem_sc_u': ('-.', 'darkorange', 'ceiling × (rate)$^{\\beta(u)}$'),
         'mem_sc_k': ((0, (3, 1, 1, 1)), 'purple', '+ per-preparation energy coupling'),
         'mem_empir': ((0, (5, 2)), '0.45', 'per-preparation empirical resampling')}
g_sim = np.arange(len(sm['model4']['curves_u'][0])) * sm['rec'] * 1e-5
for j, col in [(0, plt.cm.viridis(0.0)), (len(rates) - 1, plt.cm.viridis(0.92))]:
    L.plot(g_data, cu_data[j], color=col, lw=2.2, label='MD, %.2g K/s' % rates[j])
    for a in ['model4', 'mem_sc']:
        L.plot(g_sim, np.array(sm[a]['curves_u'])[j], ls=style[a][0], color=col, lw=1.6)
L.plot([], [], ls=':', color='k', label='README model')
L.plot([], [], ls='--', color='k', label='ceiling × (rate)$^{\\beta}$')
L.set_xlabel('strain $\\gamma$')
L.set_ylabel('mean $u$ of the preparation')
L.set_title('Slowest and fastest preparations: data vs two models')
L.legend(fontsize=8.5)


def spread_curve(curves, g):
    c = np.array(curves)
    return g, c.max(0) / c.min(0) - 1


gd, sd = spread_curve(cu_data, g_data)
Rr.plot(gd, 100 * sd, color='k', lw=2.4, label='MD data')
for a in arms:
    gs, ss = spread_curve(sm[a]['curves_u'], g_sim)
    Rr.plot(gs, 100 * ss, ls=style[a][0], color=style[a][1], lw=1.7, label=style[a][2])
Rr.set_ylim(0, 60)
Rr.set_xlim(0.05, 0.5)
Rr.set_xlim(0.05, 0.5)
Rr.set_xlabel('strain $\\gamma$')
Rr.set_ylabel('spread of mean $u$ across preparations, max/min − 1  (%)')
Rr.set_title('The failure mode (Fig. 13) and its repair')
Rr.legend(fontsize=8)
save(fig, 'fig17_memory_repair.png')

# ---- 18: ceiling vs sample maximum ----
fig, A = plt.subplots(figsize=(7.0, 5.4))
mk = {1: ('o', 'remove largest 1 event'), 5: ('s', 'remove largest 5'), 'pct': ('^', 'remove largest 1%')}


def pts(rows, key):
    x, y = [], []
    for r in rows:
        for k, v in r['drops'].items():
            kk = int(k)
            kind = 1 if kk == 1 else (5 if kk == 5 else 'pct')
            if kind == key:
                x.append(v['smax_ratio']); y.append(v['sc_ratio'])
    return np.array(x), np.array(y)


for key in [1, 5, 'pct']:
    x, y = pts(pins['tpl'], key)
    A.plot(x, y, mk[key][0], color='royalblue', ms=6, alpha=0.8, mec='none')
    x, y = pts(pins['invpow4'], key)
    A.plot(x, y, mk[key][0], color='seagreen', ms=6, alpha=0.8, mec='none')
    x, y = pts(pin, key)
    A.plot(x, y, mk[key][0], color='crimson', ms=7, mec='k')
A.plot([0.2, 1.05], [0.2, 1.05], 'k-', lw=1)
A.set_xscale('log'); A.set_yscale('log')
A.set_xlim(0.22, 1.05); A.set_ylim(1e-3, 1.3)
A.set_xticks([0.25, 0.3, 0.4, 0.5, 0.7, 1.0], ['0.25', '0.3', '0.4', '0.5', '0.7', '1'], minor=False)
A.set_xticks([], minor=True)
A.set_xlabel('largest remaining event / original largest event')
A.set_ylabel('refitted ceiling $s_c$ / original $s_c$')
for key in [1, 5, 'pct']:
    A.plot([], [], mk[key][0], color='0.3', label=mk[key][1])
A.plot([], [], 's', color='crimson', mec='k', label='MD data, 8 largest cells')
A.plot([], [], 's', color='seagreen', label='synthetic: true ceiling law, same n')
A.plot([], [], 's', color='royalblue', label='synthetic: true TPL (no ceiling), same n')
A.legend(fontsize=8, loc='lower right')
A.set_title('Is the ceiling just the largest event?  Data behave like a true ceiling')
save(fig, 'fig18_ceiling_pinning.png')


# ---- 19: the memory relaxes with strain ----
m2 = json.load(open(SCRATCH + '/memory_2d.json'))
ml = json.load(open(SCRATCH + '/memory_late.json'))
fig, (L, Rr) = plt.subplots(1, 2, figsize=(12.5, 4.8))
wins = list(m2['by_window'].keys()); gc = np.array([np.mean([float(x) for x in w.split('-')]) for w in wins])
b = np.array([m2['by_window'][w]['beta'] for w in wins]); se = np.array([m2['by_window'][w]['se'] for w in wins])
L.errorbar(gc, b, se, fmt='o-', color='k', lw=2, capsize=3, ms=7, label='all cells (fixed effect per cell × window)')
cols = {1: 'royalblue', 2: 'darkorange', 3: 'crimson'}
names = {1: 'u ∈ [0.016, 0.022)', 2: 'u ∈ [0.022, 0.026)', 3: 'u ∈ [0.026, 0.031)'}
for bnd in [1, 2, 3]:
    xs, ys, es = [], [], []
    for g in range(len(wins)):
        t = m2['table'].get('u%d_g%d' % (bnd, g))
        if t and t['ngroups'] >= 4:
            xs.append(gc[g] + 0.01 * (bnd - 2)); ys.append(t['beta']); es.append(t['se'])
    L.errorbar(xs, ys, es, fmt='s--', color=cols[bnd], capsize=2, ms=5, lw=1.2, label=names[bnd])
L.axhline(0, color='0.5', lw=1)
L.axhline(sm['beta'], color='crimson', ls=':', lw=1.2)
L.text(0.47, sm['beta'] + 0.006, 'global β used in Fig. 17', color='crimson', fontsize=8, ha='right')
L.set_xlabel('strain window'); L.set_ylabel('memory exponent β of the mean event size')
L.set_title('The ceiling memory relaxes with strain, also at fixed u')
L.legend(fontsize=8, loc='lower right')
# right: where the late-strain memory sits (early vs late channel table)
chan = ['hazard p_drop', 'mean size <s>', 'plastic rate q']
lab = {'hazard p_drop': 'hazard', 'mean size <s>': 'mean size', 'plastic rate q': 'plastic rate q'}
x = np.arange(len(chan)); wdt = 0.36
for j, (win, col) in enumerate([('early', '0.35'), ('late', 'crimson')]):
    vals = [ml[win][c]['slow_fast'] for c in chan]
    Rr.bar(x + (j - 0.5) * wdt, vals, wdt, color=col, label='%s strain (γ %s 0.3)' % (win, '<' if win == 'early' else '≥'))
    for xi, v, c in zip(x + (j - 0.5) * wdt, vals, chan):
        Rr.text(xi, v + 0.02, 'β=%+.3f' % ml[win][c]['beta'], ha='center', fontsize=7.5)
du_e, du_l = ml['early']['total <du>/step'], ml['late']['total <du>/step']
Rr.text(0.02, 0.97, 'energy step ⟨du⟩ at fixed cell, slow − fast:\n  early  %+.2f × 10⁻⁶ (typical |⟨du⟩| %.2f)\n  late   %+.2f × 10⁻⁶ (typical |⟨du⟩| %.2f)\nall of it in the event channel, both windows' %
        (1e6 * du_e['slow_fast'], 1e6 * du_e['typical'], 1e6 * du_l['slow_fast'], 1e6 * du_l['typical']),
        transform=Rr.transAxes, va='top', fontsize=8.5, bbox=dict(fc='white', ec='0.7'))
Rr.axhline(1, color='k', lw=1)
Rr.set_xticks(x, [lab[c] for c in chan]); Rr.set_ylim(0.8, 2.3)
Rr.set_ylabel('slow-3 / fast-3 at the same cell (median)')
Rr.set_title('Early vs late: the size memory fades, the hazard stays blind')
Rr.legend(fontsize=8.5, loc='upper right')
save(fig, 'fig19_memory_relaxes.png')

# ---- 20: events cluster in strain; the hazard is not Markov at the step scale ----
cl = json.load(open(SCRATCH + '/cluster_test.json'))
sc2 = json.load(open(SCRATCH + '/scatter2.json'))
me = json.load(open(SCRATCH + '/merged_events.json'))
fig, (L, Rr) = plt.subplots(1, 2, figsize=(12.5, 4.8))
lag = np.arange(1, 9); h = np.array(cl['lag_hazard'])
L.plot(lag, h / cl['baseline'], 'o-', color='crimson', lw=2, ms=7)
L.axhline(1, color='k', lw=1, ls='--', label='Markov in (u,τ): no dependence on history')
L.set_yscale('log'); L.set_xlabel('strain steps since the last event  (1 step = 10⁻⁵ strain)')
L.set_ylabel('hazard / hazard after > 8 quiet steps  (same cells)')
L.set_title('Aftershocks: the hazard is ×%.0f on the step after an event' % (h[0] / cl['baseline']))
L.text(0.97, 0.75, '%d cells\nevents in runs of ≥ 2 consecutive steps:\n  data 14%%, Markov expectation 0.4%%\nnext event 1.2× larger, not smaller' % cl['ncells'],
       transform=L.transAxes, ha='right', va='top', fontsize=8.5, bbox=dict(fc='white', ec='0.7'))
L.legend(fontsize=8.5, loc='upper right')
wins2 = list(sc2.keys()); xw = np.arange(len(wins2))
fano_d = [sc2[w]['SDN_data'] ** 2 / sc2[w]['N_data'] for w in wins2]; fano_s = [sc2[w]['SDN_sim'] ** 2 / sc2[w]['N_sim'] for w in wins2]
rel_d = [sc2[w]['SDsum_data'] / sc2[w]['sum_data'] for w in wins2]; rel_s = [sc2[w]['SDsum_sim'] / sc2[w]['sum_sim'] for w in wins2]
Rr.bar(xw - 0.2, fano_d, 0.38, color='k', label='event count, Fano factor — data')
Rr.bar(xw + 0.2, fano_s, 0.38, color='0.6', label='event count, Fano factor — model')
ax2 = Rr.twinx()
ax2.plot(xw, rel_d, 'o-', color='crimson', lw=2, label='total release per window, CV — data')
ax2.plot(xw, rel_s, 's--', color='crimson', lw=1.4, mfc='white', label='total release per window, CV — model')
ax2.set_ylim(0, 0.45); ax2.set_ylabel('CV of stress released per window', color='crimson')
Rr.set_ylim(0, 2.0); Rr.set_xticks(xw, wins2); Rr.set_xlabel('strain window'); Rr.set_ylabel('Fano factor of event count (within preparation)')
Rr.set_title('More bursts than the model, yet a steadier stress budget')
h1, l1 = Rr.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
Rr.legend(h1 + h2, l1 + l2, fontsize=7.5, loc='upper left')
save(fig, 'fig20_clustering.png')


# ---- 21: the excess variance is a renewal structure, not the size law, the hazard's memory, or the grid ----
sa = json.load(open(SCRATCH + '/sim_aftershock.json')); sfn = json.load(open(SCRATCH + '/sim_fine.json'))
sq = json.load(open(SCRATCH + '/sequence_test.json'))
fig, (L, Rr) = plt.subplots(1, 2, figsize=(12.5, 4.8))
srn = json.load(open(SCRATCH + '/sim_renewal.json')); stc = json.load(open(SCRATCH + '/sim_twoclass.json'))
tc = dict(stc['twoclass_reload']); tc['fano'] = [tc['fano_all']] * 5
labels = ['data', 'Markov model', '+ aftershock hazard', '+ renewal hazard', '+ two-class reload', 'empirical, 22×22', 'empirical, 42×42']
src = [sa['data'], sa['markov'], sa['aftershock'], srn['renewal_rel'], tc, sfn['grid20'], sfn['grid40']]
cols = ['k', '0.55', 'purple', 'royalblue', 'teal', 'seagreen', 'darkorange']
x = np.arange(len(labels))
fano = [np.mean(r['fano'][1:]) for r in src]; cvs = [np.mean(r['cv_sum'][1:]) if len(r['cv_sum']) == 5 else np.mean(r['cv_sum']) for r in src]
sdt = [r['sd_tau']['0.1'] if '0.1' in r['sd_tau'] else r['sd_tau'][0.1] for r in src]
L.bar(x - 0.28, np.array(fano) / fano[0], 0.26, color=cols, ec='k', lw=0.5)
L.bar(x, np.array(cvs) / cvs[0], 0.26, color=cols, ec='k', lw=0.5, hatch='//')
L.bar(x + 0.28, np.array(sdt) / sdt[0], 0.26, color=cols, ec='k', lw=0.5, hatch='..')
L.axhline(1, color='k', lw=1)
L.set_xticks(x, labels, rotation=20, ha='right', fontsize=8.5)
L.set_ylabel('relative to data')
from matplotlib.patches import Patch
L.legend(handles=[Patch(fc='w', ec='k', label='Fano factor of event count (γ 0.1–0.5)'),
                  Patch(fc='w', ec='k', hatch='//', label='CV of stress released per window'),
                  Patch(fc='w', ec='k', hatch='..', label='within-preparation SD of τ at γ = 0.1')], fontsize=8, loc='upper left')
L.set_ylim(0, 1.6)
L.set_title('Variance gap: survives every hazard repair and the grid')
sqc = json.load(open(SCRATCH + '/sequence_compare.json'))
for name, col, lab in [('data', 'crimson', 'MD data'), ('markov sim', '0.4', 'Markov simulation (same statistic)')]:
    gp = sqc[name]['gap_profile']; e = np.array(gp['size_edges']); xc = np.sqrt(e[1:] * e[:-1]); y = np.array(gp['rel_gap']); n = np.array(gp['n'])
    ok = n >= 200
    Rr.plot(xc[ok], y[ok], 'o-' if name == 'data' else 's--', color=col, lw=2, ms=6,
            label='%s: r = %+.2f all events, %+.2f for s > 10⁻³' % (lab, sqc[name]['cell of k+1'][0], sqc[name]['cell of k+1, s > 1e-3'][0]))
Rr.axvline(1e-3, color='0.7', lw=1, ls=':')
Rr.text(1.1e-3, 0.03, 'aftershock regime →', fontsize=8, color='0.4')
Rr.set_xscale('log'); Rr.set_yscale('log')
Rr.set_xlabel('size of event k'); Rr.set_ylabel('median strain to the next event  /  cell median')
Rr.set_title('Gap after an event vs its size: mostly the aftershock regime')
Rr.legend(fontsize=8, loc='lower right')
save(fig, 'fig21_renewal.png')
