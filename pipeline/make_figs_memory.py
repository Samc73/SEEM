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
arms = [a for a in ['model4', 'mem_sc', 'mem_sc_u', 'mem_sc_k', 'mem_empir'] if a in sm]
style = {'model4': (':', 'k', 'README model (preparation-blind)'),
         'mem_sc': ('--', 'crimson', 'ceiling × (rate)$^{\\beta}$, one global $\\beta$'),
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
