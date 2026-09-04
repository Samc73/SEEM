import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Re-emit all result figures as standalone single-panel (or paired) files."""
import numpy as np, sys, json, ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist
from symreg import evaluate

SCRATCH = _os.path.join(_HERE, 'out')
FIG = _os.path.join(_HERE, '..', 'figures') + '/'
d = np.load(SCRATCH + '/model_stats.npz')
ev = np.load(SCRATCH + '/model_events.npz')
s4 = np.load(SCRATCH + '/scale4.npz')
sf = np.load(SCRATCH + '/scale_fields.npz')
qf = np.load(SCRATCH + '/qfields.npz')
sr = json.load(open(SCRATCH + '/sr_dist_results.json'))
deep = json.load(open(SCRATCH + '/sr_deep9.json'))
synth = json.load(open(SCRATCH + '/synth_check.json'))
rep = json.load(open(SCRATCH + '/scale_report.json'))
rec = json.load(open(SCRATCH + '/recon4.json'))
NB = int(d['NB'])
S, EV = ev['S'].astype(float), ev['EV']
KG, MG, EG = float(s4['K_G']), float(s4['M_G']), float(s4['E_G'])
sc4 = s4['sc4']
ue, te, uc, tc = d['ue'], d['te'], sf['uc'], sf['tc']
q_emp, q_mod4 = qf['q_emp'], s4['q_mod4']
msk = qf['msk']
floor = float(qf['floor'])
rates, cr, g_ds = d['rates'], d['cr'], d['g_ds']
sims = {arm: np.load(SCRATCH + f'/sim_{arm}.npz') for arm in ('model4', 'tpl', 'empir')}
sm = sims['model4']
g_sim = np.arange(sm['rec_t'].shape[1]) * int(sm['record_every']) * 1e-5
cmap9 = plt.cm.viridis(np.linspace(0, 0.92, len(rates)))


def knee_curve(front, x):
    p = front[-1]
    e = ast.literal_eval(p['expr'])
    with np.errstate(all='ignore'):
        return p['a'] * np.asarray(evaluate(e, x, np.array(p['consts'])), float) + p['b']


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG + name, dpi=300)
    plt.close(fig)
    print('saved', name)


# ---- 01: one cell, all three fitted laws ----
fig, A = plt.subplots(figsize=(7.2, 5.2))
iu, it = 18, 11
s = S[EV == iu * NB + it]
s = s[s >= 1e-6]
lo, hi = s.min(), s.max()
be = np.geomspace(lo, hi * 1.001, 34)
n, _ = np.histogram(s, bins=be)
xc = np.sqrt(be[:-1] * be[1:])
keep = n > 0
rho = n / np.diff(be) / len(s)
A.loglog(xc[keep], rho[keep], 'o', ms=4, color='0.25', label='measured')
g = np.geomspace(lo, hi * 1.02, 800)
c = sc4[iu, it]
ln4 = np.where(g < c, MG * np.log(np.maximum(c - g, 1e-300)) - KG * np.log(g + EG), -np.inf)
w4 = np.exp(ln4 - np.nanmax(ln4))
Z4 = np.trapezoid(np.where(np.isfinite(w4), w4, 0), g)
A.plot(g[g < c], (w4 / Z4)[g < c], '-', color='crimson', lw=2,
       label=r'$(s_c\!-\!s)^{m}/(s\!+\!\varepsilon)^{k}$ (discovered)')
rt = dist.fit('tpl', s, 1e-6)
lnt = -rt['theta'][0] * np.log(g) - g / np.exp(rt['theta'][1])
wt = np.exp(lnt - lnt.max())
A.plot(g, wt / np.trapezoid(wt, g), '--', color='royalblue', lw=1.8,
       label=r'$s^{-\kappa}e^{-s/s^*}$ (truncated power law)')
rb = dist.fit('ldw_eps', s, 1e-6)      # Budrikis et al. 2017 eq. 1, with the same eps rounding
lnb = dist._lnf('ldw_eps', g, rb['theta'], hi)
wb = np.exp(lnb - lnb.max())
A.plot(g, wb / np.trapezoid(wb, g), '-.', color='darkorange', lw=1.8,
       label=r'$s^{-\tau}e^{\,C\sqrt{u}-\frac{B}{4}u^{\delta}}$, $u=s/S_{max}$ (Budrikis et al. 2017)')
A.axvline(c, color='crimson', ls=':', lw=1)
A.annotate('ceiling $s_c$', (c, 2e-2), rotation=90, fontsize=9, color='crimson',
           xytext=(c * 0.55, 4e-3))
A.set_xlabel('event size $s$ (stress drop)')
A.set_ylabel('probability density')
A.set_ylim(1e-4, 3e4)
A.legend(fontsize=8, loc='lower left')
A.set_title('One cell')
# inset: where this cell sits in the data's probability density over (u, tau)
ins = A.inset_axes([0.695, 0.655, 0.295, 0.31])
occ = d['n_all'].reshape(NB, NB)[1:21, 1:21].astype(float)
dens = occ / occ.sum()
M = np.where(dens > 0, np.log10(dens), np.nan)
# clip the color scale to the bulk: sparse edge cells reach 1e-7 and would
# otherwise flatten the well-populated interior into one color
ins.pcolormesh(te[1:22], ue[1:22], M, cmap='viridis',
               vmin=np.nanpercentile(M, 30), vmax=np.nanmax(M))
ins.add_patch(plt.Rectangle((te[it], ue[iu]), te[it + 1] - te[it],
                            ue[iu + 1] - ue[iu], fill=False, ec='crimson', lw=1.8))
ins.set_xlabel(r'$\tau$', fontsize=7, labelpad=1)
ins.set_ylabel('$u$', fontsize=7, labelpad=1)
ins.tick_params(labelsize=6, length=2)
ins.set_title(r'data density in $(u,\tau)$; this cell', fontsize=7.5, color='crimson')
save(fig, 'fig01_size_law.png')

# ---- 02: collapse ----
fig, B = plt.subplots(figsize=(7.2, 5.2))
cmap = plt.cm.viridis(np.linspace(0, 0.95, 12))
cells = [(r['iu'], r['it']) for r in sr if r['tag'] == 'main']
for (ciu, cit), col in zip(cells, cmap):
    sv = S[EV == ciu * NB + cit]
    sv = sv[sv >= 1e-6]
    cc = sc4[ciu, cit]
    if not np.isfinite(cc):
        continue
    be = np.geomspace(sv.min(), min(sv.max(), cc * 0.999), 22)
    n, _ = np.histogram(sv, bins=be)
    keep = n >= 10
    xb = np.sqrt(be[:-1] * be[1:])[keep]
    rho = (n / np.diff(be) / len(sv))[keep]
    xi = (xb + EG) / (cc + EG)
    gg = dist._grid(1e-7, cc, True, 3000)
    lnm = MG * np.log(cc - gg) - KG * np.log(gg + EG)
    mmax = lnm.max()
    ln_scale = (MG - KG) * np.log(cc + EG) - mmax - np.log(
        np.trapezoid(np.exp(lnm - mmax), gg))
    B.loglog(xi, rho * np.exp(-ln_scale), 'o', ms=3, color=col, alpha=0.7)
xi = np.geomspace(2e-5, 0.999, 400)
B.plot(xi, (1 - xi) ** MG * xi ** -KG, 'k-', lw=2,
       label=r'$(1-\xi)^{m}\,\xi^{-k}$,  $k=0.88,\ m=2.3$')
B.set_xlabel(r'$\xi = (s+\varepsilon)/(s_c+\varepsilon)$')
B.set_ylabel('rescaled density')
B.legend(fontsize=9)
B.set_title('All 12 test cells collapse onto one curve')
save(fig, 'fig02_collapse.png')

# ---- 03: Pareto fronts ----
fig, C = plt.subplots(figsize=(7.2, 5.2))
for i, r in enumerate([r for r in sr if r['tag'] == 'main']):
    C.plot([p['complexity'] for p in r['front']], [p['rmse'] for p in r['front']],
           '-', color='0.7', lw=1, marker='.', ms=4, zorder=1,
           label=('one front per cell; heights set by each cell\'s\n'
                  'own noise floor (not comparable across cells)') if i == 0 else None)
C.plot([p['complexity'] for p in deep], [p['rmse'] for p in deep], '-',
       color='darkorange', lw=2, marker='o', ms=5, zorder=3,
       label='audit run: largest cell pushed to complexity 9\n(4,505,024 trees) — no new form appears')
C.annotate('bounded-support form\ndiscovered here', (8, 0.118), fontsize=9,
           xytext=(5.1, 0.55), arrowprops=dict(arrowstyle='->', lw=1))
C.set_yscale('log')
C.set_xlabel('expression complexity')
C.set_ylabel('weighted RMS misfit of $\\ln\\rho(s)$')
C.legend(fontsize=8.5)
C.set_title('Exhaustive symbolic-regression fronts (12 cells)')
save(fig, 'fig03_pareto.png')

# ---- 04: falsifiability ----
fig, D = plt.subplots(figsize=(7.2, 5.2))
y_tpl = [r['daic_tpl_minus_inv'] for r in synth['mle'] if r['gen'] == 'tpl']
y_inv = [r['daic_tpl_minus_inv'] for r in synth['mle'] if r['gen'] == 'invpow']
y_real = [2 * g_['n'] * g_['dnll'] - 2 for g_ in rep['glob']]
for i, (ys, col) in enumerate(((y_tpl, 'royalblue'), (y_inv, 'crimson'),
                               (y_real, '0.2'))):
    D.scatter(np.full(len(ys), i) + np.random.default_rng(1).uniform(-.08, .08, len(ys)),
              ys, s=28, color=col, zorder=3)
D.axhline(0, color='k', lw=1)
D.set_yscale('symlog', linthresh=20)
D.set_xticks([0, 1, 2], ['synthetic:\nexponential tail', 'synthetic:\nhard ceiling',
                         'measured\ncatalogs'])
D.set_ylabel(r'AIC(TPL) $-$ AIC(ceiling law)')
D.text(0.03, 0.95, 'above the line: ceiling law wins', transform=D.transAxes, fontsize=9)
D.set_title('The verdict cannot be a pipeline artifact')
save(fig, 'fig04_falsifiability.png')

# ---- 05: ceiling map ----
fig, A = plt.subplots(figsize=(7.2, 5.2))
M = np.where(np.isfinite(sc4[1:21, 1:21]), np.log10(sc4[1:21, 1:21]), np.nan)
pc = A.pcolormesh(te[1:22], ue[1:22], M, cmap='magma')
plt.colorbar(pc, ax=A, label=r'$\log_{10} s_c$')
rng = np.nanmax(10 ** M) / np.nanmin(10 ** M)
A.set_title('Ceiling field $s_c(u,\\tau)$  (spans %.0f$\\times$)' % rng)
A.set_xlabel('stress $\\tau$')
A.set_ylabel('$u$')
save(fig, 'fig05_ceiling_map.png')

# ---- 06: ceiling factors (pair) ----
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
for P, key, xv, xl, lab in ((ax[0], 'scu', uc, '$u$', '$u$-factor of $\\ln s_c$'),
                            (ax[1], 'sct', tc, 'stress $\\tau$', '$\\tau$-factor of $\\ln s_c$')):
    fac = np.array(rec['sc4']['a' if key == 'scu' else 'b'])
    w = np.array(rec['sc4']['wa' if key == 'scu' else 'wb'])
    k = w > 0
    P.plot(xv[k], fac[k], 'o', color='0.2', label='measured factor')
    xd = np.linspace(xv[k].min(), xv[k].max(), 300)
    P.plot(xd, knee_curve(rec[key], xd), '-', color='crimson', lw=2,
           label='SR knee: %s' % rec[key][-1]['string'][:46])
    P.set_xlabel(xl)
    P.set_ylabel('additive factor of $\\ln s_c$')
    P.legend(fontsize=8)
    P.set_title(lab)
save(fig, 'fig06_ceiling_factors.png')

# ---- 07: ceiling memory ----
fig, D = plt.subplots(figsize=(6.4, 5.4))
mem = rep['mem']
xs = [z['sc_fast'] for z in mem]
ys = [z['sc_slow'] for z in mem]
D.loglog(xs, ys, 'o', ms=5, color='0.2')
lim = [min(min(xs), min(ys)) * 0.8, max(max(xs), max(ys)) * 1.2]
D.plot(lim, lim, 'k-', lw=1, label='equal')
D.plot(lim, [1.71 * v for v in lim], 'r--', lw=1.5, label='1.71$\\times$ (median)')
D.set_xlabel('$s_c$ fit on fast-cooled runs')
D.set_ylabel('$s_c$ fit on slow-cooled runs')
D.legend(fontsize=9)
D.set_title('Same $(u,\\tau)$, different history: the ceiling remembers')
save(fig, 'fig07_ceiling_memory.png')

# ---- 08: q maps (pair) ----
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
for P, qq, tt in ((ax[0], q_emp, '$q$ measured directly'),
                  (ax[1], q_mod4, '$q$ rebuilt from the fitted law')):
    M = np.where(msk[1:21, 1:21] & (qq[1:21, 1:21] > 0),
                 np.log10(qq[1:21, 1:21]), np.nan)
    pc = P.pcolormesh(te[1:22], ue[1:22], M, cmap='viridis', vmin=-1.6, vmax=0.6)
    plt.colorbar(pc, ax=P, label='$\\log_{10} q$')
    P.set_xlabel('stress $\\tau$')
    P.set_ylabel('$u$')
    P.set_title(tt)
save(fig, 'fig08_q_maps.png')

# ---- 09: q agreement ----
fig, C = plt.subplots(figsize=(6.6, 5.4))
x = q_emp[1:21, 1:21][msk[1:21, 1:21]]
y = q_mod4[1:21, 1:21][msk[1:21, 1:21]]
k = (x > 0) & (y > 0) & np.isfinite(y)
C.loglog(x[k], y[k], 'o', ms=4, color='0.3')
lim = [x[k].min() * 0.7, x[k].max() * 1.4]
C.plot(lim, lim, 'k-', lw=1)
C.fill_between(lim, [v * (1 - floor) for v in lim], [v * (1 + floor) for v in lim],
               color='orange', alpha=0.25, label='split-half noise floor (17.5%)')
rel = y[k] / x[k] - 1
C.text(0.04, 0.9, 'median %+.1f%%, RMS %.0f%%' %
       (100 * np.median(rel), 100 * np.sqrt(np.mean(rel ** 2))),
       transform=C.transAxes, fontsize=9)
C.set_xlabel('$q$ measured')
C.set_ylabel('$q$ from fitted law')
C.legend(fontsize=8.5, loc='lower right')
C.set_title('Voxel-by-voxel agreement')
save(fig, 'fig09_q_agreement.png')

# ---- 10: moments ----
fig, D = plt.subplots(figsize=(6.6, 5.4))
ok = s4['ok']
se, sm4, st = qf['sbar_emp'][ok], s4['sbar4'][ok], qf['sbar_tpl'][ok]
D.loglog(se, sm4, 'o', ms=4, color='crimson', label='ceiling law')
D.loglog(se, st, '^', ms=4, mfc='none', color='royalblue', label='TPL')
lim = [se.min() * 0.7, se.max() * 1.4]
D.plot(lim, lim, 'k-', lw=1)
D.set_xlabel(r'measured mean drop $\langle s\rangle$')
D.set_ylabel(r'model $\langle s\rangle$')
D.legend(fontsize=9)
D.set_title('First moment, both fitted laws')
save(fig, 'fig10_moments.png')

# ---- 11: q factors (pair) ----
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
for P, key, xv, xl, tt in ((ax[0], 'Lam', uc, '$u$', r'$\Lambda(u)$ from both $q$ fields'),
                           (ax[1], 'f', tc, 'stress $\\tau$', r'$f(\tau)$ from both $q$ fields')):
    for tag, col, mk2 in (('emp', '0.15', 'o'), ('mod4', 'crimson', 's')):
        fr = rec['q_' + tag]
        fac = np.array(fr['a' if key == 'Lam' else 'b'])
        w = np.array(fr['wa' if key == 'Lam' else 'wb'])
        kk = w > 0
        yv = np.exp(fac[kk] - fac[kk].max())
        P.semilogy(xv[kk], yv, mk2, ms=5, color=col,
                   label=('measured $q$' if tag == 'emp' else 'model $q$'))
        xd = np.linspace(xv[kk].min(), xv[kk].max(), 300)
        yc = knee_curve(rec[key + '_' + tag], xd)
        P.semilogy(xd, np.clip(yc, 1e-3, None), '-', color=col, lw=1.4, alpha=0.8)
    P.set_xlabel(xl)
    P.set_ylabel('factor (normalized)')
    P.legend(fontsize=8.5)
    P.set_title(tt)
save(fig, 'fig11_q_factors.png')

# ---- 12: sim stress-strain ----
fig, A = plt.subplots(figsize=(7.4, 5.2))
for c, col, ct in zip(rates, cmap9, d['curves_tau']):
    A.plot(g_ds, ct, color=col, lw=1.5)
    m = cr == c
    A.plot(g_sim[g_sim <= 0.5], sm['rec_t'][m].mean(0)[g_sim <= 0.5], '--',
           color=col, lw=1.2)
A.plot([], [], 'k-', lw=1.5, label='MD data')
A.plot([], [], 'k--', lw=1.2, label='simulated from fitted model')
A.set_xlabel('strain $\\gamma$')
A.set_ylabel('ensemble mean stress $\\tau$')
A.legend(fontsize=9)
A.set_title('Stress-strain, all 9 preparations (color = cooling rate)')
save(fig, 'fig12_sim_stress.png')

# ---- 13: sim energy ----
fig, B = plt.subplots(figsize=(7.4, 5.2))
for c, col, cu in zip(rates, cmap9, d['curves_u']):
    B.plot(g_ds, cu, color=col, lw=1.5)
    m = cr == c
    B.plot(g_sim[g_sim <= 0.5], sm['rec_u'][m].mean(0)[g_sim <= 0.5], '--',
           color=col, lw=1.2)
B.plot([], [], 'k-', lw=1.5, label='MD data')
B.plot([], [], 'k--', lw=1.2, label='simulated from fitted model')
B.set_xlabel('strain $\\gamma$')
B.set_ylabel('ensemble mean $u$')
B.legend(fontsize=9, loc='lower right')
B.set_title('Energy coordinate $u(\\gamma)$: the model is too convergent')
save(fig, 'fig13_sim_energy.png')

# ---- 14: sim peaks ----
fig, C = plt.subplots(figsize=(7.4, 5.2))
x = np.arange(len(rates))
pk_d = np.array([[d['tau_peak'][cr == c].mean(), d['tau_peak'][cr == c].std()]
                 for c in rates])
C.errorbar(x - 0.12, pk_d[:, 0], pk_d[:, 1], fmt='o', color='k', capsize=3,
           label='MD data')
mk = dict(model4=('s', 'crimson', 'ceiling law'), tpl=('^', 'royalblue', 'TPL'),
          empir=('x', '0.5', 'empirical resample'))
for j, arm in enumerate(('model4', 'tpl', 'empir')):
    pk = sims[arm]['peak']
    pv = np.array([[pk[cr == c].mean(), pk[cr == c].std()] for c in rates])
    C.errorbar(x + 0.08 * j, pv[:, 0], pv[:, 1], fmt=mk[arm][0],
               color=mk[arm][1], capsize=2, ms=5, label='sim: ' + mk[arm][2])
C.set_xticks(x, ['%.2g' % c for c in rates], rotation=45, fontsize=8)
C.set_xlabel('cooling rate (K/s)')
C.set_ylabel('yield peak stress')
C.legend(fontsize=8.5)
C.set_title('The 850 yield peaks, predicted vs measured')
save(fig, 'fig14_sim_peaks.png')

# ---- 15: sim extrapolation ----
fig, D = plt.subplots(figsize=(7.4, 5.2))
for c, col in zip(rates, cmap9):
    m = cr == c
    D.plot(g_sim, sm['rec_u'][m].mean(0), color=col, lw=1.4)
D.axvline(0.5, color='k', ls=':', lw=1.4)
D.text(0.512, 0.012, 'MD runs end', rotation=90, fontsize=9)
u_at = lambda gv: np.array([sm['rec_u'][cr == c].mean(0)[np.argmin(np.abs(g_sim - gv))]
                            for c in rates])
s05, s20 = u_at(0.5), u_at(2.0)
D.text(0.98, 0.05, 'spread across preparations:\n%.1f%% at $\\gamma$=0.5  '
       '$\\rightarrow$  %.1f%% at $\\gamma$=2' %
       (100 * (s05.max() / s05.min() - 1), 100 * (s20.max() / s20.min() - 1)),
       transform=D.transAxes, ha='right', fontsize=9,
       bbox=dict(fc='white', ec='0.7'))
D.set_xlabel('strain $\\gamma$')
D.set_ylabel('simulated mean $u$')
D.set_title('Model run 4x past the data: do preparations converge?')
save(fig, 'fig15_sim_extrapolation.png')
print('ALL FIGURES DONE')
