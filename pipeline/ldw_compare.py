import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""How does the Budrikis et al. (2017) avalanche-size law (their eq. 1, the
Le Doussal-Wiese first-order correction to mean field) do on these events?
Same likelihood contest as mle_compare.py: the eight largest cells, s >= 1e-4,
AIC against the ceiling law and the TPL; plus a full-range fit with the
small-size rounding added, and a figure of the largest cell."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SCRATCH = _os.path.join(_HERE, 'out'); FIG = _os.path.join(_HERE, '..', 'figures') + '/'
ev = np.load(SCRATCH + '/model_events.npz'); d = np.load(SCRATCH + '/model_stats.npz'); mc = json.load(open(SCRATCH + '/mle_compare.json'))
S, EV = ev['S'].astype(float), ev['EV']; NB = int(d['NB']); XMIN = 1e-4
out = []
print('%-8s %6s | %9s %9s %9s | %6s %8s %8s | %8s' % ('cell', 'n', 'AIC ldw', 'AIC tpl', 'AIC ceil', 'tau', 'Smax', 'Smax/max', 'KS ldw'))
def ks(name, th, s, xmin, smax):
    s = np.sort(s[s >= xmin]); hi = dist._upper(name, th, smax)
    grid = dist._grid(xmin, hi if hi is not None else max(100 * smax, 1.0), hi is not None, 4000)
    with np.errstate(all='ignore'): lf = dist._lnf(name, grid, th, smax)
    w = np.exp(lf - np.nanmax(lf)); w[~np.isfinite(w)] = 0
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(grid)))); cdf /= cdf[-1]
    return float(np.abs(np.interp(s, grid, cdf) - (np.arange(1, len(s) + 1) - 0.5) / len(s)).max())
for c in mc['cells']:
    if c['n'] < 3000: continue
    s = S[EV == c['iu'] * NB + c['it']]; s = s[s >= XMIN]
    r = dist.fit('ldw', s, XMIN); th = r['theta']
    a_ldw, a_tpl, a_ceil = r['aic'], c['fits']['tpl']['aic'], c['fits']['invpow4']['aic']
    row = dict(iu=c['iu'], it=c['it'], n=len(s), aic_ldw=a_ldw, aic_tpl=a_tpl, aic_ceiling=a_ceil, tau=float(th[0]), Smax=float(np.exp(th[1])),
               Smax_over_max=float(np.exp(th[1]) / s.max()), ks_ldw=ks('ldw', th, s, XMIN, s.max()), ks_ceiling=c['fits']['invpow4']['ks'], ks_tpl=c['fits']['tpl']['ks'])
    print('(%2d,%2d) %6d | %9.1f %9.1f %9.1f | %6.3f %8.4f %8.3f | %8.4f   (KS ceiling %.4f, tpl %.4f)' %
          (c['iu'], c['it'], len(s), a_ldw - a_ceil, a_tpl - a_ceil, 0.0, th[0], np.exp(th[1]), np.exp(th[1]) / s.max(), row['ks_ldw'], row['ks_ceiling'], row['ks_tpl']))
    out.append(row)
print('(AIC columns are relative to the ceiling law in the same cell; positive = worse)')
# full range, two largest cells: ldw with eps vs the ceiling law
print('\nfull range (s >= 1e-6), rounding eps added to the LDW form:')
full = []
for c in mc['cells'][:2]:
    s = S[EV == c['iu'] * NB + c['it']]
    r1 = dist.fit('ldw_eps', s, 1e-6); r2 = dist.fit('invpow4', s, 1e-6)
    print('(%2d,%2d) n=%d  AIC ldw_eps - ceiling = %+.1f   tau=%.3f Smax=%.4f eps=%.1e   | ceiling k=%.3f m=%.2f' %
          (c['iu'], c['it'], len(s), r1['aic'] - r2['aic'], r1['theta'][0], np.exp(r1['theta'][1]), np.exp(r1['theta'][2]), r2['theta'][0], r2['theta'][1]))
    full.append(dict(iu=c['iu'], it=c['it'], daic=float(r1['aic'] - r2['aic']), tau=float(r1['theta'][0]), Smax=float(np.exp(r1['theta'][1])), eps=float(np.exp(r1['theta'][2]))))
json.dump(dict(window=out, fullrange=full), open(SCRATCH + '/ldw_compare.json', 'w'), indent=1)

# ---- figure: the largest cell, s >= 1e-4, all three conditional densities ----
c = mc['cells'][0]; s = S[EV == c['iu'] * NB + c['it']]; s = s[s >= XMIN]; smax = s.max()
be = np.geomspace(XMIN, smax * 1.001, 28); n_, _ = np.histogram(s, bins=be); xc = np.sqrt(be[1:] * be[:-1]); rho = n_ / np.diff(be) / len(s)
fig, A = plt.subplots(figsize=(7.2, 5.2))
A.plot(xc[n_ > 0], rho[n_ > 0], 'o', color='k', ms=4, label='measured (%d events)' % len(s))
def curve(name, th, color, ls, lab):
    hi = dist._upper(name, th, smax); g = dist._grid(XMIN, hi if hi is not None else 3 * smax, hi is not None, 3000)
    with np.errstate(all='ignore'): lf = dist._lnf(name, g, th, smax)
    w = np.exp(lf - np.nanmax(lf)); w[~np.isfinite(w)] = 0; Z = np.trapezoid(w, g)
    A.plot(g, w / Z, ls, color=color, lw=2, label=lab)
curve('invpow4', c['fits']['invpow4']['theta'], 'crimson', '-', 'ceiling law (discovered)')
curve('tpl', c['fits']['tpl']['theta'], 'royalblue', '--', 'truncated power law')
th = out[0]['tau'], np.log(out[0]['Smax'])
curve('ldw', np.array(th), 'darkorange', '-.', 'Budrikis et al. 2017, eq. 1  (τ = %.2f, S_max = %.2f)' % (th[0], np.exp(th[1])))
A.axvline(dist._upper('invpow4', c['fits']['invpow4']['theta'], smax), color='crimson', ls=':', lw=1)
A.set_xscale('log'); A.set_yscale('log'); A.set_xlim(XMIN, 1.5); A.set_ylim(rho[n_ > 0].min() / 30, rho.max() * 3)
A.set_xlabel('event size s (stress drop)'); A.set_ylabel('probability density  (s ≥ 10⁻⁴)')
A.set_title('One cell: the discovered law against the two literature forms')
A.text(0.03, 0.05, 'ΔAIC vs ceiling law:  TPL %+.0f,  Budrikis eq. 1 %+.0f' % (out[0]['aic_tpl'] - out[0]['aic_ceiling'], out[0]['aic_ldw'] - out[0]['aic_ceiling']),
       transform=A.transAxes, fontsize=9, bbox=dict(fc='white', ec='0.7'))
A.legend(fontsize=8.5, loc='upper right')
fig.tight_layout(); fig.savefig(FIG + 'fig23_ldw.png', dpi=300); print('saved fig23_ldw.png')
