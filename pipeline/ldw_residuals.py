import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Where the likelihood margin comes from.  On a log-log density plot the
ceiling law and the Budrikis et al. form look alike; the difference is in the
residuals.  Left: measured density / model density per log bin with Poisson
error bars, largest cell, s >= 1e-4 (the standard contest window).  Right:
the running log-likelihood advantage of the ceiling law over each rival,
summed over events in order of size, so the reader sees which sizes decide."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SCRATCH = _os.path.join(_HERE, 'out'); FIG = _os.path.join(_HERE, '..', 'figures') + '/'
ev = np.load(SCRATCH + '/model_events.npz'); d = np.load(SCRATCH + '/model_stats.npz')
mc = json.load(open(SCRATCH + '/mle_compare.json')); ld = json.load(open(SCRATCH + '/ldw_compare.json'))
S, EV = ev['S'].astype(float), ev['EV']; NB = int(d['NB']); XMIN = 1e-4
c = mc['cells'][0]; s = np.sort(S[EV == c['iu'] * NB + c['it']]); s = s[s >= XMIN]; smax = s.max(); n = len(s)
fams = {'ceiling law': ('invpow4', np.array(c['fits']['invpow4']['theta']), 'crimson', '-'),
        'truncated power law': ('tpl', np.array(c['fits']['tpl']['theta']), 'royalblue', '--'),
        'Budrikis et al. 2017': ('ldw', np.array([ld['window'][0]['tau'], np.log(ld['window'][0]['Smax'])]), 'darkorange', '-.')}
def logpdf(name, th, x):
    hi = dist._upper(name, th, smax); g = dist._grid(XMIN, hi if hi is not None else max(100 * smax, 1.0), hi is not None, 4000)
    with np.errstate(all='ignore'):
        lg = dist._lnf(name, g, th, smax); lx = dist._lnf(name, x, th, smax)
    m = np.nanmax(lg); lZ = m + np.log(np.trapezoid(np.exp(lg - m), g))
    return lx - lZ
fig, (L, R) = plt.subplots(1, 2, figsize=(12.5, 4.8))
be = np.geomspace(XMIN, smax * 1.001, 22); cnt, _ = np.histogram(s, bins=be); xc = np.sqrt(be[1:] * be[:-1])
rho = cnt / np.diff(be) / n; err = np.sqrt(np.maximum(cnt, 1)) / np.diff(be) / n
for lab, (name, th, col, ls) in fams.items():
    mod = np.exp(logpdf(name, th, xc)); ok = cnt > 0
    L.errorbar(xc[ok] * (1 + 0.04 * (list(fams).index(lab) - 1)), rho[ok] / mod[ok], err[ok] / mod[ok], fmt='o', ls=ls, color=col, ms=4, lw=1.3, capsize=2, label=lab)
L.axhline(1, color='k', lw=1); L.set_xscale('log'); L.set_ylim(0.4, 1.8)
L.set_xlabel('event size s'); L.set_ylabel('measured density / model density')
L.set_title('Residuals per size bin, largest cell (Poisson error bars)'); L.legend(fontsize=9)
lp0 = logpdf('invpow4', fams['ceiling law'][1], s)
for lab, (name, th, col, ls) in fams.items():
    if lab == 'ceiling law': continue
    gain = np.cumsum(lp0 - logpdf(name, th, s))
    R.plot(s, gain, ls, color=col, lw=2, label='%s   (total %+.0f; ΔAIC %+.0f)' % (lab, gain[-1], 2 * gain[-1] + 2 * (dist.NPAR[name] - 4)))
R.axhline(0, color='k', lw=1); R.set_xscale('log')
R.set_xlabel('event size s  (events accumulated in order of size)'); R.set_ylabel('cumulative log-likelihood gain of the ceiling law')
R.set_title('Where the margin is earned'); R.legend(fontsize=9, loc='upper left')
fig.tight_layout(); fig.savefig(FIG + 'fig24_ldw_residuals.png', dpi=300); print('saved fig24_ldw_residuals.png')
