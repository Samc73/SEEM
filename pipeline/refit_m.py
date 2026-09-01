import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Decoupled-ceiling refit: global (k, m, eps) from big cells, per-voxel c.
Then redo the moment/q comparison with the corrected sign convention.
"""
import numpy as np, sys, json
from scipy.optimize import minimize_scalar
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist

SCRATCH = _os.path.join(_HERE, 'out')
d = np.load(SCRATCH + '/model_stats.npz')
ev = np.load(SCRATCH + '/model_events.npz')
sf = np.load(SCRATCH + '/scale_fields.npz')
NB = int(d['NB'])
DG = float(d['DG'])
S, EV = ev['S'].astype(float), ev['EV']
XMIN = 1e-6

with open(SCRATCH + '/mle_compare.json') as f:
    cells = [(c['iu'], c['it'], c['n']) for c in json.load(f)['cells'] if c['n'] >= 3000]

pars = []
for iu, it, n in cells:
    s = S[EV == iu * NB + it]
    r = dist.fit('invpow4', s, XMIN,
                 x0=[0.96, 1.5, np.log(5e-5), np.log(0.05)])
    k, m, le, lc = r['theta']
    pars.append((k, m, np.exp(le)))
    print('cell (%d,%d): k=%.3f m=%.3f eps=%.1e c/smax=%.3f' %
          (iu, it, k, m, np.exp(le), 1 + np.exp(lc)), flush=True)
K_G = float(np.median([p[0] for p in pars]))
M_G = float(np.median([p[1] for p in pars]))
E_G = float(np.median([p[2] for p in pars]))
print('GLOBAL k=%.3f m=%.3f eps=%.2e' % (K_G, M_G, E_G), flush=True)


def fit_c4(s):
    s = s[s >= XMIN]
    if len(s) < 40:
        return np.nan
    smax = s.max()

    def f(lc):
        return dist.nll('invpow4', s, [K_G, M_G, np.log(E_G), lc], XMIN, smax,
                        npts=1500)

    r = minimize_scalar(f, bounds=(np.log(1e-4), np.log(50.0)), method='bounded',
                        options=dict(xatol=1e-4))
    return float(smax * (1 + np.exp(r.x)))


sc4 = np.full((NB, NB), np.nan)
for v in np.unique(EV):
    sc4[v // NB, v % NB] = fit_c4(S[EV == v])
print('sc4 fitted on %d voxels' % np.isfinite(sc4).sum(), flush=True)


def moment4(c, lo=1e-7):
    g = dist._grid(lo, c, True, 2000)
    ln = M_G * np.log(c - g) - K_G * np.log(g + E_G)
    w = np.exp(ln - ln.max())
    return float(np.trapezoid(g * w, g) / np.trapezoid(w, g))


n_all = d['n_all'].reshape(NB, NB)
nev = sf['nev']
with np.errstate(all='ignore'):
    mu2 = (d['sum_dtau_ne'] / d['n_ne']).reshape(NB, NB) / DG
    q_emp = 1 - (d['sum_dtau'] / d['n_all']).reshape(NB, NB) / (mu2 * DG)
    p_drop = (d['n_drop'] / d['n_all']).reshape(NB, NB)
    sbar_emp = (d['sum_s'] / d['n_drop']).reshape(NB, NB)
sbar4 = np.full((NB, NB), np.nan)
for iu in range(NB):
    for it in range(NB):
        if np.isfinite(sc4[iu, it]):
            sbar4[iu, it] = moment4(sc4[iu, it])
q_mod4 = q_emp + p_drop * (sbar4 - sbar_emp) / (mu2 * DG)

inner = np.zeros((NB, NB), bool)
inner[1:21, 1:21] = True
ok = inner & (nev >= 100) & np.isfinite(sbar4) & (q_emp > 0)
rel = (sbar4 / sbar_emp - 1)[ok]
relq = (q_mod4 / q_emp - 1)[ok]
print('<s> invpow4/emp - 1: median %+.3f IQR [%+.3f, %+.3f] RMS %.3f' %
      (np.median(rel), *np.quantile(rel, [0.25, 0.75]), np.sqrt(np.mean(rel ** 2))))
print('q  invpow4/emp - 1: median %+.3f RMS %.3f' %
      (np.median(relq), np.sqrt(np.mean(relq ** 2))))
np.savez(SCRATCH + '/scale4.npz', sc4=sc4, sbar4=sbar4, q_mod4=q_mod4,
         K_G=K_G, M_G=M_G, E_G=E_G, ok=ok, rel=rel, relq=relq)
print('DONE')
