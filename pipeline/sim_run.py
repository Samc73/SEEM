import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Run one forward-simulation arm: sizes from the fitted invpow law, the
fitted TPL rival, or empirical per-voxel resampling. Usage: python sim_run.py ARM
"""
import numpy as np, sys
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH

ARM = sys.argv[1]
K = 257
NSTEPS = 200_000
d = np.load(SCRATCH + '/model_stats.npz')
sf = np.load(SCRATCH + '/scale_fields.npz')
fields = dict(np.load(SCRATCH + '/pdmp_fields.npz'))
ev = np.load(SCRATCH + '/model_events.npz')
NB = int(d['NB'])
A_G, EPS_G, KAP_G = float(sf['A_G']), float(sf['EPS_G']), float(sf['KAP_G'])
levels = (np.arange(K) + 0.5) / K


def q_invpow(c):
    g = dist._grid(1e-7, c, True, 3000)
    ln = A_G * (np.log(c - g) - np.log(g + EPS_G))
    w = np.exp(ln - ln.max())
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(g))))
    cdf /= cdf[-1]
    return np.interp(levels, cdf, g)


def q_tpl(ss):
    g = np.geomspace(1e-7, 50 * ss, 3000)
    ln = -KAP_G * np.log(g) - g / ss
    w = np.exp(ln - ln.max())
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(g))))
    cdf /= cdf[-1]
    return np.interp(levels, cdf, g)


def q_invpow4(c, kg, mg, eg):
    g = dist._grid(1e-7, c, True, 3000)
    ln = mg * np.log(c - g) - kg * np.log(g + eg)
    w = np.exp(ln - ln.max())
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(g))))
    cdf /= cdf[-1]
    return np.interp(levels, cdf, g)


Q = np.full((NB, NB, K), np.nan)
if ARM == 'model4':
    s4 = np.load(SCRATCH + '/scale4.npz')
    KG, MG, EG = float(s4['K_G']), float(s4['M_G']), float(s4['E_G'])
    sc4 = s4['sc4']
    for iu in range(NB):
        for it in range(NB):
            if np.isfinite(sc4[iu, it]):
                Q[iu, it] = q_invpow4(sc4[iu, it], KG, MG, EG)
elif ARM == 'empir':
    S, EV = ev['S'], ev['EV']
    for v in np.unique(EV):
        s = S[EV == v]
        if len(s) >= 40:
            Q[v // NB, v % NB] = np.quantile(s, levels)
else:
    fld = sf['sc'] if ARM == 'model' else sf['sstar']
    fn = q_invpow if ARM == 'model' else q_tpl
    for iu in range(NB):
        for it in range(NB):
            if np.isfinite(fld[iu, it]):
                Q[iu, it] = fn(fld[iu, it])

# nearest-finite fill
fin = np.argwhere(np.isfinite(Q[:, :, 0]))
for i, j in np.argwhere(~np.isfinite(Q[:, :, 0])):
    nn = fin[np.argmin((fin[:, 0] - i) ** 2 + (fin[:, 1] - j) ** 2)]
    Q[i, j] = Q[nn[0], nn[1]]

rec_t, rec_u, peak = simulate(Q, fields, d['u_init'], d['tau_init'],
                              NSTEPS, seed=42, record_every=20)
np.savez_compressed(SCRATCH + f'/sim_{ARM}.npz',
                    rec_t=rec_t, rec_u=rec_u, peak=peak, cr=d['cr'],
                    record_every=20, nsteps=NSTEPS)
rates = d['rates']
cr = d['cr']
print(ARM, 'done. peak stress by rate (sim vs data):')
for c in rates:
    m = cr == c
    print('  %.3g: sim %.3f+-%.3f  data %.3f+-%.3f' %
          (c, peak[m].mean(), peak[m].std(), d['tau_peak'][m].mean(),
           d['tau_peak'][m].std()))
