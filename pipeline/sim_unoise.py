import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Is the tau variance gap downstream of a u variance gap?  The model's
within-preparation scatter of u is 45% too large already at gamma = 0.1,
and the ceiling grows with u.  Switch off the u-channel noise terms one at
a time (Gaussian residual of the energy coupling, aging) and watch both."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library')); sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH
d = np.load(SCRATCH + '/model_stats.npz'); s4 = np.load(SCRATCH + '/scale4.npz'); fields = dict(np.load(SCRATCH + '/pdmp_fields.npz'))
ms = np.load(SCRATCH + '/memory_stats.npz'); rates, cr = d['rates'], d['cr']; NB = int(d['NB'])
K = 257; levels = (np.arange(K) + 0.5) / K; NSTEPS = 50_000; REC = 20; W = 10_000; NW = 5
KG, MG, EG = float(s4['K_G']), float(s4['M_G']), float(s4['E_G']); sc4 = s4['sc4']
def q4(c):
    g = dist._grid(1e-7, c, True, 3000); ln = MG * np.log(c - g) - KG * np.log(g + EG)
    w = np.exp(ln - ln.max()); cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(g)))); cdf /= cdf[-1]
    return np.interp(levels, cdf, g)
Q = np.full((NB, NB, K), np.nan)
for iu in range(NB):
    for it in range(NB):
        if np.isfinite(sc4[iu, it]): Q[iu, it] = q4(sc4[iu, it])
fin = np.argwhere(np.isfinite(Q[:, :, 0]))
for i, j in np.argwhere(~np.isfinite(Q[:, :, 0])):
    nn = fin[np.argmin((fin[:, 0] - i) ** 2 + (fin[:, 1] - j) ** 2)]; Q[i, j] = Q[nn[0], nn[1]]
u_ds, t_ds, g_ds = ms['u_ds'], ms['tau_ds'], ms['g_ds']
within = lambda x: float(np.sqrt(np.mean([x[cr == r].var() for r in rates])))
def score(rec_t, rec_u, g_axis, ssum=None):
    at = lambda g: int(np.argmin(np.abs(g_axis - g)))
    r = dict(sd_tau={g: within(rec_t[:, at(g)].astype(float)) for g in [0.1, 0.3, 0.5]}, sd_u={g: within(rec_u[:, at(g)].astype(float)) for g in [0.1, 0.3, 0.5]})
    if ssum is not None: r['cv_sum'] = [float(np.sqrt(np.mean([ssum[cr == q, w].var() for q in rates])) / ssum[:, w].mean()) for w in range(1, NW)]
    return r
res = dict(data=score(t_ds, u_ds, g_ds))
print('ksig (u-noise SD per event): median %.2e   m_age median %.2e  p_age median %.2e' % (np.nanmedian(fields['ksig']), np.nanmedian(fields['m_age']), np.nanmedian(fields['p_age'])))
g_sim = np.arange(NSTEPS // REC + 1) * REC * 1e-5
variants = {'model4': {}, 'no u-noise (ksig=0)': dict(ksig=np.zeros_like(fields['ksig'])),
            'no aging': dict(p_age=np.zeros_like(fields['p_age'])),
            'no u-noise, no aging': dict(ksig=np.zeros_like(fields['ksig']), p_age=np.zeros_like(fields['p_age'])),
            'u frozen (drift, noise, aging off)': dict(ksig=np.zeros_like(fields['ksig']), p_age=np.zeros_like(fields['p_age']), gdrift=np.zeros_like(fields['gdrift']), ka=np.zeros_like(fields['ka']), kb=np.zeros_like(fields['kb']))}
for name, over in variants.items():
    f = dict(fields); f.update(over)
    rec_t, rec_u, peak, cnt, ssum = simulate(Q, f, d['u_init'], d['tau_init'], NSTEPS, seed=42, record_every=REC, window_steps=W)
    res[name] = score(rec_t, rec_u, g_sim[:rec_u.shape[1]], ssum[:, :NW])
for k, r in res.items():
    print('%-36s SD tau %s   SD u %s   %s' % (k, {g: round(v, 4) for g, v in r['sd_tau'].items()}, {g: round(v, 5) for g, v in r['sd_u'].items()},
          ('CV(sum) %s' % np.round(r['cv_sum'], 3)) if 'cv_sum' in r else ''))
json.dump(res, open(SCRATCH + '/sim_unoise.json', 'w'), indent=1)
