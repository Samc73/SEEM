import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""A per-sample slow variable, with no new free parameter.  Across
preparations the ceiling factor is (rate/rate_mid)^beta and the initial
energy is u0(rate); the same relation applied within a preparation gives
each trajectory a factor exp(kappa * (u0_i - u0_mean(rate))) with
kappa = beta * d ln(rate) / d u0_mean.  Scored on the within-preparation
scatter of tau and u and on the persistence of the initial u deviation."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library')); sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH
d = np.load(SCRATCH + '/model_stats.npz'); s4 = np.load(SCRATCH + '/scale4.npz'); fields = dict(np.load(SCRATCH + '/pdmp_fields.npz'))
ms = np.load(SCRATCH + '/memory_stats.npz'); mem = json.load(open(SCRATCH + '/memory_where.json'))
rates, cr = d['rates'], d['cr']; NB = int(d['NB']); R = len(rates); ntr = len(cr)
K = 257; levels = (np.arange(K) + 0.5) / K; NSTEPS = 50_000; REC = 20
KG, MG, EG = float(s4['K_G']), float(s4['M_G']), float(s4['E_G']); sc4 = s4['sc4']
beta = mem['channels']['mean size <s>']['beta']
u0 = d['u_init'].astype(float); u0m = np.array([u0[cr == r].mean() for r in rates]); lr = np.log(rates / rates[R // 2])
kappa = beta * np.polyfit(u0m, lr, 1)[0]
du0 = u0 - u0m[np.searchsorted(rates, cr)]
print('beta = %.3f,  d ln(rate)/d u0 = %.0f,  kappa = %.1f per unit u;  within-preparation SD of u0 = %.5f -> factor spread +-%.1f%%' %
      (beta, np.polyfit(u0m, lr, 1)[0], kappa, du0.std(), 100 * abs(kappa) * du0.std()))
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
lam_rate = np.exp(beta * lr[np.searchsorted(rates, cr)])
u_ds, t_ds, g_ds = ms['u_ds'], ms['tau_ds'], ms['g_ds']
within = lambda x: float(np.sqrt(np.mean([x[cr == r].var() for r in rates])))
dev = lambda x: np.concatenate([x[cr == r] - x[cr == r].mean() for r in rates])
def score(rec_t, rec_u, g_axis):
    at = lambda g: int(np.argmin(np.abs(g_axis - g)))
    x0 = dev(rec_u[:, 0].astype(float))
    return dict(sd_tau={g: within(rec_t[:, at(g)].astype(float)) for g in [0.1, 0.3, 0.5]},
                sd_u={g: within(rec_u[:, at(g)].astype(float)) for g in [0.1, 0.3, 0.5]},
                persist_u={g: float(np.corrcoef(x0, dev(rec_u[:, at(g)].astype(float)))[0, 1]) for g in [0.1, 0.2, 0.3, 0.5]},
                spread_u05=float(max(rec_u[cr == r, at(0.5)].mean() for r in rates) / min(rec_u[cr == r, at(0.5)].mean() for r in rates) - 1))
res = dict(data=score(t_ds, u_ds, g_ds))
g_sim = np.arange(NSTEPS // REC + 1) * REC * 1e-5
for name, fac in [('model4', None), ('rate memory', lam_rate), ('rate + per-sample', lam_rate * np.exp(kappa * du0)),
                  ('rate + per-sample x3', lam_rate * np.exp(3 * kappa * du0))]:
    kw = {} if fac is None else dict(size_scale=lambda i, f=fac: f)
    rec_t, rec_u, peak = simulate(Q, fields, d['u_init'], d['tau_init'], NSTEPS, seed=42, record_every=REC, **kw)
    res[name] = score(rec_t, rec_u, g_sim[:rec_u.shape[1]])
for k, r in res.items():
    print('%-22s SD tau %s   SD u %s   persist u0 %s   u-spread(0.5) %.1f%%' %
          (k, {g: round(v, 4) for g, v in r['sd_tau'].items()}, {g: round(v, 5) for g, v in r['sd_u'].items()},
           {g: round(v, 2) for g, v in r['persist_u'].items()}, 100 * r['spread_u05']))
json.dump(res, open(SCRATCH + '/sim_persample.json', 'w'), indent=1)
