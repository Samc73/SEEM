import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Cheapest model change in the repository: a two-state hazard
p(u,tau) * m(n), n = steps since the last event, with the quiet baseline
p_base(u,tau) and the lag multipliers m(1..8) measured in cluster_test.py.
Does it close the 25-30% run-to-run variance gap (scatter_test.py) without
disturbing the mean curves and yield peaks?"""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library')); sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH
d = np.load(SCRATCH + '/model_stats.npz'); ev = np.load(SCRATCH + '/model_events.npz'); s4 = np.load(SCRATCH + '/scale4.npz')
fields = dict(np.load(SCRATCH + '/pdmp_fields.npz')); af = np.load(SCRATCH + '/aftershock.npz'); ms = np.load(SCRATCH + '/memory_stats.npz')
tid = np.load(SCRATCH + '/event_tids.npz')['tid_drop']
rates, cr = d['rates'], d['cr']; NB = int(d['NB']); ntr = len(cr)
K = 257; levels = (np.arange(K) + 0.5) / K; W = 10_000; NW = 5
KG, MG, EG = float(s4['K_G']), float(s4['M_G']), float(s4['E_G']); sc4 = s4['sc4']
def q4(c):
    g = dist._grid(1e-7, c, True, 3000); ln = MG * np.log(c - g) - KG * np.log(g + EG)
    w = np.exp(ln - ln.max()); cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(g)))); cdf /= cdf[-1]
    return np.interp(levels, cdf, g)
def nearest_fill(A):
    fin = np.argwhere(np.isfinite(A if A.ndim == 2 else A[:, :, 0]))
    for i, j in np.argwhere(~np.isfinite(A if A.ndim == 2 else A[:, :, 0])):
        nn = fin[np.argmin((fin[:, 0] - i) ** 2 + (fin[:, 1] - j) ** 2)]; A[i, j] = A[nn[0], nn[1]]
    return A
Q = np.full((NB, NB, K), np.nan)
for iu in range(NB):
    for it in range(NB):
        if np.isfinite(sc4[iu, it]): Q[iu, it] = q4(sc4[iu, it])
Q = nearest_fill(Q)
p_base = nearest_fill(af['p_base'].copy()); mult = af['mult']
print('lag multipliers m(1..8):', np.round(mult[:-1], 2), ' mean baseline/pooled hazard ratio: %.3f' % np.nanmean(p_base / np.where(fields['p_drop'] > 0, fields['p_drop'], np.nan)))

# data references
S, EG_ = ev['S'].astype(float), ev['Eg'].astype(float)
wbin = np.clip((EG_ // 0.1).astype(int), 0, NW - 1)
cnt_d = np.zeros((ntr, NW)); sum_d = np.zeros((ntr, NW)); np.add.at(cnt_d, (tid, wbin), 1); np.add.at(sum_d, (tid, wbin), S)
u_ds, t_ds, g_ds = ms['u_ds'], ms['tau_ds'], ms['g_ds']
def within(x): return float(np.sqrt(np.mean([x[cr == r].var() for r in rates])))
def fano(a): return [float(np.mean([a[cr == r, w].var() for r in rates]) / np.mean([a[cr == r, w].mean() for r in rates])) for w in range(NW)]
def cv(a): return [float(np.sqrt(np.mean([a[cr == r, w].var() for r in rates])) / np.mean([a[cr == r, w].mean() for r in rates])) for w in range(NW)]
res = dict(data=dict(fano=fano(cnt_d), cv_sum=cv(sum_d), N=cnt_d.mean(0).tolist(),
                     sd_u={g: within(u_ds[:, int(np.argmin(np.abs(g_ds - g)))].astype(float)) for g in [0.1, 0.3, 0.5]},
                     sd_tau={g: within(t_ds[:, int(np.argmin(np.abs(g_ds - g)))].astype(float)) for g in [0.1, 0.3, 0.5]}))
for name, kw in [('markov', {}), ('aftershock', dict(aftershock=(p_base, mult)))]:
    rec_t, rec_u, peak, cnt, ssum = simulate(Q, fields, d['u_init'], d['tau_init'], W * NW, seed=42, record_every=20, window_steps=W, **kw)
    cnt, ssum = cnt[:, :NW], ssum[:, :NW]
    pk = np.array([peak[cr == r].mean() for r in rates]); pd_ = np.array([d['tau_peak'][cr == r].mean() for r in rates])
    j = lambda g: int(round(g / 1e-5 / 20))
    res[name] = dict(fano=fano(cnt), cv_sum=cv(ssum), N=cnt.mean(0).tolist(), peak_rms=float(np.sqrt(np.mean((pk / pd_ - 1) ** 2))),
                     sd_u={g: within(rec_u[:, min(j(g), rec_u.shape[1] - 1)].astype(float)) for g in [0.1, 0.3, 0.5]},
                     sd_tau={g: within(rec_t[:, min(j(g), rec_t.shape[1] - 1)].astype(float)) for g in [0.1, 0.3, 0.5]},
                     flow05=[float(rec_t[cr == r, -1].mean()) for r in rates],
                     curves_u=np.stack([rec_u[cr == r].mean(0) for r in rates]).tolist())
    res[name]['spread_u05'] = float(np.max(res[name]['curves_u'], axis=0)[-1] / np.min(res[name]['curves_u'], axis=0)[-1] - 1)
for k in ['data', 'markov', 'aftershock']:
    r = res[k]
    print('%-10s events/window %s   Fano %s   CV(sum) %s' % (k, np.round(r['N'], 1), np.round(r['fano'], 2), np.round(r['cv_sum'], 3)))
    print('           within-prep SD u: %s   tau: %s %s' % ({g: round(v, 5) for g, v in r['sd_u'].items()}, {g: round(v, 4) for g, v in r['sd_tau'].items()},
          ('  peak RMS %.1f%%  u-spread(0.5) %.1f%%' % (100 * r['peak_rms'], 100 * r['spread_u05'])) if 'peak_rms' in r else ''))
json.dump(res, open(SCRATCH + '/sim_aftershock.json', 'w'), indent=1)
