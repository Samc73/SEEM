import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Forward simulation with the renewal hazard measured in renewal_hazard.py
(hazard as a function of cell and stress reloaded since the last event),
against the Markov hazard, on the variance metrics of Figure 21 plus the
mean observables of Section 8."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library')); sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH
d = np.load(SCRATCH + '/model_stats.npz'); ev = np.load(SCRATCH + '/model_events.npz'); s4 = np.load(SCRATCH + '/scale4.npz')
fields = dict(np.load(SCRATCH + '/pdmp_fields.npz')); rh = np.load(SCRATCH + '/renewal_hazard.npz'); ms = np.load(SCRATCH + '/memory_stats.npz')
tid = np.load(SCRATCH + '/event_tids.npz')['tid_drop']
rates, cr = d['rates'], d['cr']; NB = int(d['NB']); NV = NB * NB; ntr = len(cr)
K = 257; levels = (np.arange(K) + 0.5) / K; W = 10_000; NW = 5
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

def hazard_table(n, e, nmin=100):
    """H[v, bin] = e/n where n >= nmin; else the cell's pooled hazard over all bins; cells with no data: nearest cell."""
    with np.errstate(divide='ignore', invalid='ignore'):
        H = np.where(n >= nmin, e / np.maximum(n, 1), np.nan)
        pooled = e.sum(1) / np.maximum(n.sum(1), 1)
    # bins without support: carry the nearest supported bin (rightwards then leftwards), else the pooled cell hazard
    for v in range(H.shape[0]):
        row = H[v]
        if np.isfinite(row).any():
            idx = np.nonzero(np.isfinite(row))[0]
            H[v] = row[idx[np.clip(np.searchsorted(idx, np.arange(len(row))), 0, len(idx) - 1)]]
            # searchsorted picks the next supported bin; use previous where past the last one
            for b in range(len(row)):
                if not np.isfinite(row[b]):
                    left = idx[idx < b]; right = idx[idx > b]
                    H[v, b] = row[right[0]] if len(right) and (not len(left) or right[0] - b <= b - left[-1]) else row[left[-1]]
        else:
            H[v] = pooled[v]
    ok = n.sum(1) >= 200
    fin2 = np.argwhere(ok.reshape(NB, NB))
    for v in np.nonzero(~ok)[0]:
        i, j = v // NB, v % NB; nn = fin2[np.argmin((fin2[:, 0] - i) ** 2 + (fin2[:, 1] - j) ** 2)]
        H[v] = H[nn[0] * NB + nn[1]]
    return H
H_abs = hazard_table(rh['n_abs'], rh['e_abs']); H_rel = hazard_table(rh['n_rel'], rh['e_rel'])
S, EG_ = ev['S'].astype(float), ev['Eg'].astype(float)
wbin = np.clip((EG_ // 0.1).astype(int), 0, NW - 1)
cnt_d = np.zeros((ntr, NW)); sum_d = np.zeros((ntr, NW)); np.add.at(cnt_d, (tid, wbin), 1); np.add.at(sum_d, (tid, wbin), S)
u_ds, t_ds, g_ds = ms['u_ds'], ms['tau_ds'], ms['g_ds']
within = lambda x: float(np.sqrt(np.mean([x[cr == r].var() for r in rates])))
fano = lambda a: [float(np.mean([a[cr == r, w].var() for r in rates]) / a[:, w].mean()) for w in range(NW)]
cv = lambda a: [float(np.sqrt(np.mean([a[cr == r, w].var() for r in rates])) / a[:, w].mean()) for w in range(NW)]
res = dict(data=dict(fano=fano(cnt_d), cv_sum=cv(sum_d), N=cnt_d.mean(0).tolist(),
                     sd_u={g: within(u_ds[:, int(np.argmin(np.abs(g_ds - g)))].astype(float)) for g in [0.1, 0.3, 0.5]},
                     sd_tau={g: within(t_ds[:, int(np.argmin(np.abs(g_ds - g)))].astype(float)) for g in [0.1, 0.3, 0.5]},
                     curves_tau=np.stack([t_ds[cr == r].mean(0) for r in rates]).tolist(), g_ds=g_ds.tolist()))
for name, kw in [('markov', {}), ('renewal_abs', dict(renewal=(H_abs, rh['RE'], 'abs'))), ('renewal_rel', dict(renewal=(H_rel, rh['QE'], 'rel')))]:
    rec_t, rec_u, peak, cnt, ssum = simulate(Q, fields, d['u_init'], d['tau_init'], W * NW, seed=42, record_every=20, window_steps=W, **kw)
    cnt, ssum = cnt[:, :NW], ssum[:, :NW]
    pk = np.array([peak[cr == r].mean() for r in rates]); pd_ = np.array([d['tau_peak'][cr == r].mean() for r in rates])
    j = lambda g: int(round(g / 1e-5 / 20))
    cu = np.stack([rec_u[cr == r].mean(0) for r in rates]); ct = np.stack([rec_t[cr == r].mean(0) for r in rates])
    res[name] = dict(fano=fano(cnt), cv_sum=cv(ssum), N=cnt.mean(0).tolist(), peak_rms=float(np.sqrt(np.mean((pk / pd_ - 1) ** 2))),
                     sd_u={g: within(rec_u[:, min(j(g), rec_u.shape[1] - 1)].astype(float)) for g in [0.1, 0.3, 0.5]},
                     sd_tau={g: within(rec_t[:, min(j(g), rec_t.shape[1] - 1)].astype(float)) for g in [0.1, 0.3, 0.5]},
                     curves_u=cu.tolist(), curves_tau=ct.tolist(), spread_u05=float(cu[:, -1].max() / cu[:, -1].min() - 1),
                     flow05=ct[:, -1].tolist())
for k in res:
    r = res[k]
    print('%-12s events/window %s   Fano %s   CV(sum) %s' % (k, np.round(r['N'], 1), np.round(r['fano'], 2), np.round(r['cv_sum'], 3)))
    print('             within-prep SD u %s  tau %s %s' % ({g: round(v, 5) for g, v in r['sd_u'].items()}, {g: round(v, 4) for g, v in r['sd_tau'].items()},
          ('  peak RMS %.1f%%  u-spread(0.5) %.1f%%  flow(0.5) %s' % (100 * r['peak_rms'], 100 * r['spread_u05'], np.round(r['flow05'][::4], 3))) if 'peak_rms' in r else
          '  flow(0.5) %s' % np.round(np.array(r['curves_tau'])[::4, -1], 3)))
json.dump(res, open(SCRATCH + '/sim_renewal.json', 'w'), indent=1)
