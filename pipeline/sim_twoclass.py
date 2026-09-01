import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Two-class hazard: small events (s <= S_BIG) at the Markov hazard, large
events at a hazard that depends on the stress reloaded since the last large
event (renewal_big.py).  Sizes from the fitted ceiling law conditioned on
the class.  Same metrics as Figure 21."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library')); sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH
d = np.load(SCRATCH + '/model_stats.npz'); ev = np.load(SCRATCH + '/model_events.npz'); s4 = np.load(SCRATCH + '/scale4.npz')
fields = dict(np.load(SCRATCH + '/pdmp_fields.npz')); rb_ = np.load(SCRATCH + '/renewal_big.npz'); ms = np.load(SCRATCH + '/memory_stats.npz')
tid = np.load(SCRATCH + '/event_tids.npz')['tid_drop']
rates, cr = d['rates'], d['cr']; NB = int(d['NB']); NV = NB * NB; ntr = len(cr); S_BIG = float(rb_['S_BIG'])
K = 257; levels = (np.arange(K) + 0.5) / K; W = 10_000; NW = 5
KG, MG, EG = float(s4['K_G']), float(s4['M_G']), float(s4['E_G']); sc4 = s4['sc4']
def qcond(c, lo, hi):
    g = dist._grid(1e-7, c, True, 3000); ln = MG * np.log(c - g) - KG * np.log(g + EG)
    w = np.exp(ln - ln.max()); w[(g < lo) | (g > hi)] = 0
    if w.sum() == 0: return np.full(K, np.clip(lo, 1e-7, c))
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(g)))); cdf /= cdf[-1]
    return np.interp(levels, cdf, g)
def fill(Q):
    fin = np.argwhere(np.isfinite(Q[:, :, 0]))
    for i, j in np.argwhere(~np.isfinite(Q[:, :, 0])):
        nn = fin[np.argmin((fin[:, 0] - i) ** 2 + (fin[:, 1] - j) ** 2)]; Q[i, j] = Q[nn[0], nn[1]]
    return Q
Q, Qs, Qb = (np.full((NB, NB, K), np.nan) for _ in range(3))
for iu in range(NB):
    for it in range(NB):
        if np.isfinite(sc4[iu, it]):
            Q[iu, it] = qcond(sc4[iu, it], 0, np.inf); Qs[iu, it] = qcond(sc4[iu, it], 0, S_BIG); Qb[iu, it] = qcond(sc4[iu, it], S_BIG, np.inf)
Q, Qs, Qb = fill(Q), fill(Qs), fill(Qb)
# hazards: small-event Markov hazard per cell from the catalog; large-event hazard vs r_big from renewal_big
S, EV = ev['S'].astype(float), ev['EV']
n_all = d['n_all'].astype(float); n_big = np.bincount(EV[S > S_BIG], minlength=NV).astype(float); n_drop = d['n_drop'].astype(float)
with np.errstate(divide='ignore', invalid='ignore'):
    Hs = np.where(n_all >= 200, (n_drop - n_big) / n_all, np.nan)
n, e = rb_['n'].astype(float), rb_['e_big'].astype(float)
with np.errstate(divide='ignore', invalid='ignore'):
    Hb = np.where(n >= 200, e / n, np.nan); pooled_big = np.where(n_all >= 200, n_big / n_all, np.nan)
NR = Hb.shape[1]
for v in range(NV):
    row = Hb[v]
    if np.isfinite(row).any():
        idx = np.nonzero(np.isfinite(row))[0]
        for b in range(NR):
            if not np.isfinite(row[b]): Hb[v, b] = row[idx[np.argmin(np.abs(idx - b))]]
    else:
        Hb[v] = pooled_big[v]
ok = np.isfinite(Hs) & np.isfinite(Hb[:, 0])
fin = np.argwhere(ok.reshape(NB, NB))
for v in np.nonzero(~ok)[0]:
    i, j = v // NB, v % NB; nn = fin[np.argmin((fin[:, 0] - i) ** 2 + (fin[:, 1] - j) ** 2)]; Hs[v] = Hs[nn[0] * NB + nn[1]]; Hb[v] = Hb[nn[0] * NB + nn[1]]
Hb_flat = np.repeat(np.nansum(e, 1)[:, None] / np.maximum(np.nansum(n, 1)[:, None], 1), NR, 1)   # same two classes, but no reload dependence
for v in np.nonzero(~ok)[0]:
    i, j = v // NB, v % NB; nn = fin[np.argmin((fin[:, 0] - i) ** 2 + (fin[:, 1] - j) ** 2)]; Hb_flat[v] = Hb_flat[nn[0] * NB + nn[1]]
print('large-event share of the hazard (flow cells): %.2f' % np.nanmedian((Hb[:, 5] / (Hs + Hb[:, 5]))[np.isin(np.arange(NV) % NB, range(12, 21))]))
EG_ = ev['Eg'].astype(float); wbin = np.clip((EG_ // 0.1).astype(int), 0, NW - 1)
def fano_thr(T, WB, Sz, thr):
    m = Sz > thr; c = np.zeros((ntr, NW)); np.add.at(c, (T[m], WB[m]), 1)
    return float(np.mean([np.mean([c[cr == r, w].var() for r in rates]) / c[:, w].mean() for w in range(1, NW)]))
cnt_d = np.zeros((ntr, NW)); sum_d = np.zeros((ntr, NW)); np.add.at(cnt_d, (tid, wbin), 1); np.add.at(sum_d, (tid, wbin), S)
u_ds, t_ds, g_ds = ms['u_ds'], ms['tau_ds'], ms['g_ds']
within = lambda x: float(np.sqrt(np.mean([x[cr == r].var() for r in rates])))
cv = lambda a: [float(np.sqrt(np.mean([a[cr == r, w].var() for r in rates])) / a[:, w].mean()) for w in range(NW)]
res = dict(data=dict(cv_sum=cv(sum_d), fano_all=fano_thr(tid, wbin, S, 0), fano_big=fano_thr(tid, wbin, S, S_BIG), fano_03=fano_thr(tid, wbin, S, 0.03),
                     sd_tau={g: within(t_ds[:, int(np.argmin(np.abs(g_ds - g)))].astype(float)) for g in [0.1, 0.3, 0.5]},
                     sd_u={g: within(u_ds[:, int(np.argmin(np.abs(g_ds - g)))].astype(float)) for g in [0.1, 0.3, 0.5]}))
arms = [('markov', {}), ('twoclass_flat', dict(twoclass=(Hs, Hb_flat, rb_['RE'], Qs, Qb))), ('twoclass_reload', dict(twoclass=(Hs, Hb, rb_['RE'], Qs, Qb)))]
for name, kw in arms:
    rec_t, rec_u, peak, cnt, ssum, log = simulate(Q, fields, d['u_init'], d['tau_init'], W * NW, seed=42, record_every=20, window_steps=W, record_events=True, **kw)
    cnt, ssum = cnt[:, :NW], ssum[:, :NW]
    T, WB, Sz = log[:, 0].astype(int), np.clip((log[:, 1] * 1e-5 // 0.1).astype(int), 0, NW - 1), log[:, 3]
    pk = np.array([peak[cr == r].mean() for r in rates]); pd_ = np.array([d['tau_peak'][cr == r].mean() for r in rates])
    j = lambda g: int(round(g / 1e-5 / 20))
    res[name] = dict(cv_sum=cv(ssum), fano_all=fano_thr(T, WB, Sz, 0), fano_big=fano_thr(T, WB, Sz, S_BIG), fano_03=fano_thr(T, WB, Sz, 0.03),
                     sd_tau={g: within(rec_t[:, min(j(g), rec_t.shape[1] - 1)].astype(float)) for g in [0.1, 0.3, 0.5]},
                     sd_u={g: within(rec_u[:, min(j(g), rec_u.shape[1] - 1)].astype(float)) for g in [0.1, 0.3, 0.5]},
                     peak_rms=float(np.sqrt(np.mean((pk / pd_ - 1) ** 2))), events_per_window=cnt.mean(0).tolist(),
                     flow05=[float(rec_t[cr == r, -1].mean()) for r in rates])
for k, r in res.items():
    print('%-16s CV(sum) %s  Fano all %.2f  >0.01 %.2f  >0.03 %.2f   SD tau %s   SD u %s %s' %
          (k, np.round(r['cv_sum'][1:], 3), r['fano_all'], r['fano_big'], r['fano_03'], {g: round(v, 4) for g, v in r['sd_tau'].items()},
           {g: round(v, 5) for g, v in r['sd_u'].items()}, ('  peak RMS %.1f%%  flow %s' % (100 * r['peak_rms'], np.round(r['flow05'][::4], 3))) if 'peak_rms' in r else ''))
json.dump(res, open(SCRATCH + '/sim_twoclass.json', 'w'), indent=1)
