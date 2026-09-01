import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Does the size of a large event track the stress reloaded since the last
large event?  For consecutive large events k, k+1 in one trajectory,
r_big = tau(k+1) - (tau(k) - s_k); correlation of ln s_{k+1} with ln r_big at
fixed cell of k+1, data vs the Markov and two-class simulations."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library')); sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH
d = np.load(SCRATCH + '/model_stats.npz'); ev = np.load(SCRATCH + '/model_events.npz'); s4 = np.load(SCRATCH + '/scale4.npz')
fields = dict(np.load(SCRATCH + '/pdmp_fields.npz')); rb_ = np.load(SCRATCH + '/renewal_big.npz'); tid = np.load(SCRATCH + '/event_tids.npz')['tid_drop']
rates, cr = d['rates'], d['cr']; NB = int(d['NB']); NV = NB * NB; S_BIG = float(rb_['S_BIG'])
K = 257; levels = (np.arange(K) + 0.5) / K
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
Q = fill(np.array([[qcond(sc4[iu, it], 0, np.inf) if np.isfinite(sc4[iu, it]) else np.full(K, np.nan) for it in range(NB)] for iu in range(NB)]))
*_, log = simulate(Q, fields, d['u_init'], d['tau_init'], 50_000, seed=42, record_every=50_000, record_events=True)
log = log[np.lexsort((log[:, 1], log[:, 0]))]
seqs = {'data': (tid, ev['EV'].astype(int), ev['S'].astype(float), ev['Et'].astype(float)),
        'markov sim': (log[:, 0].astype(int), log[:, 2].astype(int), log[:, 3], log[:, 4])}
out = {}
for name, (T, V, S, TAU) in seqs.items():
    big = S > S_BIG; T, V, S, TAU = T[big], V[big], S[big], TAU[big]
    same = T[1:] == T[:-1]
    rbig = TAU[1:][same] - (TAU[:-1][same] - S[:-1][same]); s1 = S[1:][same]; v1 = V[1:][same]; ok = rbig > 1e-4
    cs = []; prof = []
    for v in np.unique(v1):
        m = (v1 == v) & ok
        if m.sum() >= 100: cs.append((np.corrcoef(np.log(s1[m]), np.log(rbig[m]))[0, 1], m.sum()))
    cs = np.array(cs); r = float(np.average(cs[:, 0], weights=cs[:, 1]))
    # consecutive large-event sizes at fixed cell of k+1
    s0 = S[:-1][same]; c2 = []
    for v in np.unique(v1):
        m = (v1 == v)
        if m.sum() >= 100: c2.append((np.corrcoef(np.log(s0[m]), np.log(s1[m]))[0, 1], m.sum()))
    c2 = np.array(c2); r2 = float(np.average(c2[:, 0], weights=c2[:, 1]))
    print('%-11s corr(ln s_k, ln s_{k+1}) consecutive large events at fixed cell: %+.3f (%d cells)' % (name, r2, len(c2)))
    # binned: median size of the large event vs reloaded stress, flow cells (tau bin >= 12), pooled
    fl = ok & (v1 % NB >= 12); edges = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0])
    for a, b in zip(edges[:-1], edges[1:]):
        m = fl & (rbig >= a) & (rbig < b); prof.append(float(np.median(s1[m])) if m.sum() >= 50 else None)
    out[name] = dict(corr=r, ncells=len(cs), corr_consecutive_sizes=r2, edges=edges.tolist(), median_size_vs_rbig=prof)
    print('%-11s corr(ln s_{k+1}, ln r_big) at fixed cell, large events: %+.3f (%d cells);  median size vs reloaded stress (flow cells): %s' %
          (name, r, len(cs), [None if p is None else round(p, 3) for p in prof]))
json.dump(out, open(SCRATCH + '/size_reload.json', 'w'), indent=1)
