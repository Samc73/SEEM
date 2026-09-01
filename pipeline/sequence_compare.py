import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""sequence_test.py measured corr(ln s_k, ln gap_k) at fixed cell of event
k+1 in the data and called the Markov value zero.  That is wrong: fixing the
cell where the *next* event happens selects long reloads after big drops
(the trajectory starts lower).  The fair comparison is the same statistic
on the Markov simulation's own event sequence, conditioned both ways."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library')); sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH
d = np.load(SCRATCH + '/model_stats.npz'); ev = np.load(SCRATCH + '/model_events.npz'); s4 = np.load(SCRATCH + '/scale4.npz')
fields = dict(np.load(SCRATCH + '/pdmp_fields.npz')); tid = np.load(SCRATCH + '/event_tids.npz')['tid_drop']
NB = int(d['NB']); K = 257; levels = (np.arange(K) + 0.5) / K
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
*_, log = simulate(Q, fields, d['u_init'], d['tau_init'], 50_000, seed=42, record_every=50_000, record_events=True)
log = log[np.lexsort((log[:, 1], log[:, 0]))]
seqs = {'data': (tid, np.round(ev['Eg'].astype(float) / 1e-5).astype(int), ev['EV'].astype(int), ev['S'].astype(float)),
        'markov sim': (log[:, 0].astype(int), log[:, 1].astype(int), log[:, 2].astype(int), log[:, 3])}
out = {}
for name, (T, STEP, V, S) in seqs.items():
    same = T[1:] == T[:-1]
    s0 = np.log(S[:-1][same]); gap = np.log(np.maximum(STEP[1:][same] - STEP[:-1][same], 1)); v0 = V[:-1][same]; v1 = V[1:][same]
    res = {}
    for cond, vv in [('cell of k+1', v1), ('cell of k', v0)]:
        cs = []
        for v in np.unique(vv):
            m = vv == v
            if m.sum() >= 100: cs.append((np.corrcoef(s0[m], gap[m])[0, 1], m.sum()))
        cs = np.array(cs); res[cond] = (float(np.average(cs[:, 0], weights=cs[:, 1])), len(cs))
    # size-binned median gap, each cell normalised by its own median gap (Figure 21)
    edges = np.geomspace(1e-4, 1.0, 9); prof = np.zeros(len(edges) - 1); cnt = np.zeros(len(edges) - 1)
    for v in np.unique(v1):
        m = v1 == v
        if m.sum() < 200: continue
        g = np.exp(gap[m]) / np.median(np.exp(gap[m])); b = np.clip(np.searchsorted(edges, np.exp(s0[m]), side='right') - 1, 0, len(edges) - 2)
        for k in range(len(edges) - 1):
            mm = b == k
            if mm.sum() >= 20: prof[k] += np.median(g[mm]) * mm.sum(); cnt[k] += mm.sum()
    with np.errstate(invalid='ignore'):
        res['gap_profile'] = dict(size_edges=edges.tolist(), rel_gap=(prof / cnt).tolist(), n=cnt.tolist())
    # the same correlation with the tiny (aftershock-regime) events removed
    big = s0 > np.log(1e-3); cs = []
    for v in np.unique(v1):
        m = (v1 == v) & big
        if m.sum() >= 100: cs.append((np.corrcoef(s0[m], gap[m])[0, 1], m.sum()))
    cs = np.array(cs); res['cell of k+1, s > 1e-3'] = (float(np.average(cs[:, 0], weights=cs[:, 1])), len(cs))
    print('             restricted to s_k > 1e-3: %+.3f (%d cells)' % res['cell of k+1, s > 1e-3'])
    out[name] = res
    print('%-11s corr(ln s_k, ln gap) at fixed %s: %+.3f (%d cells)   at fixed %s: %+.3f (%d cells)' %
          (name, 'cell of k+1', *res['cell of k+1'], 'cell of k', *res['cell of k']))
json.dump(out, open(SCRATCH + '/sequence_compare.json', 'w'), indent=1)
