import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Are the LARGE events more regular in the data than in the model?  Within-
preparation Fano factor of the count of events above a size threshold per
strain window of 0.1, data vs the Markov simulation's own event log."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library')); sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH
d = np.load(SCRATCH + '/model_stats.npz'); ev = np.load(SCRATCH + '/model_events.npz'); s4 = np.load(SCRATCH + '/scale4.npz')
fields = dict(np.load(SCRATCH + '/pdmp_fields.npz')); tid = np.load(SCRATCH + '/event_tids.npz')['tid_drop']
rates, cr = d['rates'], d['cr']; NB = int(d['NB']); ntr = len(cr); NW = 5
K = 257; levels = (np.arange(K) + 0.5) / K
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
np.savez_compressed(SCRATCH + '/sim_eventlog.npz', log=log)
seqs = {'data': (tid, np.clip((ev['Eg'].astype(float) // 0.1).astype(int), 0, NW - 1), ev['S'].astype(float)),
        'markov': (log[:, 0].astype(int), np.clip((log[:, 1] * 1e-5 // 0.1).astype(int), 0, NW - 1), log[:, 3])}
out = {}
print('within-preparation Fano factor of the count of events with s > threshold, per 0.1 strain window (gamma 0.1-0.5 pooled):')
print('%-10s' % 'threshold' + ''.join('%14s' % k for k in seqs) + '      mean count/window (data, model)')
for thr in [0, 0.003, 0.01, 0.03, 0.1, 0.2]:
    row = {}
    for name, (T, WB, S) in seqs.items():
        m = S > thr; c = np.zeros((ntr, NW)); np.add.at(c, (T[m], WB[m]), 1)
        f = np.mean([np.mean([c[cr == r, w].var() for r in rates]) / c[:, w].mean() for w in range(1, NW)])
        row[name] = dict(fano=float(f), mean=float(c[:, 1:].mean()))
    out[thr] = row
    print('s > %-6g' % thr + ''.join('%14.2f' % row[k]['fano'] for k in seqs) + '      %.1f, %.1f' % (row['data']['mean'], row['markov']['mean']))
json.dump(out, open(SCRATCH + '/large_events.json', 'w'), indent=1)
