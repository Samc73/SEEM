import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Decompose the simulation's excess run-to-run scatter (scatter_test.py)
into event timing vs event size: within-preparation variance of the number
of events and of the total stress released per strain window of 0.1, data
vs the ceiling-law simulation started from the same 850 initial states."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library')); sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH
d = np.load(SCRATCH + '/model_stats.npz'); ev = np.load(SCRATCH + '/model_events.npz')
tid = np.load(SCRATCH + '/event_tids.npz')['tid_drop']; s4 = np.load(SCRATCH + '/scale4.npz')
fields = dict(np.load(SCRATCH + '/pdmp_fields.npz'))
rates, cr = d['rates'], d['cr']; NB = int(d['NB']); ntr = len(cr)
W = 10_000; NW = 5
S, EG = ev['S'].astype(float), ev['Eg'].astype(float)
wbin = np.clip((EG // 0.1).astype(int), 0, NW - 1)
cnt_d = np.zeros((ntr, NW)); sum_d = np.zeros((ntr, NW))
np.add.at(cnt_d, (tid, wbin), 1); np.add.at(sum_d, (tid, wbin), S)
# simulation, ceiling law, same initial states
K = 257; levels = (np.arange(K) + 0.5) / K
KG, MG, EG_ = float(s4['K_G']), float(s4['M_G']), float(s4['E_G']); sc4 = s4['sc4']
def q4(c):
    g = dist._grid(1e-7, c, True, 3000); ln = MG * np.log(c - g) - KG * np.log(g + EG_)
    w = np.exp(ln - ln.max()); cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(g)))); cdf /= cdf[-1]
    return np.interp(levels, cdf, g)
Q = np.full((NB, NB, K), np.nan)
for iu in range(NB):
    for it in range(NB):
        if np.isfinite(sc4[iu, it]): Q[iu, it] = q4(sc4[iu, it])
fin = np.argwhere(np.isfinite(Q[:, :, 0]))
for i, j in np.argwhere(~np.isfinite(Q[:, :, 0])):
    nn = fin[np.argmin((fin[:, 0] - i) ** 2 + (fin[:, 1] - j) ** 2)]; Q[i, j] = Q[nn[0], nn[1]]
_, _, _, cnt_s, sum_s = simulate(Q, fields, d['u_init'], d['tau_init'], W * NW, seed=42, record_every=W, window_steps=W)
cnt_s, sum_s = cnt_s[:, :NW], sum_s[:, :NW]
def stats(a):
    m = np.array([[a[cr == r, w].mean() for w in range(NW)] for r in rates])
    v = np.array([[a[cr == r, w].var() for w in range(NW)] for r in rates])
    return m.mean(0), np.sqrt(v.mean(0))
out = {}
print('per strain window of 0.1: mean and within-preparation SD (rms over the 9 preparations)')
print('%-10s %8s %8s %8s %8s | %8s %8s %8s %8s' % ('window', 'N data', 'N sim', 'SD data', 'SD sim', 'sum dat', 'sum sim', 'SD data', 'SD sim'))
md, sd = stats(cnt_d); msim, ssim = stats(cnt_s); ud, usd = stats(sum_d); usim, ussd = stats(sum_s)
for w in range(NW):
    print('%.1f-%.1f    %8.1f %8.1f %8.2f %8.2f | %8.3f %8.3f %8.3f %8.3f   Fano data %.2f sim %.2f' %
          (w / 10, (w + 1) / 10, md[w], msim[w], sd[w], ssim[w], ud[w], usim[w], usd[w], ussd[w], sd[w] ** 2 / md[w], ssim[w] ** 2 / msim[w]))
    out['%.1f-%.1f' % (w / 10, (w + 1) / 10)] = dict(N_data=float(md[w]), N_sim=float(msim[w]), SDN_data=float(sd[w]), SDN_sim=float(ssim[w]),
                                                   sum_data=float(ud[w]), sum_sim=float(usim[w]), SDsum_data=float(usd[w]), SDsum_sim=float(ussd[w]))
json.dump(out, open(SCRATCH + '/scatter2.json', 'w'), indent=1)
