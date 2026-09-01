import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Is the excess run-to-run variance a grid-resolution effect?  A stress drop
smaller than a tau cell (0.10) leaves the model's state unchanged, so the
model regulates the stress budget more loosely than the glass does.  Test:
the empirical-resampling arm (no fitted law needed) on the 22x22 grid vs a
42x42 grid, same metrics as scatter2.py / sim_aftershock.py."""
import numpy as np, sys, json
sys.path.insert(0, _HERE)
from simulate import simulate
OUT = _os.path.join(_HERE, 'out'); W = 10_000; NW = 5; K = 257; levels = (np.arange(K) + 0.5) / K
res = {}
for grid in [20, 40]:
    S_ = OUT if grid == 20 else _os.path.join(OUT, 'grid%d' % grid)
    d = np.load(S_ + '/model_stats.npz'); ev = np.load(S_ + '/model_events.npz'); fields = dict(np.load(S_ + '/pdmp_fields.npz'))
    NB = int(d['NB']); rates, cr = d['rates'], d['cr']
    S, EV = ev['S'].astype(float), ev['EV']
    Q = np.full((NB, NB, K), np.nan)
    for v in np.unique(EV):
        s = S[EV == v]
        if len(s) >= 40: Q[v // NB, v % NB] = np.quantile(s, levels)
    fin = np.argwhere(np.isfinite(Q[:, :, 0]))
    for i, j in np.argwhere(~np.isfinite(Q[:, :, 0])):
        nn = fin[np.argmin((fin[:, 0] - i) ** 2 + (fin[:, 1] - j) ** 2)]; Q[i, j] = Q[nn[0], nn[1]]
    rec_t, rec_u, peak, cnt, ssum = simulate(Q, fields, d['u_init'], d['tau_init'], W * NW, seed=42, record_every=20, window_steps=W)
    cnt, ssum = cnt[:, :NW], ssum[:, :NW]
    within = lambda x: float(np.sqrt(np.mean([x[cr == r].var() for r in rates])))
    j = lambda g: int(round(g / 1e-5 / 20))
    pk = np.array([peak[cr == r].mean() for r in rates]); pd_ = np.array([d['tau_peak'][cr == r].mean() for r in rates])
    r = dict(cells_with_tables=int(len(fin)),
             fano=[float(np.mean([cnt[cr == q, w].var() for q in rates]) / cnt[:, w].mean()) for w in range(NW)],
             cv_sum=[float(np.sqrt(np.mean([ssum[cr == q, w].var() for q in rates])) / ssum[:, w].mean()) for w in range(NW)],
             sd_u={g: within(rec_u[:, min(j(g), rec_u.shape[1] - 1)].astype(float)) for g in [0.1, 0.3, 0.5]},
             sd_tau={g: within(rec_t[:, min(j(g), rec_t.shape[1] - 1)].astype(float)) for g in [0.1, 0.3, 0.5]},
             peak_rms=float(np.sqrt(np.mean((pk / pd_ - 1) ** 2))))
    res['grid%d' % grid] = r
    print('grid %dx%d (+ring): %d cells with tables   Fano %s   CV(sum) %s' % (grid, grid, len(fin), np.round(r['fano'], 2), np.round(r['cv_sum'], 3)))
    print('    within-prep SD u %s  tau %s  peak RMS %.1f%%' % ({g: round(v, 5) for g, v in r['sd_u'].items()}, {g: round(v, 4) for g, v in r['sd_tau'].items()}, 100 * r['peak_rms']))
ref = json.load(open(OUT + '/sim_aftershock.json'))['data']
print('data:               Fano %s   CV(sum) %s' % (np.round(ref['fano'], 2), np.round(ref['cv_sum'], 3)))
print('    within-prep SD u %s  tau %s' % ({g: round(v, 5) for g, v in ref['sd_u'].items()}, {g: round(v, 4) for g, v in ref['sd_tau'].items()}))
json.dump(res, open(OUT + '/sim_fine.json', 'w'), indent=1)
