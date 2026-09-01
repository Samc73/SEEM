import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Within-cell correlations between consecutive events that a Markov model
in (u,tau) cannot have: size of event k vs the reloading gap to event k+1,
and vs the size of event k+1.  Data only (the model's values are zero by
construction within a cell)."""
import numpy as np, json
SCRATCH = _os.path.join(_HERE, 'out')
ev = np.load(SCRATCH + '/model_events.npz'); tid = np.load(SCRATCH + '/event_tids.npz')['tid_drop']
S, EV, EG = ev['S'].astype(float), ev['EV'], ev['Eg'].astype(float)
same = tid[1:] == tid[:-1]
s0, s1, v1, gap = np.log(S[:-1][same]), np.log(S[1:][same]), EV[1:][same], np.round((EG[1:][same] - EG[:-1][same]) / 1e-5)
out = {}
for name, y in [('ln size(k+1)', s1), ('ln gap(k -> k+1)', np.log(gap))]:
    num = den = 0.0; nc = 0; cs = []
    for v in np.unique(v1):
        m = v1 == v
        if m.sum() < 100: continue
        a, b = s0[m], y[m]
        c = np.corrcoef(a, b)[0, 1]; cs.append((c, m.sum())); nc += 1
    cs = np.array(cs); w = cs[:, 1]
    r = float(np.average(cs[:, 0], weights=w))
    print('corr(ln size(k), %s) at fixed cell of event k+1: %+.3f  (16-84%%: %+.2f..%+.2f, %d cells)' %
          (name, r, *np.percentile(cs[:, 0], [16, 84]), nc))
    out[name] = dict(corr=r, p1684=[float(x) for x in np.percentile(cs[:, 0], [16, 84])], ncells=nc)
# gap after a large vs a small event, at fixed cell
big = s0 > np.log(0.05); small = s0 < np.log(0.005)
ratios = []
for v in np.unique(v1):
    m = v1 == v
    if (m & big).sum() >= 30 and (m & small).sum() >= 30:
        ratios.append((np.median(gap[m & big]) / np.median(gap[m & small]), m.sum()))
ratios = np.array(ratios)
print('median reloading gap after an event > 0.05 / after an event < 0.005, same cell: %.2f (%d cells)' %
      (np.average(ratios[:, 0], weights=ratios[:, 1]), len(ratios)))
out['gap_ratio_big_small'] = float(np.average(ratios[:, 0], weights=ratios[:, 1]))
# size-binned median reloading gap, each cell normalised by its own median gap (for Figure 21)
edges = np.geomspace(1e-4, 1.0, 9); prof = np.zeros(len(edges) - 1); cnt = np.zeros(len(edges) - 1)
for v in np.unique(v1):
    m = v1 == v
    if m.sum() < 200: continue
    g = gap[m] / np.median(gap[m]); b = np.clip(np.searchsorted(edges, np.exp(s0[m]), side='right') - 1, 0, len(edges) - 2)
    for k in range(len(edges) - 1):
        mm = b == k
        if mm.sum() >= 20: prof[k] += np.median(g[mm]) * mm.sum(); cnt[k] += mm.sum()
with np.errstate(invalid='ignore'):
    out['gap_profile'] = dict(size_edges=edges.tolist(), rel_gap=(prof / cnt).tolist(), n=cnt.tolist())
json.dump(out, open(SCRATCH + '/sequence_test.json', 'w'), indent=1)
