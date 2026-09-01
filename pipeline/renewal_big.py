import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""renewal_hazard.py reset the reloading coordinate at *every* event, and
half of all events are tiny aftershocks - so r was mostly the strain since
the last tiny event and told us little.  Here the reference is the last
LARGE event (s > S_BIG): r_big = tau - tau just after it.  Tabulate the
hazard of large and of small events against r_big at fixed cell."""
import numpy as np, pandas as pd, json
SCRATCH = _os.path.join(_HERE, 'out')
u0c = -4.60751861; N = 49999; S_BIG = 0.01
d = np.load(SCRATCH + '/model_stats.npz'); ue, te, NB = d['ue'], d['te'], int(d['NB']); NV = NB * NB
df = pd.read_pickle(_os.path.join(_HERE, '..', 'df_clean.pkl')).sort_values(['index', 'strain_index'])
u = (df['pe'].to_numpy() - u0c).reshape(-1, N); tau = (df['stress'].to_numpy() / 1e4).reshape(-1, N); del df
ntr = tau.shape[0]
dtau = np.diff(tau, axis=1); drop = dtau < 0; bigev = dtau < -S_BIG
idx = np.arange(N - 1)[None, :]
last = np.maximum.accumulate(np.where(bigev, idx, -1), axis=1)
prev_last = np.concatenate([np.full((ntr, 1), -1), last[:, :-1]], axis=1)
row = np.arange(ntr)[:, None]
tau_after = np.where(prev_last >= 0, tau[row, np.maximum(prev_last, 0) + 1], tau[:, :1])
r = (tau[:, :-1] - tau_after).ravel()
has_ref = (prev_last >= 0).ravel()
iu = np.clip(np.searchsorted(ue, u[:, :-1], side='right') - 1, 0, NB - 1); it = np.clip(np.searchsorted(te, tau[:, :-1], side='right') - 1, 0, NB - 1)
vox = (iu * NB + it).ravel(); drop = drop.ravel(); bigev = bigev.ravel(); small = drop & ~bigev
print('r_big: median %.3f  90%% %.3f  99%% %.3f   (steps with a reference: %.1f%%)' % (np.median(r[has_ref]), *np.percentile(r[has_ref], [90, 99]), 100 * has_ref.mean()))
RE = np.r_[np.arange(0, 0.5001, 0.02), np.inf]; NR = len(RE) - 1
rb = np.clip(np.searchsorted(RE, r, side='right') - 1, 0, NR - 1)
key = vox * NR + rb
n = np.bincount(key[has_ref], minlength=NV * NR).reshape(NV, NR)
e_big = np.bincount(key[has_ref & bigev], minlength=NV * NR).reshape(NV, NR)
e_small = np.bincount(key[has_ref & small], minlength=NV * NR).reshape(NV, NR)
np.savez_compressed(SCRATCH + '/renewal_big.npz', RE=RE, n=n, e_big=e_big, e_small=e_small, S_BIG=S_BIG)
def dev(nn, ee):
    with np.errstate(divide='ignore', invalid='ignore'):
        p = ee / nn; ll = np.where(ee > 0, ee * np.log(p), 0) + np.where(nn - ee > 0, (nn - ee) * np.log(1 - p), 0)
    return -2 * np.nansum(ll)
for lab, e in [('large (s > %.2f)' % S_BIG, e_big), ('small', e_small)]:
    D_null, D_cell, D_r = dev(n.sum(), e.sum()), dev(n.sum(1), e.sum(1)), dev(n, e)
    print('%s events: deviance explained by cell %.1f%%, by cell x r_big %.1f%%' % (lab, 100 * (D_null - D_cell) / D_null, 100 * (D_null - D_r) / D_null))
sel = np.array([(v % NB) >= 12 for v in range(NV)])
hb = e_big[sel].sum(0) / np.maximum(n[sel].sum(0), 1); hs = e_small[sel].sum(0) / np.maximum(n[sel].sum(0), 1)
print('pooled hazard vs r_big (bins of 0.02 from 0), tau cells >= 12, x1e-3:')
print('  large:', np.round(hb[:16] * 1e3, 2)); print('  small:', np.round(hs[:16] * 1e3, 2))
# at fixed cell: ratio of the large-event hazard at r_big in [0.1,0.3) to that at r_big in [0,0.04)
lo = slice(0, 2); hi = slice(5, 15)
with np.errstate(divide='ignore', invalid='ignore'):
    h_lo = e_big[:, lo].sum(1) / n[:, lo].sum(1); h_hi = e_big[:, hi].sum(1) / n[:, hi].sum(1)
ok = (e_big[:, lo].sum(1) >= 30) & (e_big[:, hi].sum(1) >= 30)
print('fixed-cell ratio  h_large(r_big 0.10-0.30) / h_large(r_big < 0.04):  median %.2f  (16-84%%: %.2f-%.2f, %d cells)' %
      (np.median(h_hi[ok] / h_lo[ok]), *np.percentile(h_hi[ok] / h_lo[ok], [16, 84]), ok.sum()))
json.dump(dict(RE=RE[:-1].tolist(), h_big=hb.tolist(), h_small=hs.tolist(), S_BIG=S_BIG,
               fixed_cell_ratio=float(np.median(h_hi[ok] / h_lo[ok])), ncells=int(ok.sum())), open(SCRATCH + '/renewal_big.json', 'w'), indent=1)
