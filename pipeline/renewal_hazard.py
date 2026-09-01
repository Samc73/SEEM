import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""The hazard as a function of the stress reloaded since the last event,
r = tau - tau(just after the last drop), at fixed (u,tau) cell; and the
same with r normalised by the size of the last event.  Both tables go to
the simulator (sim_renewal.py).  Also reports how much of the hazard's
variance across steps the reloading coordinate explains beyond the cell."""
import numpy as np, pandas as pd, json
SCRATCH = _os.path.join(_HERE, 'out')
u0c = -4.60751861; N = 49999; DG = 1e-5
d = np.load(SCRATCH + '/model_stats.npz'); ue, te, NB = d['ue'], d['te'], int(d['NB']); NV = NB * NB
df = pd.read_pickle(_os.path.join(_HERE, '..', 'df_clean.pkl')).sort_values(['index', 'strain_index'])
u = (df['pe'].to_numpy() - u0c).reshape(-1, N); tau = (df['stress'].to_numpy() / 1e4).reshape(-1, N); del df
ntr = tau.shape[0]
dtau = np.diff(tau, axis=1); drop = dtau < 0
idx = np.arange(N - 1)[None, :]
last = np.maximum.accumulate(np.where(drop, idx, -1), axis=1)            # index of the last drop step at or before this step
prev_last = np.concatenate([np.full((ntr, 1), -1), last[:, :-1]], axis=1)  # last drop strictly before this step
row = np.arange(ntr)[:, None]
tau_after = np.where(prev_last >= 0, tau[row, np.maximum(prev_last, 0) + 1], tau[:, :1])   # stress right after that drop
s_last = np.where(prev_last >= 0, -dtau[row, np.maximum(prev_last, 0)], np.nan)
r = tau[:, :-1] - tau_after                                                  # stress reloaded since
uu, tt = u[:, :-1], tau[:, :-1]
iu = np.clip(np.searchsorted(ue, uu, side='right') - 1, 0, NB - 1); it = np.clip(np.searchsorted(te, tt, side='right') - 1, 0, NB - 1)
vox = (iu * NB + it).ravel(); drop = drop.ravel(); r = r.ravel(); s_last = s_last.ravel()
print('r: min %.4f  median %.4f  99%% %.3f' % (r.min(), np.median(r), np.percentile(r, 99)))
# absolute reloading, 0.01 bins to 0.6 + overflow
RE = np.r_[np.arange(0, 0.6001, 0.01), np.inf]; rb = np.clip(np.searchsorted(RE, r, side='right') - 1, 0, len(RE) - 2)
NR = len(RE) - 1
n_abs = np.bincount(vox * NR + rb, minlength=NV * NR).reshape(NV, NR); e_abs = np.bincount((vox * NR + rb)[drop], minlength=NV * NR).reshape(NV, NR)
# relative reloading r / s_last, log bins
QE = np.r_[0, np.geomspace(0.01, 100, 41), np.inf]; ok = np.isfinite(s_last)
q = np.where(ok, r / np.where(ok, s_last, 1), 0); qb = np.clip(np.searchsorted(QE, q, side='right') - 1, 0, len(QE) - 2)
NQ = len(QE) - 1
n_rel = np.bincount((vox * NQ + qb)[ok], minlength=NV * NQ).reshape(NV, NQ); e_rel = np.bincount((vox * NQ + qb)[ok & drop], minlength=NV * NQ).reshape(NV, NQ)
# large events only: does the hazard of a *large* drop depend on reloading even if the total hazard does not?
S_step = np.where(drop, -np.diff(tau, axis=1).ravel(), 0.0)
big = {}
for thr in [0.01, 0.03, 0.1]:
    big[thr] = np.bincount((vox * NR + rb)[drop & (S_step > thr)], minlength=NV * NR).reshape(NV, NR)
np.savez_compressed(SCRATCH + '/renewal_hazard.npz', RE=RE, n_abs=n_abs, e_abs=e_abs, QE=QE, n_rel=n_rel, e_rel=e_rel,
                    e_big01=big[0.01], e_big03=big[0.03], e_big10=big[0.1])
sel_flow = np.array([(v % NB) >= 12 for v in range(NV)])
print('pooled hazard of LARGE events (tau cells >= 12) vs reloaded stress r (bins of 0.01):')
for thr in [0.01, 0.03, 0.1]:
    hb = big[thr][sel_flow].sum(0) / np.maximum(n_abs[sel_flow].sum(0), 1)
    print('  s > %.2f: %s' % (thr, np.round(hb[:15] * 1e3, 2)), ' (x1e-3)')
# how much does each coordinate explain?  deviance of the Bernoulli event indicator: cell only vs cell x r-bin vs cell x q-bin
def dev(n, e):
    with np.errstate(divide='ignore', invalid='ignore'):
        p = e / n; ll = np.where(e > 0, e * np.log(p), 0) + np.where(n - e > 0, (n - e) * np.log(1 - p), 0)
    return -2 * np.nansum(ll)
n_cell = n_abs.sum(1); e_cell = e_abs.sum(1)
D0 = dev(n_cell, e_cell); Da = dev(n_abs, e_abs); Dq = dev(n_rel[:, :], e_rel[:, :]) + dev(n_abs.sum(1) - n_rel.sum(1), e_abs.sum(1) - e_rel.sum(1))
D_null = dev(n_cell.sum(), e_cell.sum())
print('Bernoulli deviance of the event indicator (lower = better):')
print('  no state          %.0f' % D_null)
print('  cell only         %.0f   (explains %.1f%% of the null)' % (D0, 100 * (D_null - D0) / D_null))
print('  cell x r (abs)    %.0f   (%.1f%%)' % (Da, 100 * (D_null - Da) / D_null))
print('  cell x r/s_last   %.0f   (%.1f%%)' % (Dq, 100 * (D_null - Dq) / D_null))
# pooled hazard vs r for the flow-regime cells (tau > 1), for the README
sel = np.array([(v % NB) >= 12 for v in range(NV)])
h_abs = e_abs[sel].sum(0) / np.maximum(n_abs[sel].sum(0), 1); h_rel = e_rel[sel].sum(0) / np.maximum(n_rel[sel].sum(0), 1)
print('pooled hazard (tau cells >= 12) vs reloaded stress r:', np.round(h_abs[:12], 4))
print('pooled hazard vs r/s_last (bins from 0.01):', np.round(h_rel[1:41:4], 4))
json.dump(dict(D_null=D_null, D_cell=D0, D_abs=Da, D_rel=Dq, RE=RE[:-1].tolist(), h_abs=h_abs.tolist(),
               QE=QE[1:-1].tolist(), h_rel=h_rel.tolist()), open(SCRATCH + '/renewal_hazard.json', 'w'), indent=1)
