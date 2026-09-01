import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Are events clustered in strain beyond what the state (u,tau) explains?
At fixed cell, compare the hazard on the step after an event with the hazard
on the step after a quiet step, and the sizes of the two kinds of event.
A Markov-in-(u,tau) model predicts equal hazards."""
import numpy as np, pandas as pd, json
SCRATCH = _os.path.join(_HERE, 'out')
u0c = -4.60751861; N = 49999
d = np.load(SCRATCH + '/model_stats.npz'); ue, te, NB = d['ue'], d['te'], int(d['NB']); NV = NB * NB
df = pd.read_pickle(_os.path.join(_HERE, '..', 'df_clean.pkl')).sort_values(['index', 'strain_index'])
u = (df['pe'].to_numpy() - u0c).reshape(-1, N); tau = (df['stress'].to_numpy() / 1e4).reshape(-1, N)
strain = df['strain'].to_numpy()[:N]; del df
dtau = np.diff(tau, axis=1); drop = dtau < 0; s = np.where(drop, -dtau, 0.0)
uu, tt = u[:, :-1], tau[:, :-1]
iu = np.clip(np.searchsorted(ue, uu, side='right') - 1, 0, NB - 1); it = np.clip(np.searchsorted(te, tt, side='right') - 1, 0, NB - 1)
vox = iu * NB + it
prev = np.zeros_like(drop); prev[:, 1:] = drop[:, :-1]          # was the previous step an event?
def acc(mask, w=None): return np.bincount(vox[mask], weights=(None if w is None else w[mask]), minlength=NV)
n_after_e, e_after_e = acc(prev), acc(prev & drop)
n_after_q, e_after_q = acc(~prev), acc(~prev & drop)
s_after_e, s_after_q = acc(prev & drop, s), acc(~prev & drop, s)
ok = (n_after_e >= 200) & (e_after_q >= 40)
h_e, h_q = e_after_e[ok] / n_after_e[ok], e_after_q[ok] / n_after_q[ok]
w = n_after_e[ok]
rat = np.exp(np.average(np.log(h_e / h_q), weights=w))
print('%d cells: hazard after an event / hazard after a quiet step = %.2f (16-84%%: %.2f-%.2f)' %
      (ok.sum(), rat, *np.percentile(h_e / h_q, [16, 84])))
ok2 = ok & (e_after_e >= 40)
ms_e = s_after_e[ok2] / e_after_e[ok2]; ms_q = s_after_q[ok2] / e_after_q[ok2]
srat = np.exp(np.average(np.log(ms_e / ms_q), weights=e_after_e[ok2]))
print('%d cells: mean size of an event following an event / following a quiet step = %.2f (16-84%%: %.2f-%.2f)' %
      (ok2.sum(), srat, *np.percentile(ms_e / ms_q, [16, 84])))
# run-length statistics: consecutive-event runs vs geometric expectation at the pooled hazard
runs = []
for i in range(drop.shape[0]):
    x = np.diff(np.r_[0, drop[i].astype(int), 0]); st = np.nonzero(x == 1)[0]; en = np.nonzero(x == -1)[0]; runs.append(en - st)
runs = np.concatenate(runs)
p = drop.mean()
print('fraction of events that are part of a run of >=2 consecutive event steps: data %.3f, Markov expectation %.3f' %
      ((runs[runs >= 2].sum() / runs.sum()), 1 - (1 - p) ** 1 * 1))
print('runs >= 3 steps: data %d, Markov expectation %.0f' % ((runs >= 3).sum(), len(runs) * p ** 2))
# the hazard-by-time-since-event curve (pooled over the flow regime cells)
lag_n = np.zeros(8); lag_e = np.zeros(8)
since = np.full(drop.shape, 999, int)
for k in range(1, drop.shape[1]):
    since[:, k] = np.where(drop[:, k - 1], 1, since[:, k - 1] + 1)
sel_cells = np.isin(vox, np.nonzero(ok)[0])
for L in range(1, 9):
    m = (since == L) & sel_cells; lag_n[L - 1] = m.sum(); lag_e[L - 1] = (m & drop).sum()
base = (drop & sel_cells & (since > 8)).sum() / (sel_cells & (since > 8)).sum()
print('hazard vs steps since last event (same cells; baseline = quiet for > 8 steps: %.4f):' % base)
for L in range(1, 9): print('  lag %d: %.4f  (x%.2f)' % (L, lag_e[L - 1] / lag_n[L - 1], lag_e[L - 1] / lag_n[L - 1] / base))
# per-cell quiet-baseline hazard (steps with > 8 quiet steps behind them) and the pooled lag multipliers,
# for a two-state hazard in the simulator: p(v, n) = p_base(v) * m(n), m(n>8) = 1
quiet = since > 8
n_q = acc(quiet); e_q = acc(quiet & drop)
with np.errstate(divide='ignore', invalid='ignore'):
    p_base = np.where(n_q >= 200, e_q / n_q, np.nan).reshape(NB, NB)
mult = np.r_[lag_e / lag_n / base, 1.0]                     # m(1..8), then 1 beyond
np.savez(SCRATCH + '/aftershock.npz', p_base=p_base, mult=mult, size_mult_lag1=srat)
json.dump(dict(hazard_ratio=float(rat), size_ratio=float(srat), ncells=int(ok.sum()),
               lag_hazard=(lag_e / lag_n).tolist(), baseline=float(base)), open(SCRATCH + '/cluster_test.json', 'w'), indent=1)
