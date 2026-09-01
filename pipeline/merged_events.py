import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""cluster_test.py shows that consecutive strain steps with stress drops are
strongly clustered (hazard x20 on the step after an event).  If a run of
consecutive drop-steps is one physical avalanche split by the 1e-5 strain
resolution, the per-step event catalog under-counts large events.  Build a
merged catalog (run -> one event, size = summed drop, state = at run start)
and repeat the likelihood contest of Section 4 on the largest cells."""
import numpy as np, pandas as pd, json, sys
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist
SCRATCH = _os.path.join(_HERE, 'out')
u0c = -4.60751861; N = 49999; XMIN = 1e-4
d = np.load(SCRATCH + '/model_stats.npz'); ue, te, NB = d['ue'], d['te'], int(d['NB'])
mc = json.load(open(SCRATCH + '/mle_compare.json'))
df = pd.read_pickle(_os.path.join(_HERE, '..', 'df_clean.pkl')).sort_values(['index', 'strain_index'])
u = (df['pe'].to_numpy() - u0c).reshape(-1, N); tau = (df['stress'].to_numpy() / 1e4).reshape(-1, N); del df
dtau = np.diff(tau, axis=1); drop = dtau < 0
S_m, EV_m, L_m = [], [], []
for i in range(drop.shape[0]):
    x = np.diff(np.r_[0, drop[i].astype(int), 0]); st = np.nonzero(x == 1)[0]; en = np.nonzero(x == -1)[0]
    sz = np.add.reduceat(-dtau[i] * drop[i], st)[:len(st)] if len(st) else np.array([])
    # reduceat sums from st[k] to st[k+1]; restrict to the run by subtracting nothing (steps between runs are non-drops -> 0)
    iu = np.clip(np.searchsorted(ue, u[i, st], side='right') - 1, 0, NB - 1); it = np.clip(np.searchsorted(te, tau[i, st], side='right') - 1, 0, NB - 1)
    S_m.append(sz); EV_m.append(iu * NB + it); L_m.append(en - st)
S_m, EV_m, L_m = np.concatenate(S_m), np.concatenate(EV_m), np.concatenate(L_m)
print('per-step events: %d   merged avalanches: %d   (runs >= 2 steps: %d, longest %d steps)' %
      (drop.sum(), len(S_m), (L_m >= 2).sum(), L_m.max()))
np.savez_compressed(SCRATCH + '/merged_events.npz', S=S_m.astype(np.float32), EV=EV_m.astype(np.int32), L=L_m.astype(np.int16))
out = []
print('%-8s %6s %6s | %8s %8s %8s | %7s %7s %8s %8s' % ('cell', 'n_step', 'n_merg', 'dAIC tpl', 'dAIC lgn', 'dAIC pow', 'k', 'm', 's_c', 'c/smax'))
for c in mc['cells']:
    if c['n'] < 3000: continue
    s = S_m[EV_m == c['iu'] * NB + c['it']]; s = s[s >= XMIN]
    fits = {f: dist.fit(f, s, XMIN) for f in ['tpl', 'lognormal', 'powerlaw', 'invpow4']}
    a = {f: r['aic'] for f, r in fits.items()}; best = min(a, key=a.get)
    th = fits['invpow4']['theta']; sc = dist._upper('invpow4', th, s.max())
    print('(%2d,%2d) %6d %6d | %8.1f %8.1f %8.1f | %7.3f %7.2f %8.4f %8.3f   winner %s' %
          (c['iu'], c['it'], c['n'], len(s), a['tpl'] - a['invpow4'], a['lognormal'] - a['invpow4'], a['powerlaw'] - a['invpow4'],
           th[0], th[1], sc, sc / s.max(), best))
    out.append(dict(iu=c['iu'], it=c['it'], n_step=c['n'], n_merged=len(s), daic={f: a[f] - a['invpow4'] for f in a},
                    k=float(th[0]), m=float(th[1]), sc=float(sc), c_over_smax=float(sc / s.max()), winner=best,
                    sc_step=float(dist._upper('invpow4', c['fits']['invpow4']['theta'], 1.0) if False else np.nan)))
json.dump(out, open(SCRATCH + '/merged_events.json', 'w'), indent=1)
