import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Does the energy released per event, at fixed state AND fixed event size,
depend on preparation?  Per-voxel linear fits du = a + b*s for slow-3 and
fast-3 cooling rates, compared at the voxel's pooled mean size."""
import numpy as np, json, sys
WIN = sys.argv[1] if len(sys.argv) > 1 else 'all'      # all | early (gamma<0.3) | late (gamma>=0.3)
SCRATCH = _os.path.join(_HERE, 'out')
ev = np.load(SCRATCH + '/model_events.npz'); d = np.load(SCRATCH + '/model_stats.npz')
S, DU, EV, ECR = ev['S'].astype(float), ev['DUe'].astype(float), ev['EV'], ev['Ecr']
EU, ET = ev['Eu'].astype(float), ev['Et'].astype(float)
EG_ = ev['Eg'].astype(float)
keep = (EG_ < 0.3) if WIN == 'early' else ((EG_ >= 0.3) if WIN == 'late' else np.ones(len(S), bool))
S, DU, EV, ECR, EU, ET = S[keep], DU[keep], EV[keep], ECR[keep], EU[keep], ET[keep]
print('window:', WIN)
rates = d['rates']; slow = np.isin(ECR, rates[:3]); fast = np.isin(ECR, rates[-3:])
rows = []
for v in np.unique(EV):
    m = EV == v; ms, mf = m & slow, m & fast
    if ms.sum() < 40 or mf.sum() < 40: continue
    sbar = S[m].mean(); ubar, tbar = EU[m].mean(), ET[m].mean()
    def fit(mm):
        # within-voxel position (u, tau) as covariates so a sub-voxel offset
        # between preparations cannot masquerade as a coupling difference
        A = np.c_[np.ones(mm.sum()), S[mm], EU[mm] - ubar, ET[mm] - tbar]
        coef, *_ = np.linalg.lstsq(A, DU[mm], rcond=None)
        return coef
    cs, cf = fit(ms), fit(mf)
    rows.append(dict(v=int(v), n=int(min(ms.sum(), mf.sum())), sbar=float(sbar),
                     uncond=float(DU[ms].mean() - DU[mf].mean()),
                     at_sbar=float((cs[0] + cs[1] * sbar) - (cf[0] + cf[1] * sbar)),
                     b_slow=float(cs[1]), b_fast=float(cf[1]), a_slow=float(cs[0]), a_fast=float(cf[0]),
                     s_slow=float(S[ms].mean()), s_fast=float(S[mf].mean())))
w = np.array([r['n'] for r in rows], float)
def wm(k): return np.average([r[k] for r in rows], weights=w)
def wsd(k):
    x = np.array([r[k] for r in rows]); return np.sqrt(np.average((x - wm(k))**2, weights=w) / len(x))
print('%d voxels with >=40 events in both slow-3 and fast-3' % len(rows))
print('slow-3 minus fast-3 mean du per event (x1e-6):')
print('  unconditional         %+8.2f +- %.2f' % (1e6*wm('uncond'), 1e6*wsd('uncond')))
print('  at fixed s, u, tau (voxel means)  %+8.2f +- %.2f' % (1e6*wm('at_sbar'), 1e6*wsd('at_sbar')))
print('  mean size slow/fast   %.3f' % (wm('s_slow')/wm('s_fast')))
print('  slope b slow %.5f  fast %.5f   intercept a slow %.2e fast %.2e' % (wm('b_slow'), wm('b_fast'), wm('a_slow'), wm('a_fast')))
json.dump(dict(window=WIN, rows=rows, uncond=wm('uncond'), at_sbar=wm('at_sbar'), se_uncond=wsd('uncond'), se_at_sbar=wsd('at_sbar')),
          open(SCRATCH + ('/coupling_memory.json' if WIN == 'all' else '/coupling_memory_%s.json' % WIN), 'w'), indent=1)
