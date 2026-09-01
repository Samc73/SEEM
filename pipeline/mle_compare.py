import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""MLE adjudication of candidate size-distribution families, per cell.

AIC + trajectory-blocked 2-fold CV + KS distance, xmin sensitivity scan.
"""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist

SCRATCH = _os.path.join(_HERE, 'out')
d = np.load(SCRATCH + '/model_stats.npz')
ev = np.load(SCRATCH + '/model_events.npz')
tid = np.load(SCRATCH + '/event_tids.npz')['tid_drop']
NB = int(d['NB'])
S, EV = ev['S'], ev['EV']
XMIN = 1e-4
FAMS = ['tpl', 'tpl_beta', 'lognormal', 'powerlaw', 'sqrtlog', 'invpow', 'invpow4']

with open(SCRATCH + '/sr_dist_results.json') as f:
    sr = json.load(f)
cells = [(r['iu'], r['it']) for r in sr if r['tag'] == 'main']


def ks(name, th, s, xmin, smax):
    s = np.sort(s[s >= xmin])
    hi = dist._upper(name, th, smax)
    grid = dist._grid(xmin, hi if hi is not None else max(s.max(), xmin * 1.01),
                      hi is not None, 4000)
    with np.errstate(all='ignore'):
        lf = dist._lnf(name, grid, th, smax)
    w = np.exp(lf - np.nanmax(lf))
    w[~np.isfinite(w)] = 0
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(grid))))
    cdf /= cdf[-1]
    F = np.interp(s, grid, cdf)
    return float(np.abs(F - (np.arange(1, len(s) + 1) - 0.5) / len(s)).max())


out = dict(cells=[], xmin_scan=[])
for iu, it in cells:
    m = EV == iu * NB + it
    s, t = S[m].astype(float), tid[m]
    smax = s[s >= XMIN].max()
    row = dict(iu=iu, it=it, n=int((s >= XMIN).sum()), fits={})
    for fam in FAMS:
        r = dist.fit(fam, s, XMIN)
        cv = dist.cv_nll(fam, s, t, XMIN, smax=smax)
        row['fits'][fam] = dict(theta=list(map(float, r['theta'])),
                                nll=r['nll'], aic=r['aic'], cv=float(cv),
                                ks=ks(fam, r['theta'], s, XMIN, smax))
    aics = {f: row['fits'][f]['aic'] for f in FAMS}
    b = min(aics, key=aics.get)
    row['winner_aic'] = b
    row['daic'] = {f: aics[f] - aics[b] for f in FAMS}
    cvs = {f: row['fits'][f]['cv'] for f in FAMS}
    row['winner_cv'] = min(cvs, key=cvs.get)
    out['cells'].append(row)
    print('cell (%d,%d) n=%d  AIC winner=%s  CV winner=%s  dAIC=%s' %
          (iu, it, row['n'], b, row['winner_cv'],
           {f: round(row['daic'][f], 1) for f in FAMS}), flush=True)

# xmin sensitivity on the two largest cells
for iu, it in cells[:2]:
    m = EV == iu * NB + it
    s = S[m].astype(float)
    for xm in [3e-5, 1e-4, 3e-4, 1e-3]:
        rt = dist.fit('tpl', s, xm)
        rq = dist.fit('sqrtlog', s, xm)
        rec = dict(iu=iu, it=it, xmin=xm, n=int((s >= xm).sum()),
                   tpl_kappa=float(rt['theta'][0]),
                   tpl_sstar=float(np.exp(rt['theta'][1])),
                   sqrtlog_a=float(abs(rq['theta'][0])),
                   sqrtlog_c=float(s[s >= xm].max() * (1 + np.exp(rq['theta'][1]))),
                   d_nll_per_ev=float(rt['nll'] - rq['nll']))
        out['xmin_scan'].append(rec)
        print('xmin=%.0e n=%d  kappa=%.3f s*=%.3f | a=%.2f c=%.3f | tpl-sqrtlog nll/ev=%+.4f'
              % (xm, rec['n'], rec['tpl_kappa'], rec['tpl_sstar'],
                 rec['sqrtlog_a'], rec['sqrtlog_c'], rec['d_nll_per_ev']), flush=True)

# full-range test: can invpow describe ALL sizes (no xmin) via its rounding eps?
out['fullrange'] = []
for iu, it in cells[:3]:
    m = EV == iu * NB + it
    s = S[m].astype(float)
    for fam in ('invpow', 'tpl', 'lognormal'):
        r = dist.fit(fam, s, 1e-6)
        rec = dict(iu=iu, it=it, fam=fam, n=int((s >= 1e-6).sum()),
                   theta=list(map(float, r['theta'])), nll=r['nll'], aic=r['aic'],
                   ks=ks(fam, r['theta'], s[s >= 1e-6], 1e-6, s[s >= 1e-6].max()))
        out['fullrange'].append(rec)
        print('fullrange (%d,%d) %s  nll/ev=%.4f aic=%.0f ks=%.3f theta=%s' %
              (iu, it, fam, r['nll'], r['aic'], rec['ks'],
               np.round(r['theta'], 3)), flush=True)

with open(SCRATCH + '/mle_compare.json', 'w') as f:
    json.dump(out, f, indent=1)
print('DONE')
