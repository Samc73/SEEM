import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Is the fitted ceiling just the largest event in the cell?  Refit the
ceiling law after removing the largest 1, 5 and 1% of events; a ceiling that
is a property of the distribution should move by less than the removed
sample maximum does.  (Control: for TPL-generated synthetic catalogs the
fit pins to the sample maximum, c/smax = 1.002, README Section 4.)"""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist
SCRATCH = _os.path.join(_HERE, 'out')
ev = np.load(SCRATCH + '/model_events.npz'); d = np.load(SCRATCH + '/model_stats.npz')
mc = json.load(open(SCRATCH + '/mle_compare.json'))
S, EV = ev['S'].astype(float), ev['EV']; NB = int(d['NB'])
XMIN = 1e-4
rows = []
print('%-8s %6s   %8s %7s | %12s %12s %12s   (s_c relative to full-sample fit; smax ratio in brackets)' %
      ('cell', 'n', 's_c', 'c/smax', 'drop top-1', 'drop top-5', 'drop top-1%'))
for c in mc['cells']:
    if c['n'] < 3000: continue
    s = np.sort(S[EV == c['iu'] * NB + c['it']]); s = s[s >= XMIN]
    r0 = dist.fit('invpow4', s, XMIN); c0 = dist._upper('invpow4', r0['theta'], s.max())
    row = dict(iu=c['iu'], it=c['it'], n=len(s), sc=float(c0), c_over_smax=float(c0 / s.max()), drops={})
    line = '(%2d,%2d) %6d   %8.4f %7.3f |' % (c['iu'], c['it'], len(s), c0, c0 / s.max())
    for k in [1, 5, max(1, len(s) // 100)]:
        ss = s[:-k]
        r = dist.fit('invpow4', ss, XMIN, x0=r0['theta']); ck = dist._upper('invpow4', r['theta'], ss.max())
        row['drops'][k] = dict(sc_ratio=float(ck / c0), smax_ratio=float(ss.max() / s.max()), c_over_smax=float(ck / ss.max()))
        line += '  %.3f [%.3f]' % (ck / c0, ss.max() / s.max())
    print(line, flush=True); rows.append(row)
json.dump(rows, open(SCRATCH + '/ceiling_pinning.json', 'w'), indent=1)
