import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Does the ceiling memory relax with strain, or is it a property of the
state?  Fixed-effects regression of the per-(cell, rate, strain-window) mean
event size on ln(cooling rate), with one fixed effect per (cell, window) so
that any strain dependence of the size itself is absorbed:

    ln <s>(v, r, g) = a_{v,g} + beta_g * ln(rate_r / rate_mid)

beta_g per strain window uses all nine rates and every cell (far more power
than slow/fast pairs).  Then the same with beta indexed by (u band, window)
to see whether the strain trend survives at fixed u."""
import numpy as np, json
SCRATCH = _os.path.join(_HERE, 'out')
ev = np.load(SCRATCH + '/model_events.npz'); d = np.load(SCRATCH + '/model_stats.npz')
S, EV, EG, ECR = ev['S'].astype(float), ev['EV'], ev['Eg'].astype(float), ev['Ecr']
NB = int(d['NB']); rates = d['rates']; R = len(rates); ue = d['ue']
ridx = np.searchsorted(rates, ECR); lr = np.log(rates / rates[R // 2])
NMIN = 30
gedges = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
gbin = np.clip(np.searchsorted(gedges, EG, side='right') - 1, -1, len(gedges) - 2)
ubands = [(1, 12), (12, 16), (16, 19), (19, 22)]
uband_of = np.full(NB, -1)
for b, (lo, hi) in enumerate(ubands): uband_of[lo:hi] = b
# group sums: key = (v, r, g)
G = len(gedges) - 1
ok = gbin >= 0
key = (EV[ok].astype(np.int64) * R + ridx[ok]) * G + gbin[ok]
n = np.bincount(key, minlength=NB * NB * R * G).reshape(NB * NB, R, G)
sm = np.bincount(key, weights=S[ok], minlength=NB * NB * R * G).reshape(NB * NB, R, G)
with np.errstate(divide='ignore', invalid='ignore'):
    y = np.log(sm / n)

def beta_fe(sel_v, sel_g):
    """weighted FE slope pooled over groups (v,g) with v in sel_v, g in sel_g"""
    num = den = 0.0; ngrp = 0; resid = []
    for v in sel_v:
        for g in sel_g:
            o = n[v, :, g] >= NMIN
            if o.sum() < 3: continue
            w = n[v, o, g].astype(float); xx = lr[o]; yy = y[v, o, g]
            xm, ym = np.average(xx, weights=w), np.average(yy, weights=w)
            num += np.sum(w * (xx - xm) * (yy - ym)); den += np.sum(w * (xx - xm) ** 2); ngrp += 1
            resid.append((w, xx - xm, yy - ym))
    if den == 0: return np.nan, np.nan, 0
    b = num / den
    s2 = sum(np.sum((w * dx * (dy - b * dx)) ** 2) for w, dx, dy in resid)
    return float(b), float(np.sqrt(s2) / den), ngrp

allv = range(NB * NB)
print('memory exponent beta by strain window (fixed effect per cell x window; all 9 rates):')
out = dict(by_window={}, by_uband={}, table={})
for g in range(G):
    b, se, ng = beta_fe(allv, [g])
    out['by_window']['%.1f-%.1f' % (gedges[g], gedges[g + 1])] = dict(beta=b, se=se, ngroups=ng)
    print('  gamma %.1f-%.1f   beta = %+.3f +- %.3f   (%d cell-groups)' % (gedges[g], gedges[g + 1], b, se, ng))
print('by u band (fixed effect per cell x window):')
for bnd, (lo, hi) in enumerate(ubands):
    vs = [v for v in allv if uband_of[v // NB] == bnd]
    b, se, ng = beta_fe(vs, range(G))
    out['by_uband']['u %.4f-%.4f' % (ue[lo], ue[hi])] = dict(beta=b, se=se, ngroups=ng)
    print('  u %.4f-%.4f   beta = %+.3f +- %.3f   (%d)' % (ue[lo], ue[hi], b, se, ng))
print('table beta(u band, strain window):')
print('  %-18s' % 'u band \\ gamma' + ''.join('%14s' % ('%.1f-%.1f' % (gedges[g], gedges[g + 1])) for g in range(G)))
for bnd, (lo, hi) in enumerate(ubands):
    vs = [v for v in allv if uband_of[v // NB] == bnd]
    row = '  %-18s' % ('%.4f-%.4f' % (ue[lo], ue[hi]))
    for g in range(G):
        b, se, ng = beta_fe(vs, [g])
        out['table']['u%d_g%d' % (bnd, g)] = dict(beta=b, se=se, ngroups=ng)
        row += ('%+7.3f±%.3f(%2d)' % (b, se, ng)) if ng else '%14s' % '.'
    print(row)
json.dump(out, open(SCRATCH + '/memory_2d.json', 'w'), indent=1)
