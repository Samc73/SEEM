import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""The same channel-by-channel memory test as memory_where.py, split into
early (gamma < 0.3) and late (gamma >= 0.3) strain.  Question: when the size
memory has faded (memory_2d.py), what keeps the preparations apart?"""
import numpy as np, json, warnings
warnings.filterwarnings('ignore')
SCRATCH = _os.path.join(_HERE, 'out')
m = np.load(SCRATCH + '/memory_stats.npz'); d = np.load(SCRATCH + '/model_stats.npz')
NB = int(d['NB']); NV = NB * NB; cr, rates = m['cr'], m['rates']; R = len(rates)
lr = np.log(rates / rates[R // 2]); DG = 1e-5
KEYS = ['n_all', 'sum_du', 'sum_dtau', 'n_ne', 'sum_du_ne', 'sum_dtau_ne', 'n_drop', 'sum_s', 'sum_du_drop', 'sum_lns', 'n_age', 'sum_du_age']

def group_sums(window):
    out = {}
    for k in KEYS:
        a = m[k].astype(np.float64); late = m[k + '_late'].astype(np.float64)
        a = late if window == 'late' else a - late
        out[k] = np.stack([a[cr == r].sum(0) for r in rates])
    return out

def channels(g):
    with np.errstate(divide='ignore', invalid='ignore'):
        mu2 = g['sum_dtau_ne'].sum(0) / g['n_ne'].sum(0) / DG
        return {
            'hazard p_drop': (g['n_drop'] / g['n_all'], g['n_all'], True),
            'mean size <s>': (g['sum_s'] / g['n_drop'], g['n_drop'], True),
            'plastic rate q': (1 - g['sum_dtau'] / g['n_all'] / (mu2 * DG), g['n_all'], True),
            'elastic du/step': (g['sum_du_ne'] / g['n_ne'], g['n_ne'], False),
            'drop du/event': (g['sum_du_drop'] / g['n_drop'], g['n_drop'], False),
            'aging du/step': (-g['sum_du_age'] / g['n_all'], g['n_all'], False),
            'total <du>/step': (g['sum_du'] / g['n_all'], g['n_all'], False),
        }

def beta_fit(F, W, log, nmin=40):
    ok = (W >= nmin) & np.isfinite(F) & ((F > 0) if log else True)
    keep = ok.sum(0) >= 4; y = np.log(F) if log else F
    num = den = 0.0; parts = []
    for v in np.nonzero(keep)[0]:
        o = ok[:, v]; w = W[o, v]; xx = lr[o]; yy = y[o, v]
        xm, ym = np.average(xx, weights=w), np.average(yy, weights=w)
        num += np.sum(w * (xx - xm) * (yy - ym)); den += np.sum(w * (xx - xm) ** 2); parts.append((w, xx - xm, yy - ym))
    if den == 0: return np.nan, np.nan, 0
    b = num / den; s2 = sum(np.sum((w * dx * (dy - b * dx)) ** 2) for w, dx, dy in parts)
    return float(b), float(np.sqrt(s2) / den), int(keep.sum())

def slow_fast(F, W, log, nmin=40):
    a = np.nansum(F[:3] * W[:3], 0) / W[:3].sum(0); b = np.nansum(F[-3:] * W[-3:], 0) / W[-3:].sum(0)
    ok = (W[:3].sum(0) >= nmin) & (W[-3:].sum(0) >= nmin) & np.isfinite(a) & np.isfinite(b)
    if log:
        ok &= (a > 0) & (b > 0); return float(np.median(a[ok] / b[ok])), int(ok.sum())
    w = np.minimum(W[:3].sum(0), W[-3:].sum(0))[ok]
    return float(np.average(a[ok] - b[ok], weights=w)), int(ok.sum())

report = {}
for window in ['early', 'late']:
    G = channels(group_sums(window))
    print('\n== %s strain (%s) ==' % (window, 'gamma < 0.3' if window == 'early' else 'gamma >= 0.3'))
    print('%-18s %9s %8s %5s   %s' % ('channel', 'beta', 'SE', 'nvox', 'slow3 vs fast3 (ratio, or diff x1e-6 for du channels)'))
    report[window] = {}
    for name, (F, W, log) in G.items():
        b, se, nv = beta_fit(F, W, log); sf, n = slow_fast(F, W, log)
        # typical magnitude for the du channels
        typ = float(np.nanmedian(np.abs(np.nansum(F * W, 0) / W.sum(0))[W.sum(0) >= 200])) if not log else None
        print('%-18s %+9.4f %8.4f %5d   %s' % (name, b, se, nv,
              ('ratio %.3f (%d cells)' % (sf, n)) if log else ('diff %+.3f  [typical |value| %.3f]  (%d cells)' % (sf * 1e6, typ * 1e6, n))))
        report[window][name] = dict(beta=b, se=se, nvox=nv, slow_fast=sf, n=n, typical=typ)
json.dump(report, open(SCRATCH + '/memory_late.json', 'w'), indent=1)
