import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Does the ceiling memory fade with strain, and is it uniform over the
state plane?  Mean event size slow-3 / fast-3 at fixed voxel, split by the
strain at which the event occurred, and the memory exponent beta fitted
separately in bands of u and of tau."""
import numpy as np, json
SCRATCH = _os.path.join(_HERE, 'out')
ev = np.load(SCRATCH + '/model_events.npz'); d = np.load(SCRATCH + '/model_stats.npz')
S, EV, EG, ECR = ev['S'].astype(float), ev['EV'], ev['Eg'].astype(float), ev['Ecr']
NB = int(d['NB']); rates = d['rates']; R = len(rates)
ridx = np.searchsorted(rates, ECR); lr = np.log(rates / rates[R // 2])
slow, fast = ridx <= 2, ridx >= R - 3

def ratio(sel, nmin=40):
    out = []
    for v in np.unique(EV[sel]):
        m = sel & (EV == v); a, b = m & slow, m & fast
        if a.sum() >= nmin and b.sum() >= nmin:
            out.append((S[a].mean() / S[b].mean(), min(a.sum(), b.sum())))
    if not out: return np.nan, 0, [np.nan, np.nan]
    r = np.array(out); return float(np.exp(np.average(np.log(r[:, 0]), weights=r[:, 1]))), len(r), [float(x) for x in np.percentile(r[:, 0], [16, 84])]

print('mean-size ratio slow-3/fast-3 at fixed voxel, by strain window (weighted geometric mean over voxels):')
fade = {}
for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]:
    r, n, pr = ratio((EG >= lo) & (EG < hi))
    fade['%.1f-%.1f' % (lo, hi)] = dict(ratio=r, nvox=n, p1684=pr)
    print('  gamma %.1f-%.1f   %.3f  [%.2f, %.2f]  (%d voxels)' % (lo, hi, r, pr[0], pr[1], n))

def beta(sel, nmin=40):
    num = den = 0.0; nv = 0
    for v in np.unique(EV[sel]):
        m = sel & (EV == v)
        cnt = np.bincount(ridx[m], minlength=R); ok = cnt >= nmin
        if ok.sum() < 4: continue
        y = np.array([np.log(S[m & (ridx == r)].mean()) if ok[r] else np.nan for r in range(R)])
        w = cnt[ok].astype(float); xx = lr[ok]; yy = y[ok]
        xm, ym = np.average(xx, weights=w), np.average(yy, weights=w)
        num += np.sum(w * (xx - xm) * (yy - ym)); den += np.sum(w * (xx - xm) ** 2); nv += 1
    return (num / den if den else np.nan), nv

ue, te = d['ue'], d['te']; iu, it = EV // NB, EV % NB
print('\nmemory exponent beta by u band (events binned by voxel row):')
bands = {}
for lo, hi in [(1, 8), (8, 12), (12, 15), (15, 18), (18, 21)]:
    b, nv = beta((iu >= lo) & (iu < hi))
    bands['u %.4f-%.4f' % (ue[lo], ue[hi])] = dict(beta=b, nvox=nv)
    print('  u in [%.4f, %.4f)  beta = %+.3f  (%d voxels)' % (ue[lo], ue[hi], b, nv))
print('memory exponent beta by tau band:')
for lo, hi in [(1, 8), (8, 12), (12, 16), (16, 21)]:
    b, nv = beta((it >= lo) & (it < hi))
    bands['tau %.2f-%.2f' % (te[lo], te[hi])] = dict(beta=b, nvox=nv)
    print('  tau in [%.2f, %.2f)  beta = %+.3f  (%d voxels)' % (te[lo], te[hi], b, nv))
json.dump(dict(fade=fade, bands=bands), open(SCRATCH + '/memory_fading.json', 'w'), indent=1)

# ---- same-voxel early vs late: fading (hidden variable relaxes) or state dependence (beta(u))? ----
early, late = EG < 0.3, EG >= 0.3
rows = []
for v in np.unique(EV):
    m = EV == v
    cnts = [(m & early & slow).sum(), (m & early & fast).sum(), (m & late & slow).sum(), (m & late & fast).sum()]
    if min(cnts) >= 30:
        re = S[m & early & slow].mean() / S[m & early & fast].mean()
        rl = S[m & late & slow].mean() / S[m & late & fast].mean()
        rows.append((v, re, rl, min(cnts)))
rows = np.array(rows)
print('\nsame-voxel test (%d voxels with >=30 events in all four early/late x slow/fast groups):' % len(rows))
if len(rows):
    w = rows[:, 3]
    ge = np.exp(np.average(np.log(rows[:, 1]), weights=w)); gl = np.exp(np.average(np.log(rows[:, 2]), weights=w))
    print('  slow/fast mean-size ratio  early (gamma<0.3): %.3f   late (gamma>=0.3): %.3f   (same voxels)' % (ge, gl))
    for v, re, rl, n in rows:
        print('    voxel (%2d,%2d) n>=%4d   early %.2f   late %.2f' % (v // NB, v % NB, n, re, rl))
    json.dump(dict(fade=fade, bands=bands, same_voxel=dict(early=float(ge), late=float(gl), rows=rows.tolist())),
              open(SCRATCH + '/memory_fading.json', 'w'), indent=1)
