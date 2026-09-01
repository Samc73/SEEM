import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Fit the invpow model's fields: global (a, eps), per-voxel ceiling s_c.

Then: rank-1 factorization of ln s_c, SR on the two factors, and a
preparation-memory test (ceiling refit on slow- vs fast-cooled subsets).
Also fits the per-voxel TPL s* (global kappa) as the rival arm.
"""
import numpy as np, sys, json
from scipy.optimize import minimize_scalar
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist
from symreg import enumerate_search, make_checker

SCRATCH = _os.path.join(_HERE, 'out')
d = np.load(SCRATCH + '/model_stats.npz')
ev = np.load(SCRATCH + '/model_events.npz')
NB = int(d['NB'])
S, EV, Ecr = ev['S'].astype(float), ev['EV'], ev['Ecr']
XMIN = 1e-6
rates = np.array(sorted(np.unique(Ecr)))

with open(SCRATCH + '/mle_compare.json') as f:
    cells = [(c['iu'], c['it'], c['n']) for c in json.load(f)['cells']]

# ---- global shape from full-range per-cell fits ----
glob = []
for iu, it, n in cells:
    s = S[EV == iu * NB + it]
    r3 = dist.fit('invpow', s, XMIN)
    rt = dist.fit('tpl', s, XMIN)
    glob.append(dict(iu=iu, it=it, n=n, a=float(r3['theta'][0]),
                     eps=float(np.exp(r3['theta'][1])),
                     dnll=float(rt['nll'] - r3['nll']),
                     kappa=float(rt['theta'][0])))
    print('cell (%d,%d) a=%.3f eps=%.1e  kappa=%.3f  d_nll/ev(tpl-invpow)=%+.4f'
          % (iu, it, glob[-1]['a'], glob[-1]['eps'], glob[-1]['kappa'],
             glob[-1]['dnll']), flush=True)
big = [g for g in glob if g['n'] >= 3000]
A_G = float(np.median([g['a'] for g in big]))
EPS_G = float(np.median([g['eps'] for g in big]))
KAP_G = float(np.median([g['kappa'] for g in big]))
print('GLOBAL a=%.3f eps=%.2e kappa=%.3f' % (A_G, EPS_G, KAP_G), flush=True)


def fit_sc(s, a=A_G, eps=EPS_G):
    """1-D MLE of the ceiling with global shape."""
    s = s[s >= XMIN]
    if len(s) < 40:
        return np.nan
    smax = s.max()

    def f(lc):
        return dist.nll('invpow', s, [a, np.log(eps), lc], XMIN, smax, npts=1500)

    r = minimize_scalar(f, bounds=(np.log(1e-4), np.log(50.0)), method='bounded',
                        options=dict(xatol=1e-4))
    return float(smax * (1 + np.exp(r.x)))


def fit_sstar(s, kappa=KAP_G):
    s = s[s >= XMIN]
    if len(s) < 40:
        return np.nan

    def f(ls):
        return dist.nll('tpl', s, [kappa, ls], XMIN, s.max(), npts=1500)

    r = minimize_scalar(f, bounds=(np.log(1e-5), np.log(20.0)), method='bounded',
                        options=dict(xatol=1e-4))
    return float(np.exp(r.x))


sc = np.full((NB, NB), np.nan)
sstar = np.full((NB, NB), np.nan)
nev = np.zeros((NB, NB), int)
for v in np.unique(EV):
    s = S[EV == v]
    iu, it = v // NB, v % NB
    nev[iu, it] = len(s)
    sc[iu, it] = fit_sc(s)
    sstar[iu, it] = fit_sstar(s)
print('sc fitted on %d voxels' % np.isfinite(sc).sum(), flush=True)

# ---- bootstrap SE on a spread of voxels ----
rng = np.random.default_rng(0)
boot = []
vs = [v for v in np.unique(EV) if (EV == v).sum() >= 300][::4]
for v in vs[:24]:
    s = S[EV == v]
    vals = [np.log(fit_sc(s[rng.integers(0, len(s), len(s))])) for _ in range(30)]
    boot.append(dict(v=int(v), n=len(s), se=float(np.std(vals))))
print('median bootstrap SE on ln s_c: %.3f' %
      np.median([b['se'] for b in boot]), flush=True)

# ---- rank-1 factorization of ln s_c on the inner grid ----
L = np.log(sc[1:21, 1:21])
W = np.where(np.isfinite(L), np.sqrt(np.clip(nev[1:21, 1:21], 0, None)), 0.0)
L0 = np.where(np.isfinite(L), L, 0.0)
au = np.zeros(20)
bt = np.zeros(20)
for _ in range(200):
    for i in range(20):
        w = W[i]
        if w.sum() > 0:
            au[i] = np.sum(w * (L0[i] - bt)) / np.maximum(np.sum(w), 1e-9)
    for j in range(20):
        w = W[:, j]
        if w.sum() > 0:
            bt[j] = np.sum(w * (L0[:, j] - au)) / np.maximum(np.sum(w), 1e-9)
m = W > 0
resid = (au[:, None] + bt[None, :] - L0)[m]
tot = (L0 - np.average(L0[m], weights=W[m]))[m]
r1 = 1 - np.sum(W[m] * resid ** 2) / np.sum(W[m] * tot ** 2)
print('rank-1 fraction of weighted ln s_c variance: %.3f' % r1, flush=True)

# ---- SR on the two factors ----
uc = 0.5 * (d['ue'][1:21] + d['ue'][2:22])
tc = 0.5 * (d['te'][1:21] + d['te'][2:22])
sr_out = {}
for name, xv, yv, wv in (('sc_u', uc, au, W.sum(1)), ('sc_t', tc, bt, W.sum(0))):
    k = wv > 0
    sig = 1 / np.sqrt(wv[k])
    chk = make_checker(np.linspace(xv[k].min(), xv[k].max(), 80))
    par, res = enumerate_search(xv[k], yv[k], sigma=sig, max_complexity=8,
                                max_consts=2, checker=chk, n_restarts=4,
                                verbose=False)
    sr_out[name] = [dict(complexity=p['complexity'], rmse=p['rmse'],
                         string=p['string']) for p in par]
    print(name, 'front:', flush=True)
    for p in par:
        print('   C=%2d rmse=%.4f  %s' % (p['complexity'], p['rmse'], p['string']),
              flush=True)

# ---- preparation memory in the ceiling ----
slow = np.isin(Ecr, rates[:3])
fast = np.isin(Ecr, rates[-3:])
mem = []
for v in np.unique(EV):
    m1, m2 = (EV == v) & slow, (EV == v) & fast
    if m1.sum() >= 150 and m2.sum() >= 150:
        c1, c2 = fit_sc(S[m1]), fit_sc(S[m2])
        if np.isfinite(c1) and np.isfinite(c2):
            mem.append(dict(v=int(v), n1=int(m1.sum()), n2=int(m2.sum()),
                            sc_slow=c1, sc_fast=c2))
ratios = np.array([z['sc_slow'] / z['sc_fast'] for z in mem])
# within-slow control: even/odd trajectory halves
tid = np.load(SCRATCH + '/event_tids.npz')['tid_drop']
ctrl = []
for v in [z['v'] for z in mem]:
    m1 = (EV == v) & slow & (tid % 2 == 0)
    m2 = (EV == v) & slow & (tid % 2 == 1)
    if m1.sum() >= 75 and m2.sum() >= 75:
        c1, c2 = fit_sc(S[m1]), fit_sc(S[m2])
        if np.isfinite(c1) and np.isfinite(c2):
            ctrl.append(c1 / c2)
print('ceiling memory: %d voxels, median sc_slow/sc_fast=%.3f (16-84%%: %.2f-%.2f)'
      % (len(mem), np.median(ratios), *np.quantile(ratios, [0.16, 0.84])), flush=True)
if ctrl:
    print('control (within-slow halves): median |log ratio| %.3f vs memory %.3f'
          % (np.median(np.abs(np.log(ctrl))),
             np.median(np.abs(np.log(ratios)))), flush=True)

np.savez(SCRATCH + '/scale_fields.npz', sc=sc, sstar=sstar, nev=nev,
         A_G=A_G, EPS_G=EPS_G, KAP_G=KAP_G, au=au, bt=bt, uc=uc, tc=tc, r1=r1)
with open(SCRATCH + '/scale_report.json', 'w') as f:
    json.dump(dict(glob=glob, A_G=A_G, EPS_G=EPS_G, KAP_G=KAP_G,
                   boot=boot, rank1=float(r1), sr=sr_out,
                   mem=mem, mem_median=float(np.median(ratios)),
                   ctrl_absmed=float(np.median(np.abs(np.log(ctrl)))) if ctrl else None),
              f, indent=1)
print('DONE')
