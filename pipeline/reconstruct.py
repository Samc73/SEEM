import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Rebuild the plastic-rate field q from the fitted size law and compare with
the measured q. Then factorize both and run SR on both sets of factors.

Identity: q = 1 - <dtau>/(2 mu dg). Replacing the measured mean drop size
with the model's mean gives q_mod = q_emp + p_drop*(sbar_emp - sbar_mod)/(2 mu dg),
so any q difference is exactly a statement about the law's first moment.
"""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist
from symreg import enumerate_search, make_checker

SCRATCH = _os.path.join(_HERE, 'out')
d = np.load(SCRATCH + '/model_stats.npz')
sf = np.load(SCRATCH + '/scale_fields.npz')
NB = int(d['NB'])
DG = float(d['DG'])
A_G, EPS_G, KAP_G = float(sf['A_G']), float(sf['EPS_G']), float(sf['KAP_G'])
sc, sstar, nev = sf['sc'], sf['sstar'], sf['nev']

n_all = d['n_all'].reshape(NB, NB)
with np.errstate(all='ignore'):
    mu2 = (d['sum_dtau_ne'] / d['n_ne']).reshape(NB, NB) / DG
    q_emp = 1 - (d['sum_dtau'] / d['n_all']).reshape(NB, NB) / (mu2 * DG)
    p_drop = (d['n_drop'] / d['n_all']).reshape(NB, NB)
    sbar_emp = (d['sum_s'] / d['n_drop']).reshape(NB, NB)


def moment_invpow(c, a=A_G, eps=EPS_G, lo=1e-7):
    g = dist._grid(lo, c, True, 2000)
    ln = a * (np.log(c - g) - np.log(g + eps))
    w = np.exp(ln - ln.max())
    return float(np.trapezoid(g * w, g) / np.trapezoid(w, g))


def moment_tpl(ss, kap=KAP_G, lo=1e-7):
    g = np.geomspace(lo, 50 * ss, 2000)
    ln = -kap * np.log(g) - g / ss
    w = np.exp(ln - ln.max())
    return float(np.trapezoid(g * w, g) / np.trapezoid(w, g))


sbar_mod = np.full((NB, NB), np.nan)
sbar_tpl = np.full((NB, NB), np.nan)
for iu in range(NB):
    for it in range(NB):
        if np.isfinite(sc[iu, it]):
            sbar_mod[iu, it] = moment_invpow(sc[iu, it])
        if np.isfinite(sstar[iu, it]):
            sbar_tpl[iu, it] = moment_tpl(sstar[iu, it])

q_mod = q_emp + p_drop * (sbar_emp - sbar_mod) / (mu2 * DG)
q_tplm = q_emp + p_drop * (sbar_emp - sbar_tpl) / (mu2 * DG)

# ---- moment accuracy and q agreement on well-sampled inner voxels ----
inner = np.zeros((NB, NB), bool)
inner[1:21, 1:21] = True
ok = inner & (nev >= 100) & np.isfinite(sbar_mod) & (q_emp > 0)
rel_m = (sbar_mod / sbar_emp - 1)[ok]
rel_t = (sbar_tpl / sbar_emp - 1)[ok]
rel_q = (q_mod / q_emp - 1)[ok]
print('voxels compared: %d' % ok.sum())
print('<s> model/emp - 1: median %+.3f  IQR [%+.3f, %+.3f]  RMS %.3f' %
      (np.median(rel_m), *np.quantile(rel_m, [0.25, 0.75]),
       np.sqrt(np.mean(rel_m ** 2))))
print('<s> tpl/emp   - 1: median %+.3f  IQR [%+.3f, %+.3f]  RMS %.3f' %
      (np.median(rel_t), *np.quantile(rel_t, [0.25, 0.75]),
       np.sqrt(np.mean(rel_t ** 2))))
print('q  model/emp  - 1: median %+.3f  RMS %.3f' %
      (np.median(rel_q), np.sqrt(np.mean(rel_q ** 2))))

# ---- split-half statistical floor for q on this grid ----
tn = d['tn_all']; tdt = d['tsum_dtau']
cr = d['cr']; ntr = len(cr)
h1 = np.arange(ntr) % 2 == 0
qh = []
for h in (h1, ~h1):
    n_h = tn[h].sum(0).reshape(NB, NB)
    dt_h = tdt[h].sum(0).reshape(NB, NB)
    with np.errstate(all='ignore'):
        qh.append(1 - (dt_h / n_h) / (mu2 * DG))
fl = (qh[0] / qh[1] - 1)[ok & (n_all >= 5000)]
floor = np.sqrt(np.mean(fl ** 2)) / 2
print('split-half q floor (n>=5000 voxels, %d): %.3f' % ((ok & (n_all >= 5000)).sum(), floor))

# ---- factorize both q fields, SR on factors ----
uc, tc = sf['uc'], sf['tc']


def als(logq, w):
    L0 = np.where(np.isfinite(logq), logq, 0.0)
    W = np.where(np.isfinite(logq), w, 0.0)
    a = np.zeros(L0.shape[0]); b = np.zeros(L0.shape[1])
    for _ in range(300):
        for i in range(L0.shape[0]):
            if W[i].sum() > 0:
                a[i] = np.sum(W[i] * (L0[i] - b)) / np.maximum(W[i].sum(), 1e-9)
        for j in range(L0.shape[1]):
            if W[:, j].sum() > 0:
                b[j] = np.sum(W[:, j] * (L0[:, j] - a)) / np.maximum(W[:, j].sum(), 1e-9)
    m = W > 0
    res = (a[:, None] + b[None, :] - L0)[m]
    tot = (L0 - np.average(L0[m], weights=W[m]))[m]
    return a, b, 1 - np.sum(W[m] * res ** 2) / np.sum(W[m] * tot ** 2)


msk = ok & (n_all >= 5000)
w = np.sqrt(n_all[1:21, 1:21]) * msk[1:21, 1:21]
out = {}
for tag, qf in (('emp', q_emp), ('mod', q_mod)):
    lq = np.where(msk[1:21, 1:21] & (qf[1:21, 1:21] > 0),
                  np.log(qf[1:21, 1:21]), np.nan)
    a, b, r1 = als(lq, w)
    out[tag] = dict(rank1=float(r1), Lam=list(np.exp(a - a.max())),
                    f=list(np.exp(b + a.max())))
    print('%s: rank-1 fraction %.3f' % (tag, r1))
    for nm, xv, yv, wv, mono in (('Lam_' + tag, uc, a, w.sum(1), 'inc'),
                                 ('f_' + tag, tc, b, w.sum(0), 'inc')):
        k = wv > 0
        y = np.exp(yv[k] - yv[k].max())
        chk = make_checker(np.linspace(xv[k].min(), xv[k].max(), 80),
                           positive=True, increasing=True)
        par, res = enumerate_search(xv[k], y, sigma=1 / np.sqrt(wv[k]),
                                    max_complexity=8, max_consts=2,
                                    checker=chk, n_restarts=4, verbose=False)
        out[nm] = [dict(complexity=p['complexity'], rmse=p['rmse'],
                        string=p['string']) for p in par]
        print(' SR %s knee: %s' % (nm, par[-1]['string'] if par else 'none'))

np.savez(SCRATCH + '/qfields.npz', q_emp=q_emp, q_mod=q_mod, q_tplm=q_tplm,
         sbar_emp=sbar_emp, sbar_mod=sbar_mod, sbar_tpl=sbar_tpl,
         mu2=mu2, p_drop=p_drop, ok=ok, msk=msk, floor=floor)
with open(SCRATCH + '/reconstruct.json', 'w') as f:
    json.dump(dict(rel_m=list(map(float, rel_m)), rel_t=list(map(float, rel_t)),
                   rel_q=list(map(float, rel_q)), floor=float(floor), sr=out),
              f, indent=1, default=float)
print('DONE')
