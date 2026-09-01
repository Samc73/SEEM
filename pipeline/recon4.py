import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Final-field factor analysis: ALS + SR on q_emp and q_mod4 (decoupled law),
and ALS + SR on the sc4 ceiling factors. Saves factors + fronts for figures.
"""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
from symreg import enumerate_search, make_checker

SCRATCH = _os.path.join(_HERE, 'out')
d = np.load(SCRATCH + '/model_stats.npz')
qf = np.load(SCRATCH + '/qfields.npz')
s4 = np.load(SCRATCH + '/scale4.npz')
sf = np.load(SCRATCH + '/scale_fields.npz')
NB = int(d['NB'])
n_all = d['n_all'].reshape(NB, NB)
uc, tc = sf['uc'], sf['tc']
msk = qf['msk']


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


out = {}
w_q = np.sqrt(n_all[1:21, 1:21]) * msk[1:21, 1:21]
for tag, qfield in (('emp', qf['q_emp']), ('mod4', s4['q_mod4'])):
    lq = np.where(msk[1:21, 1:21] & (qfield[1:21, 1:21] > 0),
                  np.log(qfield[1:21, 1:21]), np.nan)
    a, b, r1 = als(lq, w_q)
    out['q_' + tag] = dict(rank1=float(r1), a=list(a), b=list(b),
                           wa=list(w_q.sum(1)), wb=list(w_q.sum(0)))
    print('q_%s rank-1 %.3f' % (tag, r1), flush=True)
    for nm, xv, yv, wv in (('Lam_' + tag, uc, a, w_q.sum(1)),
                           ('f_' + tag, tc, b, w_q.sum(0))):
        k = wv > 0
        y = np.exp(yv[k] - yv[k].max())
        chk = make_checker(np.linspace(xv[k].min(), xv[k].max(), 80),
                           positive=True, increasing=True)
        par, _ = enumerate_search(xv[k], y, sigma=1 / np.sqrt(wv[k]),
                                  max_complexity=8, max_consts=2,
                                  checker=chk, n_restarts=4, verbose=False)
        out[nm] = [dict(complexity=p['complexity'], rmse=p['rmse'],
                        string=p['string'], a=float(p['a']), b=float(p['b']),
                        consts=list(map(float, p['consts'])),
                        expr=repr(p['expr'])) for p in par]
        print(' SR %s knee: %s' % (nm, par[-1]['string'] if par else 'none'),
              flush=True)

# ceiling factors from the final field
nev = sf['nev']
L = np.log(s4['sc4'][1:21, 1:21])
w_c = np.where(np.isfinite(L), np.sqrt(np.clip(nev[1:21, 1:21], 0, None)), 0.0)
a, b, r1 = als(L, w_c)
out['sc4'] = dict(rank1=float(r1), a=list(a), b=list(b),
                  wa=list(w_c.sum(1)), wb=list(w_c.sum(0)))
print('sc4 rank-1 %.3f' % r1, flush=True)
for nm, xv, yv, wv in (('scu', uc, a, w_c.sum(1)), ('sct', tc, b, w_c.sum(0))):
    k = wv > 0
    chk = make_checker(np.linspace(xv[k].min(), xv[k].max(), 80))
    par, _ = enumerate_search(xv[k], yv[k], sigma=1 / np.sqrt(wv[k]),
                              max_complexity=8, max_consts=2, checker=chk,
                              n_restarts=4, verbose=False)
    out[nm] = [dict(complexity=p['complexity'], rmse=p['rmse'],
                    string=p['string'], a=float(p['a']), b=float(p['b']),
                    consts=list(map(float, p['consts'])),
                    expr=repr(p['expr'])) for p in par]
    print(' SR %s knee: %s' % (nm, par[-1]['string'] if par else 'none'), flush=True)

with open(SCRATCH + '/recon4.json', 'w') as f:
    json.dump(out, f, indent=1)
print('DONE')
