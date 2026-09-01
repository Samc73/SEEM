import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Pipeline falsifiability check on synthetic catalogs.

Draw sizes from (i) a known TPL (exponential tail) and (ii) a known invpow
(hard ceiling), with parameters matched to the data fits, then run the exact
same MLE adjudication (5 seeds) and the exact same binned-SR search (1 seed
each). The pipeline passes only if neither stage ever claims a ceiling on
TPL-generated data, and both recover the ceiling on invpow-generated data.
"""
import numpy as np, sys, time, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist
from symreg import enumerate_search, make_checker

SCRATCH = _os.path.join(_HERE, 'out')
N = 6000
XMIN = 1e-6
TPL_TH = [0.74, np.log(0.16)]
INV_TH = None  # built from (a, eps, c) below via smax trick at sample time
A, EPS, C = 0.956, 5.3e-5, 0.45

report = dict(mle=[], sr={})
for gen in ('tpl', 'invpow'):
    for seed in range(5):
        rng = np.random.default_rng(seed)
        if gen == 'tpl':
            s = dist.sample('tpl', TPL_TH, XMIN, 1.0, N, rng)
        else:
            th = [A, np.log(EPS), np.log(C / (C / 2) - 1)]
            s = dist.sample('invpow', th, XMIN, C / 2, N, rng)
        r_t = dist.fit('tpl', s, XMIN)
        r_i = dist.fit('invpow', s, XMIN)
        tid = rng.integers(0, 100, N)  # iid events: parity split is a plain 2-fold
        cv_t = dist.cv_nll('tpl', s, tid, XMIN, smax=s.max())
        cv_i = dist.cv_nll('invpow', s, tid, XMIN, smax=s.max())
        rec = dict(gen=gen, seed=seed,
                   daic_tpl_minus_inv=float(r_t['aic'] - r_i['aic']),
                   dcv_tpl_minus_inv=float(cv_t - cv_i),
                   inv_c_over_smax=float(s.max() * (1 + np.exp(r_i['theta'][2])) / s.max()),
                   tpl_kappa=float(r_t['theta'][0]), inv_a=float(r_i['theta'][0]))
        report['mle'].append(rec)
        print('%serated seed %d: dAIC(tpl-inv)=%+.1f  dCV=%+.5f  c/smax=%.3f' %
              (gen, seed, rec['daic_tpl_minus_inv'], rec['dcv_tpl_minus_inv'],
               rec['inv_c_over_smax']), flush=True)

# binned-SR stage, one seed per generator
for gen in ('tpl', 'invpow'):
    rng = np.random.default_rng(0)
    if gen == 'tpl':
        s = dist.sample('tpl', TPL_TH, XMIN, 1.0, N, rng)
    else:
        th = [A, np.log(EPS), np.log(C / (C / 2) - 1)]
        s = dist.sample('invpow', th, XMIN, C / 2, N, rng)
    s = s[s >= 1e-4]
    lo, hi = np.quantile(s, 0.005), s.max() * 0.999
    be = np.geomspace(lo, hi, 20)
    n, _ = np.histogram(s, bins=be)
    keep = n >= 25
    xc = np.sqrt(be[:-1] * be[1:])[keep]
    rho = n[keep] / np.diff(be)[keep] / len(s)
    y, sig = np.log(rho), 1 / np.sqrt(n[keep])
    chk = make_checker(np.geomspace(xc.min(), xc.max(), 80), decreasing=True)
    t0 = time.time()
    par, res = enumerate_search(xc, y, sigma=sig, max_complexity=8, max_consts=2,
                                checker=chk, n_restarts=4, verbose=False,
                                probe=np.geomspace(xc.min(), xc.max(), 17))
    report['sr'][gen] = [dict(complexity=p['complexity'], rmse=p['rmse'],
                              string=p['string']) for p in par]
    print('SR on %s-generated (%.0fs):' % (gen, time.time() - t0), flush=True)
    for p in par:
        print('   C=%2d rmse=%.4f  %s' % (p['complexity'], p['rmse'], p['string']),
              flush=True)

with open(SCRATCH + '/synth_check.json', 'w') as f:
    json.dump(report, f, indent=1)
print('DONE')
