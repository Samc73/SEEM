import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Symbolic-regression discovery of the yield-event size distribution form.

Per (u,tau) cell: build the binned log-density of event sizes with Poisson
errors, then exhaustively enumerate all expression trees up to a complexity
bound and fit ln rho(s) = a*f(s) + b. Variants probe the two dominant
systematics: bin weighting and the lower fit window.
"""
import numpy as np
import sys, time, json, pickle
from multiprocessing import Pool

sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
from symreg import enumerate_search, make_checker, to_string

SCRATCH = _os.path.join(_HERE, 'out')
d = np.load(SCRATCH + '/model_stats.npz')
ev = np.load(SCRATCH + '/model_events.npz')
NB = int(d['NB'])
nd = d['n_drop'].reshape(NB, NB)
S, EV = ev['S'], ev['EV']

# ---- cell selection: top-8 by count + 4 spread across the occupied grid ----
inner = [(iu, it, nd[iu, it]) for iu in range(1, 21) for it in range(1, 21)]
inner.sort(key=lambda z: -z[2])
cells = [(iu, it) for iu, it, n in inner[:8]]
cand = [(iu, it) for iu, it, n in inner if n >= 800 and (iu, it) not in cells]
for _ in range(4):
    if not cand:
        break
    best = max(cand, key=lambda c: min((c[0]-a)**2 + (c[1]-b)**2 for a, b in cells))
    cells.append(best)
    cand.remove(best)
print('cells:', cells, flush=True)


def target(iu, it, s_lo):
    s = S[EV == iu * NB + it]
    s = s[s >= s_lo]
    lo, hi = np.quantile(s, 0.005), s.max() * 0.999
    be = np.geomspace(lo, hi, 20)
    n, _ = np.histogram(s, bins=be)
    keep = n >= 25
    xc = np.sqrt(be[:-1] * be[1:])[keep]
    rho = n[keep] / np.diff(be)[keep] / len(s)
    return xc, np.log(rho), 1 / np.sqrt(n[keep]), len(s)


def run(job):
    iu, it, mc, weighted, s_lo, tag = job
    t0 = time.time()
    xc, y, sig, nev = target(iu, it, s_lo)
    chk = make_checker(np.geomspace(xc.min(), xc.max(), 80), decreasing=True)
    par, res = enumerate_search(
        xc, y, sigma=(sig if weighted else None), max_complexity=mc,
        max_consts=2, checker=chk, n_restarts=4, verbose=False,
        probe=np.geomspace(xc.min(), xc.max(), 17))
    # split-half floor: same bins, trajectory-parity halves are not stored per
    # cell here; use Poisson sigma as the statistical floor proxy
    out = dict(iu=iu, it=it, mc=mc, weighted=weighted, s_lo=s_lo, tag=tag,
               nev=nev, nbins=len(xc), n_admissible=len(res),
               floor=float(np.sqrt(np.mean(sig ** 2))),
               front=[dict(complexity=p['complexity'], rmse=p['rmse'],
                           string=p['string'],
                           expr=repr(p['expr']), consts=list(map(float, p['consts'])),
                           a=float(p['a']), b=float(p['b'])) for p in par],
               secs=time.time() - t0)
    print('done %s iu=%d it=%d mc=%d w=%d lo=%g  %.0fs  knee: %s' %
          (tag, iu, it, mc, weighted, s_lo,
           out['secs'], out['front'][-1]['string']), flush=True)
    return out


jobs = [(iu, it, 8, True, 1e-4, 'main') for iu, it in cells]
jobs += [(cells[0][0], cells[0][1], 8, False, 1e-4, 'unweighted'),
         (cells[1][0], cells[1][1], 8, False, 1e-4, 'unweighted'),
         (cells[0][0], cells[0][1], 8, True, 3e-5, 'window'),
         (cells[1][0], cells[1][1], 8, True, 3e-5, 'window')]

if __name__ == '__main__':
    with Pool(8) as p:
        results = p.map(run, jobs)
    with open(SCRATCH + '/sr_dist_results.json', 'w') as f:
        json.dump(results, f, indent=1)
    print('ALL DONE')
