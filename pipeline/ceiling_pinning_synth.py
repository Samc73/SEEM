import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Control for ceiling_pinning.py: the same drop-the-largest refits on
synthetic catalogs of the same size drawn from (a) the fitted ceiling law
and (b) the fitted TPL, so the observed sensitivity can be compared with
what a true ceiling, and a true exponential tail, produce."""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
import dist
SCRATCH = _os.path.join(_HERE, 'out')
mc = json.load(open(SCRATCH + '/mle_compare.json'))
XMIN = 1e-4
out = {}
for gen in ['invpow4', 'tpl']:
    print('--- synthetic catalogs from', gen)
    res = []
    for c in mc['cells'][:4]:
        th = np.array(c['fits'][gen]['theta']); n = c['n']
        ev = np.load(SCRATCH + '/model_events.npz'); d = np.load(SCRATCH + '/model_stats.npz')
        s_real = ev['S'][ev['EV'] == c['iu'] * int(d['NB']) + c['it']].astype(float); smax_real = s_real[s_real >= XMIN].max()
        c_true = dist._upper(gen, th, smax_real)
        for seed in range(3):
            rng = np.random.default_rng(seed)
            s = np.sort(dist.sample(gen, th, XMIN, smax_real, n, rng))
            r0 = dist.fit('invpow4', s, XMIN); c0 = dist._upper('invpow4', r0['theta'], s.max())
            row = dict(cell=[c['iu'], c['it']], seed=seed, n=n, c_true=(float(c_true) if c_true is not None else None),
                       sc=float(c0), c_over_smax=float(c0 / s.max()), drops={})
            line = '(%2d,%2d) seed %d  true c=%s  fit s_c=%.4f  c/smax=%.3f |' % (
                c['iu'], c['it'], seed, ('%.4f' % c_true if c_true else '  none'), c0, c0 / s.max())
            for k in [1, 5, max(1, n // 100)]:
                ss = s[:-k]
                r = dist.fit('invpow4', ss, XMIN, x0=r0['theta']); ck = dist._upper('invpow4', r['theta'], ss.max())
                row['drops'][k] = dict(sc_ratio=float(ck / c0), smax_ratio=float(ss.max() / s.max()))
                line += '  %.3f [%.3f]' % (ck / c0, ss.max() / s.max())
            print(line, flush=True); res.append(row)
    out[gen] = res
json.dump(out, open(SCRATCH + '/ceiling_pinning_synth.json', 'w'), indent=1)
