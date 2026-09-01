import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Within-preparation trajectory scatter: a test of the size law's variance.
The yield-peak and mean-curve comparisons feel only the mean event size;
the run-to-run scatter of u and tau at fixed strain feels the second moment
of P(s|u,tau) and the hazard's Bernoulli noise.  Compare data with the three
Section-8 arms (all start from the actual initial states of the 850 runs)."""
import numpy as np, json
SCRATCH = _os.path.join(_HERE, 'out')
d = np.load(SCRATCH + '/model_stats.npz'); ms = np.load(SCRATCH + '/memory_stats.npz')
rates, cr = d['rates'], d['cr']
u_ds, t_ds, g_ds = ms['u_ds'], ms['tau_ds'], ms['g_ds']
def within(x):   # rms of within-rate SD over the nine preparations
    return float(np.sqrt(np.mean([x[cr == r].var() for r in rates])))
out = {}
print('%-8s %-8s %10s %10s %10s %10s' % ('gamma', 'coord', 'data', 'model4', 'tpl', 'empir'))
for gt in [0.1, 0.2, 0.3, 0.4, 0.5]:
    jd = int(np.argmin(np.abs(g_ds - gt)))
    row = {}
    for coord, arr in [('u', u_ds), ('tau', t_ds)]:
        vals = {'data': within(arr[:, jd].astype(float))}
        for arm in ['model4', 'tpl', 'empir']:
            s = np.load(SCRATCH + f'/sim_{arm}.npz'); rec = s['rec_u'] if coord == 'u' else s['rec_t']
            js = int(round(gt / 1e-5 / int(s['record_every'])))
            vals[arm] = within(rec[:, min(js, rec.shape[1] - 1)].astype(float))
        row[coord] = vals
        print('%-8.1f %-8s %10.5f %10.5f %10.5f %10.5f' % (gt, coord, vals['data'], vals['model4'], vals['tpl'], vals['empir']))
    out[gt] = row
# initial scatter, for reference
print('initial (gamma=0): u %.5f  tau %.5f' % (within(u_ds[:, 0].astype(float)), within(t_ds[:, 0].astype(float))))
json.dump(out, open(SCRATCH + '/scatter_test.json', 'w'), indent=1)
