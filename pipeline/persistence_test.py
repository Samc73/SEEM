import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Is a trajectory's stress deviation from its preparation mean persistent
(a per-sample property) or a random walk?  Autocorrelation of the deviation
across strain, and the mean-reversion slope, data vs the Markov simulation."""
import numpy as np, json
SCRATCH = _os.path.join(_HERE, 'out')
d = np.load(SCRATCH + '/model_stats.npz'); ms = np.load(SCRATCH + '/memory_stats.npz'); sm = np.load(SCRATCH + '/sim_model4.npz')
rates, cr = d['rates'], d['cr']
def dev(arr, j):
    x = arr[:, j].astype(float); return np.concatenate([x[cr == r] - x[cr == r].mean() for r in rates])
def at(arr, g, g_axis): return int(np.argmin(np.abs(g_axis - g)))
g_d = ms['g_ds']; g_s = np.arange(sm['rec_t'].shape[1]) * int(sm['record_every']) * 1e-5
out = {}
print('%-8s %-14s %8s %8s %8s | %10s %10s' % ('coord', 'pair', 'data', 'model', '', 'slope data', 'slope model'))
for coord, A_d, A_s in [('tau', ms['tau_ds'], sm['rec_t']), ('u', ms['u_ds'], sm['rec_u'])]:
    for g1, g2 in [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.3, 0.5), (0.2, 0.5)]:
        xd, yd = dev(A_d, at(A_d, g1, g_d)), dev(A_d, at(A_d, g2, g_d)); xs, ys = dev(A_s, at(A_s, g1, g_s)), dev(A_s, at(A_s, g2, g_s))
        cd, cs = np.corrcoef(xd, yd)[0, 1], np.corrcoef(xs, ys)[0, 1]
        bd = np.polyfit(xd, yd - xd, 1)[0]; bs = np.polyfit(xs, ys - xs, 1)[0]
        out['%s %.1f-%.1f' % (coord, g1, g2)] = dict(corr_data=float(cd), corr_model=float(cs), slope_data=float(bd), slope_model=float(bs))
        print('%-8s %.1f -> %.1f     %8.3f %8.3f %8s | %+10.3f %+10.3f' % (coord, g1, g2, cd, cs, '', bd, bs))
json.dump(out, open(SCRATCH + '/persistence_test.json', 'w'), indent=1)

# short lags: is the restoring force on a stress fluctuation stronger in the data?
print('\nshort-lag mean reversion of tau deviations (slope of d(dev) on dev):')
short = {}
for g0 in [0.3, 0.45]:
    for dg in [0.002, 0.005, 0.01, 0.02, 0.05]:
        A_d, A_s = ms['tau_ds'], sm['rec_t']
        xd, yd = dev(A_d, at(A_d, g0, g_d)), dev(A_d, at(A_d, g0 + dg, g_d)); xs, ys = dev(A_s, at(A_s, g0, g_s)), dev(A_s, at(A_s, g0 + dg, g_s))
        bd = np.polyfit(xd, yd - xd, 1)[0]; bs = np.polyfit(xs, ys - xs, 1)[0]
        short['%.2f+%.3f' % (g0, dg)] = dict(slope_data=float(bd), slope_model=float(bs), sd_data=float(xd.std()), sd_model=float(xs.std()))
        print('  gamma %.2f, lag %.3f:  data %+.3f   model %+.3f     (SD of deviation: data %.3f, model %.3f)' % (g0, dg, bd, bs, xd.std(), xs.std()))
out['short_lag'] = short
json.dump(out, open(SCRATCH + '/persistence_test.json', 'w'), indent=1)

# from the initial state: how fast is a trajectory's initial deviation forgotten?
print('\npersistence of the INITIAL deviation (corr of dev at gamma=0 with dev at gamma):')
init = {}
for coord, A_d, A_s in [('tau', ms['tau_ds'], sm['rec_t']), ('u', ms['u_ds'], sm['rec_u'])]:
    x_d, x_s = dev(A_d, 0), dev(A_s, 0)
    line = '  %-4s' % coord
    for g in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:
        cd = np.corrcoef(x_d, dev(A_d, at(A_d, g, g_d)))[0, 1]; cs = np.corrcoef(x_s, dev(A_s, at(A_s, g, g_s)))[0, 1]
        init['%s %.2f' % (coord, g)] = dict(data=float(cd), model=float(cs)); line += '   g=%.2f: %.2f / %.2f' % (g, cd, cs)
    print(line + '   (data / model)')
# cross: does a high initial stress predict a lower stress later (over-relaxation) in the data?
out['initial'] = init
json.dump(out, open(SCRATCH + '/persistence_test.json', 'w'), indent=1)
