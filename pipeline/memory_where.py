import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Where does preparation memory act?  Every channel of the step process is
re-estimated per cooling rate at fixed (u,tau) voxel and regressed on
ln(cooling rate) with voxel fixed effects:

    ln F(v, rate) = ln F(v) + beta * ln(rate / rate_ref)

beta = 0 means the channel is preparation-blind at fixed state (what the
model assumes); |beta| > 0 means the hidden variable acts through it.
A within-ensemble control (trajectory-parity halves, rate composition
identical) gives the null spread of beta. Also decomposes the mean u-step
<du>(v, rate) - the direct driver of the u(gamma) spread - into its
elastic / drop / aging terms."""
import numpy as np, json, warnings
warnings.filterwarnings('ignore')

SCRATCH = _os.path.join(_HERE, 'out')
m = np.load(SCRATCH + '/memory_stats.npz')
d = np.load(SCRATCH + '/model_stats.npz')
NB = int(d['NB']); NV = NB * NB
cr, rates = m['cr'], m['rates']
R = len(rates)
lr = np.log(rates / rates[R // 2])           # centred log cooling rate
DG = 1e-5

def group_sums(sel):
    """Per-rate sums of every statistic over the selected trajectories."""
    out = {}
    for k in ['n_all', 'sum_du', 'sum_dtau', 'n_ne', 'sum_du_ne', 'sum_dtau_ne',
              'n_drop', 'sum_s', 'sum_du_drop', 'sum_lns', 'n_age', 'sum_du_age']:
        a = m[k].astype(np.float64)
        out[k] = np.stack([a[sel & (cr == r)].sum(0) for r in rates])   # (R, NV)
    return out

def channels(g):
    """Channel values F(rate, voxel) and their weights (counts)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        mu2 = g['sum_dtau_ne'].sum(0) / g['n_ne'].sum(0) / DG      # pooled 2mu per voxel
        ch = {
            'hazard p_drop':      (g['n_drop'] / g['n_all'],              g['n_all']),
            'mean size <s>':      (g['sum_s'] / g['n_drop'],              g['n_drop']),
            'geo-mean size':      (np.exp(g['sum_lns'] / g['n_drop']),    g['n_drop']),
            'plastic rate q':     (1 - g['sum_dtau'] / g['n_all'] / (mu2 * DG), g['n_all']),
            'elastic du/step':    (g['sum_du_ne'] / g['n_ne'],            g['n_ne']),
            'aging hazard p_age': (g['n_age'] / g['n_all'],               g['n_all']),
            'aging mean |du|':    (g['sum_du_age'] / g['n_age'],          g['n_age']),
            'drop du coupling':   (g['sum_du_drop'] / g['n_drop'],        g['n_drop']),
            'total <du>/step':    (g['sum_du'] / g['n_all'],              g['n_all']),
        }
    return ch

def beta_fit(F, W, nmin=40, log=True):
    """Voxel-fixed-effect weighted regression of F on ln(rate). Returns beta,
    its SE, and the number of voxels used."""
    ok = (W >= nmin) & np.isfinite(F) & ((F > 0) if log else True)
    keep = ok.sum(0) >= 4                           # voxel seen at >= 4 rates
    num = den = 0.0
    y = np.log(F) if log else F
    for v in np.nonzero(keep)[0]:
        o = ok[:, v]; w = W[o, v]; yy = y[o, v]; xx = lr[o]
        xm = np.average(xx, weights=w); ym = np.average(yy, weights=w)
        num += np.sum(w * (xx - xm) * (yy - ym)); den += np.sum(w * (xx - xm) ** 2)
    if den == 0:
        return float('nan'), float('nan'), int(keep.sum())
    beta = num / den
    # SE from voxel-wise scatter of residuals (heteroscedastic-robust)
    s2 = 0.0
    for v in np.nonzero(keep)[0]:
        o = ok[:, v]; w = W[o, v]; yy = y[o, v]; xx = lr[o]
        xm = np.average(xx, weights=w); ym = np.average(yy, weights=w)
        r = (yy - ym) - beta * (xx - xm)
        s2 += np.sum((w * (xx - xm) * r) ** 2)
    return float(beta), float(np.sqrt(s2) / den), int(keep.sum())

def slow_fast(F, W, nmin=40, log=True):
    """Median ratio (or difference) slow-3 vs fast-3 across voxels."""
    a = np.nansum(F[:3] * W[:3], 0) / W[:3].sum(0); wa = W[:3].sum(0)
    b = np.nansum(F[-3:] * W[-3:], 0) / W[-3:].sum(0); wb = W[-3:].sum(0)
    ok = (wa >= nmin) & (wb >= nmin) & np.isfinite(a) & np.isfinite(b)
    if log:
        ok &= (a > 0) & (b > 0)
        r = a[ok] / b[ok]
    else:
        r = a[ok] - b[ok]
    return float(np.median(r)), [float(x) for x in np.percentile(r, [16, 84])], int(ok.sum())

all_sel = np.ones(len(cr), bool)
par = (np.arange(len(cr)) % 2 == 0)
G = channels(group_sums(all_sel))
# control: relabel rates by trajectory parity inside each rate -> compare even vs odd halves
GA, GB = channels(group_sums(par)), channels(group_sums(~par))

report = {}
print('%-20s %8s %8s %6s   %-26s %s' % ('channel', 'beta', 'SE', 'nvox', 'slow3/fast3 [16-84%]', 'control even/odd'))
for name, (F, W) in G.items():
    log = name not in ('elastic du/step', 'drop du coupling', 'total <du>/step')
    nmin = 15 if 'aging' in name else 40
    b, se, nv = beta_fit(F, W, nmin=nmin, log=log)
    med, pr, n = slow_fast(F, W, nmin=nmin, log=log)
    Fa, Wa = GA[name]; Fb, Wb = GB[name]
    # control: even-half vs odd-half, pooled over all rates, same nmin
    a = np.nansum(Fa * Wa, 0) / Wa.sum(0); bb = np.nansum(Fb * Wb, 0) / Wb.sum(0)
    ok = (Wa.sum(0) >= nmin) & (Wb.sum(0) >= nmin) & np.isfinite(a) & np.isfinite(bb)
    if log:
        ok &= (a > 0) & (bb > 0); c = a[ok] / bb[ok]
    else:
        c = a[ok] - bb[ok]
    cmed, cpr = float(np.median(c)), [float(x) for x in np.percentile(c, [16, 84])]
    unit = 'ratio' if log else 'diff(1e-6)'
    sc = 1 if log else 1e6
    print('%-20s %8.4f %8.4f %6d   %s %.3f [%.3f, %.3f]   %s %.3f [%.3f, %.3f]' %
          (name, b, se, nv, unit, med*sc, pr[0]*sc, pr[1]*sc, unit, cmed*sc, cpr[0]*sc, cpr[1]*sc))
    report[name] = dict(beta=b, se=se, nvox=nv, slow_fast=med, slow_fast_1684=pr, n=n,
                        control=cmed, control_1684=cpr, log=log)

# ---- decomposition of <du>/step into channel terms, slow-3 vs fast-3 ----
g = group_sums(all_sel)
with np.errstate(divide='ignore', invalid='ignore'):
    def terms(sl):
        n = g['n_all'][sl].sum(0)
        return dict(elastic=g['sum_du_ne'][sl].sum(0) / n, drop=g['sum_du_drop'][sl].sum(0) / n,
                    aging=-g['sum_du_age'][sl].sum(0) / n, total=g['sum_du'][sl].sum(0) / n, n=n)
    ts, tf = terms(slice(0, 3)), terms(slice(R - 3, R))
ok = (ts['n'] >= 200) & (tf['n'] >= 200)
w = np.minimum(ts['n'], tf['n'])[ok]
dec = {}
print('\n<du>/step at fixed voxel, slow-3 minus fast-3 (count-weighted mean over %d voxels), x1e-6:' % ok.sum())
for k in ['elastic', 'drop', 'aging', 'total']:
    diff = np.average(ts[k][ok] - tf[k][ok], weights=w)
    mag = np.average(np.abs(0.5 * (ts[k][ok] + tf[k][ok])), weights=w)
    dec[k] = dict(diff=float(diff), typical=float(mag))
    print('  %-8s  diff %+8.3f   typical |term| %8.3f' % (k, diff * 1e6, mag * 1e6))

# ---- per-trajectory u(gamma) spread: between-rate vs within-rate ----
u_ds, g_ds = m['u_ds'], m['g_ds']
spread = {}
for gt in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    j = int(np.argmin(np.abs(g_ds - gt)))
    x = u_ds[:, j]
    means = np.array([x[cr == r].mean() for r in rates])
    within = np.sqrt(np.mean([x[cr == r].var() for r in rates]))
    spread[gt] = dict(rate_means=means.tolist(), between_sd=float(means.std()),
                      between_range_rel=float((means.max() - means.min()) / means.mean()),
                      within_sd=float(within))
    print('gamma=%.1f  rate-mean u: %.4f..%.4f (range %.1f%% of mean)  between-SD %.5f  within-rate SD %.5f' %
          (gt, means.min(), means.max(), 100 * spread[gt]['between_range_rel'], means.std(), within))

json.dump(dict(rates=rates.tolist(), channels=report, du_decomposition=dec, spread=spread),
          open(SCRATCH + '/memory_where.json', 'w'), indent=1)
