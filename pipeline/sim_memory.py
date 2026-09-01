import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Repair test for the model's failure mode (the 5x-too-small preparation
spread of u at gamma = 0.5).  memory_where.py located the preparation memory
in the event-size law (and, secondarily, in the energy released per event);
here the forward simulation is re-run with that memory put back in, one
channel at a time, and the u-spread is re-measured:

  model4     shared fields, shared ceiling            (the README's arm)
  mem_sc     ceiling scaled per cooling rate:  s_c(v, r) = s_c(v) (r/r_mid)^beta
  mem_sc_k   ... plus energy-release coupling refit per preparation group
  mem_empir  per-group empirical size tables + per-group coupling (upper bound)

Nothing else changes: hazard, elastic drift and aging stay preparation-blind,
which memory_where.py showed they are in the data.
"""
import numpy as np, sys, json
sys.path.insert(0, _os.path.join(_HERE, '..', 'library'))
sys.path.insert(0, _HERE)
import dist
from simulate import simulate, SCRATCH

K, NSTEPS, REC = 257, 50_000, 20
d = np.load(SCRATCH + '/model_stats.npz')
s4 = np.load(SCRATCH + '/scale4.npz')
fields = dict(np.load(SCRATCH + '/pdmp_fields.npz'))
ev = np.load(SCRATCH + '/model_events.npz')
mem = json.load(open(SCRATCH + '/memory_where.json'))
ms = np.load(SCRATCH + '/memory_stats.npz')
NB = int(d['NB']); NV = NB * NB
rates, cr = d['rates'], d['cr']
R = len(rates)
ridx = np.searchsorted(rates, cr)
grp3 = np.minimum(np.arange(R) // 3, 2)          # slow-3 / mid-3 / fast-3
levels = (np.arange(K) + 0.5) / K
KG, MG, EG = float(s4['K_G']), float(s4['M_G']), float(s4['E_G'])
sc4 = s4['sc4']
beta = mem['channels']['mean size <s>']['beta']
lam = (rates / rates[R // 2]) ** beta
print('ceiling factor per rate (slow->fast):', np.round(lam, 3))


def q_invpow4(c):
    g = dist._grid(1e-7, c, True, 3000)
    ln = MG * np.log(c - g) - KG * np.log(g + EG)
    w = np.exp(ln - ln.max())
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(g))))
    cdf /= cdf[-1]
    return np.interp(levels, cdf, g)


def nearest_fill(Q):
    fin = np.argwhere(np.isfinite(Q[:, :, 0]))
    for i, j in np.argwhere(~np.isfinite(Q[:, :, 0])):
        nn = fin[np.argmin((fin[:, 0] - i) ** 2 + (fin[:, 1] - j) ** 2)]
        Q[i, j] = Q[nn[0], nn[1]]
    return Q


def table_model(scale):
    Q = np.full((NB, NB, K), np.nan)
    for iu in range(NB):
        for it in range(NB):
            if np.isfinite(sc4[iu, it]):
                Q[iu, it] = q_invpow4(sc4[iu, it] * scale)
    return nearest_fill(Q)


S, EV, DU, ECR = ev['S'].astype(float), ev['EV'], ev['DUe'].astype(float), ev['Ecr']
egrp = grp3[np.searchsorted(rates, ECR)]


def table_empir(sel, fallback):
    Q = np.full((NB, NB, K), np.nan)
    for v in np.unique(EV[sel]):
        s = S[sel & (EV == v)]
        if len(s) >= 40:
            Q[v // NB, v % NB] = np.quantile(s, levels)
    bad = ~np.isfinite(Q[:, :, 0])
    Q[bad] = fallback[bad]
    return Q


def coupling_per_group():
    """(R,NB,NB) ka/kb/ksig: refit per preparation group where >= 40 events, else pooled."""
    ka = np.repeat(fields['ka'][None], R, 0).copy()
    kb = np.repeat(fields['kb'][None], R, 0).copy()
    ks = np.repeat(fields['ksig'][None], R, 0).copy()
    for g in range(3):
        sel = egrp == g
        for v in np.unique(EV[sel]):
            m = sel & (EV == v)
            if m.sum() < 40:
                continue
            A = np.c_[np.ones(m.sum()), S[m]]
            coef, *_ = np.linalg.lstsq(A, DU[m], rcond=None)
            sig = np.std(DU[m] - A @ coef)
            for r in np.nonzero(grp3 == g)[0]:
                ka[r, v // NB, v % NB], kb[r, v // NB, v % NB], ks[r, v // NB, v % NB] = coef[0], coef[1], sig
    return ka, kb, ks


def spread(u_at):
    means = np.array([u_at[cr == r].mean() for r in rates])
    return dict(rate_means=means.tolist(), range_rel=float((means.max() - means.min()) / means.mean()),
                between_sd=float(means.std()))


def run(name, Q, flds, group):
    rec_t, rec_u, peak = simulate(Q, flds, d['u_init'], d['tau_init'], NSTEPS, seed=42,
                                  record_every=REC, group=group)
    j = int(round(0.5 / 1e-5 / REC))
    out = dict(spread_u05=spread(rec_u[:, min(j, rec_u.shape[1] - 1)].astype(float)),
               spread_u03=spread(rec_u[:, int(round(0.3 / 1e-5 / REC))].astype(float)))
    pk = np.array([peak[cr == r].mean() for r in rates]); pd_ = np.array([d['tau_peak'][cr == r].mean() for r in rates])
    out['peak_rms_rel'] = float(np.sqrt(np.mean((pk / pd_ - 1) ** 2)))
    fl = np.array([rec_t[cr == r, min(j, rec_t.shape[1] - 1)].mean() for r in rates])
    out['flow05_by_rate'] = fl.tolist()
    out['curves_u'] = np.stack([rec_u[cr == r].mean(0) for r in rates]).tolist()
    print('%-10s u-spread(0.5) %5.1f%%  between-SD %.5f   u-spread(0.3) %5.1f%%   peak RMS %.1f%%' %
          (name, 100 * out['spread_u05']['range_rel'], out['spread_u05']['between_sd'],
           100 * out['spread_u03']['range_rel'], 100 * out['peak_rms_rel']))
    return out


# data reference with the same metric
u_ds, g_ds = ms['u_ds'], ms['g_ds']
ref = dict(spread_u05=spread(u_ds[:, int(np.argmin(np.abs(g_ds - 0.5)))].astype(float)),
           spread_u03=spread(u_ds[:, int(np.argmin(np.abs(g_ds - 0.3)))].astype(float)),
           curves_u=np.stack([u_ds[cr == r].mean(0) for r in rates]).tolist(), g_ds=g_ds.tolist())
print('%-10s u-spread(0.5) %5.1f%%  between-SD %.5f   u-spread(0.3) %5.1f%%' %
      ('data', 100 * ref['spread_u05']['range_rel'], ref['spread_u05']['between_sd'], 100 * ref['spread_u03']['range_rel']))

results = dict(data=ref, beta=beta, lam=lam.tolist(), rates=rates.tolist(), nsteps=NSTEPS, rec=REC)
Q0 = table_model(1.0)
results['model4'] = run('model4', Q0, fields, None)

QR = np.stack([table_model(l) for l in lam])
results['mem_sc'] = run('mem_sc', QR, fields, ridx)

ka, kb, ks = coupling_per_group()
fk = dict(fields); fk['ka'], fk['kb'], fk['ksig'] = ka, kb, ks
results['mem_sc_k'] = run('mem_sc_k', QR, fk, ridx)

# state-dependent memory: beta fitted per u band (memory_fading.py), applied per voxel row
fad = json.load(open(SCRATCH + '/memory_fading.json'))['bands']
ue = d['ue']
beta_u = np.full(NB, beta)
for key, val in fad.items():
    if key.startswith('u ') and np.isfinite(val['beta']):
        lo, hi = map(float, key[2:].split('-'))
        rows_ = np.nonzero((ue[:-1] >= lo - 1e-9) & (ue[:-1] < hi - 1e-9))[0]
        beta_u[rows_] = val['beta']
beta_u[:np.nonzero(beta_u != beta)[0].min()] = beta_u[np.nonzero(beta_u != beta)[0].min()]   # unsampled cold rows: nearest fitted band
print('beta by u row:', np.round(beta_u, 3))
def table_model_u(r):
    Q = np.full((NB, NB, K), np.nan)
    for iu in range(NB):
        lam_r = (rates[r] / rates[R // 2]) ** beta_u[iu]
        for it in range(NB):
            if np.isfinite(sc4[iu, it]):
                Q[iu, it] = q_invpow4(sc4[iu, it] * lam_r)
    return nearest_fill(Q)
QU = np.stack([table_model_u(r) for r in range(R)])
results['mem_sc_u'] = run('mem_sc_u', QU, fields, ridx)
results['beta_u'] = beta_u.tolist()

# relaxing memory: beta(gamma) from memory_2d.py (fixed effects per cell x strain window)
m2 = json.load(open(SCRATCH + '/memory_2d.json'))['by_window']
gw = np.array([[float(x) for x in k.split('-')] for k in m2]); bw = np.array([v['beta'] for v in m2.values()])
gc = gw.mean(1)
A_ = np.c_[np.ones(len(gc)), gc]; coef, *_ = np.linalg.lstsq(A_, np.log(-bw), rcond=None)
beta0, grelax = -np.exp(coef[0]), -1 / coef[1]
print('beta by window:', dict(zip(m2.keys(), np.round(bw, 3))), ' exp fit: beta(g) = %.3f exp(-g/%.3f)' % (beta0, grelax))
results['relax_fit'] = dict(windows=list(m2.keys()), beta=bw.tolist(), beta0=float(beta0), grelax=float(grelax))
lr_i = np.log(rates[ridx] / rates[R // 2])
def scale_piecewise(i):
    g = i * 1e-5
    b = bw[min(max(int((g - gw[0, 0]) // (gw[0, 1] - gw[0, 0])), 0), len(bw) - 1)]
    return np.exp(b * lr_i)
def scale_exp(i):
    return np.exp(beta0 * np.exp(-i * 1e-5 / grelax) * lr_i)
rec_t, rec_u, peak = simulate(Q0, fields, d['u_init'], d['tau_init'], NSTEPS, seed=42, record_every=REC, size_scale=scale_piecewise)
def summarize(name, rec_t, rec_u, peak):
    j = int(round(0.5 / 1e-5 / REC))
    out = dict(spread_u05=spread(rec_u[:, min(j, rec_u.shape[1] - 1)].astype(float)),
               spread_u03=spread(rec_u[:, int(round(0.3 / 1e-5 / REC))].astype(float)))
    pk = np.array([peak[cr == r].mean() for r in rates]); pd_ = np.array([d['tau_peak'][cr == r].mean() for r in rates])
    out['peak_rms_rel'] = float(np.sqrt(np.mean((pk / pd_ - 1) ** 2)))
    out['curves_u'] = np.stack([rec_u[cr == r].mean(0) for r in rates]).tolist()
    out['within_sd_u05'] = float(np.sqrt(np.mean([rec_u[cr == r, min(j, rec_u.shape[1] - 1)].var() for r in rates])))
    print('%-12s u-spread(0.5) %5.1f%%  between-SD %.5f   u-spread(0.3) %5.1f%%   peak RMS %.1f%%' %
          (name, 100 * out['spread_u05']['range_rel'], out['spread_u05']['between_sd'],
           100 * out['spread_u03']['range_rel'], 100 * out['peak_rms_rel']))
    return out
results['mem_relax_pw'] = summarize('mem_relax_pw', rec_t, rec_u, peak)
rec_t, rec_u, peak = simulate(Q0, fields, d['u_init'], d['tau_init'], NSTEPS, seed=42, record_every=REC, size_scale=scale_exp)
results['mem_relax_exp'] = summarize('mem_relax_exp', rec_t, rec_u, peak)

QE = np.stack([table_empir(egrp == grp3[r], Q0) for r in range(R)])
results['mem_empir'] = run('mem_empir', QE, fk, ridx)

json.dump(results, open(SCRATCH + '/sim_memory.json', 'w'), indent=1)
