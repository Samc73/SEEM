import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Full extraction pass for the distribution-first re-analysis.

Grid: 22x22 = inner 20x20 (quantile 0.002-0.998) plus an outer catch-all
ring, so every visited (u,tau) has a voxel and the fitted model can drive a
forward simulation over the whole range.
"""
import numpy as np
import pandas as pd
import time

t0 = time.time()
SCRATCH = _os.path.join(_HERE, 'out')
u0c = -4.60751861
N = 49999
DG = 1e-5

df = pd.read_pickle(_os.path.join(_HERE, '..', 'df_clean.pkl')).sort_values(['index', 'strain_index'])
cr = df['cooling_rate'].to_numpy()[::N].copy()
u = (df['pe'].to_numpy() - u0c).reshape(-1, N).astype(np.float64)
tau = (df['stress'].to_numpy() / 1e4).reshape(-1, N).astype(np.float64)
strain = df['strain'].to_numpy()[:N].copy()
del df
print('loaded %.1fs  ntraj=%d' % (time.time() - t0, len(cr)))

du = np.diff(u, axis=1)
dtau = np.diff(tau, axis=1)
uu = u[:, :-1]     # state at step start
tt = tau[:, :-1]

# ---- grid: inner 10x10 on 0.2-99.8% quantiles + outer ring ----
qs = np.quantile(uu.ravel(), [0.002, 0.998])
qt = np.quantile(tt.ravel(), [0.002, 0.998])
ue_in = np.linspace(qs[0], qs[1], 21)
te_in = np.linspace(qt[0], qt[1], 21)
ue = np.concatenate(([uu.min() - 1e-9], ue_in, [uu.max() + 1e-9]))
te = np.concatenate(([tt.min() - 1e-9], te_in, [tt.max() + 1e-9]))
NB = 22
iu = np.clip(np.searchsorted(ue, uu.ravel(), side='right') - 1, 0, NB - 1)
it = np.clip(np.searchsorted(te, tt.ravel(), side='right') - 1, 0, NB - 1)
vox = (iu * NB + it).astype(np.int32)
del iu, it
print('binned %.1fs' % (time.time() - t0))

duf = du.ravel()
dtf = dtau.ravel()
ev_drop = dtf < 0                    # stress-drop events (sizes s = -dtau)
ev_age = (duf < 0) & (dtf >= 0)      # aging events (PE drop, no stress drop)
ne = ~(ev_drop | (duf < 0))          # elastic steps

NV = NB * NB
def acc(mask=None, w=None):
    if mask is None:
        return np.bincount(vox, weights=w, minlength=NV)
    return np.bincount(vox[mask], weights=(None if w is None else w[mask]), minlength=NV)

n_all = acc()
sum_dtau = acc(w=dtf)
n_ne = acc(ne)
sum_dtau_ne = acc(ne, dtf)
sum_du_ne = acc(ne, duf)
sum_du2_ne = acc(ne, duf * duf)
n_drop = acc(ev_drop)
sum_s = acc(ev_drop, -dtf)
n_age = acc(ev_age)
sum_du_age = acc(ev_age, -duf)
print('voxel stats %.1fs' % (time.time() - t0))

# ---- per-(trajectory, voxel) stats for noise floors / per-rate fields ----
tid = np.repeat(np.arange(len(cr), dtype=np.int64), N - 1)
key = tid * NV + vox
MB = len(cr) * NV
tn_all = np.bincount(key, minlength=MB)
tsum_dtau = np.bincount(key, weights=dtf, minlength=MB)
tn_ne = np.bincount(key[ne], minlength=MB)
tsum_dtau_ne = np.bincount(key[ne], weights=dtf[ne], minlength=MB)
tn_drop = np.bincount(key[ev_drop], minlength=MB)
tsum_s = np.bincount(key[ev_drop], weights=-dtf[ev_drop], minlength=MB)
del key, tid
print('traj stats %.1fs' % (time.time() - t0))

# ---- event catalogs ----
S = (-dtf[ev_drop]).astype(np.float32)
DUe = duf[ev_drop].astype(np.float32)
EV = vox[ev_drop]
Eu = uu.ravel()[ev_drop].astype(np.float32)
Et = tt.ravel()[ev_drop].astype(np.float32)
step_of = np.tile(np.arange(N - 1, dtype=np.int32), len(cr))
Eg = (strain[:-1][step_of[ev_drop] % (N - 1)]).astype(np.float32)
Ecr = cr[(np.nonzero(ev_drop)[0] // (N - 1))].astype(np.float32)
A_du = (-duf[ev_age]).astype(np.float32)
AV = vox[ev_age]
print('catalogs %.1fs  n_drop=%d n_age=%d' % (time.time() - t0, len(S), len(A_du)))

# ---- per-trajectory / per-rate summaries for the simulation comparison ----
u_init = u[:, 0].copy()
tau_init = tau[:, 0].copy()
tau_peak = tau.max(1)
g_peak = strain[tau.argmax(1)]
rates = np.array(sorted(np.unique(cr)))
DS = 10
curves_tau = np.stack([tau[cr == c][:, ::DS].mean(0) for c in rates])
curves_u = np.stack([u[cr == c][:, ::DS].mean(0) for c in rates])
g_ds = strain[::DS]

np.savez_compressed(
    SCRATCH + '/model_stats.npz',
    ue=ue, te=te, NB=NB, u0c=u0c, DG=DG,
    n_all=n_all, sum_dtau=sum_dtau, n_ne=n_ne, sum_dtau_ne=sum_dtau_ne,
    sum_du_ne=sum_du_ne, sum_du2_ne=sum_du2_ne,
    n_drop=n_drop, sum_s=sum_s, n_age=n_age, sum_du_age=sum_du_age,
    tn_all=tn_all.reshape(len(cr), NV), tsum_dtau=tsum_dtau.reshape(len(cr), NV),
    tn_ne=tn_ne.reshape(len(cr), NV), tsum_dtau_ne=tsum_dtau_ne.reshape(len(cr), NV),
    tn_drop=tn_drop.reshape(len(cr), NV), tsum_s=tsum_s.reshape(len(cr), NV),
    cr=cr, rates=rates, u_init=u_init, tau_init=tau_init,
    tau_peak=tau_peak, g_peak=g_peak,
    curves_tau=curves_tau, curves_u=curves_u, g_ds=g_ds)
np.savez_compressed(
    SCRATCH + '/model_events.npz',
    S=S, DUe=DUe, EV=EV, Eu=Eu, Et=Et, Eg=Eg, Ecr=Ecr,
    A_du=A_du, AV=AV)
print('saved %.1fs' % (time.time() - t0))
