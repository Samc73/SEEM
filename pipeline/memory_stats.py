import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Per-(trajectory, voxel) sufficient statistics for every channel of the
step process, so each field can be re-estimated per preparation (cooling
rate) and compared at fixed (u,tau). Also stores downsampled per-trajectory
u(gamma), tau(gamma) curves. Uses the grid saved by extract2.py."""
import numpy as np, pandas as pd, time

SCRATCH = _os.path.join(_HERE, 'out')
u0c = -4.60751861
N = 49999
t0 = time.time()
d = np.load(SCRATCH + '/model_stats.npz')
ue, te, NB = d['ue'], d['te'], int(d['NB'])
NV = NB * NB

df = pd.read_pickle(_os.path.join(_HERE, '..', 'df_clean.pkl')).sort_values(['index', 'strain_index'])
cols = list(df.columns)
cr = df['cooling_rate'].to_numpy()[::N].copy()
u = (df['pe'].to_numpy() - u0c).reshape(-1, N).astype(np.float64)
tau = (df['stress'].to_numpy() / 1e4).reshape(-1, N).astype(np.float64)
strain = df['strain'].to_numpy()[:N].copy()
del df
print('loaded %.1fs  columns=%s' % (time.time() - t0, cols))

du = np.diff(u, axis=1).ravel()
dtau = np.diff(tau, axis=1).ravel()
uu = u[:, :-1].ravel()
tt = tau[:, :-1].ravel()
iu = np.clip(np.searchsorted(ue, uu, side='right') - 1, 0, NB - 1)
it = np.clip(np.searchsorted(te, tt, side='right') - 1, 0, NB - 1)
vox = (iu * NB + it).astype(np.int64)
del iu, it, uu, tt

drop = dtau < 0
age = (du < 0) & (dtau >= 0)
ne = ~(drop | (du < 0))
ntr = len(cr)
tid = np.repeat(np.arange(ntr, dtype=np.int64), N - 1)
key = tid * NV + vox
MB = ntr * NV

def acc(mask=None, w=None):
    if mask is None:
        return np.bincount(key, weights=w, minlength=MB).reshape(ntr, NV)
    return np.bincount(key[mask], weights=(None if w is None else w[mask]), minlength=MB).reshape(ntr, NV)

out = dict(
    n_all=acc(), sum_du=acc(w=du), sum_dtau=acc(w=dtau),
    n_ne=acc(ne), sum_du_ne=acc(ne, du), sum_dtau_ne=acc(ne, dtau),
    n_drop=acc(drop), sum_s=acc(drop, -dtau), sum_du_drop=acc(drop, du),
    sum_lns=acc(drop, np.log(np.where(drop, -dtau, 1.0))),
    n_age=acc(age), sum_du_age=acc(age, -du),
)
DS = 100
np.savez_compressed(SCRATCH + '/memory_stats.npz', cr=cr, rates=np.array(sorted(np.unique(cr))),
                    u_ds=u[:, ::DS], tau_ds=tau[:, ::DS], g_ds=strain[::DS],
                    columns=np.array(cols), **{k: v.astype(np.float32) for k, v in out.items()})
print('saved %.1fs' % (time.time() - t0))
