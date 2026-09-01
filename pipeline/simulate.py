import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""Forward AQS simulation from the fitted PDMP model.

One strain step per iteration, vectorized over all 850 trajectories:
  elastic:  tau += 2mu(u,tau)*dg ; u += g(u,tau)
  drop   :  s ~ P(s|u,tau) from a per-voxel quantile table; tau -= s;
            u += a(u,tau) + b(u,tau)*s + sig(u,tau)*N(0,1)
  aging  :  u -= Exp(m_age(u,tau))
Arms differ only in the quantile table (fitted model vs empirical resampling).
"""
import numpy as np

SCRATCH = _os.path.join(_HERE, 'out')
DG = 1e-5


def simulate(Q, fields, u0, t0, nsteps, seed=0, record_every=10, group=None, size_scale=None, window_steps=None):
    """Q may be (NB,NB,K) shared by all trajectories, or (G,NB,NB,K) with
    `group` giving each trajectory's table index (preparation-aware arms).
    `size_scale(step)` may return a per-trajectory multiplier applied to every
    sampled event size (a strain-dependent ceiling; exact up to the eps region)."""
    NB = int(np.sqrt(fields['mu2'].size))
    ue, te = fields['ue'], fields['te']
    mu2 = fields['mu2'].ravel()
    gd = fields['gdrift'].ravel()
    pd_ = fields['p_drop'].ravel()
    pa_ = fields['p_age'].ravel()
    ma_ = np.maximum(fields['m_age'].ravel(), 1e-9)
    ka_ = fields['ka'].ravel()
    kb_ = fields['kb'].ravel()
    ks_ = fields['ksig'].ravel()
    per_group_k = ka_.size > NB * NB          # (G,NB,NB) coupling fields
    K = Q.shape[-1]
    QF = Q.reshape(-1, K)
    n = len(u0)
    off = np.zeros(n, int) if Q.ndim == 3 else np.asarray(group, int) * (NB * NB)
    u = u0.astype(float).copy()
    t = t0.astype(float).copy()
    rng = np.random.default_rng(seed)
    nrec = nsteps // record_every + 1
    rec_t = np.empty((n, nrec), np.float32)
    rec_u = np.empty((n, nrec), np.float32)
    peak = t.copy()
    j = 0
    if window_steps:
        nwin = nsteps // window_steps + 1
        cnt = np.zeros((n, nwin)); ssum = np.zeros((n, nwin))
    for i in range(nsteps):
        iu = np.clip(np.searchsorted(ue, u, side='right') - 1, 0, NB - 1)
        it = np.clip(np.searchsorted(te, t, side='right') - 1, 0, NB - 1)
        v = iu * NB + it
        r = rng.random(n)
        pdv, pav = pd_[v], pa_[v]
        ev = r < pdv
        ag = (~ev) & (r < pdv + pav)
        el = ~(ev | ag)
        t[el] += mu2[v[el]] * DG
        u[el] += gd[v[el]]
        ne = int(ev.sum())
        if ne:
            q = rng.random(ne) * (K - 1)
            k0 = q.astype(int)
            fr = q - k0
            T = QF[off[ev] + v[ev]]
            idx = np.arange(ne)
            s = T[idx, k0] * (1 - fr) + T[idx, np.minimum(k0 + 1, K - 1)] * fr
            if size_scale is not None:
                s = s * size_scale(i)[ev]
            t[ev] -= s
            if window_steps:
                cnt[ev, i // window_steps] += 1; ssum[ev, i // window_steps] += s
            vk = (off[ev] + v[ev]) if per_group_k else v[ev]
            u[ev] += ka_[vk] + kb_[vk] * s + ks_[vk] * rng.standard_normal(ne)
        na = int(ag.sum())
        if na:
            u[ag] -= rng.exponential(ma_[v[ag]])
        np.clip(u, ue[0] + 1e-12, ue[-1] - 1e-12, out=u)
        np.clip(t, te[0] + 1e-12, te[-1] - 1e-12, out=t)
        np.maximum(peak, t, out=peak)
        if i % record_every == 0:
            rec_t[:, j] = t
            rec_u[:, j] = u
            j += 1
    if window_steps:
        return rec_t[:, :j], rec_u[:, :j], peak, cnt, ssum
    return rec_t[:, :j], rec_u[:, :j], peak
