import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
"""State-dependent fields for the generative (PDMP) model, on the 22x22 grid.

Everything here is distribution-form-agnostic: modulus, elastic PE drift,
event hazards, aging sizes, and the linear stress-to-PE jump coupling.
Missing voxels are filled by nearest occupied neighbor (used only where the
simulation wanders outside the well-sampled region).
"""
import numpy as np

GRID = int(_os.environ.get('GRID', 20))
SCRATCH = _os.path.join(_HERE, 'out') if GRID == 20 else _os.path.join(_HERE, 'out', 'grid%d' % GRID)


def build_fields(min_steps=200, min_events=30):
    d = np.load(SCRATCH + '/model_stats.npz')
    ev = np.load(SCRATCH + '/model_events.npz')
    NB = int(d['NB'])
    DG = float(d['DG'])
    n_all = d['n_all'].reshape(NB, NB)
    with np.errstate(all='ignore'):
        mu2 = (d['sum_dtau_ne'] / d['n_ne']).reshape(NB, NB) / DG   # 2*mu
        gdrift = (d['sum_du_ne'] / d['n_ne']).reshape(NB, NB)       # du per elastic step
        p_drop = (d['n_drop'] / d['n_all']).reshape(NB, NB)
        p_age = (d['n_age'] / d['n_all']).reshape(NB, NB)
        m_age = (d['sum_du_age'] / d['n_age']).reshape(NB, NB)      # mean PE drop of aging event
        s_mean = (d['sum_s'] / d['n_drop']).reshape(NB, NB)         # empirical mean drop size

    # jump coupling du = a + b*s + eps, per voxel with enough events
    S, DUe, EV = ev['S'], ev['DUe'], ev['EV']
    ka = np.full((NB, NB), np.nan)
    kb = np.full((NB, NB), np.nan)
    ksig = np.full((NB, NB), np.nan)
    for v in np.unique(EV):
        m = EV == v
        if m.sum() < min_events:
            continue
        s, du = S[m].astype(float), DUe[m].astype(float)
        A = np.column_stack([np.ones_like(s), s])
        coef, *_ = np.linalg.lstsq(A, du, rcond=None)
        r = du - A @ coef
        ka[v // NB, v % NB] = coef[0]
        kb[v // NB, v % NB] = coef[1]
        ksig[v // NB, v % NB] = r.std()

    good = (n_all >= min_steps) & np.isfinite(mu2)
    fields = dict(mu2=mu2, gdrift=gdrift, p_drop=p_drop, p_age=p_age,
                  m_age=m_age, s_mean=s_mean, ka=ka, kb=kb, ksig=ksig)
    # nearest-occupied fill
    ii, jj = np.meshgrid(np.arange(NB), np.arange(NB), indexing='ij')
    occ = np.argwhere(good)
    for k, f in fields.items():
        bad = ~good | ~np.isfinite(f)
        if k in ('ka', 'kb', 'ksig'):
            bad = ~np.isfinite(f)
        for i, j in np.argwhere(bad):
            nn = occ[np.argmin((occ[:, 0] - i) ** 2 + (occ[:, 1] - j) ** 2)]
            if k in ('ka', 'kb', 'ksig') and not np.isfinite(f[nn[0], nn[1]]):
                fin = np.argwhere(np.isfinite(f))
                nn = fin[np.argmin((fin[:, 0] - i) ** 2 + (fin[:, 1] - j) ** 2)]
            f[i, j] = f[nn[0], nn[1]]
    fields['good'] = good
    fields['ue'] = d['ue']
    fields['te'] = d['te']
    return fields


if __name__ == '__main__':
    f = build_fields()
    np.savez(SCRATCH + '/pdmp_fields.npz', **f)
    g = f['good']
    print('good voxels:', g.sum(), '/', g.size)
    print('2mu range (good):', np.nanpercentile(f['mu2'][g], [5, 50, 95]))
    print('p_drop range:', np.nanpercentile(f['p_drop'][g], [5, 50, 95]))
    print('p_age  range:', np.nanpercentile(f['p_age'][g], [5, 50, 95]))
    print('kernel b (du per unit s):', np.nanpercentile(f['kb'], [5, 50, 95]))
