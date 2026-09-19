"""Result 1: the size law has a ceiling, and its shape is global.

Input : out/events.npz
Output: out/shape.json        the global (k, m, eps)
        out/ceiling_table.npy the per-cell ceiling s_c, one number per cell

Three steps:

  (a) In the most populated cells of the (u, tau) grid, fit all four parameters
      (k, m, eps, s_c) freely, and fit the standard truncated power law
      s^-kappa exp(-s/s*) to the same events.  Both are proper normalised
      densities on [1e-6, ...), so their likelihoods are directly comparable.
      The ceiling form wins in every cell -- that is the result.

  (b) The free fits return nearly the same (k, m, eps) in every cell while s_c
      moves by an order of magnitude.  So the shape is taken to be global, at
      the median of the free fits, and only the ceiling is left free.  Using
      the median rather than a joint fit keeps the estimate insensitive to a
      single badly behaved cell.

  (c) With the shape frozen, one ceiling is fitted per cell, everywhere there
      are enough events to measure one.  That table is the reference the closed
      laws of step 3 have to beat.
"""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import size_law

N_SHAPE_CELLS = 8        # the free 4-parameter fits are done in this many cells
MIN_EVENTS_CEILING = 40  # fewer events than this and a ceiling is meaningless
CEILING_LO_FACTOR = 1.0001   # the ceiling must lie above the largest event ...
CEILING_HI_FACTOR = 51.0     # ... but not absurdly far above it

events = np.load('out/events.npz')
size = events['s']
cell = events['cell']
n_bins = int(events['n_bins'])

events_per_cell = np.bincount(cell, minlength=n_bins * n_bins)
shape_cells = np.argsort(-events_per_cell)[:N_SHAPE_CELLS]


def neg_log_likelihood_free(theta, s):
    """Mean -ln P for the size law with all four parameters free.

    eps and s_c are fitted as logarithms because both are positive scales that
    range over decades; k and m are O(1) and are fitted directly.
    """
    k, m, ln_eps, ln_sc = theta
    return size_law.mean_neg_log_likelihood(s, k, m, np.exp(ln_eps), np.exp(ln_sc))


def neg_log_likelihood_tpl(theta, s):
    """Mean -ln P for the truncated power law s^-kappa exp(-s/s*)."""
    kappa, ln_s_star = theta
    return size_law.mean_neg_log_likelihood_tpl(s, kappa, ln_s_star)


# --- (a) free fits, and the contest against the truncated power law ---------
print('cell      n     k       m       eps        s_c     | lnL ceiling   lnL tpl     dAIC')
free_parameters = []
contest = []
for c in shape_cells:
    s = size[cell == c]
    start_ceiling = np.array([0.96, 1.5, np.log(5e-5), np.log(s.max() * 1.05)])
    fit_ceiling = minimize(neg_log_likelihood_free, start_ceiling, args=(s,),
                           method='Nelder-Mead',
                           options=dict(xatol=1e-4, fatol=1e-8, maxiter=2000))
    start_tpl = np.array([1.0, np.log(0.05)])
    fit_tpl = minimize(neg_log_likelihood_tpl, start_tpl, args=(s,),
                       method='Nelder-Mead',
                       options=dict(xatol=1e-4, fatol=1e-8, maxiter=2000))
    k, m, ln_eps, ln_sc = fit_ceiling.x
    free_parameters.append([k, m, np.exp(ln_eps)])
    ll_ceiling = -fit_ceiling.fun * len(s)
    ll_tpl = -fit_tpl.fun * len(s)
    delta_aic = (2 * 2 - 2 * ll_tpl) - (2 * 4 - 2 * ll_ceiling)
    contest.append(dict(cell_u=int(c) // n_bins, cell_tau=int(c) % n_bins, n=len(s),
                        k=k, m=m, eps=float(np.exp(ln_eps)), s_c=float(np.exp(ln_sc)),
                        ll_ceiling=ll_ceiling, ll_tpl=ll_tpl, delta_aic=delta_aic,
                        kappa=fit_tpl.x[0], s_star=float(np.exp(fit_tpl.x[1]))))
    print('(%2d,%2d) %5d  %.4f  %.4f  %.3e  %.4f | %10.1f %10.1f  %+8.1f'
          % (int(c) // n_bins, int(c) % n_bins, len(s), k, m, np.exp(ln_eps),
             np.exp(ln_sc), ll_ceiling, ll_tpl, delta_aic))

# --- (b) the global shape ---------------------------------------------------
free_parameters = np.array(free_parameters)
k_global = float(np.median(free_parameters[:, 0]))
m_global = float(np.median(free_parameters[:, 1]))
eps_global = float(np.median(free_parameters[:, 2]))
print('\nglobal shape: k = %.5f   m = %.5f   eps = %.4e' % (k_global, m_global, eps_global))
print('spread over cells: k %.4f-%.4f   m %.4f-%.4f'
      % (free_parameters[:, 0].min(), free_parameters[:, 0].max(),
         free_parameters[:, 1].min(), free_parameters[:, 1].max()))
print('the ceiling law beats the truncated power law in %d of %d cells'
      % (sum(1 for r in contest if r['delta_aic'] > 0), len(contest)))


# --- (c) one ceiling per cell, shape frozen ---------------------------------
def fit_ceiling_of_cell(s):
    """The ceiling that maximises the likelihood of one cell's sizes.

    One bounded scalar search: with the shape frozen the likelihood is a smooth
    function of ln s_c alone.  The ceiling can only lie above the largest event
    seen in the cell, which is what the lower bound encodes.
    """
    lo = np.log(s.max() * CEILING_LO_FACTOR)
    hi = np.log(s.max() * CEILING_HI_FACTOR)

    def objective(ln_sc):
        return size_law.mean_neg_log_likelihood(s, k_global, m_global, eps_global,
                                                np.exp(ln_sc),
                                                size_law.N_INTEGRATION_TABLE)

    result = minimize_scalar(objective, bounds=(lo, hi), method='bounded',
                             options=dict(xatol=1e-4))
    return float(np.exp(result.x))


ceiling_table = np.full(n_bins * n_bins, np.nan)
for c in np.unique(cell):
    s = size[cell == c]
    if len(s) >= MIN_EVENTS_CEILING:
        ceiling_table[c] = fit_ceiling_of_cell(s)
n_fitted = int(np.isfinite(ceiling_table).sum())
print('\nceilings fitted in %d of %d occupied cells (>= %d events each)'
      % (n_fitted, len(np.unique(cell)), MIN_EVENTS_CEILING))
print('ceiling range %.4f to %.4f' % (np.nanmin(ceiling_table), np.nanmax(ceiling_table)))

np.save('out/ceiling_table.npy', ceiling_table)
with open('out/shape.json', 'w') as handle:
    json.dump(dict(k=k_global, m=m_global, eps=eps_global,
                   n_shape_cells=N_SHAPE_CELLS, n_ceiling_cells=n_fitted,
                   cells=contest), handle, indent=1)
print('wrote out/shape.json and out/ceiling_table.npy')
