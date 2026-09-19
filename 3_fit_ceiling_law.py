"""Result 2: the ceiling itself obeys a two-parameter closed law.

Input : out/events.npz, out/shape.json, out/ceiling_table.npy
Output: out/ceiling_law.json

With the shape (k, m, eps) frozen at the step-2 values, the ceiling is
evaluated at every event's OWN state (u, tau) -- no grid, no binning -- and the
question is how few parameters it takes to do as well as the 353 measured
per-cell ceilings.  The contest:

    constant            ln s_c = a
    per-cell table      one fitted ceiling per (u, tau) cell   (the reference)
    product             ln s_c = a + b (100 u tau)
    product + rate      ... + c ln(rate / rate_mid)            (not closed: it
                        needs the sample's preparation history)
    product + tau_max   ... + c tau_max                        (closed again:
                        tau_max is a state variable of the trajectory)

Everything is scored by log-likelihood, in-sample and held out.  Held-out means
2-fold cross-validation blocked by TRAJECTORY (even trajectory ids against odd),
not by event: events within one trajectory are strongly correlated, so a random
event split would let a model memorise its training trajectories and still look
predictive.  The per-cell table has 353 parameters and the closed laws have 2
or 3, so the held-out column is the one that matters.
"""
import json
import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import size_law

U_FLOOR = 0.001       # u and tau are clipped from below before entering the law:
TAU_FLOOR = 0.02      # a handful of events sit at essentially zero u or tau and
                      # would otherwise dominate through 1/u or ln tau terms
MIN_EVENTS_CELL = 5   # cells with fewer events get no entry in the table
CEILING_SCAN_POINTS = 60
START_LIFT = 0.3      # lift the least-squares start above the measured ceilings

start_time = time.time()
events = np.load('out/events.npz')
with open('out/shape.json') as handle:
    shape = json.load(handle)
K = shape['k']
M = shape['m']
EPS = shape['eps']

size = events['s']
u = np.maximum(events['u'], U_FLOOR)
tau = np.maximum(events['tau'], TAU_FLOOR)
tau_max = events['tau_max']
trajectory = events['traj']
cell = events['cell']
rate = events['rate']
n_events = len(size)
n_bins = int(events['n_bins'])

# The cooling rate enters only as a ratio to the middle of the nine rates, so
# that its coefficient is dimensionless and the intercept keeps its meaning.
distinct_rates = np.unique(rate)
ln_rate = np.log(rate / distinct_rates[len(distinct_rates) // 2])

ln_ceiling_grid, ln_norm_table = size_law.tabulate_ln_normalisation(K, M, EPS)
print('%d events, shape k=%.5f m=%.5f eps=%.3e, leak %.0e'
      % (n_events, K, M, EPS, size_law.LEAK_FRACTION))
print('clipped: u < %.0e in %d events, tau < %.2f in %d events'
      % (U_FLOOR, int((events['u'] < U_FLOOR).sum()), TAU_FLOOR,
         int((events['tau'] < TAU_FLOOR).sum())))


# --- the models: each maps parameters and states to ln s_c ------------------
def ln_sc_constant(theta, u, tau, tau_max, ln_rate):
    return np.full(len(u), theta[0])


def ln_sc_product(theta, u, tau, tau_max, ln_rate):
    """The closed law: the ceiling grows exponentially in the product u tau.

    u is multiplied by 100 only to make b an O(1) number.
    """
    return theta[0] + theta[1] * 100.0 * u * tau


def ln_sc_product_rate(theta, u, tau, tau_max, ln_rate):
    """The product law times a power of the cooling rate: memory of preparation."""
    return theta[0] + theta[1] * 100.0 * u * tau + theta[2] * ln_rate


def ln_sc_product_taumax(theta, u, tau, tau_max, ln_rate):
    """The product law times exp(c tau_max): memory of the largest stress so far."""
    return theta[0] + theta[1] * 100.0 * u * tau + theta[2] * tau_max


# --- likelihood, fitting, scoring -------------------------------------------
def log_likelihood(model_function, theta, mask):
    """Total ln L over the selected events, and how many of them lie above
    their own predicted ceiling (those are carried by the leak)."""
    ln_sc = model_function(theta, u[mask], tau[mask], tau_max[mask], ln_rate[mask])
    ln_p, inside = size_law.ln_density_with_leak(size[mask], ln_sc, K, M, EPS,
                                                 ln_ceiling_grid, ln_norm_table)
    return ln_p.sum(), int((~inside).sum())


def fit(model_function, theta_start, mask):
    """Maximise the likelihood over the selected events.

    Nelder-Mead first because the leak makes the surface mildly non-smooth,
    then Powell from its answer to polish along the parameter axes.
    """
    def objective(theta):
        return -log_likelihood(model_function, theta, mask)[0]

    result = minimize(objective, np.asarray(theta_start, float), method='Nelder-Mead',
                      options=dict(xatol=1e-5, fatol=1e-4, maxiter=6000,
                                   maxfev=6000, adaptive=True))
    result = minimize(objective, result.x, method='Powell',
                      options=dict(xtol=1e-5, ftol=1e-9))
    return result.x, -result.fun


# --- the per-cell reference table -------------------------------------------
def fit_cell_ceilings(mask):
    """One ceiling per cell, fitted on the selected events, with the same leak.

    The search starts at the 90th percentile of the cell's sizes rather than at
    its maximum: with a leak present the best ceiling may sit BELOW the largest
    event, which is then explained as background.  That makes the likelihood
    non-monotonic in ln s_c, hence the coarse scan before the local search.
    """
    table = np.full(n_bins * n_bins, np.nan)
    index = np.nonzero(mask)[0]
    for c in np.unique(cell[index]):
        in_cell = index[cell[index] == c]
        if len(in_cell) < MIN_EVENTS_CELL:
            continue

        def objective(ln_sc):
            return -log_likelihood(ln_sc_constant, [ln_sc], in_cell)[0]

        lo = np.log(max(np.percentile(size[in_cell], 90), 1e-4))
        hi = np.log(50.0)
        scan = np.linspace(lo, hi, CEILING_SCAN_POINTS)
        values = [objective(x) for x in scan]
        j = int(np.argmin(values))
        bracket = (scan[max(j - 1, 0)], scan[min(j + 1, CEILING_SCAN_POINTS - 1)])
        result = minimize_scalar(objective, bounds=bracket, method='bounded',
                                 options=dict(xatol=1e-4))
        table[c] = result.x
    return table


def score_cell_ceilings(table, mask):
    """Score a per-cell table on the selected events.

    A held-out event can land in a cell the training fold never visited; it is
    given the median ceiling rather than being dropped, so that every model is
    scored on exactly the same events.
    """
    ln_sc = table[cell[mask]]
    ln_sc = np.where(np.isfinite(ln_sc), ln_sc, np.nanmedian(table))
    ln_p, inside = size_law.ln_density_with_leak(size[mask], ln_sc, K, M, EPS,
                                                 ln_ceiling_grid, ln_norm_table)
    return ln_p.sum(), int((~inside).sum())


# --- starting values --------------------------------------------------------
# Least squares of the measured per-cell ln ceilings on the three regressors,
# then lifted by START_LIFT so the starting surface sits above most events.
# This is only a starting point; the likelihood fit moves freely from here.
measured_ceiling = np.load('out/ceiling_table.npy')
ln_measured = np.log(measured_ceiling[cell])
has_ceiling = np.isfinite(ln_measured)
design = np.column_stack([np.ones(n_events), 100.0 * u * tau, ln_rate, tau_max])
coefficients, _, _, _ = np.linalg.lstsq(design[has_ceiling], ln_measured[has_ceiling],
                                        rcond=None)
coefficients[0] += START_LIFT

everything = np.ones(n_events, bool)
even_trajectories = (trajectory % 2) == 0
results = {}


def run(name, model_function, theta_start):
    """Fit in-sample, then refit on each fold and score on the other."""
    theta, ll = fit(model_function, theta_start, everything)
    _, violations = log_likelihood(model_function, theta, everything)
    held_out = 0.0
    held_out_violations = 0
    for train in (even_trajectories, ~even_trajectories):
        theta_fold, _ = fit(model_function, theta, train)
        ll_fold, violations_fold = log_likelihood(model_function, theta_fold, ~train)
        held_out += ll_fold
        held_out_violations += violations_fold
    results[name] = dict(n_parameters=len(theta), ll=float(ll), violations=violations,
                         held_out_ll=float(held_out), held_out_violations=held_out_violations,
                         theta=[float(x) for x in theta])
    return results[name]


run('constant', ln_sc_constant, [np.log(np.nanmedian(measured_ceiling))])

# the per-cell table: fitted on everything, then once per fold
table = fit_cell_ceilings(everything)
ll_cells, violations_cells = score_cell_ceilings(table, everything)
held_out_cells = 0.0
held_out_violations_cells = 0
for train in (even_trajectories, ~even_trajectories):
    fold_table = fit_cell_ceilings(train)
    ll_fold, violations_fold = score_cell_ceilings(fold_table, ~train)
    held_out_cells += ll_fold
    held_out_violations_cells += violations_fold
results['per-cell table'] = dict(n_parameters=int(np.isfinite(table).sum()),
                                 ll=float(ll_cells), violations=violations_cells,
                                 held_out_ll=float(held_out_cells),
                                 held_out_violations=held_out_violations_cells,
                                 theta=None)

run('product', ln_sc_product, coefficients[[0, 1]])
run('product + cooling rate', ln_sc_product_rate, coefficients[[0, 1, 2]])
run('product + tau_max', ln_sc_product_taumax, coefficients[[0, 1, 3]])

# --- report -----------------------------------------------------------------
ll_constant = results['constant']['ll']
held_out_constant = results['constant']['held_out_ll']
print('\nconstant-ceiling reference: lnL %.1f in sample, %.1f held out\n'
      % (ll_constant, held_out_constant))
print('%-24s %5s %12s %12s %8s' % ('model', 'par', 'gain lnL', 'gain held-out', 'above s_c'))
for name, row in results.items():
    print('%-24s %5d %12.0f %12.0f %8d'
          % (name, row['n_parameters'], row['ll'] - ll_constant,
             row['held_out_ll'] - held_out_constant, row['held_out_violations']))
print()
theta = results['product']['theta']
print('product          s_c = %.5f * exp(%.4f * 100 u tau)' % (np.exp(theta[0]), theta[1]))
theta = results['product + cooling rate']['theta']
print('  + cooling rate s_c = %.5f * exp(%.4f * 100 u tau) * (rate/rate_mid)^%.4f'
      % (np.exp(theta[0]), theta[1], theta[2]))
theta = results['product + tau_max']['theta']
print('  + tau_max      s_c = %.5f * exp(%.4f * 100 u tau) * exp(%.4f * tau_max)'
      % (np.exp(theta[0]), theta[1], theta[2]))

with open('out/ceiling_law.json', 'w') as handle:
    json.dump(dict(n_events=n_events, shape=dict(k=K, m=M, eps=EPS),
                   leak=size_law.LEAK_FRACTION, models=results), handle, indent=1)
print('\nwrote out/ceiling_law.json  (%.0fs)' % (time.time() - start_time))
