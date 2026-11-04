"""Joint catalogue of stress drops and PE drops, for the copula analysis.

Input : df_clean.pkl
Output: out/joint_events.npz -- one entry per event, with SIGNED du and dtau

Event definition (the notebook's, not step 1's):
    a step is an event when dtau < 0 OR du < 0
so an event can be a stress drop, a PE drop, or both.  du and dtau are kept
signed, so later steps can choose their own subsets:
    both drop      du < 0 and dtau < 0   (the copula sample)
    stress only    dtau < 0, du >= 0     ("aging": PE rises during the drop)
    PE only        du < 0, dtau >= 0
No size cut is applied here; the lower truncation bounds are chosen in step 6.

Conventions are those of 1_extract_events.py: u = pe - U_REFERENCE,
tau = stress / 1e4, the state is taken at the START of the step.  The (u, tau)
grid is rebuilt with exactly step 1's code and constants, so cell numbers here
are the same cells as in out/events.npz and step 2.  Visit counts per cell
(all steps, events or not) are saved so that "most visited" can be found.

Runs one pass over the pickle.  Needs ~10 GB of RAM for a few minutes.
"""
import os
import time
import numpy as np
import pandas as pd

PICKLE_PATH = 'df_clean.pkl'
U_REFERENCE = -4.60751861
N_STEPS = 49999
STRESS_UNIT = 1e4
GRID_INNER = 20
QUANTILE_LO = 0.002
QUANTILE_HI = 0.998

start_time = time.time()

# --- load, and reshape into (trajectory, step) -- identical to step 1 -------
frame = pd.read_pickle(PICKLE_PATH).sort_values(['index', 'strain_index'])
cooling_rate_of_trajectory = frame['cooling_rate'].to_numpy()[::N_STEPS].copy()
strain_axis = frame['strain'].to_numpy()[:N_STEPS].copy()
u_grid = (frame['pe'].to_numpy() - U_REFERENCE).reshape(-1, N_STEPS).astype(np.float64)
tau_grid = (frame['stress'].to_numpy() / STRESS_UNIT).reshape(-1, N_STEPS).astype(np.float64)
del frame
n_trajectories = len(cooling_rate_of_trajectory)
print('loaded %.0fs  %d trajectories x %d steps' % (time.time() - start_time, n_trajectories, N_STEPS))

u_step = u_grid[:, :-1].ravel()
tau_step = tau_grid[:, :-1].ravel()
d_tau = np.diff(tau_grid, axis=1).ravel()
d_u = np.diff(u_grid, axis=1).ravel()
del u_grid, tau_grid

# --- the grid -- identical to step 1 ----------------------------------------
u_quantiles = np.quantile(u_step, [QUANTILE_LO, QUANTILE_HI])
tau_quantiles = np.quantile(tau_step, [QUANTILE_LO, QUANTILE_HI])
u_edges = np.concatenate(([u_step.min() - 1e-9],
                          np.linspace(u_quantiles[0], u_quantiles[1], GRID_INNER + 1),
                          [u_step.max() + 1e-9]))
tau_edges = np.concatenate(([tau_step.min() - 1e-9],
                            np.linspace(tau_quantiles[0], tau_quantiles[1], GRID_INNER + 1),
                            [tau_step.max() + 1e-9]))
n_bins = GRID_INNER + 2


def cell_of(u, tau):
    bin_u = np.clip(np.searchsorted(u_edges, u, side='right') - 1, 0, n_bins - 1)
    bin_tau = np.clip(np.searchsorted(tau_edges, tau, side='right') - 1, 0, n_bins - 1)
    return (bin_u * n_bins + bin_tau).astype(np.int32)


# visits: every step counts, so the most-visited cell is about where the
# system spends its time, not where events happen
visits = np.bincount(cell_of(u_step, tau_step), minlength=n_bins * n_bins)
print('grid %.0fs' % (time.time() - start_time))

# --- select the events -------------------------------------------------------
is_event = (d_tau < 0) | (d_u < 0)
du = d_u[is_event]
dtau = d_tau[is_event]
u_event = u_step[is_event]
tau_event = tau_step[is_event]
del d_u, d_tau, u_step, tau_step

step_index = np.tile(np.arange(N_STEPS - 1, dtype=np.int32), n_trajectories)[is_event]
trajectory = np.repeat(np.arange(n_trajectories, dtype=np.int32), N_STEPS - 1)[is_event]
strain_event = strain_axis[step_index]
rate_event = cooling_rate_of_trajectory[trajectory]
cell = cell_of(u_event, tau_event)
del is_event, step_index

os.makedirs('out', exist_ok=True)
np.savez_compressed('out/joint_events.npz',
                    du=du, dtau=dtau, u=u_event, tau=tau_event,
                    strain=strain_event, traj=trajectory, rate=rate_event,
                    cell=cell, visits=visits, n_bins=n_bins,
                    u_edges=u_edges, tau_edges=tau_edges)


# --- report -----------------------------------------------------------------
def quadrant_counts(mask):
    both = int(((du < 0) & (dtau < 0) & mask).sum())
    stress_only = int(((dtau < 0) & (du >= 0) & mask).sum())
    pe_only = int(((du < 0) & (dtau >= 0) & mask).sum())
    return both, stress_only, pe_only


def report(label, mask):
    both, stress_only, pe_only = quadrant_counts(mask)
    total = both + stress_only + pe_only
    print('%-22s total %8d | both drop %8d (%4.1f%%) | stress only %8d (%4.1f%%) | PE only %8d (%4.1f%%)'
          % (label, total, both, 100 * both / total, stress_only, 100 * stress_only / total,
             pe_only, 100 * pe_only / total))


print('\n%d events (dtau < 0 or du < 0)' % len(du))
report('all cells', np.ones(len(du), bool))
top = int(np.argmax(visits))
in_top = cell == top
print('most-visited cell %d = (%d, %d): u in [%.5f, %.5f), tau in [%.4f, %.4f), %d visits'
      % (top, top // n_bins, top % n_bins,
         u_edges[top // n_bins], u_edges[top // n_bins + 1],
         tau_edges[top % n_bins], tau_edges[top % n_bins + 1], visits[top]))
report('most-visited cell', in_top)
print('\nwrote out/joint_events.npz  (%.0fs)' % (time.time() - start_time))
