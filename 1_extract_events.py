"""Turn the raw trajectories into a flat catalogue of stress-drop events.

Input : df_clean.pkl -- 850 athermal quasistatic shear trajectories of a glass
        (9 cooling rates), 49,999 strain steps each, strain step 1e-5, with
        columns index, strain_index, strain, stress, pe, cooling_rate.
Output: out/events.npz -- one entry per stress-drop event.

Conventions (used by every later step):
    u   = pe - U_REFERENCE      potential energy above the deep-glass reference
    tau = stress / 1e4          stress in the units the size law is written in
    a step is an event when dtau < 0, and its size is s = -dtau
    the state (u, tau) of an event is taken at the START of the step, i.e.
    just before the drop, which is the state the drop is a response to
    events with s < 1e-6 are discarded as numerical noise

The (u, tau) grid is only used for the per-cell reference numbers in steps 2
and 3; the closed laws never touch it.  It is 22 x 22: an inner 20 x 20 on the
0.2%-99.8% quantiles of all visited step states, plus an outer ring so that
every event, however extreme its state, lands in some cell.

Runs one pass over the pickle.  Needs ~10 GB of RAM for a few minutes.
"""
import time
import numpy as np
import pandas as pd

PICKLE_PATH = 'df_clean.pkl'            # 2.4 GB, not in git
U_REFERENCE = -4.60751861
N_STEPS = 49999
STRESS_UNIT = 1e4
XMIN = 1e-6
GRID_INNER = 20                        # inner bins per axis; +2 for the outer ring
QUANTILE_LO = 0.002
QUANTILE_HI = 0.998

start_time = time.time()

# --- load, and reshape into (trajectory, step) ------------------------------
frame = pd.read_pickle(PICKLE_PATH).sort_values(['index', 'strain_index'])
cooling_rate_of_trajectory = frame['cooling_rate'].to_numpy()[::N_STEPS].copy()
strain_axis = frame['strain'].to_numpy()[:N_STEPS].copy()
u_grid = (frame['pe'].to_numpy() - U_REFERENCE).reshape(-1, N_STEPS).astype(np.float64)
tau_grid = (frame['stress'].to_numpy() / STRESS_UNIT).reshape(-1, N_STEPS).astype(np.float64)
del frame
n_trajectories = len(cooling_rate_of_trajectory)
print('loaded %.0fs  %d trajectories x %d steps' % (time.time() - start_time, n_trajectories, N_STEPS))

# The running maximum of tau is taken over ALL steps of the trajectory, not
# only over the steps that happen to be events: it is the largest stress the
# sample has ever carried, which is what the memory term in step 3 tests.
tau_running_max = np.maximum.accumulate(tau_grid, axis=1)

u_step = u_grid[:, :-1].ravel()                  # state at the start of each step
tau_step = tau_grid[:, :-1].ravel()
tau_max_step = tau_running_max[:, :-1].ravel()
d_tau = np.diff(tau_grid, axis=1).ravel()
del u_grid, tau_grid, tau_running_max

# --- the grid, from the distribution of visited states ----------------------
u_quantiles = np.quantile(u_step, [QUANTILE_LO, QUANTILE_HI])
tau_quantiles = np.quantile(tau_step, [QUANTILE_LO, QUANTILE_HI])
u_edges = np.concatenate(([u_step.min() - 1e-9],
                          np.linspace(u_quantiles[0], u_quantiles[1], GRID_INNER + 1),
                          [u_step.max() + 1e-9]))
tau_edges = np.concatenate(([tau_step.min() - 1e-9],
                            np.linspace(tau_quantiles[0], tau_quantiles[1], GRID_INNER + 1),
                            [tau_step.max() + 1e-9]))
n_bins = GRID_INNER + 2
print('grid %.0fs  u in [%.4f, %.4f]  tau in [%.4f, %.4f]'
      % (time.time() - start_time, u_quantiles[0], u_quantiles[1], tau_quantiles[0], tau_quantiles[1]))

# --- select the events ------------------------------------------------------
is_event = d_tau < 0
n_drops = int(is_event.sum())
size = -d_tau[is_event]
u_event = u_step[is_event]
tau_event = tau_step[is_event]
tau_max_event = tau_max_step[is_event]
del d_tau, u_step, tau_step, tau_max_step

step_index = np.tile(np.arange(N_STEPS - 1, dtype=np.int32), n_trajectories)[is_event]
trajectory = np.repeat(np.arange(n_trajectories, dtype=np.int32), N_STEPS - 1)[is_event]
strain_event = strain_axis[step_index]
rate_event = cooling_rate_of_trajectory[trajectory]
del is_event, step_index

keep = size >= XMIN
print('stress drops %d, of which %d have s >= %.0e' % (n_drops, keep.sum(), XMIN))
size = size[keep]
u_event = u_event[keep]
tau_event = tau_event[keep]
tau_max_event = tau_max_event[keep]
strain_event = strain_event[keep]
trajectory = trajectory[keep]
rate_event = rate_event[keep]

bin_u = np.clip(np.searchsorted(u_edges, u_event, side='right') - 1, 0, n_bins - 1)
bin_tau = np.clip(np.searchsorted(tau_edges, tau_event, side='right') - 1, 0, n_bins - 1)
cell = (bin_u * n_bins + bin_tau).astype(np.int32)

np.savez_compressed('out/events.npz',
                    s=size, u=u_event, tau=tau_event, tau_max=tau_max_event,
                    strain=strain_event, traj=trajectory, rate=rate_event,
                    cell=cell, n_bins=n_bins, u_edges=u_edges, tau_edges=tau_edges)
print('wrote out/events.npz  %d events in %d occupied cells  (%.0fs)'
      % (len(size), len(np.unique(cell)), time.time() - start_time))
