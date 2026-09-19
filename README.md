# Stress-drop statistics of a sheared glass

850 athermal quasistatic shear trajectories of a glass, at 9 cooling rates,
49,999 strain steps each (strain step 1e-5).  Two results, five files.

## Data conventions

| symbol | definition |
|---|---|
| `u`   | `pe - (-4.60751861)`, potential energy above the deep-glass reference |
| `tau` | `stress / 1e4` |
| event | a strain step with `dtau < 0`; its size is `s = -dtau` |
| state | `(u, tau)` at the **start** of the step, i.e. just before the drop |
| cut   | events with `s < 1e-6` are discarded as numerical noise (404 of 187,468) |

That leaves **187,064 events**.  A 22x22 `(u, tau)` grid (inner 20x20 on the
0.2%-99.8% quantiles of all visited step states, plus an outer catch-all ring)
is used only for the per-cell reference numbers; the closed laws never use it.

## Result 1 - the size distribution has a ceiling, and its shape is global

At a fixed state,

    P(s | u, tau) = (s_c - s)^m / (s + eps)^k / Z      for 1e-6 <= s < s_c

with a **global** shape and one ceiling per state:

    k = 0.87918      m = 2.27327      eps = 2.745e-5

The shape is the median of free four-parameter fits in the eight most populated
cells (>= 5,000 events each).  In those same cells the ceiling form beats the
standard truncated power law `s^-kappa exp(-s/s*)` on likelihood **in all eight**,
by dAIC = +194 to +386.  The ceiling is then fitted per cell with the shape
frozen: 277 cells have the >= 40 events needed to measure one, and their ceilings
span 0.021 to 2.99.

## Result 2 - the ceiling obeys a closed two-parameter law

With the shape frozen, the ceiling is evaluated at every event's **own** state,
so no grid enters.  Scores are log-likelihood, in sample and held out by 2-fold
cross-validation blocked by trajectory (even vs odd trajectory id).  Gains are
quoted against a constant ceiling (`lnL = 602,479` in sample, `602,478` held out).

| model | par | gain lnL | gain held-out | events above `s_c` |
|---|---|---|---|---|
| per-cell table (reference) | 353 | +8,367 | +7,354 | 104 |
| `s_c = 0.0619 exp(0.914 * 100 u tau)` | 2 | +7,262 | +7,262 | 72 |
| ... `* (rate/rate_mid)^-0.110` | 3 | +8,320 | +8,316 | 31 |
| ... `* exp(0.646 tau_max)` | 3 | +8,439 | +8,435 | 20 |

Two parameters reproduce 353 measured ceilings to within 1.3% of the held-out
likelihood gain.  Adding one memory variable beats the per-cell table outright:
the cooling rate does it, but the cooling rate is not a state variable, so the
law is no longer closed.  `tau_max`, the largest `tau` the trajectory has
reached so far (running maximum over **all** steps, not only over events), does
it better and keeps the law closed:

    s_c = 0.02587 * exp(80.50 * u * tau) * exp(0.6461 * tau_max)

In `figures/ceiling_law.png` the law runs above the per-cell ceilings at small
`u tau`.  Those cells hold few events, and a ceiling measured from few events
sits just above the largest event seen, which is an underestimate.  The law is
fitted to the events, not to those points.

Every model carries the same fixed leak, `P = (1 - PI) P_ceiling + PI * log-uniform
on [1e-6, 5]` with `PI = 1e-3` never fitted: a smooth surface cannot sit above
every one of 187,064 events, and without the leak a single outlier would send
the likelihood to `-inf`.  The last column counts the events that the leak is
carrying.

## Running it

    ./run_all.sh

Step 1 reads `df_clean.pkl` (2.4 GB; ~10 GB of RAM while it runs) and is the only
step that touches it.  Its path is the constant `PICKLE_PATH` at the top of
`1_extract_events.py`.  Everything afterwards reads `out/events.npz` (6 MB).

| file | lines | what it does | runtime |
|---|---|---|---|
| `size_law.py` | 151 | the only shared module: the size law, its numerical normalisation, the leak mixture, the truncated power law | - |
| `1_extract_events.py` | 106 | pickle -> `out/events.npz` (s, u, tau, tau_max, strain, traj, rate, cell), one pass | 4 s |
| `2_fit_size_law.py` | 137 | Result 1 -> `out/shape.json`, `out/ceiling_table.npy` | 1 s |
| `3_fit_ceiling_law.py` | 245 | Result 2 -> `out/ceiling_law.json` | 11 s |
| `4_figures.py` | 116 | `figures/size_law.png`, `figures/ceiling_law.png` | 1 s |

(numpy 2.3, scipy 1.16, pandas 2.3, matplotlib 3.10; times on an Apple M-series laptop.)

## Numerical choices that matter

They are named constants at the top of each file.  The ones that move the
answer: the size cut `XMIN = 1e-6`; the leak `PI = 1e-3` and its background
range `[1e-6, 5]`; the 1500-point log integration grid used for the tabulated
`ln Z` and the 240-point ceiling grid `1e-4 .. 50` it is tabulated on; a
4000-point grid for the free fits, which vary the shape and need it; `u` and
`tau` clipped from below at 1e-3 and 0.02 before entering the ceiling law
(41 and 135 events); and even/odd trajectory id as the cross-validation folds.

The eight cells of Result 1 are chosen as **the eight most populated cells** on
the grid.  The rule "every cell with >= 3,000 events" is not equivalent: it
selects 20 cells and gives `k = 0.882, m = 2.035, eps = 2.52e-5`.  The `m` of a
single-cell fit is poorly determined when the cell's largest events are few
(one of the eight returns `m = 25.9`), which is why the global shape is a
median and not a joint fit.
