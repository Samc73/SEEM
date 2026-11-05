# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research analysis of stress-drop events in athermal quasistatic (AQS) shear simulations of a glass. The data is 850 trajectories (9 cooling rates) × 49,999 strain steps (step 1e-5). The goal is a statistical law for stress-drop sizes as a function of the state `(u, tau)`. There is no package, build system, or test suite: the analysis is plain Python scripts plus Jupyter notebooks.

## Running the pipeline

The numbered scripts form a pipeline. Run them in order from this directory. Each one reads the outputs of the earlier steps from `out/`:

```
python 1_extract_events.py    # df_clean.pkl            -> out/events.npz   (~10 GB RAM, a few minutes)
python 2_fit_size_law.py      # events                  -> out/shape.json, out/ceiling_table.npy
python 3_fit_ceiling_law.py   # events, shape, table    -> out/ceiling_law.json   (slowest fitting step)
python 4_figures.py           # all of out/             -> figures/size_law.png, figures/ceiling_law.png

# copula analysis (branch copula-analysis); independent of steps 1-4
python 5_extract_joint_events.py   # df_clean.pkl  -> out/joint_events.npz   (~10 GB RAM, ~15 s)
python 6_fit_marginals.py          # joint events  -> out/marginals.json, out/marginal_table.npz
python test_copulas.py             # synthetic checks of copulas.py; run before step 7 after any change
python 7_fit_copulas.py            # marginals     -> out/copulas.json   (most-visited cell only)
```

- Scripts 2–4 `import size_law` from this folder. `C:\Users\scohe\Downloads` holds another copy of `size_law.py` and scripts 1–4 with its own `out/` and `figures/`. That copy is separate: changes made here don't reach it.
- Dependencies: numpy, pandas, scipy, matplotlib (scikit-learn for the notebook). `python` on PATH is the standalone Python 3.14 in `AppData\Local\Programs\Python\Python314`.
- There is no test suite. `python -c` checks on synthetic samples (draw from a known law, refit, compare) are how fitting code has been validated.

## Data files (not in git)

- `df_clean.pkl` (2.4 GB): the raw input. Columns `index` (trajectory id), `strain_index`, `strain`, `stress`, `pe`, `cooling_rate`. Rows are sorted by `(index, strain_index)`, and step 1 reshapes them to `(trajectory, step)`. That reshape assumes every trajectory has exactly `N_STEPS = 49999` rows.
- `df_clean_with_deltas.pkl.gz` (1.3 GB): written once by the notebook and never read back by any code.
- `tmp_dict.pkl`: old scratch file that no current code reads or writes.

Never read these with tools that load the whole file into context. `*.pkl`, `*.pkl.gz` and `out/` are in `.gitignore`. Don't commit the notebooks' PNGs or `Untitled.ipynb`.

## Conventions shared across all steps

These are defined in `1_extract_events.py` and assumed by everything after it:
- `u = pe - U_REFERENCE` (with `U_REFERENCE = -4.60751861`) and `tau = stress / 1e4`.
- An event is a step with `dtau < 0`. Its size is `s = -dtau`, and events with `s < 1e-6` (`XMIN`) are dropped as noise. `pe_drop = -du`.
- The state `(u, tau, tau_max)` of an event is taken at the start of the step, just before the drop. `tau_max` is the running maximum of tau over all steps, not only over events.
- The `(u, tau)` grid is 22 × 22: an inner 20 × 20 on the 0.2–99.8% quantiles, plus an outer ring so that every event lands in a cell. Cells are flattened as `cell = bin_u * n_bins + bin_tau`. The grid is only for per-cell reference fits. The closed laws are evaluated at each event's own state.

## The model (`size_law.py`)

`P(s | u, tau) = (s_c - s)^m / (s + eps)^k / Z(s_c)` for `XMIN <= s < s_c`. This is a power law with a hard ceiling `s_c`. `Z` is integrated numerically on a log grid that is refined near the ceiling. Only `s_c` depends on the state. `(k, m, eps)` are global.

- Step 2 fits all four parameters freely in the 8 most populated cells and compares them against a truncated power law `s^-kappa exp(-s/s*)` by AIC. It then freezes the shape at the median `(k, m, eps)` and fits one `s_c` per cell. That gives the reference table.
- Step 3 tests closed laws for `ln s_c`: constant, `a + b·100·u·tau`, then `+ c·ln(rate/rate_mid)` or `+ c·tau_max`. These are scored against the per-cell table. With the shape frozen, `ln Z` is tabulated once over a grid of ceilings and interpolated. A small log-uniform background "leak" (`LEAK_FRACTION = 1e-3`) keeps events above a predicted ceiling from giving `-inf`.
- Held-out scoring is 2-fold cross-validation blocked by trajectory (even ids against odd), never a random event split, because events within one trajectory are strongly correlated. The held-out log-likelihood is the number that decides between models.
- Optimisation is Nelder-Mead, then Powell. Positive scales (`eps`, `s_c`, `s*`) are fitted as logarithms.

## Copula analysis (steps 5+)

The joint distribution of stress drop `-dtau` and PE drop `-du`. It uses different conventions from steps 1–3:
- **Event definition:** the notebook's `dtau < 0 OR du < 0`, not step 1's `dtau < 0`. `du` and `dtau` are stored signed. Step 5 rebuilds step 1's grid with the same code, so cell ids match step 1 and step 2. Stress drops at cutoff 1e-6 are exactly step 2's events, and step 6's TPL fits on them reproduce `out/shape.json`.
- **Cutoffs:** PE ≥ 1e-6 (below it is a numerical-noise bump near 3e-8 that holds nearly all "PE only" events). Stress ≥ 1e-3, which excludes most stress-only "aging" drops (PE rises during the drop). Step 6 also stores the stress fits at 1e-6 and 1e-4 as a sensitivity check.
- **Separate subsets:** each marginal is fitted on all of its own drops, and the copula only on paired drops (both negative). The PIT of paired events under the marginals is therefore not exactly uniform, so rank-based (pseudo-observation) copula fits are the reference and IFM fits are secondary.
- **Marginal:** truncated power law `s^-kappa exp(-s/s*)` on `s >= xmin`, in `marginals.py` (free `xmin`, unlike `size_law.py`'s fixed 1e-6). Per-cell fits need ≥ 40 drops, and maps show only cells with ≥ 100.
- **Copula fits** are done in the most-visited cell, (18, 11) = cell 407. Families: independence, gaussian, t, clayton, gumbel, frank, bb1 and bb7, plus 180° rotations (`survival_*`). BB1 contains Gumbel (θ → 0) and Clayton (δ = 1), and BB7 contains Clayton (θ = 1), so a BB fit with a bound flag on those parameters has collapsed to that one-parameter family. That is a finding, not an optimiser failure. Fits that fail outright are flagged `FIT FAILED`.
- **Testing:** `test_copulas.py` checks family selection at n = 4000 and parameter accuracy (5%) at n = 40000. At n = 4000 a single estimate scatters by up to about 9%, so 5% can't be checked there.

## Other files

- `SEEM-model-correct.ipynb`: the main exploratory notebook. It covers delta computation, regressions, correlation and independence checks, and polynomial fits of `kappa`, and produced most of the loose PNGs here.
- `seem_library.py`: a SINDy-style candidate-function library `Theta(u, tau)` (polynomial, log/exp, and RBF terms) for regressing `du/dgamma`. Use `build_SEEM_library_from_metadata` to rebuild the same features on new data. The pipeline scripts do not use it.
- `C:\Users\scohe\Research\SEEM` is an older copy of this repo (notebooks and `seem_library.py` only). This folder is the current one.
