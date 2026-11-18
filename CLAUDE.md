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
python 8_copula_figures.py         # steps 5-7     -> figures/{marginal_maps,marginal_fits,pit_histograms,copula_shapes,
                                   #                  copula_scores,tail_comparison}.png (+ figures/panels/),
                                   #                  out/voxel_dependence.json   (~3 min)
python test_dependence_law.py      # synthetic checks of dependence_law.py (~4 min); run before step 9 after any change
python 9_fit_dependence_law.py ranks=voxel_rate    # steps 5, 6, 8 -> out/dependence_law_scoreboard.json (main)
python 9_fit_dependence_law.py ranks=voxel_rate3   #   -> out/dependence_law_scoreboard_voxel_rate3.json (sensitivity)
python 9_fit_dependence_law.py ranks=voxel         #   -> out/dependence_law_scoreboard_pooled.json (comparison)
                                   # ~9 min each alone, ~16 min when the three run in parallel; `checks` = pre-fit checks only
python 9b_repeated_splits.py halving   ranks=<mode> start=0 stop=50   # 50/50 splits, all three rank modes (~22 s each)
python 9b_repeated_splits.py split80   ranks=voxel_rate start=0 stop=50   # 80/20 splits (~35 s each)
python 9b_repeated_splits.py bootstrap ranks=voxel_rate start=0 stop=100  # refitting bootstrap; needs step 9's main output
python 9b_repeated_splits.py merge                                  # out/replicates/*.json -> out/dependence_law_replicates.json
                                   # replicates are seeded individually: split start/stop into chunks and run them in parallel
python 9c_dependence_diagnostics.py # steps 9, 9b -> out/dependence_law.json, the FINAL step 9 result (~4 min)
python 10_dependence_figures.py    # out/dependence_law.json -> figures/{voxel_dependence,dependence_scoreboard,
                                   #   dependence_tau_residuals,dependence_tail_residuals}.png (+ figures/panels/)
python 11_copula_benchmark.py      # kernel copula vs Gumbel in cell (18, 11) -> out/copula_benchmark.json (~2 min)
```

- Scripts 2–4 `import size_law` from this folder. `C:\Users\scohe\Downloads` holds another copy of `size_law.py` and scripts 1–4 with its own `out/` and `figures/`. That copy is separate: changes made here don't reach it.
- Dependencies: numpy, pandas, scipy, matplotlib (scikit-learn for the notebook). `python` on PATH is the standalone Python 3.14 in `AppData\Local\Programs\Python\Python314`.
- Figures for steps 8 and 10 share `figstyle.py` (fonts >= 10 pt at the saved size, palette, panel letters, `save()`, which also writes each panel of a multi-panel figure to `figures/panels/`).
- The automated tests are `test_copulas.py` (about 5 minutes) and `test_dependence_law.py` (about 4 minutes). Each draws synthetic data from a known model, refits it, and checks selection and parameter recovery. Other fitting code was validated with one-off `python -c` checks on synthetic samples: draw from a known law, refit, compare.

## Data files (not in git)

- `df_clean.pkl` (2.4 GB): the raw input. Columns `index` (trajectory id), `strain_index`, `strain`, `stress`, `pe`, `cooling_rate`. Rows are sorted by `(index, strain_index)`, and step 1 reshapes them to `(trajectory, step)`. That reshape assumes every trajectory has exactly `N_STEPS = 49999` rows.
- `df_clean_with_deltas.pkl.gz` (1.3 GB): written once by the notebook and never read back by any code.
- `tmp_dict.pkl`: old scratch file that no current code reads or writes.

Never read these with tools that load the whole file into context. `*.pkl`, `*.pkl.gz` and `out/` are in `.gitignore`. `figures/` is committed: the pipeline's figures go at its top level, and the notebook's older PNGs are in `figures/exploratory/`. Don't commit the notebooks (`SEEM-model-correct.ipynb` changes are the user's; `Untitled.ipynb` is scratch).

## Conventions shared across all steps

These are defined in `1_extract_events.py` and assumed by everything after it:
- `u = pe - U_REFERENCE` (with `U_REFERENCE = -4.60751861`) and `tau = stress / 1e4`.
- An event is a step with `dtau < 0`. Its size is `s = -dtau`, and events with `s < 1e-6` (`XMIN`) are dropped as noise. `pe_drop = -du`.
- The state `(u, tau, tau_max)` of an event is taken at the start of the step, just before the drop. `tau_max` is the running maximum of tau over all steps, not only over events.
- The `(u, tau)` grid is 22 × 22: an inner 20 × 20 on the 0.2–99.8% quantiles, plus an outer ring so that every event lands in a cell. Cells are flattened as `cell = bin_u * n_bins + bin_tau`. The grid is only for per-cell reference fits. The closed laws are evaluated at each event's own state.
- **Figures:** after generating or changing any figure, open the PNG with the Read tool and check legibility, colour scales and that plotted numbers match the json outputs before reporting it done.

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
- **Ranks within voxel × cooling rate (steps 9+):** the size marginals shift with cooling rate inside a voxel (fast-cooled drops are smaller on both quantities: PE ×0.56, stress ×0.77 median fast/slow), so ranking a voxel's events together adds spurious concordance. Rank-based fits rank within voxel × rate; a group needs ≥ 10 training events (`dependence_law.MIN_RANK_TRAIN`), else the event falls back to voxel × 3 rate groups (slow / mid / fast), then to the voxel, and every fallback is counted (about 2% of events at 50% training). At 20 the fallbacks left a spurious rate effect in synthetic data (test case E). Training ranks are r/(n+1); held-out events get (k+0.5)/(n+1) against the training values of the same group (`dependence_law.RANK_RULE`). Steps 7 and 8 still rank within the voxel (pooled over rates).
- **Marginal:** truncated power law `s^-kappa exp(-s/s*)` on `s >= xmin`, in `marginals.py` (free `xmin`, unlike `size_law.py`'s fixed 1e-6). Per-cell fits need ≥ 40 drops, and maps show only cells with ≥ 100.
- **Copula fits** are done in the most-visited cell, (18, 11) = cell 407. Families: independence, gaussian, t, clayton, gumbel, frank, bb1 and bb7, plus 180° rotations (`survival_*`). BB1 contains Gumbel (θ → 0) and Clayton (δ = 1), and BB7 contains Clayton (θ = 1), so a BB fit with a bound flag on those parameters has collapsed to that one-parameter family. That is a finding, not an optimiser failure. Fits that fail outright are flagged `FIT FAILED`.
- **Testing:** `test_copulas.py` checks family selection at n = 4000 and parameter accuracy (5%) at n = 40000. At n = 4000 a single estimate scatters by up to about 9%, so 5% can't be checked there.

### Findings so far (2026-11-04)

- **Marginals:** the TPL fits are stable but only approximate. In voxels with ≥ 500 drops, κ is 0.61–0.82 for PE and 0.77–1.00 for stress. In cell (18, 11), KS is 0.030 (PE) and 0.029 (stress), both rejected at this n. Moving the stress cutoff from 1e-3 to 1e-6 lowers κ by a median of 0.12 per voxel.
- **Best copula, cell (18, 11), 3906 paired events, ranks:** Gumbel (held-out gain 3166 over independence) is tied with flipped Clayton (−33 [−93, +24]), flipped BB1 (−22 [−70, +23]) and BB7 (−30 [−85, +22]). BB1 collapses to Gumbel, and flipped BB7 to flipped Clayton. The second parameter does not help. Frank, t, Gaussian and the lower-tail families are clearly worse. Dependence is strong (Kendall τ 0.726) and concentrated in the large-drop corner.
- **τ depends on state:** across the 119 voxels with ≥ 100 paired events (92% of all 91,532 paired events), τ runs from 0.51 to 0.87 and rank-based Gumbel θ from 2.1 to 5.7. In 80 voxels the τ interval does not overlap cell (18, 11)'s. τ rises with stress at fixed u and with u. A single copula parameter does not describe the whole state space.
- **Open caveats** (updated 2026-11-18):
  - *Upper-tail overshoot is everywhere:* after the null correction, Gumbel overstates P(V > 0.95 | U > 0.95) by about 0.065 in every part of the state space, by the same amount for the formula and the voxel table. It is the family's shape, not the θ law; a one- or two-parameter family can't fix it in cell (18, 11) (step 11: a kernel copula beats Gumbel by +314).
  - *The family may vary with state:* at the highest τ_K, Frank beats Gumbel in three of three voxels (sampling-only intervals). A per-voxel family comparison is untested.
  - *Marginals depend on cooling rate within voxels* (step 9 pre-fit check 3: 58 and 56 of 83 voxels differ fast vs slow), which steps 6–8 do not model. Steps 7 and 8 rank within the voxel, pooled over rates.
  - *Stress marginal misfit:* the paired events' stress PIT is far from uniform (KS 0.21, mean 0.63). Paired drops are scarce between 1e-3 and about 1e-2, where stress-only drops dominate. IFM results are secondary for this reason.
  - *Intervals:* step 7's and step 8's intervals resample trajectories without refitting. In step 9 the deciding evidence is repeated random splits (a stability guide, not a confidence interval), and coefficient intervals come from the splits by the delete-d jackknife; the refitting bootstrap (duplicated trajectories → tied ranks) is kept as secondary only.

### Step 9: Gumbel θ(u, τ, rate) with clean ranks (2026-11-18)

Final result in `out/dependence_law.json` (assembled by `9c_dependence_diagnostics.py`). Model: θ = 1 + exp(η), on the 84,273 paired events in the 119 voxels with ≥ 100 each, ranks within voxel × rate. Forms: constant, product (100uτ), linear, interaction (linear + z_u z_τ), quadratic, each ± g ln(rate/rate_mid); references: one θ per voxel (`voxel_table`, 119) and `voxel_table+rate` (120). Deciding evidence: 50 random 50/50 and 50 random 80/20 splits of each rate's trajectories (step 9b); sampling-only intervals are ~1.4× too narrow for near-ties. The earlier pooled-rank run (2026-11-06) is superseded; it is kept as the `voxel` sensitivity mode.
- **Candidate formula: interaction + rate (5 parameters)**, η = 0.902 + 0.210 z_u + 0.253 z_τ + 0.067 z_u z_τ − 0.051 ln(rate/rate_mid). 95% intervals: intercept [0.890, 0.914], z_u [0.193, 0.228], z_τ [0.238, 0.267], z_u z_τ [0.055, 0.079], rate [−0.060, −0.042]. The quadratic + rate adds nothing clear: +3.7 ± 5.8 per test half (won 36/50) at 50/50, +2.1 ± 2.3 (43/50) at 80/20.
- **How coefficient intervals are computed:** estimate ± 1.96 sd, with sd from the coefficient's spread over the 50 random 50/50 splits scaled by √(m/d) (delete-d jackknife: m training, d held-out trajectories per rate; = 1 for halves). The 80/20 splits (× √4 = 2) give the same sd within 0.81–1.15× (median 0.92) over all 38 coefficients; √(m/n) would not (it ignores that the splits come from one sample). Intercepts are about 0.010–0.016 higher when fitted on half the data (coarser rank groups); slopes shift by < 0.006. The refitting bootstrap's percentiles are kept in the json as secondary only: tied ranks from duplicated trajectories bias them (its intercept intervals exclude the estimates; its τ′ median is 0.244 against 0.253). The checkpoint-3 values 0.883 / 0.211 / 0.244 / 0.066 / −0.050 were those bootstrap medians, not estimates.
- **Against the table, the gap depends on training size.** interaction + rate minus voxel_table + rate per 1,000 held-out events: −0.28 ± 0.61 (won 15/50) at 50/50; −1.23 ± 0.66 (won 3/50) at 80/20; 5 folds −1.54 [−2.10, −0.97]. With half the data the formula ties; with 80% the table holds structure the formula misses. Pooled ranks gave −1.44 ± 0.60 at 50/50 (won 0/50): the clean ranks closed most of the earlier gap.
- **Effect size:** gain over a single θ per 1,000 held-out events: table + rate 19.3 ± 1.3 (50/50) and 20.7 ± 2.0 (80/20); interaction + rate recovers 98.6% (50/50), 94.1% (80/20) and 92.5% (5 folds) of it.
- **Cooling rate:** two measures. (i) Within-voxel slope (τ_K of each (voxel, rate) group against ln rate, centred within each voxel, pooled): −0.0129 [−0.0151, −0.0108] per unit ln rate (−0.072 over the ×256 range; 33 of 83 voxels clearly lower when fast, 0 higher). (ii) Group regression (τ_K of 577 (voxel, rate) groups on their median z_u, z_τ and ln rate, controlling for the state within voxels): z_u +0.066, z_τ +0.054, ln rate −0.0158 [−0.0174, −0.0140] (−0.079 in η units). The fitted rate coefficient is −0.054 [−0.064, −0.045] (table + rate), 70–80% of those targets. Rate correlates with u (r = +0.52), so leaving it out hides part of the u effect: adding it raises the z_u coefficient from 0.164 to 0.210 (interaction). Every form except constant gains 1.4–2.5 per 1,000 events from the rate term (50/50 of 50).
- **Unpooled per-voxel τ_K** (event-weighted mean over rate groups): 0.52–0.88, within 0.004 ± 0.012 of step 8's pooled values (1 of 119 outside step 8's interval). "τ_K rises with u" survives: +0.051 [+0.048, +0.055] per sd across voxels.
- **Calibration** (implied vs unpooled measured τ_K, in-sample): outside the interval 37/119 (interaction + rate) and 36 (quadratic + rate) vs 28 for table + rate. Mean bias −0.013 for all (a correct model: 0.000). Trend of the bias on implied τ_K: −0.22 [−0.29, −0.14] (interaction + rate), −0.16 [−0.19, −0.12] (table + rate), correct model −0.02. Residual structure: both too weak in the dense high-u band (stress 1.0–1.5); the formula is also too strong at the low-u, high-stress edge and along the lowest-stress row.
- **Upper tail** P(V > 0.95 | U > 0.95): models +0.103 above the empirical value on average, 21–23 of 119 voxels above the interval, the formula and the table alike. A correct model measured the same way scores +0.038 and lands above in 2 voxels (small rank groups bias the empirical tail low). So about **+0.065 is a genuine overshoot, everywhere**: it does not correlate with u, τ, rate mix or measured τ_K (|r| ≤ 0.17).
- **Family check (hint, not finding):** in the three highest-τ_K voxels with ≥ 300 paired events, Frank (no tail dependence) beats Gumbel: −33.8 [−67.5, −4.5] (17, 15), −59.9 [−92.9, −28.9] (19, 14), −73.0 [−108.3, −40.6] (20, 13), sampling-only intervals. Lowest τ_K (13, 9): flipped Clayton best, Gumbel tied. Cell (18, 11) with clean ranks: Gumbel still best (3138 vs pooled 3166), same ties as step 7.
- **Hypothesis, not a finding:** the product form shares step 3's ceiling-law variable 100uτ. It is not the best form (product + rate −1.18 ± 0.64 per 1,000 events at 50/50, won 2/50), and the product form depends on U_REFERENCE through its zero crossing. Testing a shared cause would need step 3's outputs here and η = a + b ln s_c.

### Step 11: could any copula beat Gumbel? (2026-11-10)
Benchmark: a transformation kernel copula (`copulas.kernel_log_density`: Gaussian kernel density of the normal scores, no family shape). It uses step 7's sample, ranks and even/odd split, with the bandwidth chosen inside each training half by trajectory blocks (0.270 in both halves). On true Gumbel data it scores *below* Gumbel (test_copulas.py section 5: −58 [−93, −31]), so a kernel win means the Gumbel shape is wrong.
- **Kernel beats Gumbel by +314 held-out ln L [+259, +368]** (kernel gain 3480 vs Gumbel 3166). That is about 10% of Gumbel's gain over independence, or +80 per 1,000 events. Kernel minus flipped Clayton is +348 and minus t is +656.
- **Where:** most of the gain (+270 [+223, +317]) is in the bulk of the square, not the corners. Upper corner (both > 0.9): +63 [+37, +94]. Lower corner (both < 0.1): −19 [−31, −8].
- **Corners at q = 0.05:** data lower 0.256 and upper 0.686; kernel 0.278 and 0.699; Gumbel 0.503 and 0.778. The kernel reproduces both; Gumbel overstates both.
- **Conclusion:** a fixed one-parameter family is not enough in this cell. Trying further named families is unlikely to close a gap this size without knowing its cause (next: step 12, the conditional / physical-relation check in the plan).

## Other files

- `SEEM-model-correct.ipynb`: the main exploratory notebook. It covers delta computation, regressions, correlation and independence checks, and polynomial fits of `kappa`. Its PNGs are in `figures/exploratory/`.
- `seem_library.py`: a SINDy-style candidate-function library `Theta(u, tau)` (polynomial, log/exp, and RBF terms) for regressing `du/dgamma`. Use `build_SEEM_library_from_metadata` to rebuild the same features on new data. The pipeline scripts do not use it.
- `figstyle.py`: the shared figure style (steps 8 and 10). `PLAN_step9.md`: the original step 9 plan (superseded in places by the step 9 findings above).
- `C:\Users\scohe\Research\SEEM` is an older copy of this repo (notebooks and `seem_library.py` only). This folder is the current one.
