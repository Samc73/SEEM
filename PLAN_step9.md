# Plan: step 9, a state-dependent Gumbel θ(u, τ)

Status: proposed 2026-11-04, not started. No fits have been run for this step.

## Why

Step 7 compared copula families for the joint law of PE drop (−Δu) and stress drop (−Δτ) in one voxel, the most-visited cell (18, 11). Gumbel came out on top on held-out data, statistically tied with flipped Clayton, flipped BB1 and BB7.

Step 8 then showed that the dependence is not the same everywhere. Across the 119 voxels with at least 100 paired events:
- Kendall τ runs from 0.51 to 0.87, and a rank-based Gumbel θ from 2.1 to 5.7.
- In 80 voxels the τ interval does not overlap cell (18, 11)'s.
- τ rises with stress at fixed u, and with u.

A single copula parameter therefore does not describe the whole state space. The question for step 9 is whether a closed formula θ(u, τ) with a few parameters captures this variation as well as fitting one θ per voxel (119 parameters). Step 3 asked the same kind of question about the size-law ceiling.

## Data

- **Events:** all paired events (du < 0 and dtau < 0, with −du ≥ 1e-6 and −dtau ≥ 1e-3, the cutoffs of step 6). That is 91,532 events from all 850 trajectories, read from `out/joint_events.npz`.
- **State:** each event's own (u, τ) at the start of its step, not its voxel centre.
- **Main scoring set:** the 84,273 events (92%) in the 119 voxels with at least 100 paired events. On this set the one-θ-per-voxel reference can score every event. The closed forms are also scored on all events as a secondary check.

## Turning sizes into uniforms

- Rank each event's −Δu and −Δτ within its own voxel (conditional pseudo-observations). Ranking all events together would mix marginals that change with state (step 6 maps κ and s* across voxels), which would distort the dependence.
- For held-out events, use the training fold's ranks within each voxel (`copulas.ecdf_transform`), the same rule as step 7.
- Ranking within voxels brings the grid back in. Add a version that smooths over state instead (a kernel-weighted ranking in (u, τ)) only if the results turn out to depend on the grid.

## Model

Gumbel copula with an event-specific parameter:

θᵢ = 1 + exp(η(uᵢ, τᵢ)), which keeps θ ≥ 1 for any coefficients.

u and τ are standardised over the paired events (z_u, z_τ), except in the product form, which keeps step 3's variable 100·u·τ.

| Form | η(u, τ) | Parameters |
|---|---|---|
| Constant | a | 1 |
| Product | a + b·100uτ | 2 |
| Linear | a + b·z_u + c·z_τ | 3 |
| Quadratic | a + b·z_u + c·z_τ + d·z_u² + e·z_u·z_τ + f·z_τ² | 6 |
| **Reference** | one θ per voxel | 119 |

The product form uses the same variable as step 3's closed law for the size-law ceiling, ln s_c = a + b·100uτ.

**Possible extension:** η = a + b·ln s_c(u, τ), which ties the dependence directly to step 3's ceiling. It needs `out/ceiling_law.json` from step 3, and steps 1–3 have not been run in this folder yet; their outputs exist only in the `C:\Users\scohe\Downloads` copy. Leave it until those steps are run here.

## Fitting

- Maximum likelihood over all events together: maximise Σ log c_Gumbel(Uᵢ, Vᵢ; θᵢ). `copulas._ln_gumbel` already works element by element, so it accepts a θ per event.
- Optimiser: L-BFGS-B from several starting points:
  - a least-squares fit of the per-voxel log(θ − 1) values on each voxel's median state, like step 3's starting values;
  - a flat start, with a = log(θ_pooled − 1) and the other coefficients 0;
  - small random perturbations of both.
- Flag any coefficients that run off, and any form that pins θ near 1 across large regions of the state space.
- The voxel reference uses `copulas.fit_family('gumbel', ...)` per voxel, the same code as step 8.

## Test first, on synthetic data

Extend `test_copulas.py` (or add `test_dependence_law.py`) before touching real data:
- Generate events whose θ depends on state in a known way, for example in the product form, at realistic (u, τ). Sample them with the frailty method the tests already have: Gumbel = positive-stable frailty with α = 1/θ.
- Check that the coefficients come back within 5% at large n (as in test 3b).
- Check that the true form wins on held-out data at the real sample size.
- Check that when θ is constant, the linear and quadratic forms do not beat the constant form beyond noise.

## Holding out whole trajectories

Events within one trajectory are strongly correlated, so data is always held out by trajectory, never by event.

- **Main split:** 2 folds, even trajectory ids against odd, as in steps 3 and 7 (45,733 vs. 45,799 paired events).
- **Everything is refitted on the training fold:** the within-voxel ranks, the closed-form coefficients and the per-voxel θ table. If a held-out event's voxel has no training events, it gets the pooled θ, as step 3 does with its median ceiling.
- **Second split:** 5 folds, trajectory id mod 5, for a steadier comparison (2 folds give only 2 scores). Report both.
- **Intervals:** resample trajectories with replacement and re-sum the held-out per-event scores, paired across models (2000 resamples, as in step 7).
- **Refitting noise:** step 7's intervals only cover sampling noise. Add a second bootstrap that refits the closed forms and the voxel table on 100–200 trajectory resamples. That gives intervals for the coefficients, and score intervals that include refitting noise.

## What counts as an improvement over one θ per voxel

1. **Necessary:** the formula beats a constant θ on held-out data, with a paired interval that excludes 0.
2. **Main test:** a formula with at most 6 parameters scores as well as the 119-parameter voxel table on held-out data (the interval for formula − table includes 0), or better. That would mean the formula captures the state dependence. If the table clearly wins, the formula is missing structure, and a residual map shows where.
3. **Simplest wins:** among forms within noise of the best one, take the one with the fewest parameters.
4. **Calibration:** in each voxel, compare the model-implied Kendall τ (Gumbel τ = 1 − 1/θ, averaged over the voxel's events) with the measured τ and its trajectory-bootstrap interval from `out/voxel_dependence.json`. Count the voxels outside their interval. About 6 of 119 are expected by chance, and the voxel table's own count is the benchmark.
5. **Consistency:** in cell (18, 11), the formula should not score worse on held-out data than step 7's Gumbel beyond the noise.

## Limits

- **The family is fixed as Gumbel.** Gumbel was only tied for best, and only in one voxel. The best family might change with state, for example toward more lower-tail dependence at low stress. Checking that would need a per-voxel family comparison, which is a separate fit.
- **It will not fix the upper-tail overshoot.** In cell (18, 11) every family puts the upper 5% corner at 0.78–0.84, against 0.69 [0.64, 0.74] in the data. That is a problem with the family's shape, not with how θ varies.
- **Ranking within voxels** removes the marginal variation but keeps the grid binning (see "Turning sizes into uniforms").

## Deliverables

- `9_fit_dependence_law.py`: reads `out/joint_events.npz` and `out/voxel_dependence.json`, writes `out/dependence_law.json` (coefficients, held-out scores for both splits, intervals, bound flags, calibration counts).
- The synthetic test described above.
- One added figure: the held-out scoreboard and a map of τ residuals per voxel (model-implied minus measured), either in `8_copula_figures.py` or a new step.

## Expected cost

Each fit should take seconds on about 90,000 events. The refitting bootstrap is the slow part, roughly 10–20 minutes.
