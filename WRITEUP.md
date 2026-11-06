# Joint statistics of PE drops and stress drops: copula analysis, steps 5–9

Branch `copula-analysis`, November 2026. Every number below comes from the pipeline's saved outputs (`out/*.json`), the scripts' printed output, or `CLAUDE.md`.

**Notation.** I write the state as (u, τ) with u = pe − u₀ (u₀ = −4.60751861) and τ = stress/10⁴, as in SEEM. To avoid a clash, the rank correlation Kendall's tau is written **τ_K**.

## Summary

- **The two drops are strongly linked, mainly when both are large.** In the most-visited voxel, cell (18, 11), 3,906 events where both stress and PE drop have Kendall τ_K = 0.726. (τ_K is a rank correlation: 0 means no link, 1 means the two always order the same way.) Big PE drops almost always come with big stress drops, while small drops of the two kinds are only loosely tied.
- **Several copula families fit equally well, and none fits exactly.** A copula is a model of how two quantities move together after each one's own size distribution has been factored out. Gumbel and "flipped" Clayton, both of which put the link in the large-drop corner, are statistically tied on held-out data, as are two two-parameter families (flipped BB1 and BB7). The extra parameter does not help. All of them overstate how often the very largest drops coincide, and they understate τ_K slightly.
- **The strength of the link depends on state.** Across the 119 voxels with at least 100 paired events, τ_K runs from 0.51 to 0.87. It is higher at higher stress and higher u.
- **A formula captures most of that variation, but not all.** A Gumbel copula whose parameter is a quadratic function of (u, τ) recovers about 94% of the held-out improvement that one parameter per voxel achieves over a single constant (gains of 1571 vs. 1671). In 50 of 50 repeated random splits of the trajectories, though, the 119-parameter voxel table still predicts held-out data better.
- **The marginal fits are approximate.** A truncated power law describes each drop size only roughly. For stress, the stress-only "aging" events distort what the copula sees, so the results that depend on these fits are treated as secondary.

## 1. Question and approach

SEEM draws each event's jumps (Δu, Δτ) together from empirical lists per voxel. The notebook had already found them correlated over events (Spearman 0.48, notebook cell 73). The question here is whether a compact model of the joint law can replace the empirical lists: a size law for each drop plus a copula for the link.

**Events.** A step is an event when dtau < 0 or du < 0, the notebook's definition, taken at the state just before the step. Over all 850 trajectories that gives 292,656 events, matching the notebook's count:

| | Events | Share |
|---|---|---|
| Both drop | 100,091 | 34% |
| Stress drop only ("aging": PE rises) | 87,377 | 30% |
| PE drop only | 105,188 | 36% |

**Cutoffs.**
- *PE drops ≥ 1e-6.* Below it there is a separate numerical-noise bump near 3e-8, which holds nearly all "PE only" events.
- *Stress drops ≥ 1e-3.* Below about 1e-3 the stress drops are mostly aging events, a different population from the paired drops. Pooled over all voxels, the fit error (KS distance, the largest gap between the fitted and the observed cumulative distribution) falls from 0.065 at a 1e-6 cutoff to 0.017 at 1e-3. The 1e-3 cutoff excludes 53,479 of the 87,377 aging drops (61%).
- After both cuts there are 91,532 paired events.

**Voxel grid.** This is step 1's grid: 22 × 22 cells in (u, τ), an inner 20 × 20 on the 0.2–99.8% quantiles plus an outer ring, so voxel ids match the existing pipeline. The most-visited voxel, (18, 11), covers u in [0.0242, 0.0255) and τ in [1.027, 1.131) (step 5 output).

## 2. Size laws for each drop (marginals)

Each drop size is fitted per voxel with a truncated power law, P(s) ∝ s^−κ exp(−s/s*) for s at or above the cutoff. Here κ sets how strongly small drops dominate, and s* sets the scale where large drops are cut off.

- **The parameters vary smoothly with state.** In voxels with at least 500 drops, κ is 0.61–0.82 for PE and 0.77–1.00 for stress. The median κ over fitted voxels is 0.72 (PE) and 0.82 (stress).
- **The stress fit depends on the cutoff.** Moving it from 1e-3 to 1e-6 lowers κ by a median of 0.12 per voxel and s* by about 20%.
- **Consistency check:** at the 1e-6 cutoff, the stress fits reproduce step 2's truncated-power-law fits.
- **The fits are approximate in the most-visited voxel.** KS is 0.030 for PE (κ 0.738, s* 1.06e-3, 4,042 drops) and 0.029 for stress (κ 0.872, s* 0.176, 5,437 drops). Both are rejected at this sample size. Step 2's ceiling law fits stress drops better than this form: it wins by AIC in all 8 cells step 2 tested, including (18, 11). That run is in the `Downloads` copy's `out/shape.json`; steps 1–3 haven't been run in this folder.
- **The aging events distort the stress marginal.** The stress marginal is fitted on all stress drops, while the copula uses only paired ones. In cell (18, 11), 1,284 aging drops above 1e-3 remain in the stress fit (3,367 in the cell, 2,083 below the cutoff). Paired stress drops are larger: median 0.033 against 0.015 for all stress drops.

The consequence shows up in the PIT. A PIT (probability integral transform) pushes each size through its fitted distribution to get a percentile between 0 and 1; if the fit describes the data, the percentiles are uniform. For the paired events they are nearly uniform for PE (KS 0.039, mean 0.509) but far from uniform for stress (KS 0.213, mean 0.631).

## 3. Which copula, in cell (18, 11)

**Families tested:** independence (the baseline), Gaussian, t, Frank, Clayton (link in the small-drop corner), Gumbel (link in the large-drop corner), the 180° "flipped" versions of Clayton and Gumbel, and the two-parameter BB1 and BB7 with their flips.

**Two ways of fitting each family:**
- **Ranks (the reference):** each event's size is replaced by its rank in the sample, divided by n + 1. This needs no size law at all.
- **IFM** ("inference functions for margins"): convert each size to its percentile under the fitted truncated power law (the PIT), then fit the copula to those percentiles.

Ranks are the reference because IFM inherits the marginal misfit. With the distorted stress PIT, no IFM fit implies a τ_K above 0.60 (the leading families give 0.57–0.60), against 0.726 in the data.

**Scoring.** Every family is fitted on half the trajectories (even vs. odd ids) and scored on the other half, by held-out log-likelihood: how probable the model makes data it was not fitted on, where higher is better. Intervals are 95% ranges from resampling whole trajectories.

| Family (ranks) | Held-out gain over independence | Minus the best [95%] |
|---|---|---|
| Gumbel | 3166 | best |
| BB1 | 3165 | collapses to Gumbel |
| Flipped BB1 | 3143 | −22 [−70, +23] |
| BB7 | 3136 | −30 [−85, +22] |
| Flipped Clayton | 3132 | −33 [−93, +24] |
| Flipped BB7 | 3132 | collapses to flipped Clayton |
| Frank | 3049 | −117 [−172, −63] |
| t | 2825 | −341 |
| Gaussian | 2681 | −485 |
| Flipped Gumbel, Clayton | 2143, 1375 | below −1000 |

- **Every family with the link in the large-drop corner is tied with Gumbel.** Every family with no tail link or a symmetric one (Frank, t, Gaussian), and every family with the link in the small-drop corner, is clearly worse.
- **BB1 and BB7 don't help.** BB1's extra parameter goes to its Gumbel limit, and flipped BB7's to its Clayton limit. The two BB fits that stay inside their parameter range land between Gumbel and flipped Clayton, within noise of both.
- **IFM ranks flipped Clayton above Gumbel** (by 270) and has the same BB collapses. That's consistent with its distorted stress percentiles.

**Corner check** (ranks; the data's 95% intervals come from resampling trajectories):

| | τ_K | Lower 5% corner | Upper 5% corner |
|---|---|---|---|
| Data | 0.726 [0.717, 0.735] | 0.256 [0.192, 0.324] | 0.686 [0.636, 0.739] |
| Gumbel | 0.702 | 0.503 | 0.778 |
| Flipped Clayton | 0.669 | 0.210 | 0.842 |
| BB7 | 0.671 | 0.267 | 0.843 |
| Flipped BB1 | 0.678 | 0.332 | 0.836 |

The lower corner is the probability that, when one drop is in its smallest 5%, the other is too; the upper corner is the same for the largest 5%. Independence would give 0.05 for both.

## 4. Does the link depend on state?

**Yes.** In the 119 voxels with at least 100 paired events (84,273 events, 92% of all paired events):
- τ_K ranges from 0.51 to 0.87 (median 0.71).
- In 80 voxels the τ_K interval does not overlap cell (18, 11)'s.
- τ_K rises with u (Spearman 0.54 across voxels) and with stress (0.36).

**Step 9** fits a Gumbel copula over these voxels with θ = 1 + exp(η(u, τ)), where η is a constant, a function of the product 100·u·τ, a linear function, or a quadratic function of (u, τ). It compares each against one θ per voxel (119 parameters), always holding out whole trajectories.

| Model (parameters) | Gain over constant θ (2 folds) | Minus voxel table, 50 random halvings: mean (sd) [2.5–97.5%], wins |
|---|---|---|
| Quadratic (6) | +1571 [1448, 1700] | −45 (20) [−81, −18], 0 of 50 |
| Product (2) | +1467 [1352, 1596] | −93 (24) [−140, −52], 0 of 50 |
| Linear (3) | +1294 [1181, 1418] | −176 (32) [−238, −125], 0 of 50 |
| Voxel table (119) | +1671 [1539, 1814] | — |

- The product form is θ = 1 + exp(−0.35 + 0.47·100uτ), with slope interval [0.45, 0.50], giving θ from 2.3 to 5.8.
- **Calibration:** the number of voxels whose model-implied τ_K falls outside the measured interval is 38 of 119 for the voxel table, 47 for the quadratic and 52 for the product. Every model, the voxel table included, understates τ_K by about 0.015 on average.

## 5. Limitations and open questions

1. **The large-drop corner is overstated by every family.** All the leading families put the upper 5% corner at 0.78–0.84, against 0.69 [0.64, 0.74] in the data, and τ_K at 0.67–0.70, against 0.73. More parameters don't fix this. One possible cause, not tested, is the hard ceiling on drop sizes.
2. **The stress marginal does not describe the paired events** (PIT KS 0.21), because aging events remain in its fit. That's why IFM understates τ_K, and why nothing that relies on the fitted marginals should be used yet.
3. **Neighboring voxels share trajectories.** Each voxel's interval resamples trajectories within that voxel, but comparisons across voxels (the 80 non-overlapping intervals, the Spearman trends) treat voxels as independent. Their p-values are overstated.
4. **The error bars mostly cover sampling only.** The step 7 and step 8 intervals and step 9's 2-fold intervals resample trajectories without refitting. Step 9's refitting bootstrap made the quadratic look tied with the table (+3 [−115, +124]), but its resamples keep only about 63% of distinct trajectories, which handicaps the 119-parameter table. It shouldn't be used to compare models with very different parameter counts.
5. **The 50 random halvings overlap.** The splits share trajectories, so their spread shows how stable the result is, not a formal confidence interval. Winning 50 of 50 is strong but is not 50 independent tests.
6. **The link to step 3's ceiling law is a hypothesis.** The product form uses the same variable, 100uτ, as step 3's ceiling law. That suggests a common cause, but nothing here tests one.
7. **The all-events comparison is easy to misread.** On all 91,532 paired events, the formulas beat a 362-voxel table (product +328, quadratic +421).
   - *What it shows:* where voxels are thin, a smooth formula predicts better than fitting each voxel separately. Several small voxels' θ ran to a bound in that table.
   - *What it doesn't show:* that the formula is right. The reference there is a deliberately noisy one; against the 119 well-filled voxels the table wins.
8. **The family choice rests on one voxel.** Gumbel was carried into step 9 from cell (18, 11) alone, where it was only tied for best.

## 6. Possible next steps

- **Per-voxel family comparison.** Run step 7's contest in each of the 119 voxels on ranks. This needs no new code, only time. It would show whether the best family changes with state, and so whether "Gumbel with a varying θ" is the right model at all.
- **Fix the stress marginal.** Fit it on paired events only, or use step 2's ceiling law for stress. This is a prerequisite for anything that uses the fitted marginals, including IFM and sampling.
- **Sample paired events inside SEEM.** Replace the empirical (Δu, Δτ) lists with draws from the size laws plus a copula. That needs the stress-marginal fix first, plus a decision on how to handle the aging and PE-only events, which a paired-event copula doesn't describe. It matters because a parametric joint law could fill in sparse voxels and smooth across voxel edges.
- **Test the ceiling link.** Run steps 1–3 in this folder and fit η = a + b·ln s_c(u, τ). That would turn the hypothesis in limitation 6 into a test.

## 7. Figures for the meeting

**Recommended, ranked by importance:**

| Rank | File | One-sentence takeaway | What to say |
|---|---|---|---|
| 1 | `voxel_dependence.png` | The link between the two drops varies strongly with state. | τ_K ranges from 0.51 to 0.87 and is higher at high stress and high u, so a single copula parameter can't serve all of SEEM. |
| 2 | `copula_shapes.png` | Large drops of both kinds go together tightly; small ones only loosely. | Gaussian can't make this lopsided shape. Gumbel and flipped Clayton can, but each in only one corner. |
| 3 | `dependence_law.png` | A formula for θ(u, τ) gets most of the way, but one θ per voxel still wins. | The quadratic is short by about 1 log-likelihood unit per 1,000 events in all 50 splits. The misses sit at the edges of the occupied region. |
| 4 | `tail_comparison.png` | Every family overstates how often the largest drops coincide. | The data's upper corner is 0.69; the models give 0.78–0.84. This is the main thing no family fixes. |
| 5 | `copula_scores.png` | Families with the link in the large-drop corner are tied for best; the extra parameter doesn't help. | Keep this brief and point to the large-drop-corner families at the top. Mention IFM only if asked. |

**What could confuse a first-time viewer, and the fixes (not yet made):**
- **`voxel_dependence.png`**
  - "Kendall τ" on the colorbar clashes with "τ = stress / 10⁴" on the y-axis. *Fix:* label it τ_K.
  - The Gumbel θ panel repeats the τ_K panel without saying why. *Fix:* drop it or caption it.
  - The event counts are very small and dense. *Fix:* enlarge them or leave them for backup.
  - There's no sign of which voxels differ significantly. *Fix:* mark the voxels whose interval excludes (18, 11)'s.
  - u has no units. *Fix:* add units, or note that u is per the simulation's energy units.
- **`copula_shapes.png`**
  - The axes "Φ⁻¹(rank …)" are undefined. *Fix:* relabel as "rank of PE drop (normal-score scale)".
  - The contour numbers are densities with no meaning to a viewer. *Fix:* remove them.
  - The data shading has no colorbar. *Fix:* add "darker = more events".
  - "Held-out gain" in the panel titles is undefined. *Fix:* define it or remove it.
  - Six panels are a lot. *Fix:* keep the data, Gaussian, Gumbel and flipped Clayton.
- **`dependence_law.png`**
  - The left panel shows totals while the right shows per 1,000 events. *Fix:* use one unit.
  - The orange refitting-bootstrap series is biased and is explained only in the legend. *Fix:* remove it.
  - The τ symbol clashes in the maps. *Fix:* use τ_K.
  - "(119)" and "(6)" are parameter counts but aren't labeled as such. *Fix:* say so.
  - The outlines need a caption.
- **`tail_comparison.png`**
  - τ_K and the corner probabilities share one axis but measure different things. *Fix:* separate panels.
  - U and V in the axis labels are undefined. *Fix:* say "both drops in their largest 5%".
  - Only the data bars have intervals. *Fix:* note that the model values are fitted points.
  - The independence reference appears only in the title. *Fix:* add a dashed line at 0.05.
- **`copula_scores.png`**
  - It shows 11 families, including two that collapse onto others (BB1 = Gumbel, flipped BB7 = flipped Clayton). *Fix:* drop those two.
  - The hatched IFM bars sit next to the rank bars. *Fix:* show ranks only.
  - The right panel's "best" differs between the two methods. *Fix:* use ranks only.
  - The best family's zero-length bar is invisible. *Fix:* mark it.
  - "Held-out ln L" is undefined. *Fix:* label it "held-out log-likelihood gain".

**Leave out, or keep as backup:**
- `marginal_fits.png`: backup. It's the clearest explanation of the stress-marginal problem, if IFM or the marginals come up.
- `pit_histograms.png`: backup. It repeats `marginal_fits.png` for this audience, and the PIT needs explaining.
- `marginal_maps.png`: backup. It has four panels with different color scales, and the κ and s* trends aren't central to the copula story.
- `figures/exploratory/*`: leave out. These are notebook-era plots on a different grid, and some use a different event selection.

## 8. Claims I would avoid

- **"Gumbel is the copula for these events."** It's tied with three other families, in one voxel.
- **"θ follows the ceiling law" or "the dependence is set by 100uτ."** That's untested; the product form is not even the best formula.
- **"A formula describes the state dependence."** The 119-voxel table beats every formula in 50 of 50 splits.
- **"The formula beats the voxel table."** That's true only against the noisy 362-voxel table on all events (limitation 7).
- **"The drop sizes follow a truncated power law."** It's approximate and statistically rejected in the busiest voxel.
- **"IFM confirms the result."** IFM inherits the stress misfit and understates τ_K.
- **"The upper-tail dependence is λ_U ≈ 0.77–0.84."** These are model values. The data only measure corners at a finite level (5%), and there the models overshoot.
- **"τ_K rises with u and stress at p < 1e-9."** Voxels share trajectories, so those p-values are overstated.
- **"The intervals include model uncertainty."** Most are sampling-only, and the repeated-split spread is not a confidence interval.
