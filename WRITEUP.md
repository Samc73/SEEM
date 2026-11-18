# Joint statistics of PE drops and stress drops: copula analysis, steps 5–9

Branch `copula-analysis`, November 2026 (step 9 updated 18 November). Every number below comes from the pipeline's saved outputs (`out/*.json`; step 9's final result is `out/dependence_law.json`), the scripts' printed output, or `CLAUDE.md`.

**Notation.** I write the state as (u, τ) with u = pe − u₀ (u₀ = −4.60751861) and τ = stress/10⁴, as in SEEM. To avoid a clash, the rank correlation Kendall's tau is written **τ_K**.

## Summary

- **The two drops are strongly linked, mainly when both are large.** In the most-visited voxel, cell (18, 11), 3,906 events where both stress and PE drop have Kendall τ_K = 0.726. (τ_K is a rank correlation: 0 means no link, 1 means the two always order the same way.) Big PE drops almost always come with big stress drops, while small drops of the two kinds are only loosely tied.
- **Several copula families fit equally well, and none fits exactly.** A copula is a model of how two quantities move together after each one's own size distribution has been factored out. Gumbel and "flipped" Clayton, both of which put the link in the large-drop corner, are statistically tied on held-out data, as are two two-parameter families (flipped BB1 and BB7). The extra parameter does not help. All of them overstate how often the very largest drops coincide, and they understate τ_K slightly.
- **The strength of the link depends on state and on cooling rate.** Across the 119 voxels with at least 100 paired events, τ_K runs from 0.52 to 0.88. It is higher at higher stress and higher u, and lower for faster-cooled glasses at the same state.
- **A five-parameter formula captures nearly all of that variation.** A Gumbel copula with η = log(θ − 1) linear in u and τ, their product, and the log cooling rate recovers 94–99% of the held-out improvement that one parameter per voxel (plus a rate term) achieves over a single constant. With half the data it ties the 120-parameter voxel table; trained on 80% of the trajectories, the table leads by about 1.2 log-likelihood units per 1,000 events, so some state structure remains that the formula misses.
- **The Gumbel shape is the bigger limitation.** Everywhere in the state space the fitted copulas, formula and voxel table alike, overstate how often both drops are in their largest 5%, by about 0.065 after correcting for how this is measured. At the highest τ_K, Frank (a family with no tail link) beats Gumbel in all three voxels checked.
- **The marginal fits are approximate.** A truncated power law describes each drop size only roughly. For stress, the stress-only "aging" events distort what the copula sees, so the results that depend on these fits are treated as secondary. Both size laws also shift with cooling rate inside a voxel, which the size-law fits do not model.

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
- **The size laws shift with cooling rate inside a voxel.** Comparing fast-cooled with slow-cooled trajectories in the same voxel (83 voxels with enough of both), the PE drops differ in 58 voxels and the stress drops in 56 (two-sample KS with trajectory-permutation p < 0.05; about 4 expected by chance). Fast-cooled drops are smaller: by a factor of 0.56 (PE) and 0.77 (stress) in the median voxel.

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
- **Ranking within each cooling rate changes nothing here.** Step 9 found that pooling ranks over cooling rates adds spurious concordance (section 4). Rerunning this comparison with ranks within voxel × rate, Gumbel is still best (3138 against 3166 pooled), tied with the same four families.

**Corner check** (ranks; the data's 95% intervals come from resampling trajectories):

| | τ_K | Lower 5% corner | Upper 5% corner |
|---|---|---|---|
| Data | 0.726 [0.717, 0.735] | 0.256 [0.192, 0.324] | 0.686 [0.636, 0.739] |
| Gumbel | 0.702 | 0.503 | 0.778 |
| Flipped Clayton | 0.669 | 0.210 | 0.842 |
| BB7 | 0.671 | 0.267 | 0.843 |
| Flipped BB1 | 0.678 | 0.332 | 0.836 |

The lower corner is the probability that, when one drop is in its smallest 5%, the other is too; the upper corner is the same for the largest 5%. Independence would give 0.05 for both.

## 4. How the link depends on state and cooling rate (step 9)

**Data.** The 84,273 paired events (92%) in the 119 voxels with at least 100 each.

**Ranks within each cooling rate.** The size laws shift with cooling rate inside a voxel (section 2), so ranking a voxel's events together mixes rates: fast-cooled events sit low on both ranks at once, which looks like extra dependence. With ranks pooled over rates, a fitted rate term even came out with the wrong sign. Step 9 therefore ranks within voxel × cooling rate, falling back to three merged rate groups and then to the voxel when a group has fewer than 10 training events (about 2% of events at 50% training). A synthetic test confirms that this removes a spurious rate effect that pooled ranks create.

**Measured dependence.** Measured within each rate and averaged, τ_K runs from 0.52 to 0.88 across voxels, within 0.004 ± 0.012 of the pooled values. A model-free regression of the τ_K of 577 (voxel, rate) groups with at least 30 events each, on the group's median u and τ (standardised) and its ln rate, weighted by group size, gives:

| Effect | Change in τ_K [95%] |
|---|---|
| u (energy), per sd | +0.066 [+0.062, +0.070] |
| τ (stress), per sd | +0.054 [+0.051, +0.057] |
| cooling rate, per unit ln rate | −0.0158 [−0.0174, −0.0140] |

**Two cooling-rate numbers, defined.** The rate effect is quoted two ways:
- **−0.0158 [−0.0174, −0.0140] per unit ln rate: the group regression above**, which controls for the state (u, τ) of each group. This is the number used in the summary and the one to quote.
- **−0.0129 [−0.0151, −0.0108] per unit ln rate: the within-voxel slope**, from regressing each (voxel, rate) group's τ_K on ln rate after centring both within the voxel, pooled over voxels. It does not adjust for the small drift of the state with rate inside a voxel.

Both say that faster cooling weakens the link, by about 0.07–0.09 in τ_K over the ×256 range of rates. Per voxel, 33 of 83 voxels are clearly lower when fast and none higher. Faster-cooled glasses also sit at higher u, so leaving rate out of a model hides part of the u effect. Intervals come from resampling trajectories within each rate.

**Models.** Gumbel with θ = 1 + exp(η), where η is constant, a function of 100uτ, linear, linear plus the u·τ interaction, or quadratic in standardised (u, τ), each with and without a term g·ln(rate/rate_mid). References: one θ per voxel (119 parameters), and one θ per voxel plus a shared rate term (120). Scoring holds out whole trajectories; the deciding evidence is 50 random 50/50 and 50 random 80/20 splits of each rate's trajectories.

**The candidate formula** is interaction + rate (5 parameters):

η = 0.902 + 0.210 u′ + 0.253 τ′ + 0.067 u′τ′ − 0.051 ln(rate/rate_mid),

with u′, τ′ the standardised energy and stress. 95% intervals: intercept [0.890, 0.914], u′ [0.193, 0.228], τ′ [0.238, 0.267], u′τ′ [0.055, 0.079], rate [−0.060, −0.042]. The quadratic adds nothing clear (it wins 36 and 43 of 50 splits against the interaction form, by 0.1 per 1,000 events).

**How the coefficient intervals are computed.** Each interval is the estimate ± 1.96 sd. The sd comes from the spread of the coefficient over the 50 random 50/50 splits, which never duplicate a trajectory. Because the splits are drawn without replacement from one sample, the full-sample sd is the split sd × √(m/d), with m training and d held-out trajectories per rate (the delete-d jackknife); for halves this factor is 1. The 80/20 splits (factor √4 = 2) give the same sd within a factor 0.81–1.15 (median 0.92) across all 38 coefficients, which supports the scaling. Intercepts come out about 0.01–0.016 higher when fitted on half the data, probably because the rank groups are smaller; the slopes and rate terms shift by less than 0.006. The refitting bootstrap's intervals are not used: duplicated trajectories create tied ranks, which shift its intercepts below the estimates and also move the slopes (its τ′ median is 0.244 against 0.253).

| Model (parameters) | Gain over constant θ, per 1,000 held-out events (50/50; 80/20) | Minus its voxel table per 1,000 events: 50/50, 80/20 (wins of 50) | 5 folds [95%] |
|---|---|---|---|
| Voxel table + rate (120) | 19.3; 20.7 | — | — |
| Quadratic + rate (7) | 19.1; 19.6 | −0.20 ± 0.63 (19); −1.11 ± 0.67 (4) | −1.38 [−1.94, −0.81] |
| **Interaction + rate (5)** | **19.0; 19.4** | **−0.28 ± 0.61 (15); −1.23 ± 0.66 (3)** | **−1.54 [−2.10, −0.97]** |
| Product + rate (3) | 18.1; 18.7 | −1.18 ± 0.64 (2); −1.99 ± 0.73 (0) | — |
| Interaction, no rate (4) | 17.0; 17.5 | −0.32 ± 0.61 (14); −1.24 ± 0.68 (2) | — |

- **Effect size:** interaction + rate recovers 98.6% (50/50), 94.1% (80/20) and 92.5% (5 folds) of the voxel table + rate's gain over a single θ.
- **Against the table, the answer depends on training size.** With half the data the formula ties the table (it wins 15 of 50 splits). With 80% it loses in 47 of 50: the table's estimation noise shrinks with more data and real structure appears that the formula misses. With ranks pooled over rates the gap at 50/50 was −1.44 per 1,000 events (0 of 50 won); the clean ranks closed most of it.
- **The rate term helps every form** except the constant one: 1.4–2.5 per 1,000 events, in 50 of 50 splits. The fitted rate coefficient, −0.054 [−0.064, −0.045] in the table + rate model, is 70–80% of what the direct τ_K measurements imply (−0.065 from the within-voxel slope and −0.079 from the group regression, in η units).

**Calibration and residuals** (in-sample, against the τ_K measured within each rate):
- Voxels whose implied τ_K falls outside the measured interval: **37 of 119 for interaction + rate, against 28 for the voxel table + rate**.
- Every model understates τ_K by 0.013 on average (a correct model measured the same way shows 0.000), and compresses its range: the bias falls with implied τ_K by 0.22 per unit for the formula and 0.16 for the table, against 0.02 for a correct model.
- Where the formula misses (figure 6): both models are too weak in the dense high-u band at stress 1.0–1.5; the formula is also too strong at the low-u, high-stress edge and along the lowest-stress row. Those edges are the structure the table captures.

**Upper tail.** The probability that the stress drop is in its largest 5% given that the PE drop is: the fitted copulas exceed the data by +0.103 on average, and lie above the data's interval in 21 (formula) and 23 (table) of 119 voxels. Small rank groups bias the measured value low, so a correct model would show +0.038 and land above the interval in 2 voxels. The remaining **+0.065 is a genuine overshoot, present everywhere**: it does not correlate with u, τ, rate mix or τ_K (|r| ≤ 0.17), and the formula and the table show it equally. It is the Gumbel shape, not the θ(u, τ, rate) law.

**Family check (a hint, not a finding).** Rerunning the family comparison with clean ranks:

| Voxel (measured τ_K, events) | Best family | Gumbel minus best [95%, sampling only] |
|---|---|---|
| (13, 9), lowest τ_K (0.567, 302) | flipped Clayton | −1.1 [−12.3, +10.6], tied |
| (18, 11), median (0.726, 3,906) | Gumbel | best |
| (17, 15), highest (0.854, 413) | Frank | −33.8 [−67.5, −4.5] |
| (19, 14), 2nd highest (0.837, 488) | Frank | −59.9 [−92.9, −28.9] |
| (20, 13), 3rd highest (0.830, 591) | Frank | −73.0 [−108.3, −40.6] |

Frank, which has no tail link, beats Gumbel beyond noise in all three of the highest-τ_K voxels. This fits the upper-tail overshoot, but it rests on three voxels of 400–600 events each and on intervals that do not include refitting noise.

## 5. Limitations and open questions

1. **The large-drop corner is overstated everywhere.** In cell (18, 11) all the leading families put the upper 5% corner at 0.78–0.84 against 0.69 [0.64, 0.74]; across the state space the overshoot is about +0.065 after the measurement correction, for the formula and the voxel table alike. More parameters in θ cannot fix it. In cell (18, 11) a kernel copula with no family shape beats Gumbel by 314 held-out log-likelihood (step 11), so the shape, not the parameter, is the limit. One possible cause, not tested, is the hard ceiling on drop sizes.
2. **The best family may change with state.** Frank wins in the three highest-τ_K voxels checked. Gumbel was carried into step 9 from one voxel.
3. **The size laws depend on cooling rate inside a voxel**, which the size-law fits (steps 6–8) don't model, and steps 7–8 rank within the voxel, pooled over rates. Step 9 ranks within voxel × rate; the remaining fallbacks (about 2% of events) mix rates again, slightly.
4. **The stress marginal does not describe the paired events** (PIT KS 0.21), because aging events remain in its fit. That's why IFM understates τ_K, and why nothing that relies on the fitted marginals should be used yet.
5. **Neighboring voxels share trajectories.** Each voxel's interval resamples trajectories within that voxel, but comparisons across voxels (the τ_K trends, the counts of voxels outside an interval) treat voxels as independent. Their p-values are overstated.
6. **The error bars.** Step 7 and 8 intervals, and step 9's fold scores, resample trajectories without refitting; a synthetic coverage study found them about 1.4× too narrow for near-ties. The repeated splits include refitting but overlap, so their spread shows stability, not a formal confidence interval. The refitting bootstrap handicaps the voxel tables and biases coefficients (duplicated trajectories create tied ranks), so it is kept only as a secondary record; coefficient intervals come from the random splits instead.
7. **The link to step 3's ceiling law is a hypothesis.** The product form uses the same variable, 100uτ, as step 3's ceiling law. It is not the best form (product + rate trails the table by 1.2 per 1,000 events at 50/50 and wins 2 of 50 splits), and its zero crossing depends on the reference energy u₀.

## 6. Possible next steps

- **Per-voxel family comparison.** Run the family comparison in every well-filled voxel with clean ranks. The Frank result at high τ_K suggests the family, not only θ, varies with state.
- **A copula with a lighter upper tail.** The overshoot is uniform, so a family (or a mixture such as Gumbel–Frank) with a weaker upper corner could be tested across all voxels at once, with θ(u, τ, rate) as here.
- **Rate-aware size laws.** Fit the marginals per voxel × rate (or with a rate term), and fix the stress marginal (paired events only, or step 2's ceiling law). Both are prerequisites for IFM and for sampling.
- **Sample paired events inside SEEM.** Replace the empirical (Δu, Δτ) lists with draws from the size laws plus a copula. That needs the marginal fixes first, plus a decision on the aging and PE-only events, which a paired-event copula doesn't describe.
- **Test the ceiling link.** Run steps 1–3 in this folder and fit η = a + b·ln s_c(u, τ).

## 7. Figures for the meeting

All figures are at the top level of `figures/`; each panel of the multi-panel figures is also saved on its own in `figures/panels/` for slides.

**Recommended, in order:**

| Rank | File | Takeaway |
|---|---|---|
| 1 | `voxel_dependence.png` | The link varies strongly with state. |
| 2 | `copula_shapes.png` | Large drops of both kinds go together tightly; small ones only loosely. |
| 3 | `dependence_scoreboard.png` | A five-parameter formula captures nearly all of the variation; one θ per voxel still leads with enough data. |
| 4 | `dependence_tail_residuals.png` | The Gumbel tail is too heavy everywhere, for formula and table alike. |
| 5 | `dependence_tau_residuals.png` | Where the formula misses: the edges of the occupied region. |
| 6 | `tail_comparison.png` | In the busiest voxel, every family overstates how often the largest drops coincide. |
| 7 | `copula_scores.png` | Families with the link in the large-drop corner are tied for best. |

**Backup:** `marginal_fits.png` and `pit_histograms.png` (the stress-marginal problem), `marginal_maps.png` (the size-law parameters). Leave out `figures/exploratory/*` (notebook-era plots on a different grid).

**Captions.**
1. **`voxel_dependence.png`.** Strength of the link between PE drop and stress drop (Kendall's τ_K) in the 119 voxels with at least 100 paired events, measured within each cooling rate and averaged (a), and the half-width of its 95% interval from resampling trajectories (b). τ_K runs from 0.52 to 0.88 and rises with both potential energy and stress; grey voxels have too few events, and the orange box is the most-visited voxel (18, 11).
2. **`copula_shapes.png`.** Joint shape of the two drops in voxel (18, 11) (3,906 paired events), on the normal scores of their ranks: data (blue) against fitted copulas (orange contours, the same density levels in every panel). The families with the link in the large-drop corner reproduce the narrow upper-right ridge; the Gaussian cannot.
3. **`dependence_scoreboard.png`.** (a) Held-out log-likelihood gain over a single Gumbel θ per 1,000 held-out events, mean ± sd over 50 random 50/50 and 50 random 80/20 splits of the trajectories (stratified by cooling rate); percentages are the share of the voxel table + rate's gain. (b) Each formula minus its voxel table: every split (dots), mean ± sd, and the 5-fold score with its sampling-only 95% interval. The five-parameter interaction + rate form recovers 94–99% of the table's gain and ties it with half the data, but trails by about 1.2 per 1,000 events when trained on 80%.
4. **`dependence_tail_residuals.png`.** Model minus empirical probability that the stress drop is in its largest 5% given that the PE drop is, per voxel, for interaction + rate (a) and the voxel table + rate (b). Both overshoot by +0.10 on average (21 and 23 of 119 voxels above the interval, outlined), against +0.04 (2 voxels) for a correct model measured the same way; the overshoot shows no trend with state, so it comes from the Gumbel shape, not from the θ law.
5. **`dependence_tau_residuals.png`.** Model-implied minus measured τ_K per voxel for interaction + rate (a) and the voxel table + rate (b), with τ_K measured within each cooling rate. Outlined voxels lie outside the measured 95% interval (37 and 28 of 119). Both understate τ_K in the dense high-u band; the formula also overstates it at the low-u, high-stress edge and along the lowest-stress row.
6. **`tail_comparison.png`.** Kendall's τ_K and the probabilities that both drops are in their smallest or largest 5%, in voxel (18, 11): data with 95% interval against four rank-fitted families (independence gives 0.05 in each corner). Every family understates τ_K slightly and overstates how often the largest drops coincide (0.78–0.84 against 0.69).
7. **`copula_scores.png`.** Held-out log-likelihood gain over independence for every copula family in voxel (18, 11), fitted on ranks (reference) and on the fitted size laws (secondary), with 95% intervals from resampling trajectories (a), and each family's difference from the best (b). Gumbel, flipped Clayton, BB7 and flipped BB1 are tied; the second parameter of BB1 and BB7 does not help.
8. **`marginal_maps.png`** (backup). Truncated-power-law exponent κ and cutoff s* fitted to the PE drops (a, b) and stress drops (c, d) in each voxel with at least 100 drops; grey voxels have fewer. Both parameters vary smoothly across the state space.
9. **`marginal_fits.png`** (backup). Drop-size distributions in voxel (18, 11): all drops (grey, fitted) and paired drops only (orange), with the truncated-power-law fit. The fit describes the paired PE drops, but paired stress drops are scarce below about 10⁻², where stress-only (aging) drops dominate.
10. **`pit_histograms.png`** (backup). Fitted-CDF values of the 3,906 paired events under the voxel's size laws; a correct size law gives a flat histogram. PE is close (KS 0.039); stress is far from flat (KS 0.213), so results that rely on the fitted stress law are secondary.

## 8. Claims I would avoid

- **"Gumbel is the copula for these events."** It's tied with three other families in one voxel, Frank beats it at the highest τ_K, and its upper tail is too heavy everywhere.
- **"A formula describes the state dependence."** It recovers 94–99% of the table's gain and ties it with half the data, but with 80% of the data the table wins 47 of 50 splits, and the residual maps show where.
- **"Faster cooling weakens the link by g = −0.05."** The fitted coefficient is 70–80% of what the direct τ_K measurements imply; quote the direction and the measured −0.0158 per unit ln rate (group regression, at fixed state), about −0.09 in τ_K over the rate range.
- **"θ follows the ceiling law" or "the dependence is set by 100uτ."** That's untested, and the product form is not the best formula.
- **"The drop sizes follow a truncated power law."** It's approximate, rejected in the busiest voxel, and ignores the shift with cooling rate.
- **"IFM confirms the result."** IFM inherits the stress misfit and understates τ_K.
- **"The upper-tail dependence is λ_U ≈ 0.77–0.84."** These are model values. The data only measure corners at a finite level (5%), and there the models overshoot.
- **"τ_K rises with u and stress at p < 1e-9."** Voxels share trajectories, so p-values across voxels are overstated.
- **"The intervals include model uncertainty."** Most are sampling-only, and the repeated-split spread is not a confidence interval.
