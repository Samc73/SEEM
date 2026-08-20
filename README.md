# SEEM — a fitted law for yield events, and everything rebuilt from it

This is the second pass over the SEEM dataset, and it inverts the logic of the
first. Last time we *measured* the plastic response of the glass voxel by voxel
and asked what functions describe the measurements. This time we go after the
project's original target directly: **the probability distribution of yield
events**. We fit one law for the event statistics — with the functional form
*discovered by symbolic regression, not assumed* — and then re-derive every
downstream quantity (mean event sizes, the plastic-rate field, its factor
structure, and finally full synthetic stress–strain curves for all 850
simulations) **from the fitted law**, side by side with the directly measured
version of each quantity. Where the two versions agree, the law is carrying the
physics. Where they disagree, we say so.

The first-pass technical report ([REPORT.md](REPORT.md)) and its figures are
unchanged and still valid; this document supersedes the old README.

---

## 1. The data, briefly

850 athermal quasistatic (AQS) shear trajectories of a model glass: strain is
applied in steps of Δγ = 10⁻⁵ up to γ = 0.5 (49,999 steps each), with the
system energy-minimized at every step. Nine preparation histories — cooling
rates from 2×10¹⁰ to 5.12×10¹² K/s in factors of two — with 100 runs each (50
at the fastest). Each step records the shear stress and the per-atom potential
energy: 42.5 million rows total.

In AQS dynamics every strain step is one of two things: a smooth **elastic**
increment, or a discontinuous **event** (a plastic rearrangement). That
dichotomy is not a modeling choice, it is what the algorithm does — which is
why the model below has the structure it has.

## 2. The variables, all in one place

| symbol | plain reading |
|---|---|
| γ | applied shear strain; one step = 10⁻⁵ |
| τ | shear stress (raw stress ÷ 10⁴, the lab convention) |
| u | per-atom potential energy above a reference: u = pe − u₀, with u₀ = −4.60752 taken from the dataset. High u = poorly annealed, "young" glass |
| (u, τ) | the two-coordinate **state** of a run; all fields below live on a 20×20 grid over this plane |
| 2μ(u,τ) | elastic modulus: the slope dτ/dγ on elastic steps (≈ 20 in these units) |
| event, s | a step where stress drops; its **size** s = −Δτ. 187,468 of them in the dataset, spanning s ≈ 10⁻⁶ to 1.16 |
| aging event | a step where energy drops but stress does not (105,188 of them, 36% of all events); they relax u without releasing stress |
| p(u,τ) | event **hazard**: probability per step that an event fires (median ≈ 0.3%/step) |
| P(s \| u,τ) | the size distribution — *the object this whole pass is about* |
| k | the law's small-size exponent: P(s) ∼ s⁻ᵏ for small s. Fitted k = 0.88 |
| s_c(u,τ) | the **ceiling**: the largest event the state (u,τ) can produce. The law's only state-dependent parameter |
| m | the ceiling exponent: P(s) vanishes like (s_c − s)ᵐ at the ceiling. Fitted m = 2.3 |
| ε | a tiny rounding scale (≈ 3×10⁻⁵) below which the power law flattens; resolution-limited |
| ξ | rescaled size (s+ε)/(s_c+ε): in this variable every cell has the *same* distribution |
| q(u,τ) | plastic fraction of the strain rate: q = 1 − (dτ/dγ)/2μ. 0 = purely elastic, 1 = steady flow |
| Λ(u), f(τ) | the two factors of q ≈ Λ(u)·f(τ) (and analogously for ln s_c) |
| "rank-1" | how much of a field's (log) variance a product of one-variable factors captures — 96% for q, 96% for s_c |

## 3. The model: three measured fields and one law to discover

Because AQS is literally "drift, then jump", the generative model is fixed by
the dynamics itself; the only scientific freedom is in the ingredients:

1. **Elastic drift** — τ rises at 2μ(u,τ), u drifts slightly; both measured
   directly from elastic steps.
2. **When events fire** — the hazard p(u,τ), measured as a per-voxel event
   fraction.
3. **How big they are** — P(s | u,τ), *the part we fit*, plus a linear
   coupling giving the energy change per event and an exponential model for
   the (small) aging channel.

Everything is tabulated on a 20×20 grid over (u,τ) (equal-width bins between
the 0.2% and 99.8% quantiles, plus a catch-all outer ring so a simulation can
never step off the map). Ingredient 3 is where the physics lives, and it is
the one we refused to write down by hand.

## 4. Finding the form: exhaustive symbolic regression

**Expect this component to draw the most scrutiny, so here is exactly what was
done.** The search uses [library/symreg.py](library/symreg.py), a purpose-built
engine with one deliberate property: for one-variable targets it does not
sample the space of formulas, it **enumerates all of them** up to a complexity
budget, so "the search did not find X" is a checkable statement, not a claim
about luck.

**The target.** For a given grid cell (a narrow window of u and τ, inside
which the state is effectively constant), take all its event sizes, bin them
in 19 logarithmic bins, and form the log-density ln ρ(s) with Poisson error
bars (1/√count per bin). The regression target is y = ln ρ(s) vs x = s.

**The grammar.** Expressions are trees built from x, fitted constants, six
unary operators (exp, log, 1/·, (·)², √, −) and four binary ones (+, −, ×, ÷).
Each expression may carry at most two internal constants, fitted by
Levenberg–Marquardt with 4 restarts, while the outer scale and offset
a·f(x)+b are solved in closed form (variable projection) — so every candidate
is optimized, not just evaluated. Transcendentals cost 2 complexity units,
everything else 1, which makes the search prefer plain algebra unless the
data insists otherwise.

**The scale.** At complexity ≤ 8, the enumeration visits **624,936 trees per
target**; a numerical fingerprint (invariant to the outer scale/offset,
evaluated on a log-spaced probe so behavior at small s is not aliased)
collapses these to ~14,000 genuinely distinct functions, of which ~10,300
survive the admissibility filter (finite and monotonically decreasing over
the fit window). This was run on **12 cells** spread over the (u,τ) grid
(4 orders of magnitude in event size, 700–7,600 events each), plus four
variants — two cells re-run with unweighted bins and two with the fit window
extended toward smaller sizes — to expose the two systematics that dominated
form-selection in the first pass. One **deep run at complexity ≤ 9 enumerated
4,505,024 trees** (68,305 distinct, 48,725 admissible) on the largest cell to
check that nothing qualitatively new appears with more budget. It doesn't.

**What came out.** Every one of the 16 fronts tells the same story. The
canonical truncated power law s⁻ᵏe^(−s/s*) — which sits *inside* this search
space at complexity 7 — **never appears on a single Pareto front**. What
recurs instead, across cells, weightings, and windows, is the structure
`log(1/(s+ε) − C)`, which algebra rearranges into

$$\rho(s) \;\propto\; \frac{(s_c-s)^{m}}{(s+\varepsilon)^{k}},\qquad 0<s<s_c$$

a power-law decay that does not fade out exponentially but **terminates at a
finite ceiling s_c**, just above the largest observed event. Physically that
is a finite-size statement: a finite simulation cell has a largest possible
stress release, and the data are near enough to it to see it.

**Symbolic regression proposes; likelihood disposes.** Binned least squares is
a blunt instrument — it over-weights the crowded small-size bins, and one
seductive low-complexity form it produced, √log(c/s), turned out to be pure
binning artifact (likelihood demolished it at ΔAIC ≈ +20,000). So every form
the search surfaced was promoted to a properly normalized probability density
and made to compete by maximum likelihood — against the truncated power law,
stretched cutoffs, the lognormal, and a bounded pure power law — under AIC
and under **trajectory-blocked two-fold cross-validation** (events from the
same run are correlated, so runs, not events, are what we split). Figures
1–4 walk through the result of that trial, one figure at a time.

![One cell, all fitted laws](figures/fig01_size_law.png)

**Figure 1 — one cell, one distribution, three candidate laws.** A single
well-populated cell of the state grid, with its full range of event sizes —
five decades — on log-log axes. Black points: the measured density. Red: the
discovered ceiling law. Blue dashed: the best truncated power law (TPL).
Green dotted: the best lognormal. All three are honest maximum-likelihood
fits of proper densities. The ceiling law is the only one that captures both
ends — the flattening below s ≈ 3×10⁻⁵ (the ε rounding) *and* the abrupt
termination at the dotted vertical line (the ceiling s_c), where the TPL's
exponential tail decays too softly and overshoots. In numbers: on the full
range the ceiling law beats the TPL by ΔAIC ≈ 200–345 per cell, and at the
conventional analysis window (s ≥ 10⁻⁴) its free-exponent version wins AIC
in **8 of 8** well-populated cells (margins 67–120 vs TPL, 300–750 vs
lognormal) and wins cross-validation in 5 of 8, with the stretched-cutoff
TPL close behind in the other three.

![The collapse](figures/fig02_collapse.png)

**Figure 2 — the universality claim in one plot.** Every test cell's density,
replotted against the rescaled size ξ = (s+ε)/(s_c+ε); each cell contributes
*its own fitted normalization only* — no per-cell shape freedom. All 12
cells, whose raw distributions differ by two orders of magnitude in scale,
fall on the single black curve (1−ξ)ᵐ ξ⁻ᵏ, including the bend into the
ceiling at ξ → 1. This is the strongest visual statement of the result:
**one shape, one state-dependent number.**

![Symbolic-regression Pareto fronts](figures/fig03_pareto.png)

**Figure 3 — the discovery process itself, so it can be audited.** Each grey
line is one cell's exact Pareto front: the best achievable misfit at each
expression complexity. Orange is the largest cell, pushed one complexity
unit further (4.5 million trees). The fronts drop steeply until complexity
8 — where the ceiling law enters (arrow) — and gain little beyond. Because
the enumeration is exhaustive, "the TPL never made a front" is a statement
about this grammar and this data, not about search luck.

![Falsifiability control](figures/fig04_falsifiability.png)

**Figure 4 — the verdict cannot be a pipeline artifact.** The falsifiability
control: synthetic catalogs of matched size were generated from a *known*
TPL (exponential tail) and from a known ceiling law, and the identical
machinery was run on both. Each dot is one fitted catalog; height is the AIC
margin, and points above the line mean the ceiling law wins. Left column: on
TPL-generated data the verdict is TPL, five seeds out of five (the ceiling
fit betrays its own degeneracy by pinning its ceiling to the sample maximum,
and the SR knee comes out as `log(x) − c·x` — exactly the TPL — with no
bounded form in sight). Middle: on ceiling-law data the verdict is the
ceiling, five of five, with ε and s_c recovered to a few percent. Right: the
twelve measured catalogs. Eleven sit decisively on the ceiling side; one
sparse cell is a statistical tie. **The discriminator works in both
directions, and the data lands on one side of it.**

## 5. The law

$$\boxed{\;P(s \mid u,\tau) \;=\; \frac{1}{Z}\,\frac{\bigl(s_c(u,\tau)-s\bigr)^{m}}{(s+\varepsilon)^{k}}\,,\qquad
k = 0.88,\quad m = 2.3,\quad \varepsilon = 2.8\times10^{-5}\;}$$

Three global constants; **all** dependence on state — and, Section 6, on
preparation history — enters through the single ceiling field s_c(u,τ). The
law needs no lower cutoff: ε lets it describe every recorded event down to
s = 10⁻⁶.

Stated at the confidence each piece deserves:

- **Solid.** A power-law size regime with k ≈ 0.9, and a large-size
  termination *sharper than exponential* wherever the statistics can tell
  the difference. Both survive reweighting, window changes, cross-validation,
  and the synthetic controls.
- **Softer.** The exact exponents move with fitting window: k ∈ [0.85, 1.03],
  m ∈ [1.6, 2.7] across treatments. (Symbolic regression's complexity budget
  first delivered the law with m tied to k; likelihood then demanded the
  decoupling, m > k.) In sparse cells m and s_c degenerate toward the
  exponential-tail limit — a ceiling needs events near it to be visible.
- **Open.** A hard ceiling begs to be a finite-size effect, and its scaling
  with system size N is the natural test — but N is not recorded in this
  dataset. Worth chasing.

## 6. The ceiling field s_c(u,τ)

With the shape frozen, one number per grid cell — the ceiling — was fit by
maximum likelihood in each of 277 cells (median bootstrap uncertainty 6.5%
on ln s_c).

![The ceiling field](figures/fig05_ceiling_map.png)

**Figure 5 — the ceiling field itself.** The largest possible event, mapped
over the state plane: it grows from ≈ 0.02 in the cold, low-stress corner to
≈ 3 at high u and τ — a **146× range**. Read it as a map of fragility: the
same glass, depending on where it sits in (u,τ), can at most release either
a tiny flicker of stress or nearly its entire load in one avalanche.

![The ceiling's two factors](figures/fig06_ceiling_factors.png)

**Figure 6 — the ceiling factorizes.** The field is a product:
ln s_c(u,τ) = (u-factor) + (τ-factor) captures 95.6% of the weighted
variance, and the two panels show the two measured factors (points) with the
form symbolic regression finds for each (red; same exhaustive protocol as
Section 4, applied to the factors). The u-factor's discovered form contains
`exp(0.065/u)` — an activated, Boltzmann-like dependence on 1/u, the
STZ-flavored structure showing up uninvited in the *event-scale* field. The
τ-factor is a stiff saturating rise. We display these as discovered
parameterizations, not established physics: at factor level several forms
tie within noise — that ambiguity was the central lesson of the first pass,
and it has not gone away.

![Ceiling memory](figures/fig07_ceiling_memory.png)

**Figure 7 — the model's hidden-variable problem, localized.** The ceiling
was refit twice per cell — once from slow-cooled runs only, once from
fast-cooled only — at the *same* (u,τ). If (u,τ) were a sufficient state,
the points would sit on the black identity line. They sit above it: median
ratio **1.71×** (16–84% range 1.14–2.60) across 45 cells, versus 1.13× for a
within-ensemble split control. Preparation memory, which the first pass
detected as a flow-rate violation, is now a *parameter of a fitted law*: at
identical coarse state, the well-annealed glass carries a higher ceiling —
bigger maximum avalanches. Any candidate third state variable must explain
precisely this number.

## 7. Rebuilding q from the law

The first pass's central object — the plastic-rate field q(u,τ) — is now a
*derived* quantity: hazard × mean event size ÷ modulus. Replace the measured
mean size with the fitted law's first moment ⟨s⟩ = ∫ s P(s|u,τ) ds, and any
discrepancy in q is exactly a statement about the law's moment accuracy.

![The two q maps](figures/fig08_q_maps.png)

**Figure 8 — measured next to rebuilt.** The measured q field (left) beside
the q field rebuilt from the fitted law (right), on identical color scales.
They are near-duplicates across the map, including the elastic floor at low
u and the softening region (q > 1) in the high-u, high-τ corner.

![Voxel-by-voxel agreement](figures/fig09_q_agreement.png)

**Figure 9 — the same comparison, voxel by voxel.** Each dot is one grid
cell, across two decades of q. The shaded band is this grid's statistical
noise floor (17.5%, from split-half ensembles). The reconstruction lands at
median +1.8%, RMS 17% — **the fitted law reproduces the plastic-rate field
to within the data's own noise.** There is no headroom left; a better law
could not score better on this test.

![First moments](figures/fig10_moments.png)

**Figure 10 — where the ceiling matters and where it doesn't.** Model mean
sizes against measured ones, for both fitted laws. The TPL (blue) nails
first moments almost by construction — at fixed exponent, fitting its scale
s* *is* moment-matching, an accounting identity rather than evidence for the
exponential tail. The ceiling law (red) earns its moments from a shape
chosen by per-event likelihood: unbiased, with ~17% scatter. Mean-level
quantities cannot discriminate the two tails — which is exactly why Section
4 argued at the per-event level.

![The q factors from both fields](figures/fig11_q_factors.png)

**Figure 11 — the first pass's factor analysis, run twice.** Once on
measured q (black), once on model-rebuilt q (red). Both factorize at 96%;
the factors lie on top of each other; and symbolic regression, run
independently on each version, returns the **same functional knee with
nearly the same constants**. For f(τ), right panel: τ·log(1/τ − 0.44) from
measured q vs τ·log(1/τ − 0.43) from the model. For Λ(u), left panel: a
form with a **pole at u = 0.0287** (measured) vs u = 0.0280 (model) — a
divergence of plastic activity sitting just past the edge of the sampled
range (largest u-bin center 0.0275) and just above the attractor the
dynamics heads toward (u ≈ 0.0255; Figure 15). The first pass's downstream
analytic structure is fully recoverable from the fitted event law; none of
it depended on measuring q directly.

## 8. The acid test: simulating all 850 runs

If the model — two drift fields, the hazard, the jump kernel, and P(s|u,τ) —
really is the dynamics, it should be able to *replace* the MD. So we ran it
forward: 850 synthetic trajectories from the measured initial conditions,
step by step at Δγ = 10⁻⁵, drawing every event size from the fitted law. And
because the law is the part on trial, the identical simulation was run three
ways: sizes from the ceiling law, sizes from the fitted TPL, and sizes
resampled from the raw per-voxel catalogs (the "cheating" arm — as well as
any size model could possibly do here).

![Simulated stress-strain](figures/fig12_sim_stress.png)

**Figure 12 — the stress–strain curves, predicted.** Ensemble curves for all
nine preparations (color = cooling rate): MD solid, model dashed. The model
reproduces the elastic rise, the ordering and height of the stress
overshoot across a 256× range of cooling rate, the post-yield decay, and
the flow plateau (flow stresses at γ = 0.5 within ±5%). None of these
curves was fit directly — they emerge from fields measured at the
single-step level.

![Simulated energy curves](figures/fig13_sim_energy.png)

**Figure 13 — the energy coordinate, and the first crack.** The same
comparison for u(γ). The model is good early and visibly too *convergent*
late: the dashed curves pinch together faster than the data's — the real
curves keep a 23.7% spread at γ = 0.5 where the simulation keeps only 4%.
Keep this figure in mind for Figure 15.

![The yield peaks](figures/fig14_sim_peaks.png)

**Figure 14 — the 850 yield peaks.** The distribution of maximum stress per
run, by preparation. Black: MD. The three simulation arms bracket the data:
empirical resampling at 0.9% RMS, TPL at 1.4%, ceiling law at 2.5% (worst
single rate 4.3%). Two honest readings. First, the model predicts the
yield-strength distribution of a glass from its preparation history to a
few percent. Second, the *differences between size laws are smaller than
the differences between preparations* — macroscopic observables feel only
the law's mean, so they are the wrong place to identify tail shape (Section
4's per-event likelihood is the right place).

![Extrapolating past the data](figures/fig15_sim_extrapolation.png)

**Figure 15 — the model run 4× past the data.** What the model says about
the question the data cannot answer: no MD run reached steady state by
γ = 0.5. Run further, the model's nine preparations converge onto a common
attractor u∞ ≈ 0.0255 by γ ≈ 1, the preparation spread shrinking from 4.0%
to 2.6%. **Treat this as the model's opinion, not as evidence**: its fields
are preparation-blind by construction (one shared map for all histories),
so eventual convergence is baked in. The data's own verdict is Figure 13 —
the 5×-too-small u-spread is the same missing state variable as Figure 7,
seen macroscopically. This, quantitatively, is the model's failure mode,
and it is the strongest argument in this repository for runs past γ = 0.5
and for a third state variable.

## 9. The modeling choices, and why

1. **Drift + jump decomposition.** Not assumed — AQS dynamics *is* elastic
   branches punctuated by discrete events. The decomposition is exact
   bookkeeping; all modeling risk is pushed into the three ingredients,
   where each can be tested separately.
2. **20×20 state grid.** Finer than the first pass's 10×10, as befits a pass
   whose subject is distributions rather than means. The price is a higher
   per-voxel noise floor (17.5% vs 11.3% split-half); every "at the floor"
   claim above uses the recomputed floor, not the old one.
3. **Form discovery by exhaustive enumeration, not genetic search.** The
   headline claim is a *negative* one ("no exponential tail survives"), and
   negative claims require a search you can prove was complete. Counts:
   624,936 trees per target at complexity ≤ 8, spot-checked at ≤ 9 with
   4,505,024.
4. **Poisson-weighted binned target for SR; likelihood for selection.**
   Binned least squares is what makes an exhaustive search affordable; it is
   *not* how a winner gets picked (it over-weights crowded bins and produced
   one artifact form that likelihood killed at ΔAIC +20,000). Division of
   labor: SR proposes structures, normalized MLE with AIC and
   trajectory-blocked CV disposes.
5. **Trajectory-blocked validation.** Events within a run are correlated;
   splitting by event would leak information. All cross-validation splits
   are by whole trajectory, as in the first pass.
6. **Global (k, m, ε), local s_c.** Justified three ways: the fitted
   exponents are flat across cells (k spread ±0.05); the collapse in
   Fig. 2 works with all shape freedom removed; and the rank-1 structure
   of s_c says the one local parameter is itself low-dimensional. Refitting
   ε per cell moves it ±2× with no AIC gain — kept global.
7. **m decoupled from k.** SR's complexity budget first delivered the law
   with a single exponent (m = k); likelihood preferred decoupling in 8/8
   large cells, and the tied version biased mean sizes by +10%. We report
   the tied form as the *discovery* and the decoupled form as the *fit* —
   that is the honest division.
8. **No lower size cutoff.** The ε-rounding is part of the law, so the fit
   window is "all recorded events" — removing the xmin arbitrariness that
   plagued first-pass exponent estimates (the old κ drifted 1.2→1.6 with
   xmin; k now moves only 0.85→1.03 under far larger window changes, and
   the first pass's κ ≈ 0.95 is recovered as the mid-range slope).
9. **Aging events as a separate exponential channel.** 36% of events, but
   they carry no stress drop; they matter only for u-bookkeeping. An
   exponential per-voxel size model is the least structure that keeps the
   u-drift honest; nothing downstream is sensitive to this choice.
10. **Three simulation arms.** The empirical-resampling arm bounds what any
    size law could achieve; the TPL arm is the ablation. Differences between
    arms measure how much the law's *form* matters for each observable —
    which is how Fig. 14 can honestly conclude "peaks don't discriminate
    tails" instead of over-claiming a win.

## 10. What holds, what doesn't

| claim | status |
|---|---|
| Event sizes follow one universal shape with a single state-dependent scale | **supported** (Fig. 2: 12 cells spanning 100× in scale collapse) |
| The large-size cutoff is a hard ceiling, sharper than exponential | **supported where testable** (8/8 AIC, synthetic controls pass both directions); degenerate in sparse cells |
| Small-size exponent k ≈ 0.9 | **supported**, ±0.1 window systematic |
| Ceiling exponent m ≈ 2 | order-of-magnitude only |
| The ceiling field factorizes; its u-factor is activated-like | factorization **supported** (96%); the specific factor forms are best-of-front, not unique |
| The fitted law reproduces the plastic-rate field q | **at the noise floor** (Fig. 9) — this test is saturated |
| The fitted model reproduces stress–strain curves and yield peaks across 256× in cooling rate | **supported**, few-percent level (Figs. 12, 14) |
| (u,τ) is a sufficient state | **contradicted, again, independently**: 1.71× ceiling memory (Fig. 7) and 5×-underpredicted u-spread (Figs. 13, 15) |
| Steady state / convergence of preparations | not in the data; the model's convergence prediction (Fig. 15) is structurally baked in — needs longer runs |

**Next, in order of value:** (1) longer strain runs — every open question
above touches γ > 0.5; (2) the system size N, ideally a second size, to test
the finite-size reading of the ceiling; (3) any candidate third state
variable, evaluated directly against the 1.71× ceiling-memory ratio; (4)
reverse loading, unchanged from the first pass.

## 11. What's in the repository

| file | contents |
|---|---|
| [library/symreg.py](library/symreg.py) | the exhaustive symbolic-regression engine (grammar, variable projection, fingerprint dedup, exact Pareto fronts) |
| [library/dist.py](library/dist.py) | normalized-MLE machinery: candidate densities, AIC, trajectory-blocked CV, moments, sampling |
| [library/stz.py](library/stz.py), [library/avalanche.py](library/avalanche.py) | first-pass field extraction and TPL fitting (kept for comparison) |
| [REPORT.md](REPORT.md) | first-pass technical report (measured-field route + STZ scrutiny) |
| figures/fig01 … fig15 | this pass (Figures 1–15, one plot per figure) |
| figures/state_sufficiency.png … steady_state.png | first pass |

Raw data (`df_clean.pkl`, 2.4 GB) is not distributed with the repository.
