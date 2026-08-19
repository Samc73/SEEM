# Constitutive-model discovery from AQS simulations of amorphous plasticity

## Status report with adversarial assessment of STZ-consistency

*Prepared 2026-08-06. Code: `library/stz.py`, `library/symreg.py`. Figures: `figures/`.*

**Provenance note.** This work originates in the laboratory where STZ theory was
developed. All findings favorable to STZ were therefore subjected to a
pre-specified robustness battery before inclusion, and results are reported at
the strength that survives that battery — including outcomes that weaken the
STZ interpretation. One earlier intermediate conclusion ("the Boltzmann form
wins decisively") is explicitly superseded below.

---

## 1. Data

850 athermal-quasistatic shear trajectories of a model glass; 9 cooling rates,
2×10¹⁰–5.12×10¹² K/s in factor-of-2 steps (100 trajectories each; 50 at the
fastest); 49,999 strain steps of Δγ=10⁻⁵ to γ=0.5. State variables follow lab
convention: u = pe − u₀ (u₀ = −4.60751861, dataset-derived), τ = stress/10⁴.

Preparation phenomenology is textbook effective-temperature behavior: initial
u spans 60,000× across the ensemble; ensemble peak stress falls monotonically
1.96 → 1.30 with faster cooling; stress-overshoot fraction 0.45 → 0.24
(`figures/state_sufficiency.png`, panel a). **No cooling rate reaches steady
state by γ=0.5** (panel b) — u drifts upward and τ downward in every strain
window to the end. All "steady-state" quantities below are extrapolations.

## 2. Framework

We do not regress the raw derivatives (that fails: R² ≈ 0.0005 at one step).
The dynamics are decomposed as a piecewise-deterministic process — elastic
branches, event hazard, jump kernel — and the modeling target is the
dimensionless plastic-rate field

  q(u,τ) = 1 − (dτ/dγ) / 2μ(u,τ),

with 2μ measured from elastic-branch slopes (two estimators — just-after-event
and branch-median — agree to 2.3%). q has the correct limits (→0 elastic,
→1 steady flow, >1 in the softening region) on a 10×10 (u,τ) grid with ≥20k
transitions per voxel; median statistical SE per voxel is 8.7% from
per-trajectory dispersion. All train/test splits are by whole trajectory.

## 3. Findings that survive scrutiny

**S1 — Factorization.** q(u,τ) ≈ Λ(u)·f(τ): a rank-1 (additive-in-log) model
leaves 3.6–4% of log-q variance; residual RMS is ~30% multiplicative, set by
systematics (S5), not statistics.

**S2 — Form of Λ(u).** Λ rises 26× over the accessible range as a floor plus a
stiff convex rise. Growth-exponential (A·e^{u/s}) and sinh forms **lose in
every one of 11 analysis variants** — a robust family-level exclusion. Within
the surviving family, activated A·e^{−c/û}+B versus power A·û^n+B is **not
decidable at current systematics**: the weighted pipeline prefers activated
(7/9 variants, margins 1.08–1.47×); the unweighted pipeline prefers power
(margin 3.9×; 97% of 300 trajectory bootstraps; exhaustive symbolic search
returns cubic/quartic knees in 12/12 unweighted replicates but e^{−c/x} on the
weighted target). A ±0.003 shift in the u-zero (10% of χ̂; u₀ is
dataset-derived, not independently known) flips the ranking, and free-zero
fits are degenerate (9.6% vs 10.9% relRMS). Conditional on the activated form:
c_u = 1.79–1.90 (pipeline spread; bootstrap 16–84% [1.83, 1.97]), i.e. a
formation energy e_Z = c·χ̂ ≈ 0.056–0.059 per-atom PE units.

**S3 — Form of f(τ).** Same structure: floor + stiff rise (30×);
growth-exponential and sinh excluded everywhere; activated-in-1/τ̂ (weighted
pipeline, c_τ ≈ 4.2–4.6) versus power-with-floor (unweighted; a bounded
Herschel–Bulkley fit drives its threshold to x₀=0, n≈3.4) are near-degenerate.
**No robust evidence for a nonzero macroscopic stress threshold** in f at this
resolution.

**S4 — The plastic background is real.** The low-u floor (q ≈ 0.078) is not a
modulus artifact (branch vs fresh-branch modulus: 0.0785 vs 0.0773) and is
carried by genuine stress-drop events at p ≈ 0.23%/step even at the lowest u.
Aging-only events (PE drop without stress drop) are 9.1% of all events.

**S5 — Two-variable state insufficiency (headline negative result).** At fixed
(u,τ), the slowest- and fastest-cooled ensembles flow at rates differing by
1.25–2.04× (median 51%, correlation 0.988, always the same sign) against a
split-half noise floor of 11.3% (`figures/state_sufficiency.png`, panel c).
Accumulated plastic strain as a third variable absorbs ~⅓ of the gap
(51→37% vs a 16% three-variable floor). Consistently, the apparent barrier
parameters drift with preparation far outside statistical error: Δc_u spans
+0.42 to −0.49 (≈ ±25% of c_u) and Δc_τ spans ~1.3 across the nine cooling
rates (`figures/stz_scrutiny.png`, panel c). **No two-variable (u,τ) closure —
STZ-form or otherwise — can be exact for this system, and the barrier is not a
preparation-independent constant at our resolution.**

**S6 — Generalization.** Trained on the 7 slowest rates and tested on the 2
fastest, the 5-parameter factorized activated model reaches 30.7% median
error vs 32.4% for a 15-parameter quartic (which trains to 5.7%, below the
8.7% noise floor — memorization) (`figures/symbolic_regression.png`; its
panel-a title reflects the weighted pipeline only and is superseded by §4).
Both models sit at the state-insufficiency ceiling: added complexity buys
train error only.

## 4. Assessment with respect to STZ theory

**Supported.** Effective-temperature phenomenology (overshoot ordering by
preparation; u relaxing toward a common attractor ≈ 0.031, not yet reached);
the factorized structure q = Λ(u)·f(τ) that STZ posits (STZ-density factor ×
stress factor); an STZ-consistent activated Λ is one of the two surviving
forms, and on the statistically-weighted pipeline it is the best one.

**Not established.** The Boltzmann form e^{−e_Z/χ} *uniquely* (degenerate with
a power law under the extraction systematics and the u-zero uncertainty); a
preparation-independent formation energy (contradicted at ±25%, S5); a
canonical STZ stress factor (the measured f is floor-plus-stiff-rise; a
marginal-stability-flavored power law describes it at least as well).

**Contradicted.** Sufficiency of any two-variable (χ,s)-type closure, at the
1.7× level between extreme preparations. STZ's own candidate for the missing
physics — the orientational bias / back-stress variable m — is untested here
because all data are monotonic forward shear; accumulated plastic strain
accounts for only ~⅓ of the violation.

The earlier intermediate result reported in this project ("activated form wins,
found independently by symbolic search") was conditional on the weighted
extraction pipeline; the battery exposed that conditionality, and §3-S2 is the
corrected statement.

## 5. Robustness battery (summary)

| Variant | Λ winner | f winner |
|---|---|---|
| V1 weighted-stat fit | activated (1.30×) | activated (1.16×) |
| V2 unweighted ALS+fit | **power (3.9×)** | **power-with-floor** |
| V3 relative-error fit | activated (1.09×) | activated (1.02×) |
| V4 log-space fit | activated (1.08×) | tie |
| V5 pre-softening only (q<1) | activated (1.28×) | activated (1.09×) |
| V6 fresh-branch modulus | activated (1.42×) | activated (1.01×) |
| V7 rise-only (û≥0.3) | indistinguishable | — |
| V8 leave-one-out CV | activated (1.45×) | activated (1.21×) |
| V9 u-zero ±0.003 | **flips with sign** | — |
| 300× trajectory bootstrap (unweighted) | power 97% | power-floor 99% |

Margins are (2nd-best)/(best) relative residual RMS. The two pipelines are
each internally stable; the disagreement between them — 32% maximum shape
difference in the extracted Λ — is the dominant systematic, larger than any
sampling effect.

## 6. Known systematics and limitations

1. Extraction-weighting systematic on Λ (32% shape) — dominates form selection.
2. u₀ is dataset-derived; the 1/u structure is sensitive to it at the ±0.003
   level. No independent reference-energy measurement exists in this dataset.
3. Steady state not reached; χ̂ = 0.0312 is the observed maximum u, used as a
   scale only (barrier values in raw u-units are χ̂-independent).
4. Extreme-preparation comparison rests on 10 well-populated shared voxels;
   the held-out test on 24.
5. Grid edges and χ̂ were derived from the full dataset (minor leakage into the
   held-out test).
6. Event labels use a zero threshold; q is insensitive to this (mean-drift
   construction), but hazard-based quantities would not be.

## 7. Recommended next steps (ranked)

1. **Independent determination of the reference energy u₀** (e.g., inherent-
   structure energy of deeply-annealed or crystalline reference), or new
   ensembles populating lower u under stress — this alone can break the
   activated-vs-power degeneracy in Λ.
2. **Reverse-loading AQS from saved configurations.** Cheap relative to the
   original runs; directly tests the back-stress/orientational-bias variable
   (STZ's m) via the parity constraints (G even, plastic rate odd in τ), and
   is the leading candidate for the ⅔ of the state-insufficiency gap that γ_p
   does not absorb.
3. **Longer strain runs** at 2–3 cooling rates to actually reach steady state
   (χ̂, and the f floor, measured rather than extrapolated).
4. **Spatial fields** for a subset of trajectories: localization/shear-banding
   is both a candidate explanation for S5 and the object of STZ's sharpest
   untested prediction here (χ-diffusion); per-atom data would also give the
   local-threshold distribution P(x) directly, discriminating the
   marginal-stability reading of f.
5. Interim modeling: adopt the factorized activated form as the working
   2-variable closure (it is the best 5-parameter model and extrapolates at
   the ceiling), with (u₀- or γ_p-) preparation-parameterized coefficients,
   clearly labeled history-parameterized rather than constitutive.

## Figures

- `figures/state_sufficiency.png` — preparation phenomenology; q-field;
  state-insufficiency scatter.
- `figures/symbolic_regression.png` — discovered forms (weighted pipeline);
  held-out-preparation comparison vs fixed-library baselines.
- `figures/stz_scrutiny.png` — pipeline dependence of Λ and f; per-preparation
  barrier drift; bootstrap distributions.
