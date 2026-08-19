# SEEM: discovering the equations of glassy plasticity

This repository contains a data-driven search for a **constitutive model** — the
equation relating stress, deformation, and internal state — of a simulated glass,
together with an adversarial test of the leading theory in the field (the
shear-transformation-zone, or STZ, theory). This README is a ~10-minute overview
that builds from basics. The full technical detail lives in [REPORT.md](REPORT.md),
and a slide version in `SEEM_project_deck.pptx`.

---

## 1. The question

When you bend a metal past its elastic limit, we know what happens microscopically:
crystal defects called dislocations glide through the lattice, and a century of
theory turns that picture into predictive equations. **Glasses have no lattice**, so
they have no dislocations — yet they still yield, flow, and harden. Their plasticity
happens through localized rearrangements: small clusters of atoms that suddenly
snap from one packing to another.

What's missing is the *equation of motion* for this process — a constitutive law
you could hand to an engineer. The best-developed candidate is **STZ theory**
(Falk & Langer, 1998), which postulates that flow is carried by a sparse population
of "shear-transformation zones" whose abundance is controlled by an **effective
temperature**: not the thermal temperature, but a measure of how much disorder is
frozen into the structure.

This project asks two things:

1. Can we **discover** the constitutive law directly from simulation data, using
   symbolic regression, rather than postulating it?
2. Does what we find **support or contradict STZ theory**? (A disclosure that
   shapes everything below: this work comes from the lab where STZ theory was
   invented, so pro-STZ findings were deliberately subjected to extra scrutiny.)

## 2. The data

The dataset is 850 molecular-dynamics simulations of a model glass being sheared,
in the cleanest protocol available: **athermal quasistatic (AQS)** shear. Shear the
box by a tiny increment (10⁻⁵ strain), let every atom settle into mechanical
equilibrium, record, repeat — 49,999 times per sample, out to 50% strain. There is
no thermal noise and no rate dependence; every stress drop is a genuine mechanical
instability.

Crucially, the samples were prepared at **nine different cooling rates** spanning
2×10¹⁰ to 5×10¹² K/s. Cooling rate is to a glass what aging is to cheese: cool
slowly and you get a deep, stable, well-annealed structure; quench fast and you
freeze in a sloppy, high-energy one. This gives us nine materially different
starting points for the *same* substance — the key lever for testing whether a
candidate law is really a law.

Each snapshot reduces to two numbers:

- **τ** — the shear stress (how hard the sample is being pushed), and
- **u** — the potential energy per atom above a reference, i.e. *how badly packed
  the structure is*. Low u = deep in the energy landscape, well-annealed. High u =
  shallow, rejuvenated. This is the natural stand-in for STZ theory's effective
  temperature.

![Preparation phenomenology and the state-sufficiency test](figures/state_sufficiency.png)

**Panel (a)** shows the ensemble stress–strain curves: an elastic ramp, then — for
well-annealed samples only — a stress *overshoot* before settling toward steady
flow. Slower cooling ⇒ higher peak (1.96 vs 1.30 in our units): the material
remembers its preparation. **Panel (b)** shows u climbing during shear for all nine
preparations: deformation "reheats" the structure toward a common band. Note that
none of the curves has flattened by the end of the run — this matters later.
(Panels (c) and (d) are explained below.)

## 3. Why the obvious approach fails

The naive plan: fit an equation dτ/dγ = F(u, τ) to the 42 million recorded steps.
This fails spectacularly (R² ≈ 0.0005), and understanding why shapes everything
downstream. Glassy deformation is **smooth elastic loading interrupted by rare,
violent avalanches** — the strain-rate signal spans five orders of magnitude, and
least squares chases the avalanches while ignoring the physics in between.

The fix is to stop regressing and start *measuring*, after splitting the dynamics
into parts that each behave well:

- the **elastic stiffness** 2μ(u, τ) — the slope of the smooth segments (two
  independent ways of measuring it agree to 2.3%);
- the **plastic fraction**

  **q(u, τ) = 1 − (dτ/dγ) / 2μ**

  which asks: *of the strain we just applied, what fraction went into permanent
  rearrangement rather than elastic loading?* It runs from 0 (purely elastic) to 1
  (steady plastic flow), and can exceed 1 where the material is shedding stress
  faster than it's loaded (softening after the overshoot).

One more habit adopted throughout: **measure the noise floor before fitting
anything.** Splitting identically-prepared samples in half and comparing gives an
11.3% floor; per-cell statistical error is 8.7%. Any model that "fits" better than
the floor is memorizing noise — this rule ends up doing a lot of work below.

## 4. What q looks like — and a lucky break

Panel (d) of the figure above maps q over the (u, τ) plane. It behaves like a
constitutive field should: near zero in the cold/unloaded corner, rising toward 1
at flow, monotone in both variables. And it has one big structural simplicity —
it **factorizes**:

**q(u, τ) ≈ Λ(u) · f(τ)**

like a surface built from a row profile times a column profile. A rank-1 model
captures 96% of the variation in log q. That splits the discovery problem into two
one-dimensional ones:

- **Λ(u)** — how plasticity grows as the structure gets configurationally hotter
  (at fixed stress it rises 26-fold across our range), and
- **f(τ)** — how stress activates rearrangements (rises ~30-fold).

In STZ language, Λ is proportional to the population of shear-transformation
zones, and f is the stress-driven rate at which each one fires. The factorized
form is itself an STZ prediction, so finding it empirically is the first
substantive point on the theory's scoreboard.

## 5. Machine-discovering the equations

With two clean 1-D targets, we used **symbolic regression**: instead of fitting
coefficients in a fixed formula, search over the *space of formulas themselves*
(sums, products, powers, exponentials, logs...) for the best trade-off between
accuracy and simplicity. Because each target has only ~10–20 well-measured points,
we could afford **exhaustive enumeration** — every expression up to a complexity
bound, with constants optimized inside nonlinearities — rather than a stochastic
genetic search. Physics constraints (positivity, monotonicity) are hard filters:
inadmissible formulas are discarded, not penalized. The engine is
[library/symreg.py](library/symreg.py), validated on synthetic problems with known
answers before touching real data.

![Discovered forms and the generalization test](figures/symbolic_regression.png)

**Panels (a, b):** at the accuracy-vs-simplicity sweet spot, both factors come out
**activated** — Boltzmann-like:

Λ(û) ≈ 11.6·exp(−1.79/û) + 0.086,  f(τ) ≈ 43.6·exp(−4.57/τ) + 0.19

(û = u/χ\*, energy rescaled by its steady-flow scale). This is exactly STZ's
central object: zone abundance ∝ exp(−formation energy / effective temperature).
The search was not told to look for it — it emerged from ~4,600 admissible
candidate expressions.

**Panels (c, d)** show the test that matters: train on the seven slowest cooling
rates, predict the two fastest — preparations the model has never seen. The
discovered 5-parameter model does as well as a 15-parameter polynomial (30.7% vs
32.4% error) — and the polynomial's seemingly stellar training error (5.7%, *below
the noise floor*) is pure memorization. Simple, physically-shaped models
extrapolate; flexible ones flatter you on training data and betray you off it.

## 6. The stress test

Because a pro-STZ result from this lab would be suspect, we ran a pre-specified
battery of robustness checks: different statistical weightings, different modulus
estimators, restricted data ranges, shifted energy references, hundreds of
bootstrap resamples.

![Scrutiny results](figures/stz_scrutiny.png)

The outcome is genuinely mixed, and the mix is the finding:

- **Robust:** the *family* of the laws. Growth-exponentials and sinh forms lose in
  every single variant, for both Λ and f. The shape is a floor plus a stiff rise —
  that much is settled.
- **Not robust:** the *distinction between* the activated form exp(−c/u) and a
  plain power law uⁿ. It flips with the extraction weighting (panel a: the same
  data supports either) and with a small shift in the energy zero-point u₀ —
  which is dataset-derived, not independently measured. Verdict: **undecidable at
  current systematics.** The honest scoreboard: activated wins 7 of 9 weighted
  variants; the power law wins 97–99% of bootstraps on the unweighted pipeline.
- **A clean negative:** if you *do* adopt the activated form, its barrier
  parameter should be a material constant. It isn't — it drifts ±25% across the
  nine preparations, far outside statistical error (panel c).

## 7. The most important result is negative

That barrier drift is a symptom of something bigger, shown back in **panel (c) of
the first figure**: take the slowest-cooled and fastest-cooled ensembles and find
moments where they sit at *identical* (u, τ). They flow at rates differing by a
factor of ~1.7 — consistently, across every well-populated cell, at 4.5× the noise
floor.

Two glasses with the same energy and the same stress behave differently. **The
pair (u, τ) is not a complete description of the material's state** — the glass
remembers more of its history than these two numbers capture. Adding accumulated
plastic strain as a third variable absorbs about a third of the discrepancy; the
rest points at candidates we cannot test with this dataset: directional internal
stresses ("back-stress" — which is, notably, the very variable STZ theory includes
and we had to omit), or spatial localization of the flow.

This means no two-variable constitutive law — STZ-form or otherwise — can be exact
for this system. It bounds what any equation on this repository's state space can
achieve (~25–50% depending on preparation spread), and the discovered model
already sits at that bound.

## 8. The missing endpoint: steady state

One more structural caveat. Every preparation is still relaxing when the
simulations end at 50% strain:

![Steady-state extrapolation](figures/steady_state.png)

Fitting the approach to steady state (left: dashed extrapolations, using the
relaxation form the theory itself predicts) says the slowest-cooled samples still
have ~11% of their journey left, and full convergence needs runs 2–3× longer.
Worse (right): the **extrapolated endpoints don't meet** — a 10% spread remains.
Either the relaxation has slower-than-exponential tails, or different preparations
genuinely flow toward different steady states, which would be a second, deeper
violation of effective-temperature theory. Current data cannot tell these apart.

## 9. Where this leaves us

| | |
|---|---|
| **Supported** | Effective-temperature phenomenology; the factorization q = Λ(u)·f(τ); an activated Λ as one of two surviving forms |
| **Not established** | The Boltzmann form *uniquely*; a preparation-independent formation energy; a common steady-state attractor; any stress threshold in f |
| **Contradicted** | Sufficiency of any two-variable state; constancy of the barrier across preparations |

Ranked next steps: (1) measure the reference energy u₀ independently — this alone
breaks the activated-vs-power degeneracy; (2) reverse-loading AQS runs from saved
configurations — a direct, cheap test of the back-stress variable; (3) extend or
bracket the steady state; (4) spatial fields to test localization; (5) meanwhile,
use the 5-parameter factorized model as an honest, history-parameterized interim.

## 10. What's in this repository

| Path | What it is |
|---|---|
| `README.md` | This overview |
| [REPORT.md](REPORT.md) | The full technical report, with the complete robustness battery |
| `SEEM_project_deck.pptx` | 14-slide presentation version |
| [library/stz.py](library/stz.py) | Field extraction: loading, event labeling, modulus, q, hazard, factorization tests |
| [library/symreg.py](library/symreg.py) | The symbolic-regression engine (exhaustive + genetic modes, physics filters) |
| [library/library.py](library/library.py) | Fixed candidate-function library (SINDy baseline arm) |
| `SEEM-model.ipynb` | Original exploratory notebook (voxel SEEM model, first SINDy attempts) |
| `figures/` | All figures referenced above |

**Data:** the underlying trajectory file `df_clean.pkl` (2.4 GB, 42.5M rows) is
gitignored and not distributed here. All voxel-level analysis rebuilds from it in
a single ~6-second pass; the analysis scripts follow a
measure-the-noise-floor-first workflow described in REPORT.md §2 and §5.
