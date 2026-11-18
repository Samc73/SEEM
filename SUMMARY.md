# PE drops and stress drops: how they move together (summary)

November 2026. Detail and caveats: `WRITEUP.md`; numbers: `out/dependence_law.json`.

## Question

SEEM draws each event's PE and stress drops from empirical lists per voxel of the state (u, τ). Can a compact law replace them: a size law per drop plus a copula linking the two, with the link's strength a smooth function of state?

## Answer

**The drops are strongly linked, mostly when both are large.** In the busiest voxel, Kendall's τ_K is 0.73. Across the 119 well-filled voxels τ_K runs from 0.52 to 0.88, rising with both energy u and stress τ (`voxel_dependence.png`).

**A five-parameter formula captures nearly all of this variation.** We use a Gumbel copula with θ = 1 + exp(η), where

η = 0.902 + 0.210 u′ + 0.253 τ′ + 0.067 u′τ′ − 0.051 ln(rate/rate_mid)

and u′, τ′ are the standardised energy and stress. All slopes are well determined; for example, the u′ interval is [0.193, 0.228].

On held-out trajectories this formula recovers **92–99% of the gain** that a separate θ for every voxel (plus a rate term, 120 parameters) achieves over a single θ (`dependence_scoreboard.png`).

**Where it falls short:**
- With half the data it ties the 120-parameter table. With 80% the table leads by about 1.2 log-likelihood units per 1,000 events, so the table holds some structure the formula misses.
- That structure sits at the edges of the visited region. The formula is too strong at low energy and high stress, and along the lowest-stress row (`dependence_tau_residuals.png`).

## Cooling rate

At the same state, faster-cooled glasses have a weaker link. τ_K falls by 0.016 per unit ln(cooling rate) [0.014, 0.017], about 0.09 across the ×256 range of rates.

Faster-cooled glasses sit at higher energy, so ignoring rate hides part of the energy effect (u′ coefficient 0.16 without rate, 0.21 with).

The drop sizes themselves also shrink with faster cooling inside a voxel. So links are measured within each voxel and cooling rate; pooling rates fakes extra dependence.

## Limits of the Gumbel family

- **Tail overshoot everywhere.** Gumbel overstates how often both drops are in their largest 5% by about 0.065, after correcting for how this is measured. The formula and the per-voxel table show it equally, so it comes from the family's shape, not the θ law (`dependence_tail_residuals.png`).
- **Frank at high τ_K (a hint).** In the three most strongly linked voxels, Frank (no tail link) beats Gumbel. Three voxels of 400–600 events.

## Proposed next steps

1. **A state-dependent copula family.** Compare families voxel by voxel, then test a family (e.g. a Gumbel–Frank mixture) whose upper tail weakens as the link strengthens, keeping θ(u, τ, rate).
2. **Rate-dependent size laws.** Fit the drop-size distributions per voxel and cooling rate (or with a rate term). Fix the stress size law, which aging events currently distort.
3. **Then sample inside SEEM.** Replace the empirical lists with draws from the joint law; aging and PE-only events need their own treatment.

## Key figures

`voxel_dependence.png`, `dependence_scoreboard.png`, `dependence_tail_residuals.png`, `dependence_tau_residuals.png` (single panels in `figures/panels/`).
