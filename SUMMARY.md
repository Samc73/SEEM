# PE drops and stress drops: how they move together (summary)

November 2026. Detail: `WRITEUP.md`. Numbers: `out/dependence_law.json`, `out/copula_benchmark_rate.json`.

## Question and approach

SEEM draws each event's PE and stress drops from empirical lists per voxel of the state (u, τ). Can a compact law replace them, with a size law for each drop plus a copula linking the two? Models are scored on held-out trajectories.

## Result 1: the link's strength follows a five-parameter formula (step 9)

Kendall's τ_K runs from 0.52 to 0.88 across the 119 well-filled voxels. It rises with energy u and stress τ, and falls with faster cooling: −0.016 per unit ln(rate), about −0.09 over the ×256 range. A Gumbel copula with θ = 1 + exp(η) captures this with

η = 0.902 + 0.210 u′ + 0.253 τ′ + 0.067 u′τ′ − 0.051 ln(rate/rate_mid),

where u′ and τ′ are the standardised energy and stress. State dependence gains ~20 log-likelihood units per 1,000 held-out events over a single θ. The formula recovers 92–99% of what one θ per voxel (120 parameters) achieves. It falls short at the edges of the visited region (low u with high stress; the lowest-stress row), where the table leads by 1.2 per 1,000 with 80% training data.

## Result 2: the copula's shape matters more (steps 11, 11b)

A kernel copula, which has no fixed family shape, beats Gumbel by **+74 per 1,000 events [+60, +88]** in the busiest voxel, (18, 11). That is ~60× the formula's shortfall (one voxel, though, against an all-voxel average).

| Voxel (τ_K, events) | Best named family | Kernel − Gumbel | Kernel − best |
|---|---|---|---|
| (13, 9) (0.57, 302) | flipped Clayton | +12 [−36, +60] | +9 [−54, +65] |
| (18, 11) (0.73, 3,906) | Gumbel | +74 [+60, +88] | same |
| (20, 13) (0.83, 591) | Frank | +77 [+25, +130] | −46 [−89, −8] |
| (19, 14) (0.84, 488) | Frank | +112 [+57, +173] | −11 [−77, +48] |
| (17, 15) (0.85, 413) | Frank | +82 [+30, +139] | +0.5 [−40, +43] |

All values are per 1,000 events.

- **At high τ_K, Frank (no tail link) matches the kernel in 3 of 3 voxels.** Caveats: 400–600 events each; best family chosen on the same held-out data; sampling-only intervals.
- **In the busiest voxel, no named family comes close.** Gumbel is already best there, and its misfit is mostly in the bulk of the square (+69 of the 74), not the corners.
- **At low τ_K, no gain.**
- **Pooling ranks across cooling rates had added ~25 units of fake large-drop-corner dependence** (+80 → +74). Ranking within voxel × rate removes it.

Gumbel also overstates, everywhere, how often both drops are in their largest 5% (+0.065), for formula and table alike.

## Next steps, in priority order

1. **Copula shape.** Use a family that changes with state (Gumbel → Frank as τ_K rises), plus a more flexible option for the mid-τ_K bulk misfit.2. **Run steps 1–3 in this folder.** A prerequisite for rate-dependent size laws and any link to the ceiling s_c; data present, scripts match.
3. **Rate-dependent size laws.** Sizes shrink with faster cooling; aging events distort the stress law.
4. **Sample inside SEEM.** This needs 1–3 first; aging and PE-only events need their own treatment.

## Key figures

`voxel_dependence.png`, `dependence_scoreboard.png`, `dependence_tau_residuals.png`, `dependence_tail_residuals.png`, `kernel_benchmark.png` (panel d is slide-ready alone).

## Likely questions

**Why hold out whole trajectories?** Events within a trajectory are strongly correlated; a random split tests on near-copies of training data.

**Why rank within voxel × rate?** Fast-cooled drops are smaller on both measures, so ranking rates together creates fake dependence.

**Why trust a 1-per-1,000 gap?** It appears in 47 of 50 random 80/20 splits and in 5-fold scoring. With 50/50 splits it ties, so it is real but small.

**What is the kernel copula?** A smoothed density of the ranks, bandwidth tuned on training data. On true Gumbel data it scores slightly below Gumbel, so a win means the family shape is wrong.

**What is still uncertain?** Whether Frank-at-high-τ_K holds beyond 3 voxels, and what fixes the mid-τ_K bulk. Most intervals are sampling-only, and voxels share trajectories.
