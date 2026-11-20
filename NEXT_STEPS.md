# Next steps: options to decide with the PI (2026-11-20)

Goal: replace SEEM's per-voxel empirical lists of (PE drop, stress drop) with a compact law, made of size marginals in (u, τ, rate) and a copula with θ(u, τ, rate). Numbers are from `SUMMARY.md` and `CLAUDE.md`. Time estimates are rough and include analysis, not only compute.

### A. A copula family that changes with state (Gumbel → Frank at high τ_K)
- **Question:** does one switch rule (for example family by τ_K, or by u and τ) close most of the kernel − Gumbel gap without going back to per-voxel tables?
- **Do:** run step 11b's family comparison (voxel × rate ranks) on all 119 voxels with ≥ 100 paired events, not just 5. Then fit the switch, or a 2-parameter family whose shape follows the state, and score it held out against Gumbel θ(u, τ, rate).
- **Cost:** about 1 day for the 119-voxel comparison (11b takes ~1 min for 5 voxels), plus 3–5 days for the law.
- **Would change plans if:** Frank wins in a coherent high-u/high-stress band → add a family boundary to the law. If wins scatter between voxels, or (18, 11)'s +74 per 1,000 gap remains → no named family is enough; accept the gap or use a 2-parameter family.
- **Goal:** fixes the shape (the upper tail overshoots by +0.065 everywhere) while keeping the law at a few parameters.

### B. Truncation check: already run (step 11d, cell (18, 11))
- **Result:** kernel − Gumbel along the low-rank edges, per 1,000 events: +52 [+43, +62] at stress ≥ 10⁻³. It falls to +12 [+7, +18] at stress ≥ 10⁻², and to +42 at PE ≥ 10⁻⁵. The upper-tail misfit stays at +19 to +24 at every cutoff. So the edge misfit sits in drops near the cutoffs, and the upper tail is a genuine family failure.
- **Open:** whether that is selection by the cutoff or real small-drop physics; the samples differ between cutoffs. No further run is proposed unless SEEM needs drops near the cutoffs.
- **Goal:** tells A to focus on the upper tail, not the edges.

### C. Run steps 1–3 here and make the size laws depend on cooling rate
- **Question:** how much does cooling rate shift the size laws at fixed state, and does the ceiling s_c carry it?
- **Do:** run steps 1–3 in this folder (data present; step 1 needs ~10 GB RAM). Score step 3's existing `+ c·ln(rate)` ceiling form, and add a rate term to step 6's TPL (κ, s*) for PE and stress.
- **Cost:** about 1 day to run and check steps 1–3; 2–3 days for rate-dependent marginals.
- **Would change plans if:** the rate term is large held out (in (18, 11) the median fast/slow ratio is 0.68 for PE and 0.83 for stress, see `copula_marginals.png`) → rate becomes a required SEEM input for sizes, not only for θ. If it is small → keep state-only marginals. It also allows the s_c ↔ θ test (η = a + b ln s_c).
- **Goal:** the marginals are half of the compact law, and SEEM's lists currently mix rates.

### D. Sample the compact law inside SEEM
- **Question:** does the compact law reproduce SEEM's macroscopic output (stress–strain, avalanche statistics) as well as the empirical lists do?
- **Do:** write a sampler (TPL marginals + Gumbel θ(u, τ, rate)), swap it in for the lists, and compare with list-driven runs. Aging (stress-only) and PE-only events need their own treatment.
- **Cost:** 1–2 weeks. Full version needs C. If SEEM's lists can stand in for the marginals, the copula alone can be tested sooner.
- **Would change plans if:** outputs match within run-to-run scatter → the remaining copula misfit doesn't matter, and A is optional. If they don't, the mismatch shows which part (tail, marginal) to fix.
- **Goal:** this is the end-to-end test of the goal itself.

### Recommended order
1. **C:** cheap, unblocks D, and the rate shift in the marginals is already visible.
2. **D (with the current Gumbel law):** measures whether the known copula misfit matters for SEEM before we spend effort fixing it.
3. **A:** only if D shows tail-sensitive mismatch; B already says where to aim (the upper tail).
4. **B:** done; revisit only if SEEM's output is sensitive to drops near the cutoffs.

(This order differs from `SUMMARY.md`, which puts A first.)
