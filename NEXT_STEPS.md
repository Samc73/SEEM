# Next steps

Goal: replace SEEM's per-voxel lists of drop sizes with a compact law: a size law for each drop plus a copula, a function linking the two sizes.

1. **Size laws that depend on cooling rate.**
   - Do: run steps 1–3 in this folder, add cooling rate to the size laws, and decide how to treat small stress drops, where most of Gumbel's misfit came from.
   - Why: faster cooling gives smaller drops; SEEM's lists mix all rates.
   - Worked if: the rate term improves the fit on trajectories held out from fitting.

2. **Sample inside SEEM with the compact law.**
   - Do: replace SEEM's lists with draws from the size laws and the Gumbel copula.
   - Why: the real test of the compact law.
   - Worked if: SEEM reproduces the simulated drop sizes and how the two drops pair up.

3. **A copula family that changes with state (only if needed).**
   - Do: switch from Gumbel to Frank where the two drops are most strongly linked.
   - Why: Gumbel overstates how often the largest drops occur together (the upper tail).
   - Worked if: SEEM's largest-drop mismatch disappears. Skip if step 2 shows none.

Settled: Gumbel's misfit along the low-rank edges in cell (18, 11) is not a cutoff artifact. It is unchanged when the stress cutoff is lowered (+51 per 1,000 at 1e-4 vs +52 at 1e-3) and mostly disappears when stress drops below 1e-2 are excluded (+12). So small stress drops (below ~1e-2) behave differently; step 1 handles them.
