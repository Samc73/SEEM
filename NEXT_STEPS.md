# Next steps

Goal: replace SEEM's per-voxel lists of drop sizes with a compact law: a size law for each drop (PE and stress) plus a copula, a function linking the two sizes.

1. **Size laws that depend on cooling rate.**
   - Do: run steps 1–3 in this folder, add cooling rate to the size laws, and decide how to treat small stress drops, where most of Gumbel's misfit came from.
   - Why: faster-cooled glasses give smaller drops, and SEEM's lists mix all rates.
   - Worked if: the rate term improves the fit on held-out trajectories (ones not used for fitting).

2. **Sample inside SEEM with the compact law.**
   - Do: replace SEEM's lists with draws from the size laws and the Gumbel copula.
   - Why: this is the real test of whether the compact law is good enough.
   - Worked if: SEEM reproduces the simulated drops: their sizes, and how PE and stress drops pair up.

3. **A copula family that changes with state (only if needed).**
   - Do: switch from Gumbel to Frank where the two drops are most strongly linked.
   - Why: Gumbel overstates how often the largest PE and stress drops occur together (the upper tail).
   - Worked if: SEEM's mismatch in the largest drops goes away. Skip if step 2 shows none.

Open question: whether Gumbel's misfit along the low-rank edges (where one drop is small) is a truncation artifact or a real population of small stress drops (rerun the cell (18, 11) check at a stress cutoff of 1e-4).
