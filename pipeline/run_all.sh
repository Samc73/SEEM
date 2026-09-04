#!/bin/bash
# Reproduce every number and figure in README.md from df_clean.pkl.
# Order matters; each step reads the outputs of the earlier ones from pipeline/out/.
# Wall time on an 8-core laptop: ~45 min (verified clean-slate: 43 min, Figures 1-22 byte-identical) (SR fleet ~5 min, complexity-9 audit ~11 min,
# three forward-simulation arms ~2 min each).
set -e
cd "$(dirname "$0")"
mkdir -p out
run() { echo "== $*  ($(date +%H:%M:%S))"; python "$@" > "out/${1%.py}${2:+_$2}.log" 2>&1; }

run extract2.py          # event catalog + per-voxel sufficient statistics (22x22 grid)
run event_tids.py        # trajectory id of every event (blocked CV, memory test)
run discover_dist.py     # SR fleet: 12 cells + weighting/window variants -> sr_dist_results.json
run mle_compare.py       # likelihood adjudication of the SR candidates -> mle_compare.json
run ldw_compare.py       # the Budrikis et al. (2017) mean-field-corrected form on the same cells -> fig23
run scale_field.py       # global shape + per-voxel ceiling (tied exponents) -> scale_fields.npz
run refit_m.py           # decouple m from k; ceiling field s_c4 -> scale4.npz
run fields.py            # drift, hazard, jump coupling, aging fields -> pdmp_fields.npz
run reconstruct.py       # q from the law vs measured; Lambda/f factorization -> qfields.npz
run recon4.py            # same with the decoupled law; SR fronts of the factors -> recon4.json
run sim_run.py model4    # forward simulation, ceiling law
run sim_run.py tpl       # forward simulation, TPL rival
run sim_run.py empir     # forward simulation, empirical resampling
run synth_check.py       # falsifiability control (TPL-generated and ceiling-generated catalogs)
run sr_deep9.py          # audit: largest cell to complexity 9 (4.5M trees)
run make_figs_single.py  # figures/fig01 ... fig15

# ---- second pass: where the preparation memory acts (README Section 9) ----
run memory_stats.py      # per-(trajectory, voxel) sums for every channel of the step process
run memory_where.py      # memory exponent beta per channel at fixed (u,tau); control -> memory_where.json
run coupling_memory.py   # energy released per event at fixed size: residual memory
run coupling_memory.py early
run coupling_memory.py late
run memory_2d.py         # memory exponent by strain window x u band (relaxation vs state dependence)
run memory_late.py       # channel table, early vs late strain
run memory_fading.py     # memory vs strain window and vs u / tau band
run ceiling_pinning.py   # sensitivity of s_c to the largest events, MD cells
run ceiling_pinning_synth.py  # ... the same on synthetic ceiling-law and TPL catalogs
run sim_memory.py        # forward simulation with the memory put back into the ceiling
run scatter_test.py      # within-preparation run-to-run scatter, data vs arms
run scatter2.py          # ... decomposed into event counts vs total release per window
run cluster_test.py      # hazard vs steps since last event (aftershocks)
run merged_events.py     # runs of consecutive drops merged into avalanches; likelihood contest repeated
run sim_aftershock.py    # two-state (aftershock) hazard in the forward simulation
run sequence_test.py     # size -> reloading-gap correlation at fixed cell (renewal structure)
GRID=40 python extract2.py > out/extract2_grid40.log 2>&1   # 42x42 grid for the resolution test
GRID=40 python fields.py > out/fields_grid40.log 2>&1
run sim_fine.py          # empirical arm on 22x22 vs 42x42: the variance gap is not resolution
run renewal_hazard.py    # hazard vs stress reloaded since the last event (renewal tables)
run sim_renewal.py       # forward simulation with the renewal hazard (all events): no effect
run sequence_compare.py  # size->gap correlation, data vs the simulation's own event log
run large_events.py      # Fano factor of large-event counts, data vs simulation
run renewal_big.py       # hazard vs stress reloaded since the last LARGE event
run sim_twoclass.py      # two-class hazard: small events Markov, large events reload-dependent
run size_reload.py       # large-event size vs reloaded stress; consecutive sizes
run persistence_test.py  # persistence of deviations from the preparation mean, data vs model
run sim_persample.py     # per-sample ceiling factor from the initial energy, no new parameter (Fig. 22)
run sim_unoise.py        # u-channel noise terms switched off: tau scatter unchanged
run make_figs_memory.py  # figures/fig16 ... fig22
echo "== done ($(date +%H:%M:%S))"
