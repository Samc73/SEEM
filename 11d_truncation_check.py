"""Step 11d: does Gumbel's misfit in cell (18, 11) depend on the size cutoffs?

Input : out/joint_events.npz, out/marginals.json, out/dependence_law.json (via step 11b's module)
Output: out/truncation_check.json

Step 11b's benchmark (kernel copula vs every named family, ranks within voxel x
rate, even/odd trajectory split, sampling-only trajectory bootstrap) rerun in
cell (18, 11) with the paired-event cutoffs varied: stress 1e-4, 1e-3 (step 6's
choice), 1e-2; PE 1e-6 (step 6's choice), 1e-5.  Reported: kernel - Gumbel per
1,000 events and its split over copulas.square_regions (q = 0.2).  If the
low-rank-edge misfit moves with the cutoff, it is partly a truncation effect
(paired drops cut at a size threshold), not only a family failure.
"""
import importlib
import json
import time
import numpy as np

b = importlib.import_module('11b_copula_benchmark_rate')
OUT = 'out/truncation_check.json'
STRESS_CUTS = (1e-4, 1e-3, 1e-2)
PE_CUTS = (1e-6, 1e-5)

t0 = time.time()
c = b.picked['cell_18_11']['cell']
both_drop = (b.du < 0) & (b.dtau < 0)
rows = []
for pe_cut in PE_CUTS:
    for stress_cut in STRESS_CUTS:
        mask = both_drop & (-b.du >= pe_cut) & (-b.dtau >= stress_cut)
        r = b.benchmark(c, 'voxel_rate', mask)
        r.pop('_events')
        kg = r['kernel_minus_gumbel']
        reg = r['by_region_q20']['gumbel']
        rows.append(dict(pe_cut=pe_cut, stress_cut=stress_cut, n=r['n'], best=r['best_parametric'],
                         kernel_minus_gumbel=kg, kernel_minus_best=r['kernel_minus_best'], regions=reg,
                         bandwidths=[f['bandwidth'] for f in r['folds']],
                         flags=sorted({fl for f in r['folds'] for fl in f['flags'] if 'gumbel' in fl or 'bandwidth' in fl})))
        print('PE >= %.0e, stress >= %.0e: n %4d  kernel-gumbel %+6.1f [%+6.1f, %+6.1f] per 1,000  best %-16s  h %s  (%.0f s)'
              % (pe_cut, stress_cut, r['n'], kg['per_1000_events'], *kg['ci_per_1000'], r['best_parametric'],
                 '/'.join('%.3f' % h for h in rows[-1]['bandwidths']), time.time() - t0))
        print('      ' + '   '.join('%s %+.1f [%+.1f, %+.1f]' % (k, g['per_1000_events'], *g['ci_per_1000'])
                                    for k, g in reg.items()))

# the default cutoffs must reproduce step 11b
default = [r for r in rows if r['pe_cut'] == 1e-6 and r['stress_cut'] == 1e-3][0]
with open(b.OUT) as handle:
    ref = json.load(handle)['results']['cell_18_11']['kernel_minus_gumbel']['difference']
assert abs(default['kernel_minus_gumbel']['difference'] - ref) < 1e-6, 'default cutoffs do not reproduce step 11b'

with open(OUT, 'w') as handle:
    json.dump(dict(step='11d', cell=int(c), q_region=b.Q_REGION, n_boot=b.N_BOOT, rows=rows,
                   note='per 1,000 events of each sample; sampling-only intervals; samples differ between rows, '
                        'so rows are not paired comparisons'), handle, indent=1)
print('wrote %s  (%.0f s)' % (OUT, time.time() - t0))
