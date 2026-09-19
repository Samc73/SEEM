"""The two figures.

Input : out/events.npz, out/shape.json, out/ceiling_table.npy, out/ceiling_law.json
Output: figures/size_law.png     measured size distributions against the two
                                 candidate laws, in the four richest cells
        figures/ceiling_law.png  the measured per-cell ceilings against the
                                 closed law, and the held-out likelihood gains
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import size_law

N_HISTOGRAM_BINS = 40
N_CURVE_POINTS = 400

events = np.load('out/events.npz')
size = events['s']
u = events['u']
tau = events['tau']
cell = events['cell']
n_bins = int(events['n_bins'])
ceiling_table = np.load('out/ceiling_table.npy')
with open('out/shape.json') as handle:
    shape = json.load(handle)
with open('out/ceiling_law.json') as handle:
    ceiling_law = json.load(handle)
K = shape['k']
M = shape['m']
EPS = shape['eps']

# --- figure 1: the size distribution in four cells --------------------------
figure, axes = plt.subplots(2, 2, figsize=(10, 8))
for axis, record in zip(axes.ravel(), shape['cells'][:4]):
    c = record['cell_u'] * n_bins + record['cell_tau']
    s = size[cell == c]
    ceiling = ceiling_table[c]

    bin_edges = np.geomspace(size_law.XMIN, s.max() * 1.01, N_HISTOGRAM_BINS + 1)
    counts, _ = np.histogram(s, bins=bin_edges)
    density = counts / (len(s) * np.diff(bin_edges))
    centres = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    axis.plot(centres[counts > 0], density[counts > 0], 'o', color='0.3',
              markersize=4, label='measured (n = %d)' % len(s))

    curve = np.geomspace(size_law.XMIN, ceiling * 0.9999, N_CURVE_POINTS)
    ln_f, _ = size_law.ln_unnormalised(curve, K, M, EPS, ceiling)
    ln_z = size_law.ln_normalisation(K, M, EPS, ceiling, size_law.N_INTEGRATION_FREE)
    axis.plot(curve, np.exp(ln_f - ln_z), '-', color='C0', linewidth=2,
              label='ceiling law, $s_c$ = %.2f' % ceiling)

    curve_tpl = np.geomspace(size_law.XMIN, s.max(), N_CURVE_POINTS)
    ln_p_tpl = size_law.ln_density_tpl(curve_tpl, record['kappa'],
                                       np.log(record['s_star']), s.max())
    axis.plot(curve_tpl, np.exp(ln_p_tpl), '--', color='C3', linewidth=2,
              label='truncated power law')

    axis.set_xscale('log')
    axis.set_yscale('log')
    # the ceiling law drops to zero at s_c; showing all of that cliff would
    # squash the decade range where the two laws actually differ
    axis.set_ylim(density[counts > 0].min() * 0.05, density.max() * 5)
    axis.set_xlabel('stress-drop size $s$')
    axis.set_ylabel('$P(s\\,|\\,u,\\tau)$')
    axis.set_title('cell (%d, %d):  $u$ = %.4f, $\\tau$ = %.2f'
                   % (record['cell_u'], record['cell_tau'],
                      u[cell == c].mean(), tau[cell == c].mean()), fontsize=10)
    axis.legend(fontsize=8, loc='lower left')
figure.suptitle('Size law with a ceiling vs the truncated power law '
                '(shape $k$ = %.3f, $m$ = %.3f, $\\epsilon$ = %.2e fixed globally)'
                % (K, M, EPS))
figure.tight_layout()
figure.savefig('figures/size_law.png', dpi=140)
print('wrote figures/size_law.png')

# --- figure 2: the ceiling law, and the scoreboard --------------------------
figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

# each cell is placed at the mean u tau of the events it contains, which is
# more honest than the bin centre for the wide cells of the outer ring
occupied = np.nonzero(np.isfinite(ceiling_table))[0]
product_u_tau = np.array([(u[cell == c] * tau[cell == c]).mean() for c in occupied])
counts_per_cell = np.array([int((cell == c).sum()) for c in occupied])
scatter = left.scatter(100.0 * product_u_tau, ceiling_table[occupied],
                       c=np.log10(counts_per_cell), s=22, cmap='viridis')
figure.colorbar(scatter, ax=left, label='$\\log_{10}$ events in cell')

theta = ceiling_law['models']['product']['theta']
line_x = np.linspace(0, 100.0 * product_u_tau.max(), 100)
left.plot(line_x, np.exp(theta[0] + theta[1] * line_x), 'k-', linewidth=2,
          label='$s_c = %.3f\\,e^{%.3f \\times 100\\,u\\tau}$' % (np.exp(theta[0]), theta[1]))
left.set_yscale('log')
left.set_xlabel('$100\\,u\\,\\tau$')
left.set_ylabel('measured ceiling $s_c$ of the cell')
left.set_title('%d per-cell ceilings and the closed law' % len(occupied))
left.legend(loc='upper left')

names = list(ceiling_law['models'].keys())
reference = ceiling_law['models']['constant']['held_out_ll']
gains = [ceiling_law['models'][n]['held_out_ll'] - reference for n in names]
colours = ['0.6' if n in ('constant', 'per-cell table') else 'C0' for n in names]
bars = right.bar(range(len(names)), gains, color=colours)
for bar, name, gain in zip(bars, names, gains):
    right.text(bar.get_x() + bar.get_width() / 2, gain + 120,
               '%+.0f\n(%d par)' % (gain, ceiling_law['models'][name]['n_parameters']),
               ha='center', fontsize=9)
right.set_xticks(range(len(names)))
right.set_xticklabels([n.replace(' + ', '\n+ ') for n in names], fontsize=9)
right.set_ylabel('held-out $\\ln L$ gain over a constant ceiling')
right.set_title('grey = reference, blue = closed law')
right.set_ylim(0, max(gains) * 1.22)
figure.tight_layout()
figure.savefig('figures/ceiling_law.png', dpi=140)
print('wrote figures/ceiling_law.png')
