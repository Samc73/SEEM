"""Shared figure style for steps 8 and 10 (slide-ready: >= 10 pt text at the saved size).

Palette: the dataviz reference instance (light mode).  Categorical slots are
used in fixed order and every series also gets its own marker or hatch, so no
identity is carried by colour alone.  Diverging maps are blue <-> grey <-> red,
centred on zero with symmetric limits shared by compared panels.

save(fig, name, panels) writes figures/<name>.png and, for multi-panel figures,
figures/panels/<name>_<letter>.png for each panel (cut from the same render, so
the numbers are identical).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.transforms import Bbox

DPI = 170
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
FEW_EVENTS = '#d9d8d4'          # voxels with some events, but too few to show
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']      # categorical slots 1-4, fixed order
BLUE_RAMP = ListedColormap(['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
                            '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'])
BLUE_RAMP.set_bad(SURFACE)
DIVERGING = LinearSegmentedColormap.from_list(
    'blue_red', ['#184f95', '#3987e5', '#9ec5f4', '#f0efec', '#f2a7a6', '#e34948', '#a92b2b'])
DIVERGING.set_bad(SURFACE)
U_LABEL = 'potential energy u = PE − u₀'
T_LABEL = 'stress τ = σ / 10⁴'
KTAU = r'Kendall’s $\tau_K$'      # the dependence measure; τ alone is the stress

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': INK_2, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.grid': False, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.titlesize': 13,
    'legend.frameon': False, 'lines.linewidth': 2})


def letter(ax, text, x=-0.02, y=1.02):
    """Panel letter, bold, above the top-left corner of the axes."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=14, fontweight='bold', ha='right', va='bottom',
            color=INK)


def display_edges(edges):
    """The outer ring of the voxel grid spans the extremes; draw it one inner bin wide."""
    e = edges.copy()
    width = e[2] - e[1]
    e[0], e[-1] = e[1] - width, e[-2] + width
    return e


def save(fig, name, panels=None, pad=0.08):
    """figures/<name>.png, plus one PNG per panel: panels = {letter: [axes, ...]} (an axes'
    colourbar can be listed with it).  Panels are cut from the full render."""
    os.makedirs('figures/panels', exist_ok=True)
    fig.savefig('figures/%s.png' % name, dpi=DPI, bbox_inches='tight')
    print('wrote figures/%s.png' % name)
    if panels:
        # the figure title and figure-level legends belong to the whole figure; hide them so a
        # panel's crop does not catch slivers of them (the caption carries their content)
        hidden = [a for a in [fig._suptitle] + list(fig.legends) if a is not None and a.get_visible()]
        for a in hidden:
            a.set_visible(False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for key, axes in panels.items():
            box = Bbox.union([a.get_tightbbox(renderer) for a in axes])
            box = box.transformed(fig.dpi_scale_trans.inverted()).expanded(1, 1)
            box = Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
            fig.savefig('figures/panels/%s_%s.png' % (name, key), dpi=DPI, bbox_inches=box)
        for a in hidden:
            a.set_visible(True)
        print('   panels: %s' % ', '.join('figures/panels/%s_%s.png' % (name, k) for k in panels))
    plt.close(fig)
