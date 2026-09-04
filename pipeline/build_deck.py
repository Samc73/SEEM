import os as _os
"""SEEM project deck v3: the distribution-first pass and Section 9, in the plain research format of v2."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

NAVY = RGBColor(0x1E, 0x27, 0x61)
INK = RGBColor(0x21, 0x21, 0x21)
MUTED = RGBColor(0x6E, 0x7B, 0x85)
RED = RGBColor(0x99, 0x00, 0x11)
BOX = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
P = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'figures') + '/'

BASE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'deck_base.pptx')   # slides 1-3, hand-edited
prs = Presentation(BASE)
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def textbox(s, x, y, w, h, wrap=True, anchor=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    for m in ('margin_left', 'margin_right', 'margin_top', 'margin_bottom'):
        setattr(tf, m, 0)
    if anchor:
        tf.vertical_anchor = anchor
    return tf


def run(p, text, size=15, bold=False, color=INK, font='Calibri'):
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    return r


def para(tf, first=False):
    return tf.paragraphs[0] if first and not tf.paragraphs[0].runs else tf.add_paragraph()


def set_bullet(p):
    pPr = p._p.get_or_add_pPr()
    pPr.set('marL', '228600')
    pPr.set('indent', '-228600')
    bu = pPr.makeelement(qn('a:buChar'), {'char': '•'})
    pPr.append(bu)


def bullets(tf, items, size=15):
    for i, it in enumerate(items):
        if isinstance(it, tuple):
            txt, hdr = it
        else:
            txt, hdr = it, False
        p = para(tf, first=(i == 0))
        if hdr:
            run(p, txt, size=size + 1, bold=True, color=NAVY)
            p.space_after = Pt(6)
        else:
            run(p, txt, size=size, color=INK)
            set_bullet(p)
            p.space_after = Pt(8)


def title(s, t, color=NAVY):
    tf = textbox(s, 0.6, 0.32, 12.13, 0.75)
    run(para(tf, True), t, size=30, bold=True, color=color)


def takeaway(s, t):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(0.6), Inches(6.62), Inches(12.13), Inches(0.62))
    sh.adjustments[0] = 0.18
    sh.fill.solid()
    sh.fill.fore_color.rgb = BOX
    sh.line.fill.background()
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.18)
    tf.margin_right = Inches(0.18)
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    run(p, 'Takeaway: ', size=13.5, bold=True, color=NAVY)
    run(p, t, size=13.5, color=INK)


def img(s, path, x, y, w, h):
    s.shapes.add_picture(P + path, Inches(x), Inches(y), Inches(w), Inches(h))


def caption(s, t, y=5.75, color=INK, size=13.5, align=None):
    tf = textbox(s, 0.7, y, 12.0, 0.7)
    p = para(tf, True)
    run(p, t, size=size, color=color)
    if align:
        p.alignment = align

def img_fit(s, path, x, y, w, h):
    """place the image inside the (x,y,w,h) box, preserving aspect, centred"""
    from PIL import Image
    iw, ih = Image.open(P + path).size
    sc = min(w / iw, h / ih)
    ww, hh = iw * sc, ih * sc
    s.shapes.add_picture(P + path, Inches(x + (w - ww) / 2), Inches(y + (h - hh) / 2), Inches(ww), Inches(hh))


def fig_left(s, path, items, size=14, w=6.9, h=4.95, y=1.35, cap=None):
    img_fit(s, path, 0.55, y, w, h)
    tf = textbox(s, 0.55 + w + 0.35, y + 0.15, 12.78 - w - 0.35, h - 0.2)
    bullets(tf, items, size=size)
    if cap:
        caption(s, cap, y=6.05, size=12.5, color=MUTED)


def fig_wide(s, path, items, size=13.5, h=3.45, y=1.3):
    img_fit(s, path, 0.55, y, 12.2, h)
    tf = textbox(s, 0.7, y + h + 0.12, 12.0, 6.5 - (y + h + 0.12))
    bullets(tf, items, size=size)


def two_figs(s, a, b, cap=None, y=1.3, h=4.3):
    img_fit(s, a, 0.55, y, 6.0, h)
    img_fit(s, b, 6.78, y, 6.0, h)
    if cap:
        caption(s, cap, y=y + h + 0.12, size=13)



def eq(s, text, x, y, w, size=22, color=NAVY, align=PP_ALIGN.CENTER):
    tf = textbox(s, x, y, w, 0.7)
    p = para(tf, True)
    run(p, text, size=size, bold=False, color=color, font='Cambria Math')
    p.alignment = align


def terms(s, items, x, y, w, size=13):
    """term : meaning rows, term in Cambria Math"""
    tf = textbox(s, x, y, w, 4.5)
    for i, (t, m) in enumerate(items):
        p = para(tf, first=(i == 0))
        run(p, t + '   ', size=size + 1, color=NAVY, font='Cambria Math')
        run(p, m, size=size, color=INK)
        p.space_after = Pt(11)



# ---- helpers in the hand-edited style: terse bullets, bold lead phrases, no takeaway bars ----
import re as _re

def rich(p, text, size=15, color=INK, font='Calibri'):
    """**bold** segments inside a run of Calibri text"""
    for i, seg in enumerate(_re.split(r'\*\*', text)):
        if seg:
            run(p, seg, size=size, bold=(i % 2 == 1), color=color, font=font)


def block(s, x, y, w, h, header, items, size=15):
    """header (Calibri 16 bold navy) + bullets; item = str, or (str, 1) for a sub-bullet"""
    tf = textbox(s, x, y, w, h)
    p = para(tf, True)
    run(p, header, size=16, bold=True, color=NAVY)
    p.space_after = Pt(6)
    for it in items:
        txt, lvl = (it if isinstance(it, tuple) else (it, 0))
        p = tf.add_paragraph()
        rich(p, txt, size=size)
        pPr = p._p.get_or_add_pPr()
        pPr.set('marL', str(228600 + 457200 * lvl)); pPr.set('indent', '-228600')
        pPr.append(pPr.makeelement(qn('a:buChar'), {'char': '•'}))
        p.level = lvl
        p.space_after = Pt(8)
    return tf


def rows(s, items, x, y, w, size=15):
    """term rows: Cambria Math term (navy) + Calibri meaning with **bold** lead"""
    tf = textbox(s, x, y, w, 4.5)
    for i, (t, m) in enumerate(items):
        p = para(tf, first=(i == 0))
        run(p, t + '   ', size=size + 1, color=NAVY, font='Cambria Math')
        rich(p, m, size=size)
        p.space_after = Pt(11)
    return tf


def eq(s, text, x, y, w, size=22, color=NAVY, align=PP_ALIGN.CENTER):
    tf = textbox(s, x, y, w, 0.7)
    p = para(tf, True)
    run(p, text, size=size, color=color, font='Cambria Math')
    p.alignment = align


def img_fit(s, path, x, y, w, h):
    from PIL import Image
    iw, ih = Image.open(P + path).size
    sc = min(w / iw, h / ih)
    ww, hh = iw * sc, ih * sc
    s.shapes.add_picture(P + path, Inches(x + (w - ww) / 2), Inches(y + (h - hh) / 2), Inches(ww), Inches(hh))


# -------------------------------------------------------------- 5 the law
s = slide()
title(s, 'The discovered law')
eq(s, 'P(s | u, τ)  =  (1/Z) · (s_c(u, τ) − s)ᵐ / (s + ε)ᵏ ,     0 < s < s_c', 0.7, 1.2, 12.0, size=24)
eq(s, 'k = 0.88,     m = 2.3,     ε = 2.8 × 10⁻⁵', 0.7, 1.9, 12.0, size=18, color=MUTED)
rows(s, [
    ('(s + ε)⁻ᵏ', '**power-law decay** with exponent k ≈ 0.9; ε rounds it off below about 3 × 10⁻⁵, so there is no lower cutoff'),
    ('(s_c − s)ᵐ', '**hard ceiling**: density is zero at s = s_c. With m ≈ 2 it vanishes gently, so the largest observed event sits a few percent below s_c'),
    ('s_c(u, τ)', '**the only place the state enters**: 146× range over the plane (0.02 cold and unloaded, 3 in flow); ln s_c ≈ (u-part) + (τ-part) at 95.5% of the variance'),
    ('Z', 'normalization, computed numerically'),
], 0.7, 2.75, 12.0, size=15)
block(s, 0.7, 5.15, 12.0, 1.5, 'Versus the literature forms (slide 4)', [
    'Truncated power law s⁻ᵏ e^(−s/s*): an exponential tail suppresses large events but never forbids them. It sits inside the search space at complexity 7, never made a Pareto front, and loses every likelihood contest that follows',
    'Budrikis et al. (2017) eq. 1, fitted the same way: behind the TPL in all eight large cells (ΔAIC +130 to +268 vs the ceiling law; TPL +67 to +120)',
], size=13.5)

# -------------------------------------------------------------- 6b where the margin comes from (after the hand-edited one-cell slide)
s = slide()
title(s, 'Where the likelihood margin comes from')
img_fit(s, 'fig24_ldw_residuals.png', 0.55, 1.25, 12.2, 4.1)
block(s, 0.7, 5.45, 5.9, 1.9, 'Left: residuals per size bin, largest cell', [
    'Measured density divided by each fitted density, with Poisson error bars. The Budrikis form runs 30% low at 3 × 10⁻⁴ and 30% high through '
    '10⁻³ to 10⁻², the bend the exp(C√u) factor puts in; the TPL has the same wave, smaller',
], size=13)
block(s, 6.95, 5.45, 5.8, 1.9, 'Right: log-likelihood gain of the ceiling law, accumulated over events', [
    'Both rivals win below 10⁻³ and lose it back, and more, between 10⁻³ and 10⁻¹, where most events sit. '
    'Totals: **+62 over the TPL (ΔAIC +120), +106 over Budrikis (ΔAIC +208)**',
], size=13)

# -------------------------------------------------------------- 7 all cells + control
s = slide()
title(s, 'Performance: every cell, and a control')
img_fit(s, 'fig02_collapse.png', 0.55, 1.25, 6.0, 4.15)
img_fit(s, 'fig04_falsifiability.png', 6.78, 1.25, 6.0, 4.15)
block(s, 0.7, 5.5, 5.9, 1.9, 'Left: the collapse', [
    'Twelve cells, 100× apart in scale, plotted against **ξ = (s+ε)/(s_c+ε)** with the global (k, m, ε): one curve, including the bend into the ceiling',
], size=13.5)
block(s, 6.95, 5.5, 5.8, 1.9, 'Right: the same machinery on synthetic data', [
    'TPL-generated catalogs return a TPL verdict, 5 of 5; ceiling-generated return the ceiling, 5 of 5',
    'Measured cells sit on the ceiling side; in the 8 largest, **AIC 8 of 8** (ΔAIC 67 to 120 over TPL, 300 to 754 over lognormal)',
], size=13.5)

# -------------------------------------------------------------- 8 q rebuilt
s = slide()
title(s, 'Performance: the plastic-rate field, rebuilt from the law')
img_fit(s, 'fig09_q_agreement.png', 0.55, 1.3, 6.4, 5.0)
block(s, 7.3, 1.4, 5.45, 5.0, 'Measured q against q from the law', [
    'Measured: **q = 1 − (dτ/dγ)/2μ** per cell, first pass',
    'From the law: **hazard × mean event size**, the mean taken from the fitted P(s | u, τ)',
    'One dot per cell, two decades of q; band = statistical noise floor of this grid, **17.5%** (split-half ensembles)',
    'Rebuilt field: **median +1.8%, RMS 17%**, within the noise. A better law could not score better here',
    'Same factor structure: both versions factorize at 96%, same SR knee, nearly the same constants',
], size=14)

# -------------------------------------------------------------- 9 forward model
s = slide()
title(s, 'Running the model with the discovered law')
img_fit(s, 'fig12_sim_stress.png', 0.55, 1.25, 6.0, 4.15)
img_fit(s, 'fig14_sim_peaks.png', 6.78, 1.25, 6.0, 4.15)
block(s, 0.7, 5.5, 5.9, 1.9, 'All 850 runs, forward from their measured initial states', [
    'Each step at Δγ = 10⁻⁵: elastic drift at 2μ, an event with probability p(u, τ), size drawn from the law',
    'MD solid, model dashed: elastic rise, **yield overshoot across 256× in cooling rate**, post-yield decay, flow stress (±5%)',
], size=13.5)
block(s, 6.95, 5.5, 5.8, 1.9, 'The 850 yield peaks', [
    'Ceiling law **RMS 2.5%** (worst rate 4.3%); TPL 1.4%; raw-event resampling 0.9%',
    'Peaks feel only the mean event size, so they do not discriminate size laws',
], size=13.5)

# -------------------------------------------------------------- 10 the issue
s = slide()
title(s, 'The energy-convergence problem', color=RED)
img_fit(s, 'fig13_sim_energy.png', 0.55, 1.3, 6.9, 5.0)
block(s, 7.75, 1.4, 5.0, 5.0, 'The same comparison for u(γ)', [
    'Early: right. Late: the dashed curves **pinch together**',
    'Spread of mean u across preparations at γ = 0.5: **24% in the data, 4% in the model**',
    'The fields are one map for all histories, so two runs at the same (u, τ) evolve identically however they got there',
    'The data disagree: at the same (u, τ), slow-cooled glass has a **1.71× higher ceiling** than fast-cooled',
    'So (u, τ) is not a sufficient state. Which channel carries the missing information?',
], size=14)

# -------------------------------------------------------------- 11 the fix
s = slide()
title(s, 'The fix: a one-parameter preparation memory in the ceiling')
eq(s, 's_c(u, τ; rate)  =  s_c(u, τ) · (rate / rate_mid)^β ,        β = −0.117 ± 0.005', 0.7, 1.15, 12.0, size=22)
img_fit(s, 'fig17_memory_repair.png', 0.55, 1.9, 7.6, 2.95)
block(s, 8.35, 1.9, 4.45, 3.1, 'Reading the equation', [
    'Only the ceiling is scaled per preparation; hazard, drift, aging, energy coupling unchanged',
    '**β**: slope of ln(mean event size) on ln(cooling rate) at fixed cell, with cell fixed effects',
    '**rate_mid**: the middle of the nine rates; factors run 1.38 (slowest) to 0.72 (fastest)',
    'Why the ceiling: the hazard has β = −0.012 ± 0.002 and the drift does not move; the memory is in event size, in the tail',
], size=12.5)
block(s, 0.7, 5.05, 12.0, 1.5, 'Result', [
    'Spread of mean u at γ = 0.5: **4.1% → 26.5%**, against 24% in the data, tracking the data across the strain window; yield peaks unchanged (RMS 2.5% → 2.8%)',
    'Within a preparation the same relation, exp(κ·δu₀) with κ = β · d ln(rate)/du₀ = −43 and no new parameter, predicts how long a run remembers its initial energy: '
    '0.44 / 0.27 / 0.22 / 0.17 at γ = 0.1 / 0.2 / 0.3 / 0.5 (data 0.63 / 0.27 / 0.24 / 0.18). The memory relaxes with strain, γ_r ≈ 0.2',
], size=12.5)

# order: the generated 'discovered law' slide goes right after the budget table (before the hand-edited one-cell slide)
lst = prs.slides._sldIdLst
ids = list(lst)
n_base = 8                                   # hand-edited slides in deck_base.pptx
law = ids[n_base]                            # first generated slide
lst.remove(law); lst.insert(n_base - 1, law)
out = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'SEEM_project_deck.pptx')
prs.save(out)
print('saved', out, len(prs.slides), 'slides')
