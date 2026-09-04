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

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
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


# ------------------------------------------------------------------ 1 title
s = slide()
tf = textbox(s, 0.9, 1.9, 11.5, 1.6)
run(para(tf, True), 'A fitted law for yield events', size=40, bold=True, color=NAVY)
tf = textbox(s, 0.9, 3.5, 11.5, 0.9)
run(para(tf, True), 'The variables, the symbolic-regression search, the discovered size law and how it performs, '
                    'the forward model built on it, and the fix for where it fails', size=19, color=INK)
tf = textbox(s, 0.9, 4.6, 11.5, 0.45)
run(para(tf, True), 'SEEM project, September 4, 2026', size=15, color=MUTED)
tf = textbox(s, 0.9, 6.5, 11.5, 0.4)
run(para(tf, True), 'AQS shear of a model glass, 850 runs, nine cooling rates, 187,468 stress drops. '
                    'Every number here is regenerated by pipeline/run_all.sh.', size=13, color=MUTED)

# -------------------------------------------------------------- 2 variables
s = slide()
title(s, 'The variables')
terms(s, [
    ('γ', 'applied shear strain; one step is 10⁻⁵, runs go to 0.5'),
    ('τ', 'shear stress (raw stress ÷ 10⁴)'),
    ('u = pe − u₀', 'potential energy per atom above the reference u₀ = −4.60752; high u is a poorly annealed glass'),
    ('(u, τ)', 'the two-coordinate state of a run; every field lives on a 20 × 20 grid over this plane, plus a catch-all ring'),
    ('2μ(u, τ)', 'elastic modulus, the slope dτ/dγ on elastic steps (about 20)'),
    ('s = −Δτ', 'an event: a step where the stress drops, of size s; 187,468 of them, from 10⁻⁶ to 1.16'),
], 0.7, 1.35, 5.9, size=15)
terms(s, [
    ('p(u, τ)', 'event hazard: the probability per step that an event fires at this state (median about 0.3% per step)'),
    ('P(s | u, τ)', 'the size distribution of events at a given state, the object we fit'),
    ('s_c(u, τ)', 'the ceiling: the largest event a state can produce; the only state-dependent parameter of the law'),
    ('k, m, ε', 'the law’s three global constants (small-size exponent, ceiling exponent, rounding scale)'),
    ('q(u, τ)', 'plastic fraction of the strain rate, q = 1 − (dτ/dγ)/2μ; 0 is elastic, 1 is steady flow'),
    ('aging event', 'a step where u drops but τ does not (105,188); it relaxes energy without releasing stress'),
], 6.95, 1.35, 5.8, size=15)
takeaway(s, 'Each AQS step is either an elastic increment or an event. The model needs the drift, the hazard, and the size '
            'distribution of events; the last is the one we fit.')

# -------------------------------------------------------------- 3 SR library
s = slide()
title(s, 'The symbolic-regression library  (library/symreg.py)')
tf = textbox(s, 0.7, 1.35, 5.9, 4.9)
bullets(tf, [('What it does', True),
             'Given points (x, y), it searches for a formula y = a·f(x) + b over a grammar of operators '
             '(exp, log, 1/x, x², √x, and + − × ÷) with up to two fitted constants inside f',
             'For one-variable targets it enumerates every formula up to a complexity budget rather than sampling them, '
             'so "the search did not find X" is a checkable statement',
             'It returns the exact Pareto front: the best misfit at each complexity. The form at the knee is the candidate',
             'Physics is imposed as a filter, not a prior: here, only densities that decrease with s are admitted'])
tf = textbox(s, 6.95, 1.35, 5.8, 4.9)
bullets(tf, [('How it was used here', True),
             'Target: the binned log-density ln ρ(s) of event sizes in one grid cell, 19 log bins with Poisson error bars',
             'Run on 12 cells spanning the state plane (700 to 7,600 events each), plus variants with unweighted bins '
             'and a wider size window',
             'Budget: 624,936 formula trees per cell at complexity 8, about 10,300 distinct admissible functions; '
             'one audit run at complexity 9 (4.5 million trees) found nothing new',
             'The search only proposes forms. Each candidate was then turned into a normalized density and judged by '
             'per-event maximum likelihood (AIC and trajectory-blocked cross-validation)'])
takeaway(s, 'The search is exhaustive and the selection is by likelihood, so the discovered form is neither a lucky '
            'draw nor a binning artifact.')

# -------------------------------------------------------------- 4 the law
s = slide()
title(s, 'The discovered law')
eq(s, 'P(s | u, τ)  =  (1/Z) · (s_c(u, τ) − s)ᵐ / (s + ε)ᵏ ,     0 < s < s_c', 0.7, 1.25, 12.0, size=24)
eq(s, 'k = 0.88,     m = 2.3,     ε = 2.8 × 10⁻⁵', 0.7, 1.95, 12.0, size=18, color=MUTED)
terms(s, [
    ('(s + ε)⁻ᵏ', 'a power-law decay in size with exponent k ≈ 0.9: small events are far more common than large ones. '
                  'The rounding scale ε flattens it below about 3 × 10⁻⁵, so the law describes every recorded event with no lower cutoff'),
    ('(s_c − s)ᵐ', 'a hard ceiling: the density goes to zero at s = s_c and no event larger than s_c is possible. '
                   'With m ≈ 2 it vanishes gently, so the largest observed event sits a few percent below the ceiling'),
    ('s_c(u, τ)', 'the only place the state enters. It ranges 146× over the plane, from 0.02 in the cold, low-stress corner '
                  'to 3 in the flowing state, and it factorizes, ln s_c ≈ (u-part) + (τ-part), at 95.5% of the variance'),
    ('Z', 'the normalization, computed numerically'),
], 0.7, 2.75, 12.0, size=14)
tf = textbox(s, 0.7, 5.55, 12.0, 0.9)
bullets(tf, ['Compared with the usual truncated power law s⁻ᵏ e^(−s/s*): that form suppresses large events with an exponential '
             'tail but never forbids them. The search never placed it on a Pareto front, and the ceiling form beat it in every '
             'likelihood contest that follows'], size=13)
takeaway(s, 'Three global constants and one state-dependent number. Everything about the state, and (later) about the '
            'preparation history, enters through the ceiling s_c.')

# -------------------------------------------------------------- 5 one cell
s = slide()
title(s, 'How the law performs: one cell, three candidate laws')
fig_left(s, 'fig01_size_law.png', [
    'One well-populated cell (6,706 events), five decades of event size. The inset marks the cell on the state plane',
    'Black: the measured density. Red: the ceiling law. Blue dashed: the best truncated power law (TPL). Green dotted: the best lognormal',
    'All three are maximum-likelihood fits of proper densities on the same events',
    'Only the ceiling law gets both ends: the flattening below 3 × 10⁻⁵ and the abrupt stop at the dotted line (s_c). '
    'The TPL tail decays too softly and overshoots',
    'Over the full range the ceiling law beats the TPL by ΔAIC 200 to 345 in this kind of cell'], size=13)
takeaway(s, 'At the level of individual events, the data terminate where the ceiling law says they should and where the TPL says they should not.')

# -------------------------------------------------------------- 6 collapse + verdict
s = slide()
title(s, 'How the law performs: every cell, and the falsifiability control')
two_figs(s, 'fig02_collapse.png', 'fig04_falsifiability.png', y=1.3, h=4.1)
tf = textbox(s, 0.7, 5.5, 12.0, 1.0)
bullets(tf, ['Left: twelve cells whose raw distributions differ by 100× in scale, replotted against ξ = (s+ε)/(s_c+ε) with the '
             'global (k, m, ε). They fall on one curve, including the bend into the ceiling',
             'Right: the same fitting machinery on synthetic catalogs. TPL-generated data return a TPL verdict (5 of 5); ceiling-generated '
             'data return the ceiling (5 of 5). The twelve measured cells sit on the ceiling side, one sparse cell a tie. '
             'In the eight largest cells the ceiling law wins AIC 8 of 8 (ΔAIC 67 to 120 over the TPL, 300 to 754 over the lognormal)'], size=12.5)
takeaway(s, 'One shape everywhere, a discriminator that works in both directions, and the data on one side of it.')

# -------------------------------------------------------------- 7 downstream check
s = slide()
title(s, 'How the law performs: the plastic-rate field, rebuilt from it')
fig_left(s, 'fig09_q_agreement.png', [
    'q(u, τ) was measured directly in the first pass, cell by cell, as 1 − (dτ/dγ)/2μ',
    'From the law it is hazard × mean event size, with the mean taken from the fitted P(s | u, τ)',
    'Each dot is one cell over two decades of q. The band is the statistical noise floor of this grid, 17.5% from split-half ensembles',
    'The rebuilt field lands at median +1.8% and RMS 17%: within the noise. A better law could not score better on this test',
    'Its factor structure is the measured one too: both versions factorize at 96%, with the same SR knee and nearly the same constants'], size=13)
takeaway(s, 'The fitted law reproduces the central object of the first pass to within the data’s own noise.')

# -------------------------------------------------------------- 8 forward model
s = slide()
title(s, 'Running the model with the discovered law')
two_figs(s, 'fig12_sim_stress.png', 'fig14_sim_peaks.png', y=1.3, h=4.1)
tf = textbox(s, 0.7, 5.5, 12.0, 1.0)
bullets(tf, ['All 850 runs simulated forward from their measured initial states at Δγ = 10⁻⁵: elastic drift at 2μ, an event with '
             'probability p(u, τ), and the event size drawn from the law at that state',
             'Left: stress-strain curves for the nine preparations, MD solid and model dashed. The elastic rise, the yield overshoot across '
             '256× in cooling rate, the post-yield decay, and the flow stress (±5%) all come out. Right: the yield peaks, RMS 2.5% '
             '(worst rate 4.3%); the TPL arm gives 1.4% and resampling the raw events 0.9%, so peaks do not discriminate size laws'], size=12.5)
takeaway(s, 'Fields measured at the single-step level predict the yield strength of a glass from its preparation to a few percent.')

# -------------------------------------------------------------- 9 the issue
s = slide()
title(s, 'The energy-convergence problem', color=RED)
fig_left(s, 'fig13_sim_energy.png', [
    'The same comparison for the energy coordinate u(γ)',
    'Early on the model is right. Late, the dashed curves pinch together: the preparations keep a 24% spread in mean u at γ = 0.5 '
    'in the data, and 4% in the model',
    'The model’s fields are one map for all histories, so two runs at the same (u, τ) evolve identically no matter how they got there. '
    'The data say they do not: at the same (u, τ), slow-cooled glass has a 1.71× higher ceiling than fast-cooled glass',
    'So (u, τ) is not a sufficient state. The question is which channel of the step process carries the missing information'], size=13)
takeaway(s, 'A five-times-too-small preparation spread in u: the one place the two-variable model fails.')

# -------------------------------------------------------------- 10 solution
s = slide()
title(s, 'The fix: a one-parameter preparation memory in the ceiling')
eq(s, 's_c(u, τ; rate)  =  s_c(u, τ) · (rate / rate_mid)^β ,        β = −0.117 ± 0.005', 0.7, 1.2, 12.0, size=22)
img_fit(s, 'fig17_memory_repair.png', 0.55, 1.95, 7.6, 2.95)
tf = textbox(s, 8.35, 1.95, 4.45, 3.0)
bullets(tf, ['Only the ceiling is scaled per preparation; hazard, drift, aging and the energy coupling stay as measured',
             'β is the memory exponent: the slope of ln(mean event size) on ln(cooling rate) at fixed cell, with cell fixed effects. '
             'rate_mid is the middle of the nine rates, so the factors run from 1.38 (slowest) to 0.72 (fastest)',
             'Why the ceiling: measured the same way, the hazard has β = −0.012 ± 0.002 and the elastic drift does not move; '
             'the memory is in the event size, and in its tail'], size=11.5)
tf = textbox(s, 0.7, 5.05, 12.0, 1.4)
bullets(tf, ['Result (right panel): the spread of mean u at γ = 0.5 goes from 4.1% to 26.5% against 24% in the data, and the model tracks '
             'the data over the whole strain window; yield peaks are unchanged (RMS 2.5% → 2.8%)',
             'The same relation applied within a preparation, a per-run factor exp(κ·δu₀) with κ = β · d ln(rate)/du₀ = −43 and no new '
             'parameter, predicts how long a run remembers its initial energy (0.44 / 0.27 / 0.22 / 0.17 at γ = 0.1 / 0.2 / 0.3 / 0.5, '
             'against 0.63 / 0.27 / 0.24 / 0.18 in the data). The memory also relaxes with strain, with γ_r ≈ 0.2'], size=12)
takeaway(s, 'To the resolution of this dataset, the missing state variable is a single multiplicative factor on the ceiling, '
            'set by the preparation and erased by plastic strain.')

out = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'SEEM_project_deck.pptx')
prs.save(out)
print('saved', out, len(prs.slides), 'slides')
