"""State-dependent Gumbel copula: theta = 1 + exp(eta(u, tau[, rate])).

Used by 9_fit_dependence_law.py, 9b_repeated_splits.py and
test_dependence_law.py.  See PLAN_step9.md.

Forms for eta, built from named terms (z_u, z_tau standardised with the
TRAINING events' mean and sd; the product form keeps step 3's raw variable
100 u tau, which depends on U_REFERENCE through u = pe - U_REFERENCE):
    constant      a
    product       a + b 100 u tau
    linear        a + b z_u + c z_tau
    interaction   linear + e z_u z_tau
    quadratic     linear + d z_u^2 + e z_u z_tau + f z_tau^2
    <form>+rate   <form> + g ln(rate / rate_mid)      (when ln_rate is given)
References:
    voxel_table        one theta per voxel (copulas.fit_family on that voxel's ranks)
    voxel_table+rate   eta = a_voxel + g ln(rate / rate_mid), fitted jointly:
                       the like-for-like reference for the +rate forms

Uniforms are ranks WITHIN groups (RANK_RULE): the marginals change with state
(and with cooling rate), so ranking events of different groups together would
mix marginals and add spurious concordance.  rank_levels = [finest, ..., voxel]
gives candidate groupings, e.g. [voxel x rate, voxel x 3 rate groups, voxel];
each event is ranked in the finest group with >= MIN_RANK_TRAIN training
events (rank_levels_of), the same group for training and held-out events, and
every fallback to a coarser level is counted.  Default: [voxel].

Every fit runs after the forms it contains, starting (among other starts)
from their fits padded with zeros, so a larger form's maximised training
ln L cannot fall below a form it contains unless the optimiser fails;
nesting_violations() checks this on every fit.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize
import copulas

BASE_FORMS = ('constant', 'product', 'linear', 'interaction', 'quadratic')
RATE = '+rate'
RATE_FORMS = tuple(f + RATE for f in BASE_FORMS)
RATE_TERM = 'ln(rate/rate_mid)'
TERMS = dict(constant=['1'], product=['1', '100 u tau'], linear=['1', 'z_u', 'z_tau'],
             interaction=['1', 'z_u', 'z_tau', 'z_u z_tau'],
             quadratic=['1', 'z_u', 'z_tau', 'z_u^2', 'z_u z_tau', 'z_tau^2'])
TERMS.update({f + RATE: TERMS[f] + [RATE_TERM] for f in BASE_FORMS})
FORMS = BASE_FORMS                  # rate-free forms (always fitted)
N_COEFFICIENTS = {f: len(t) for f, t in TERMS.items()}
COEFFICIENT_NAMES = TERMS
TABLE, TABLE_RATE = 'voxel_table', 'voxel_table' + RATE
MODELS = FORMS + (TABLE,)           # without rate
MODELS_WITH_RATE = FORMS + RATE_FORMS + (TABLE, TABLE_RATE)

ETA_RANGE = (-15.0, np.log(90.0))   # theta in (1 + 3e-7, 91); a fit pushed to the edge is flagged
H_ETA = 1e-5                        # step for the per-event derivative in eta
RUNAWAY = 20.0                      # |coefficient| above this is flagged
THETA_NEAR_ONE = 1.05
MIN_TRAIN_PER_VOXEL = 5             # fewer training events: the voxel gets the pooled theta
MAX_RESTARTS = 5                    # L-BFGS-B is restarted from where it stopped if it ends abnormally
MIN_RANK_TRAIN = 10                 # a rank group needs this many training events, else a coarser level.
                                    # Fallbacks mix rates again: at 20 (~6% of events) synthetic case E kept a
                                    # spurious rate effect; at 10 (~2%) it is gone
GRAD_TOL_PER_EVENT = 1e-6           # converged if max |d lnL / d beta| < this x n, whatever L-BFGS-B says:
                                    # its gtol is below the finite-difference gradient's noise at large n

# (smaller, larger): the larger contains the smaller (its extra coefficients = 0)
NESTING = ((('constant', 'product'), ('constant', 'linear'), ('linear', 'interaction'),
            ('interaction', 'quadratic'))
           + tuple((a + RATE, b + RATE) for a, b in (('constant', 'product'), ('constant', 'linear'),
                                                     ('linear', 'interaction'), ('interaction', 'quadratic')))
           + tuple((f, f + RATE) for f in BASE_FORMS)
           + ((TABLE, TABLE_RATE),))
NESTING_TOLERANCE = 1e-6            # relative; a larger form's maximised ln L may not fall below this


# --- design --------------------------------------------------------------------
def scaling(u_state, tau_state):
    return dict(mu_u=float(u_state.mean()), sd_u=float(u_state.std()),
                mu_t=float(tau_state.mean()), sd_t=float(tau_state.std()))


def design(form, u_state, tau_state, scale, ln_rate=None):
    zu = (u_state - scale['mu_u']) / scale['sd_u']
    zt = (tau_state - scale['mu_t']) / scale['sd_t']
    columns = {'1': lambda: np.ones(len(u_state)), '100 u tau': lambda: 100.0 * u_state * tau_state,
               'z_u': lambda: zu, 'z_tau': lambda: zt, 'z_u^2': lambda: zu * zu,
               'z_u z_tau': lambda: zu * zt, 'z_tau^2': lambda: zt * zt}
    if form not in TERMS:
        raise ValueError(form)
    if RATE_TERM in TERMS[form]:
        if ln_rate is None:
            raise ValueError('%s needs ln_rate' % form)
        columns[RATE_TERM] = lambda: np.asarray(ln_rate, float)
    return np.column_stack([columns[t]() for t in TERMS[form]])


def theta_of(beta, x):
    return 1.0 + np.exp(np.clip(x @ beta, *ETA_RANGE))


# --- uniforms --------------------------------------------------------------------
RANK_RULE = ('training events: u = r / (n + 1), r = rank among the n training events of the same '
             'rank group (ties averaged; copulas.pseudo_observations), so the grid is symmetric: mean 0.5, '
             'min = 1 - max.  Held-out events: u = (k + 0.5) / (n + 1), k = number of training values '
             '<= x in that rank group (copulas.ecdf_transform), i.e. midway between the neighbouring '
             'training ranks; also symmetric, range [0.5/(n+1), (n+0.5)/(n+1)].  Rank group: the finest of '
             'rank_levels with >= MIN_RANK_TRAIN training events (step 9: voxel x rate, then voxel x 3 rate '
             'groups, then voxel)')


def within_voxel_ranks(values, cell):
    """Pseudo-observations computed separately inside each voxel (RANK_RULE)."""
    out = np.empty(len(values))
    for c in np.unique(cell):
        m = cell == c
        out[m] = copulas.pseudo_observations(values[m])
    return out


def rank_levels_of(levels, counted, min_n=None):
    """levels = [finest, ..., coarsest] group ids per event (the last should be the
    voxel).  Each event gets the finest level whose group has >= min_n `counted`
    (training) events; the coarsest level is used regardless.  Returns the level
    index per event."""
    k = len(levels)
    min_n = MIN_RANK_TRAIN if min_n is None else min_n
    level = np.full(len(levels[0]), -1)
    for i, g in enumerate(levels):
        ids, inv = np.unique(np.asarray(g), return_inverse=True)
        enough = np.bincount(inv[counted], minlength=len(ids)) >= min_n
        take = (level < 0) & (enough[inv] | (i == k - 1))
        level[take] = i
    return level


def grouped_uniforms(values, levels, train, min_n=None):
    """Uniforms for training and held-out events by RANK_RULE, inside each event's
    rank group (rank_levels_of).  An event at level i is ranked against ALL
    training events of its level-i group (the reference), whatever level those
    events themselves use.  Returns u_train, u_test, level (per event) and the
    number of held-out events with no training reference (ranked among held-out)."""
    level = rank_levels_of(levels, train, min_n)
    u = np.empty(len(values))
    fallback = 0
    for i, g in enumerate(levels):
        g = np.asarray(g)
        for c in np.unique(g[level == i]):
            members = g == c
            ref = members & train
            mine = members & (level == i)
            if ref.any():
                ranks = copulas.pseudo_observations(values[ref])
                pos = np.flatnonzero(ref)
                take_tr = np.isin(pos, np.flatnonzero(mine))
                u[pos[take_tr]] = ranks[take_tr]
                te = mine & ~train
                if te.any():
                    u[te] = copulas.ecdf_transform(values[ref], values[te])
            else:
                te = mine & ~train
                u[te] = copulas.pseudo_observations(values[te])
                fallback += int(te.sum())
    return u[train], u[~train], level, fallback


def held_out_uniforms(values, cell, train):
    """Training events: ranks within their voxel among training events.
    Held-out events: the training ECDF of their own voxel.  A held-out event
    whose voxel has no training events is ranked among the held-out events of
    that voxel (counted and reported as a fallback)."""
    u_train = within_voxel_ranks(values[train], cell[train])
    test = ~train
    u_test = np.empty(test.sum())
    test_values, test_cells = values[test], cell[test]
    train_values, train_cells = values[train], cell[train]
    fallback = 0
    for c in np.unique(test_cells):
        m = test_cells == c
        reference = train_values[train_cells == c]
        if len(reference) == 0:
            u_test[m] = copulas.pseudo_observations(test_values[m])
            fallback += int(m.sum())
        else:
            u_test[m] = copulas.ecdf_transform(reference, test_values[m])
    return u_train, u_test, fallback


# --- likelihood and fitting ----------------------------------------------------------
def log_density(beta, x, u, v):
    return copulas.log_density('gumbel', [theta_of(beta, x)], u, v)


def _neg_ll_and_grad(beta, x, u, v):
    eta = np.clip(x @ beta, *ETA_RANGE)

    def ll(e):
        return copulas.log_density('gumbel', [1.0 + np.exp(e)], u, v)
    value = ll(eta)
    slope = (ll(eta + H_ETA) - ll(eta - H_ETA)) / (2 * H_ETA)    # d ln c_i / d eta_i
    return -float(value.sum()), -np.asarray(x.T @ slope).ravel()


def _minimise(start, x, u, v):
    """L-BFGS-B, restarted from its own end point (fresh curvature memory) while it
    stops abnormally -- a line-search failure, which with the finite-difference
    gradient can happen far from the optimum."""
    r = None
    for _ in range(MAX_RESTARTS + 1):
        r = minimize(_neg_ll_and_grad, np.asarray(start if r is None else r.x, float),
                     args=(x, u, v), jac=True, method='L-BFGS-B',
                     options=dict(maxiter=1000, gtol=1e-6))
        if r.success:
            break
    r.max_grad = float(np.abs(_neg_ll_and_grad(r.x, x, u, v)[1]).max())
    r.converged = bool(r.success or r.max_grad < GRAD_TOL_PER_EVENT * len(u))
    return r


def fit_form(x, u, v, starts):
    """L-BFGS-B from each start; keep the best.  Returns beta, ll and flags."""
    best = None
    for start in starts:
        r = _minimise(start, x, u, v)
        if best is None or r.fun < best.fun:
            best = r
    beta = best.x
    eta = x @ beta
    flags = []
    if np.any(np.abs(beta) > RUNAWAY):
        flags.append('coefficient beyond |%g|' % RUNAWAY)
    if np.any((eta <= ETA_RANGE[0]) | (eta >= ETA_RANGE[1])):
        flags.append('eta clipped for %d events' % int(((eta <= ETA_RANGE[0]) | (eta >= ETA_RANGE[1])).sum()))
    if not best.converged:
        flags.append('optimiser: %s, max |gradient| %.3g' % (best.message, best.max_grad))
    theta = theta_of(beta, x)
    return dict(beta=beta, ll=float(-best.fun), flags=flags,
                frac_theta_near_one=float(np.mean(theta < THETA_NEAR_ONE)),
                theta_range=[float(theta.min()), float(theta.max())])


# --- voxel tables ----------------------------------------------------------------------
def fit_voxel_table(u, v, cell):
    """One Gumbel theta per voxel with enough events, plus the pooled theta."""
    pooled = float(copulas.fit_family('gumbel', u, v)['params'][0])
    table, flags = {}, []
    for c in np.unique(cell):
        m = cell == c
        if m.sum() < MIN_TRAIN_PER_VOXEL:
            continue
        fit = copulas.fit_family('gumbel', u[m], v[m])
        table[int(c)] = float(fit['params'][0])
        flags += ['voxel %d: %s' % (c, f) for f in fit['at_bound']]
    return table, pooled, flags


def table_theta(table, pooled, cell):
    return np.array([table.get(int(c), pooled) for c in cell])


def table_rate_design(cell, ln_rate, cells):
    """Sparse design for voxel_table+rate: one indicator per voxel in `cells`, one
    shared indicator for every other voxel (pooled intercept), and ln_rate."""
    position = {int(c): i for i, c in enumerate(cells)}
    col = np.array([position.get(int(c), len(cells)) for c in cell])
    n = len(cell)
    rows = np.concatenate([np.arange(n), np.arange(n)])
    cols = np.concatenate([col, np.full(n, len(cells) + 1)])
    vals = np.concatenate([np.ones(n), np.asarray(ln_rate, float)])
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(cells) + 2))


def fit_voxel_table_rate(u, v, cell, ln_rate, table, pooled, g_starts=(0.0,)):
    """eta = a_voxel + g ln(rate/rate_mid), all a_voxel and g fitted jointly.  Voxels
    outside `table` (too few events) share one intercept.  Starts: the table's
    ln(theta - 1) (so g = 0 reproduces the rate-free table) with each g in g_starts."""
    cells = np.array(sorted(table))
    x = table_rate_design(cell, ln_rate, cells)
    a0 = [np.log(max(table[c] - 1, 1e-3)) for c in cells] + [np.log(max(pooled - 1, 1e-3))]
    fit = fit_form(x, u, v, [np.array(a0 + [g]) for g in g_starts])
    fit.update(cells=cells, g=float(fit['beta'][-1]))
    return fit


def table_rate_eta(fit, cell, ln_rate):
    return table_rate_design(cell, ln_rate, fit['cells']) @ fit['beta']


# --- starting points and nesting --------------------------------------------------------
def nested_starts(form, fits):
    """For every already-fitted form that this form contains: its coefficients
    placed on the matching terms, zeros elsewhere.  Such a start cannot end below
    that smaller fit if the optimiser works."""
    out = []
    for small, large in NESTING:
        if large == form and small in fits and small in TERMS:
            beta = np.zeros(N_COEFFICIENTS[form])
            for term, b in zip(TERMS[small], fits[small]):
                beta[TERMS[form].index(term)] = b
            out.append(beta)
    return out


def nesting_violations(lls):
    """lls = {model: maximised TRAINING ln L}.  A larger model below a model it
    contains means an optimiser failure, whether or not anything was flagged."""
    out = []
    for small, large in NESTING:
        if small in lls and large in lls and lls[large] < lls[small] - NESTING_TOLERANCE * abs(lls[small]):
            out.append('NESTING VIOLATION: %s ln L %.3f < %s ln L %.3f (optimiser failure)'
                       % (large, lls[large], small, lls[small]))
    return out


def starting_points(form, table, cell, u_state, tau_state, scale, pooled, rng, n_perturb=3):
    """(i) least squares of ln(theta_voxel - 1) on each voxel's median state,
    weighted by events; (ii) flat at the pooled theta; (iii) perturbations of (i).
    A +rate form uses its base form's points with the rate coefficient at 0.
    Callers add nested_starts()."""
    if form.endswith(RATE):
        return [np.append(s, 0.0) for s in starting_points(form[:-len(RATE)], table, cell, u_state,
                                                           tau_state, scale, pooled, rng, n_perturb)]
    k = N_COEFFICIENTS[form]
    flat = np.zeros(k)
    flat[0] = np.log(pooled - 1)
    cells = np.array(sorted(table))
    rows, targets, weights = [], [], []
    for c in cells:
        m = cell == c
        rows.append(design(form, np.array([np.median(u_state[m])]),
                           np.array([np.median(tau_state[m])]), scale)[0])
        targets.append(np.log(max(table[c] - 1, 1e-3)))
        weights.append(np.sqrt(m.sum()))
    rows, targets, weights = np.array(rows), np.array(targets), np.array(weights)
    lsq = np.linalg.lstsq(rows * weights[:, None], targets * weights, rcond=None)[0]
    return [lsq, flat] + [lsq + rng.normal(0, 0.3, k) for _ in range(n_perturb)]


def _fit_all(u, v, u_state, tau_state, cell, ln_rate, rng, n_perturb, warm):
    """Every model on one set of (training) events.  Returns the voxel table, the
    pooled theta, the scale and {model: fit}; fits run after the forms they contain."""
    scale = scaling(u_state, tau_state)
    table, pooled, table_flags = fit_voxel_table(u, v, cell)
    theta_t = table_theta(table, pooled, cell)
    fits = {TABLE: dict(ll=float(copulas.log_density('gumbel', [theta_t], u, v).sum()),
                        flags=table_flags)}
    forms = FORMS + (RATE_FORMS if ln_rate is not None else ())
    for form in forms:
        x = design(form, u_state, tau_state, scale, ln_rate)
        starts = ([warm[form]] if warm is not None and form in warm else
                  starting_points(form, table, cell, u_state, tau_state, scale, pooled, rng, n_perturb))
        starts = starts + nested_starts(form, {f: fits[f]['beta'] for f in fits if f in TERMS})
        fits[form] = fit_form(x, u, v, starts)
    if ln_rate is not None:
        # g starts: 0 (the rate-free table) and the constant+rate fit's g
        g_starts = (0.0, float(fits['constant' + RATE]['beta'][-1]))
        if warm is not None and TABLE_RATE in warm:
            g_starts = (float(warm[TABLE_RATE]),)
        fits[TABLE_RATE] = fit_voxel_table_rate(u, v, cell, ln_rate, table, pooled, g_starts)
    violations = nesting_violations({m: f['ll'] for m, f in fits.items()})
    return table, pooled, scale, fits, violations


# --- cross-validation ------------------------------------------------------------------
def cross_validate(x_size, y_size, u_state, tau_state, cell, fold_id, rng, warm=None,
                   n_perturb=3, held_out_folds=None, ln_rate=None, rank_levels=None):
    """Held-out ln c per event for every model.  fold_id[i] = fold of event i;
    each fold is held out once, or only the folds in held_out_folds (events
    of other folds are then left at 0).  With ln_rate, the +rate forms and
    voxel_table+rate are fitted too.  warm = {model: beta (g for the rate
    table)} gives a single warm start (refitting bootstrap) instead of the full set.
    rank_levels: see the module docstring (default [cell])."""
    n = len(x_size)
    models = MODELS if ln_rate is None else MODELS_WITH_RATE
    held = {m: np.zeros(n) for m in models}
    folds = []
    for k in (np.unique(fold_id) if held_out_folds is None else held_out_folds):
        train = fold_id != k
        test = ~train
        levels = rank_levels if rank_levels is not None else [cell]
        u_tr, u_te, level, fb_u = grouped_uniforms(x_size, levels, train)
        v_tr, v_te, _, fb_v = grouped_uniforms(y_size, levels, train)
        lr_tr = None if ln_rate is None else ln_rate[train]
        table, pooled, scale, fits, violations = _fit_all(u_tr, v_tr, u_state[train], tau_state[train],
                                                          cell[train], lr_tr, rng, n_perturb, warm)
        held[TABLE][test] = copulas.log_density('gumbel', [table_theta(table, pooled, cell[test])], u_te, v_te)
        fold = dict(fold=int(k), n_train=int(train.sum()), n_test=int(test.sum()),
                    rank_fallback_events=max(fb_u, fb_v), voxels_in_table=len(table),
                    rank_level_events=dict(train=np.bincount(level[train], minlength=len(rank_levels or [0])).tolist(),
                                           test=np.bincount(level[test], minlength=len(rank_levels or [0])).tolist()),
                    table_flags=fits[TABLE]['flags'], forms={}, nesting_violations=violations)
        lr_te = None if ln_rate is None else ln_rate[test]
        for form in [m for m in models if m in TERMS]:
            x_te = design(form, u_state[test], tau_state[test], scale, lr_te)
            held[form][test] = log_density(fits[form]['beta'], x_te, u_te, v_te)
            fold['forms'][form] = dict(beta=fits[form]['beta'].tolist(), flags=fits[form]['flags'],
                                       train_ll=fits[form]['ll'])
        if ln_rate is not None:
            f = fits[TABLE_RATE]
            eta = np.clip(table_rate_eta(f, cell[test], lr_te), *ETA_RANGE)
            held[TABLE_RATE][test] = copulas.log_density('gumbel', [1.0 + np.exp(eta)], u_te, v_te)
            fold['forms'][TABLE_RATE] = dict(g=f['g'], flags=f['flags'], train_ll=f['ll'])
        fold['forms'][TABLE] = dict(train_ll=fits[TABLE]['ll'])
        folds.append(fold)
    return held, folds


def fit_in_sample(x_size, y_size, u_state, tau_state, cell, rng, n_perturb=3, ln_rate=None,
                  rank_levels=None, warm=None):
    """All models on all events (ranks within the groups of rank_levels)."""
    everything = np.ones(len(x_size), bool)
    levels = rank_levels if rank_levels is not None else [cell]
    u, _, level, _ = grouped_uniforms(x_size, levels, everything)
    v, _, _, _ = grouped_uniforms(y_size, levels, everything)
    table, pooled, scale, fits, violations = _fit_all(u, v, u_state, tau_state, cell, ln_rate,
                                                      rng, n_perturb, warm)
    return dict(u=u, v=v, scale=scale, table=table, pooled=pooled, table_flags=fits[TABLE]['flags'],
                fits={m: f for m, f in fits.items() if m in TERMS}, ll_table=fits[TABLE]['ll'],
                table_rate=fits.get(TABLE_RATE), nesting_violations=violations,
                rank_level_events=np.bincount(level, minlength=len(rank_levels or [0])).tolist())


# --- trajectory bootstrap of held-out scores (sampling noise only) ---------------------
def bootstrap_scores(held, traj, n_boot, rng):
    """Per-model held-out totals under trajectory resampling; one shared set of
    resamples, so differences between models are paired."""
    ids, index = np.unique(traj, return_inverse=True)
    weights = np.stack([np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
                        for _ in range(n_boot)])
    per_traj = {m: np.bincount(index, weights=held[m], minlength=len(ids)) for m in held}
    return {m: weights @ per_traj[m] for m in held}


def compare(held, boot, a, b):
    point = float(held[a].sum() - held[b].sum())
    diff = boot[a] - boot[b]
    return dict(difference=point, ci=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                p_positive=float(np.mean(diff > 0)))


def gumbel_tau(theta):
    return 1.0 - 1.0 / np.asarray(theta)
