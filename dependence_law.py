"""State-dependent Gumbel copula: theta(u, tau) = 1 + exp(eta(u, tau)).

Used by 9_fit_dependence_law.py and test_dependence_law.py.  See PLAN_step9.md.

Forms for eta (z_u, z_tau standardised with the TRAINING events' mean and sd;
the product form keeps step 3's raw variable 100 u tau):
    constant    a
    product     a + b 100 u tau
    linear      a + b z_u + c z_tau
    quadratic   linear + d z_u^2 + e z_u z_tau + f z_tau^2
Reference: one theta per voxel (copulas.fit_family on that voxel's ranks).

Uniforms are ranks WITHIN each voxel: the marginals change with state, so
ranking all events together would mix them.  Held-out events are ranked
against the training events of their own voxel (copulas.ecdf_transform).
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import copulas

FORMS = ('constant', 'product', 'linear', 'quadratic')
N_COEFFICIENTS = dict(constant=1, product=2, linear=3, quadratic=6)
COEFFICIENT_NAMES = dict(constant=['a'], product=['a', 'b(100u tau)'],
                         linear=['a', 'b(z_u)', 'c(z_tau)'],
                         quadratic=['a', 'b(z_u)', 'c(z_tau)', 'd(z_u^2)', 'e(z_u z_tau)', 'f(z_tau^2)'])
ETA_RANGE = (-15.0, np.log(90.0))   # theta in (1 + 3e-7, 91); a fit pushed to the edge is flagged
H_ETA = 1e-5                        # step for the per-event derivative in eta
RUNAWAY = 20.0                      # |coefficient| above this is flagged
THETA_NEAR_ONE = 1.05
MIN_TRAIN_PER_VOXEL = 5             # fewer training events: the voxel gets the pooled theta


# --- design --------------------------------------------------------------------
def scaling(u_state, tau_state):
    return dict(mu_u=float(u_state.mean()), sd_u=float(u_state.std()),
                mu_t=float(tau_state.mean()), sd_t=float(tau_state.std()))


def design(form, u_state, tau_state, scale):
    ones = np.ones(len(u_state))
    if form == 'constant':
        return ones[:, None]
    if form == 'product':
        return np.column_stack([ones, 100.0 * u_state * tau_state])
    zu = (u_state - scale['mu_u']) / scale['sd_u']
    zt = (tau_state - scale['mu_t']) / scale['sd_t']
    if form == 'linear':
        return np.column_stack([ones, zu, zt])
    if form == 'quadratic':
        return np.column_stack([ones, zu, zt, zu * zu, zu * zt, zt * zt])
    raise ValueError(form)


def theta_of(beta, x):
    return 1.0 + np.exp(np.clip(x @ beta, *ETA_RANGE))


# --- uniforms --------------------------------------------------------------------
def within_voxel_ranks(values, cell):
    """Pseudo-observations computed separately inside each voxel."""
    out = np.empty(len(values))
    for c in np.unique(cell):
        m = cell == c
        out[m] = copulas.pseudo_observations(values[m])
    return out


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
    return -float(value.sum()), -(x.T @ slope)


def fit_form(x, u, v, starts):
    """L-BFGS-B from each start; keep the best.  Returns beta, ll and flags."""
    best = None
    for start in starts:
        r = minimize(_neg_ll_and_grad, np.asarray(start, float), args=(x, u, v), jac=True,
                     method='L-BFGS-B', options=dict(maxiter=500, gtol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    beta = best.x
    eta = x @ beta
    flags = []
    if np.any(np.abs(beta) > RUNAWAY):
        flags.append('coefficient beyond |%g|' % RUNAWAY)
    if np.any((eta <= ETA_RANGE[0]) | (eta >= ETA_RANGE[1])):
        flags.append('eta clipped for %d events' % int(((eta <= ETA_RANGE[0]) | (eta >= ETA_RANGE[1])).sum()))
    if not best.success:
        flags.append('optimiser: %s' % best.message)
    theta = theta_of(beta, x)
    return dict(beta=beta, ll=float(-best.fun), flags=flags,
                frac_theta_near_one=float(np.mean(theta < THETA_NEAR_ONE)),
                theta_range=[float(theta.min()), float(theta.max())])


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


def starting_points(form, table, cell, u_state, tau_state, scale, pooled, rng, n_perturb=3):
    """(i) least squares of ln(theta_voxel - 1) on each voxel's median state,
    weighted by events; (ii) flat at the pooled theta; (iii) perturbations of (i)."""
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


# --- cross-validation ------------------------------------------------------------------
MODELS = FORMS + ('voxel_table',)


def cross_validate(x_size, y_size, u_state, tau_state, cell, fold_id, rng, warm=None,
                   n_perturb=3):
    """Held-out ln c per event for every model.  fold_id[i] = fold of event i;
    each fold is held out once.  warm = {form: beta} gives a single warm start
    (used by the refitting bootstrap) instead of the full set of starts."""
    n = len(x_size)
    held = {m: np.zeros(n) for m in MODELS}
    folds = []
    for k in np.unique(fold_id):
        train = fold_id != k
        test = ~train
        u_tr, u_te, fb_u = held_out_uniforms(x_size, cell, train)
        v_tr, v_te, fb_v = held_out_uniforms(y_size, cell, train)
        scale = scaling(u_state[train], tau_state[train])
        table, pooled, table_flags = fit_voxel_table(u_tr, v_tr, cell[train])
        held['voxel_table'][test] = copulas.log_density(
            'gumbel', [table_theta(table, pooled, cell[test])], u_te, v_te)
        fold = dict(fold=int(k), n_train=int(train.sum()), n_test=int(test.sum()),
                    rank_fallback_events=max(fb_u, fb_v), voxels_in_table=len(table),
                    table_flags=table_flags, forms={})
        for form in FORMS:
            x_tr = design(form, u_state[train], tau_state[train], scale)
            x_te = design(form, u_state[test], tau_state[test], scale)
            starts = ([warm[form]] if warm is not None else
                      starting_points(form, table, cell[train], u_state[train], tau_state[train],
                                      scale, pooled, rng, n_perturb))
            fit = fit_form(x_tr, u_tr, v_tr, starts)
            held[form][test] = log_density(fit['beta'], x_te, u_te, v_te)
            fold['forms'][form] = dict(beta=fit['beta'].tolist(), flags=fit['flags'])
        folds.append(fold)
    return held, folds


def fit_in_sample(x_size, y_size, u_state, tau_state, cell, rng, n_perturb=3):
    """All models on all events (ranks within voxel)."""
    u = within_voxel_ranks(x_size, cell)
    v = within_voxel_ranks(y_size, cell)
    scale = scaling(u_state, tau_state)
    table, pooled, table_flags = fit_voxel_table(u, v, cell)
    fits = {}
    for form in FORMS:
        x = design(form, u_state, tau_state, scale)
        starts = starting_points(form, table, cell, u_state, tau_state, scale, pooled, rng, n_perturb)
        fits[form] = fit_form(x, u, v, starts)
    theta_table = table_theta(table, pooled, cell)
    ll_table = float(copulas.log_density('gumbel', [theta_table], u, v).sum())
    return dict(u=u, v=v, scale=scale, table=table, pooled=pooled, table_flags=table_flags,
                fits=fits, ll_table=ll_table)


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
