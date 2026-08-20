"""
Symbolic regression for constitutive-model discovery.

Two search modes over one grammar:

  enumerate()  exhaustive search over all expression trees up to a complexity
               bound, deduplicated by numerical fingerprint. Complete within
               the grammar -- the returned Pareto front is exact, not a sample.
               Use for 1-D targets (<= ~10^5 structures).

  evolve()     genetic programming over the same grammar, for targets where
               enumeration is infeasible (2-D and up).

Both return a Pareto front of (complexity, error, expression).

Design notes
------------
Constants are handled by variable projection: an expression carries at most
`max_consts` internal fitted constants, and the outer affine map a*f(x)+b is
profiled out in closed form at every evaluation. So the nonlinear optimiser
only ever sees a 1-2 parameter problem, which makes constant fitting fast and
robust enough to run inside an exhaustive loop.

Physics constraints (monotonicity, positivity, limit anchors) are applied as
hard filters rather than soft penalties, so that surviving expressions are
admissible by construction and the Pareto front means what it says.
"""

import numpy as np
from scipy.optimize import least_squares
from itertools import product

# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

UNARY = {
    "exp":  (np.exp,            1),
    "log":  (np.log,            1),
    "inv":  (lambda z: 1.0/z,   1),
    "sq":   (lambda z: z*z,     1),
    "sqrt": (np.sqrt,           1),
    "neg":  (lambda z: -z,      1),
}

BINARY = {
    "+": (np.add,      1),
    "-": (np.subtract, 1),
    "*": (np.multiply, 1),
    "/": (np.divide,   1),
}

# operators that cost more than one unit of complexity -- discourages the
# search from reaching for transcendentals when a power law will do
OP_COST = {"exp": 2, "log": 2, "sqrt": 2, "inv": 1, "sq": 1, "neg": 1,
           "+": 1, "-": 1, "*": 1, "/": 1}


def _is_leaf(h):
    """Leaves are the constant 'c' and any variable 'x', 'x0', 'x1', ...
    No operator name begins with 'x', so the prefix test is unambiguous."""
    return h == "c" or h.startswith("x")


def complexity(e):
    """Total complexity of an expression tree."""
    h = e[0]
    if _is_leaf(h):
        return 1
    if h in UNARY:
        return OP_COST[h] + complexity(e[1])
    return OP_COST[h] + complexity(e[1]) + complexity(e[2])


def n_consts(e):
    h = e[0]
    if h == "c":
        return 1
    if _is_leaf(h):
        return 0
    if h in UNARY:
        return n_consts(e[1])
    return n_consts(e[1]) + n_consts(e[2])


def evaluate(e, x, consts, _i=None):
    """Evaluate tree `e` at `x`. `consts` is consumed left-to-right."""
    if _i is None:
        _i = [0]
    h = e[0]
    if h == "x":
        return x
    if h == "c":
        v = consts[_i[0]]
        _i[0] += 1
        return np.full_like(x, v, dtype=float)
    if h in UNARY:
        return UNARY[h][0](evaluate(e[1], x, consts, _i))
    a = evaluate(e[1], x, consts, _i)
    b = evaluate(e[2], x, consts, _i)
    return BINARY[h][0](a, b)


def to_string(e, consts=None, _i=None, var="x", varnames=None):
    if _i is None:
        _i = [0]
    h = e[0]
    if h.startswith("x"):
        if h == "x":
            return var
        i = int(h[1:])
        return varnames[i] if varnames else h
    if h == "c":
        if consts is None:
            return "c"
        s = f"{consts[_i[0]]:.4g}"
        _i[0] += 1
        return s
    if h in UNARY:
        inner = to_string(e[1], consts, _i, var)
        if h == "sq":
            return f"({inner})^2"
        if h == "inv":
            return f"1/({inner})"
        if h == "neg":
            return f"-({inner})"
        return f"{h}({inner})"
    a = to_string(e[1], consts, _i, var)
    b = to_string(e[2], consts, _i, var)
    return f"({a} {h} {b})"


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def _build(max_c, max_consts):
    """All trees indexed by complexity, up to max_c."""
    by_c = {1: [("x",), ("c",)]}
    for c in range(2, max_c + 1):
        out = []
        for op in UNARY:
            k = c - OP_COST[op]
            if k >= 1 and k in by_c:
                out += [(op, s) for s in by_c[k]]
        for op in BINARY:
            rem = c - OP_COST[op]
            for cl in range(1, rem):
                cr = rem - cl
                if cl in by_c and cr in by_c:
                    out += [(op, l, r) for l in by_c[cl] for r in by_c[cr]]
        by_c[c] = [e for e in out if n_consts(e) <= max_consts]
    return by_c


def _fingerprint(e, probe, max_consts):
    """Numerical signature, invariant to the outer affine map, used to collapse
    structurally different but numerically identical expressions."""
    rng = np.random.default_rng(0)
    sigs = []
    for _ in range(2):
        cs = rng.uniform(0.3, 2.0, size=max(n_consts(e), 1))
        with np.errstate(all="ignore"):
            v = evaluate(e, probe, cs)
        v = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(v)):
            return None
        s = np.std(v)
        if s < 1e-12:
            return ("const",)
        v = (v - np.mean(v)) / s        # quotient out a*f + b
        sigs.append(np.round(v, 6).tobytes())
    return tuple(sigs)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _fit_one(e, x, y, w, n_restarts=4, seed=0):
    """Fit internal constants (nonlinear) with the outer affine profiled out.

    Returns (rmse_weighted, consts, a, b) or None if the expression cannot be
    evaluated finitely on the data.
    """
    k = n_consts(e)
    rng = np.random.default_rng(seed)

    def affine(v):
        """Closed-form weighted least squares for y ~ a*v + b."""
        A = np.column_stack([v, np.ones_like(v)])
        W = w[:, None]
        try:
            coef, *_ = np.linalg.lstsq(A * W, y * w, rcond=None)
        except np.linalg.LinAlgError:
            return None
        return coef

    def resid(p):
        with np.errstate(all="ignore"):
            v = np.asarray(evaluate(e, x, p), dtype=float)
        if not np.all(np.isfinite(v)) or np.std(v) < 1e-14:
            return np.full_like(y, 1e6)
        coef = affine(v)
        if coef is None:
            return np.full_like(y, 1e6)
        return (coef[0] * v + coef[1] - y) * w

    if k == 0:
        r = resid(np.array([]))
        if np.any(r >= 1e5):
            return None
        v = np.asarray(evaluate(e, x, np.array([])), dtype=float)
        coef = affine(v)
        return float(np.sqrt(np.mean(r**2))), np.array([]), coef[0], coef[1]

    best = None
    for t in range(n_restarts):
        p0 = rng.uniform(0.2, 2.0, size=k) * (1.0 if t % 2 == 0 else -1.0) ** t
        try:
            sol = least_squares(resid, p0, method="lm", max_nfev=300)
        except Exception:
            continue
        r = resid(sol.x)
        if np.any(~np.isfinite(r)) or np.any(r >= 1e5):
            continue
        e_rms = float(np.sqrt(np.mean(r**2)))
        if best is None or e_rms < best[0]:
            v = np.asarray(evaluate(e, x, sol.x), dtype=float)
            coef = affine(v)
            best = (e_rms, sol.x, coef[0], coef[1])
    return best


def _predict(e, consts, a, b, x):
    with np.errstate(all="ignore"):
        v = np.asarray(evaluate(e, x, consts), dtype=float)
    return a * v + b


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def make_checker(x_dense, positive=False, increasing=False, decreasing=False,
                 anchors=None, anchor_tol=0.25):
    """Build a hard admissibility filter evaluated on a dense grid.

    anchors: list of (x_value, y_value) the expression must pass near, e.g.
             the elastic limit q(u->0) -> 0.
    """
    def check(e, consts, a, b):
        v = _predict(e, consts, a, b, x_dense)
        if not np.all(np.isfinite(v)):
            return False
        if positive and np.any(v <= 0):
            return False
        d = np.diff(v)
        if increasing and np.any(d < -1e-12):
            return False
        if decreasing and np.any(d > 1e-12):
            return False
        if anchors:
            for xa, ya in anchors:
                pa = _predict(e, consts, a, b, np.array([xa]))[0]
                if not np.isfinite(pa):
                    return False
                if abs(pa - ya) > anchor_tol * max(abs(ya), 1e-3):
                    return False
        return True
    return check


# ---------------------------------------------------------------------------
# Pareto bookkeeping
# ---------------------------------------------------------------------------

def _pareto(results):
    """Keep only entries not dominated on (complexity, error)."""
    out = []
    for r in sorted(results, key=lambda z: (z["complexity"], z["rmse"])):
        if not out or r["rmse"] < out[-1]["rmse"] - 1e-15:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enumerate_search(x, y, sigma=None, max_complexity=7, max_consts=2,
                     checker=None, n_restarts=4, verbose=True, probe=None):
    """Exhaustive symbolic regression for a 1-D target.

    x, y    : data. sigma: per-point standard errors (weights = 1/sigma).
    checker : optional admissibility filter from make_checker().
    probe   : grid used for the duplicate-collapsing fingerprint. Defaults to
              linspace over the data range; pass a log-spaced grid when x
              spans decades, so expressions differing only at small x are not
              falsely merged.

    Returns the exact Pareto front within the grammar.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(y) if sigma is None else 1.0 / np.maximum(np.asarray(sigma, float), 1e-12)
    w = w / np.mean(w)

    by_c = _build(max_complexity, max_consts)
    if probe is None:
        probe = np.linspace(x.min() * 0.9 + 1e-6, x.max() * 1.1, 17)

    seen, results, n_tot, n_uniq = set(), [], 0, 0
    for c in range(1, max_complexity + 1):
        for e in by_c.get(c, []):
            n_tot += 1
            fp = _fingerprint(e, probe, max_consts)
            if fp is None or fp in seen:
                continue
            seen.add(fp)
            n_uniq += 1
            fit = _fit_one(e, x, y, w, n_restarts=n_restarts)
            if fit is None:
                continue
            rmse, consts, a, b = fit
            if checker is not None and not checker(e, consts, a, b):
                continue
            results.append({
                "complexity": c, "rmse": rmse, "expr": e,
                "consts": consts, "a": a, "b": b,
                "string": to_string(e, consts),
            })
        if verbose:
            print(f"    complexity {c}: {n_tot} trees, {n_uniq} unique, "
                  f"{len(results)} admissible")
    return _pareto(results), results


def evolve(x, y, sigma=None, n_vars=2, pop=400, gens=60, max_complexity=14,
           max_consts=3, checker=None, seed=0, verbose=True, varnames=None):
    """Genetic programming for multi-dimensional targets.

    x : (n_points, n_vars). Uses the same grammar, with leaves x0..x{n-1}.
    """
    rng = np.random.default_rng(seed)
    X = np.atleast_2d(np.asarray(x, float))
    if X.shape[0] != len(y):
        X = X.T
    y = np.asarray(y, float)
    w = np.ones_like(y) if sigma is None else 1.0/np.maximum(np.asarray(sigma, float), 1e-12)
    w = w / np.mean(w)
    leaves = [(f"x{i}",) for i in range(n_vars)] + [("c",)]

    def ev(e, cs, _i=None):
        if _i is None:
            _i = [0]
        h = e[0]
        if h.startswith("x") and len(h) > 1:
            return X[:, int(h[1:])]
        if h == "c":
            v = cs[_i[0]]; _i[0] += 1
            return np.full(len(y), v)
        if h in UNARY:
            return UNARY[h][0](ev(e[1], cs, _i))
        return BINARY[h][0](ev(e[1], cs, _i), ev(e[2], cs, _i))

    def rand_tree(d=0):
        if d >= 3 or rng.random() < 0.3:
            return leaves[rng.integers(len(leaves))]
        if rng.random() < 0.4:
            op = list(UNARY)[rng.integers(len(UNARY))]
            return (op, rand_tree(d+1))
        op = list(BINARY)[rng.integers(len(BINARY))]
        return (op, rand_tree(d+1), rand_tree(d+1))

    def nodes(e, path=()):
        out = [(path, e)]
        if e[0] in UNARY:
            out += nodes(e[1], path+(1,))
        elif e[0] in BINARY:
            out += nodes(e[1], path+(1,)) + nodes(e[2], path+(2,))
        return out

    def replace(e, path, sub):
        if not path:
            return sub
        if e[0] in UNARY:
            return (e[0], replace(e[1], path[1:], sub))
        i = path[0]
        if i == 1:
            return (e[0], replace(e[1], path[1:], sub), e[2])
        return (e[0], e[1], replace(e[2], path[1:], sub))

    def score(e):
        if complexity(e) > max_complexity or n_consts(e) > max_consts:
            return np.inf, None, 0.0, 0.0
        k = n_consts(e)

        def resid(p):
            with np.errstate(all="ignore"):
                v = np.asarray(ev(e, p), float)
            if not np.all(np.isfinite(v)) or np.std(v) < 1e-14:
                return np.full_like(y, 1e6)
            A = np.column_stack([v, np.ones_like(v)])
            try:
                coef, *_ = np.linalg.lstsq(A*w[:, None], y*w, rcond=None)
            except np.linalg.LinAlgError:
                return np.full_like(y, 1e6)
            return (coef[0]*v + coef[1] - y)*w

        if k == 0:
            r = resid(np.array([]))
            if np.any(r >= 1e5):
                return np.inf, None, 0.0, 0.0
            v = np.asarray(ev(e, np.array([])), float)
            A = np.column_stack([v, np.ones_like(v)])
            coef, *_ = np.linalg.lstsq(A*w[:, None], y*w, rcond=None)
            return float(np.sqrt(np.mean(r**2))), np.array([]), coef[0], coef[1]
        best = (np.inf, None, 0.0, 0.0)
        for t in range(3):
            try:
                sol = least_squares(resid, rng.uniform(0.2, 2.0, k), method="lm", max_nfev=200)
            except Exception:
                continue
            r = resid(sol.x)
            if np.any(~np.isfinite(r)) or np.any(r >= 1e5):
                continue
            em = float(np.sqrt(np.mean(r**2)))
            if em < best[0]:
                v = np.asarray(ev(e, sol.x), float)
                A = np.column_stack([v, np.ones_like(v)])
                coef, *_ = np.linalg.lstsq(A*w[:, None], y*w, rcond=None)
                best = (em, sol.x, coef[0], coef[1])
        return best

    P = [rand_tree() for _ in range(pop)]
    hall = []
    for g in range(gens):
        sc = []
        for e in P:
            em, cs, a, b = score(e)
            if np.isfinite(em):
                hall.append({"complexity": complexity(e), "rmse": em, "expr": e,
                             "consts": cs, "a": a, "b": b,
                             "string": to_string(e, cs, varnames=varnames)})
            # parsimony pressure
            sc.append(em + 0.002*complexity(e) if np.isfinite(em) else np.inf)
        sc = np.array(sc)
        order = np.argsort(sc)
        elite = [P[i] for i in order[:max(2, pop//10)]]
        newP = list(elite)
        while len(newP) < pop:
            # tournament
            cand = rng.integers(0, pop, 4)
            a_ = P[cand[np.argmin(sc[cand])]]
            if rng.random() < 0.6:
                cand2 = rng.integers(0, pop, 4)
                b_ = P[cand2[np.argmin(sc[cand2])]]
                na = nodes(a_); nb = nodes(b_)
                pa = na[rng.integers(len(na))][0]
                sb = nb[rng.integers(len(nb))][1]
                child = replace(a_, pa, sb)
            else:
                na = nodes(a_)
                pa = na[rng.integers(len(na))][0]
                child = replace(a_, pa, rand_tree(2))
            if complexity(child) <= max_complexity:
                newP.append(child)
        P = newP
        if verbose and (g+1) % 20 == 0:
            hp = _pareto(hall)
            print(f"    gen {g+1}: best rmse={min(h['rmse'] for h in hall):.4f}, "
                  f"pareto size={len(hp)}")
    return _pareto(hall), hall
