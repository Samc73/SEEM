"""
Constitutive-model discovery for AQS amorphous plasticity.

Core idea: rather than fitting an ODE to (du/dgamma, dtau/dgamma) directly,
measure the dimensionless plastic-rate field

    q(u, tau) = 1 - (dtau/dgamma) / (2 mu(u, tau))

which is the object STZ theory actually makes predictions about, then test
those predictions.

State variables follow the existing notebook convention:
    u   = pe - u0      (inherent-structure energy above reference; ~ chi)
    tau = stress / 1e4

NOTE: untested against real data as of writing -- df_clean.pkl was not
available. Expect to iterate once it is.
"""

import numpy as np
import pandas as pd

U0_DEFAULT = -4.60751861
TAU_SCALE = 1e4
DGAMMA = 1e-5


# ---------------------------------------------------------------------------
# Loading and diagnostics
# ---------------------------------------------------------------------------

def load(path, u0=U0_DEFAULT):
    """Load raw data and attach state variables and one-step transitions.

    Transitions are taken within trajectories only; the last row of each
    trajectory gets NaN.
    """
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_pickle(path)

    df = df.sort_values(["index", "strain_index"]).reset_index(drop=True)

    df["u"] = df["pe"] - u0
    df["tau"] = df["stress"] / TAU_SCALE

    g = df.groupby("index", sort=False)
    df["du"] = g["u"].shift(-1) - df["u"]
    df["dtau"] = g["tau"].shift(-1) - df["tau"]
    df["dstrain"] = g["strain"].shift(-1) - df["strain"]

    return df


def diagnose(df):
    """Facts we need before any modelling. Prints, and returns a dict."""
    g = df.groupby("index")
    u0_first = g["u"].first()
    strain_first = g["strain"].first()
    lengths = g.size()

    out = {
        "n_trajectories": int(df["index"].nunique()),
        "n_rows": int(len(df)),
        "cooling_rates": sorted(df["cooling_rate"].unique().tolist())
        if "cooling_rate" in df
        else None,
        "u0_spread": (float(u0_first.min()), float(u0_first.max())),
        "u0_ratio": float(u0_first.max() / max(u0_first.min(), 1e-12)),
        "strain_starts_at_zero": bool(np.allclose(strain_first, strain_first.iloc[0])),
        "traj_lengths": (int(lengths.min()), int(lengths.max())),
        "dstrain_unique": np.unique(
            np.round(df["dstrain"].dropna().values[:100000], 12)
        ).tolist()[:5],
    }

    print("trajectories:      ", out["n_trajectories"])
    print("rows:              ", f'{out["n_rows"]:,}')
    print("cooling rates:     ", out["cooling_rates"])
    print("initial u range:   ", out["u0_spread"], f'(ratio {out["u0_ratio"]:.1f}x)')
    print("all start same eps:", out["strain_starts_at_zero"])
    print("traj lengths:      ", out["traj_lengths"])
    print("dstrain values:    ", out["dstrain_unique"])
    return out


def split_by_trajectory(df, frac=0.7, seed=0):
    """Train/test split that does NOT leak adjacent strain steps across the
    boundary. Every metric in the original notebook used random row splits,
    which leaks heavily -- neighbouring steps are near-identical states.
    """
    ids = df["index"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    cut = int(frac * len(ids))
    train_ids, test_ids = set(ids[:cut]), set(ids[cut:])
    return (
        df[df["index"].isin(train_ids)].copy(),
        df[df["index"].isin(test_ids)].copy(),
    )


# ---------------------------------------------------------------------------
# Event labelling
# ---------------------------------------------------------------------------

def label_events(df, drop_threshold=0.0):
    """Label plastic events.

    In AQS an event is a genuine instability: energy and stress both drop.
    `drop_threshold` lets you sweep the cutoff to check that downstream
    results are not artefacts of labelling numerical noise as events --
    this sweep is a prerequisite for trusting anything built on top.
    """
    df = df.copy()
    df["event"] = (df["dtau"] < -drop_threshold) | (df["du"] < -drop_threshold)
    return df


def threshold_sweep(df, thresholds=None):
    """Event rate vs labelling threshold. We want a plateau, not a slope."""
    if thresholds is None:
        thresholds = np.logspace(-9, -3, 13)
    rows = []
    for t in thresholds:
        ev = (df["dtau"] < -t) | (df["du"] < -t)
        rows.append({"threshold": t, "rate": float(ev.mean()), "n": int(ev.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Elastic modulus, measured the AQS-correct way
# ---------------------------------------------------------------------------

def branch_modulus(df, n_after=5, min_count=50, u_edges=None, tau_edges=None):
    """Measure 2*mu(u, tau) from the first `n_after` steps of each elastic
    branch, i.e. immediately following an event.

    Why not just fit all no-event rows (what `G_pred` does): in AQS a branch
    legitimately softens as it approaches its instability, since the relevant
    Hessian eigenvalue goes to zero. Averaging over the whole branch folds
    that softening into "the modulus", and then q = 1 - (dtau/dgamma)/(2mu)
    double-counts it as plasticity. Sampling just after an event keeps us
    far from the next instability.
    """
    df = df.sort_values(["index", "strain_index"])
    ev = df["event"].to_numpy()
    traj = df["index"].to_numpy()

    # steps since the last event, within trajectory
    since = np.full(len(df), np.iinfo(np.int32).max, dtype=np.int64)
    counter = np.iinfo(np.int32).max
    prev_traj = None
    for i in range(len(df)):
        if traj[i] != prev_traj:
            counter = np.iinfo(np.int32).max
            prev_traj = traj[i]
        since[i] = counter
        if ev[i]:
            counter = 0
        elif counter < np.iinfo(np.int32).max:
            counter += 1

    df = df.copy()
    df["since_event"] = since

    fresh = df[(df["since_event"] <= n_after) & (~df["event"])].copy()
    fresh["slope"] = fresh["dtau"] / fresh["dstrain"]

    return _bin_field(
        fresh, "slope", u_edges, tau_edges, min_count=min_count, how="median"
    )


# ---------------------------------------------------------------------------
# Voxel machinery
# ---------------------------------------------------------------------------

def make_grid(df, n_u=20, n_tau=20, qlo=0.001, qhi=0.999):
    u_edges = np.linspace(df["u"].quantile(qlo), df["u"].quantile(qhi), n_u + 1)
    tau_edges = np.linspace(
        df["tau"].quantile(qlo), df["tau"].quantile(qhi), n_tau + 1
    )
    return u_edges, tau_edges


def assign_bins(df, u_edges, tau_edges):
    df = df.copy()
    df["u_bin"] = np.searchsorted(u_edges, df["u"].to_numpy(), side="right") - 1
    df["tau_bin"] = np.searchsorted(tau_edges, df["tau"].to_numpy(), side="right") - 1
    ok = (
        (df["u_bin"] >= 0)
        & (df["u_bin"] < len(u_edges) - 1)
        & (df["tau_bin"] >= 0)
        & (df["tau_bin"] < len(tau_edges) - 1)
    )
    return df.loc[ok].copy()


def _bin_field(df, col, u_edges, tau_edges, min_count=50, how="mean"):
    """Reduce `col` onto the (u, tau) grid. Returns a 2D array with NaN where
    the voxel is under-populated."""
    if "u_bin" not in df:
        df = assign_bins(df, u_edges, tau_edges)
    n_u, n_tau = len(u_edges) - 1, len(tau_edges) - 1

    grp = df.groupby(["u_bin", "tau_bin"])[col]
    agg = grp.median() if how == "median" else grp.mean()
    cnt = grp.size()

    field = np.full((n_u, n_tau), np.nan)
    for (i, j), v in agg.items():
        if cnt.loc[(i, j)] >= min_count:
            field[i, j] = v
    return field


# ---------------------------------------------------------------------------
# The plastic-rate field
# ---------------------------------------------------------------------------

def plastic_rate_field(df, u_edges, tau_edges, mu2_field=None, min_count=200):
    """q(u, tau) = 1 - <dtau/dgamma> / (2 mu(u, tau)).

    <dtau/dgamma> is the mean over ALL transitions in the voxel (elastic
    branches and events together) -- that average is what the coarse-grained
    derivative measures.

    Sanity checks that must pass before trusting anything downstream:
      - q -> 0 in the low-u, low-tau corner (elastic loading)
      - q -> 1 at steady-state flow (dtau/dgamma -> 0)
      - q stays within [0, 1] essentially everywhere
    """
    df = assign_bins(df, u_edges, tau_edges)
    df = df.dropna(subset=["dtau", "dstrain"])
    df["dtau_dgamma"] = df["dtau"] / df["dstrain"]
    df["du_dgamma"] = df["du"] / df["dstrain"]

    drift_tau = _bin_field(df, "dtau_dgamma", u_edges, tau_edges, min_count)
    drift_u = _bin_field(df, "du_dgamma", u_edges, tau_edges, min_count)

    if mu2_field is None:
        mu2_field = branch_modulus(df, u_edges=u_edges, tau_edges=tau_edges)

    with np.errstate(invalid="ignore", divide="ignore"):
        q = 1.0 - drift_tau / mu2_field

    return {"q": q, "drift_tau": drift_tau, "drift_u": drift_u, "mu2": mu2_field}


def factorization_test(q):
    """Test STZ's structural prediction q(u, tau) = Lambda(u) * f(tau).

    In logs that is additive separability, so log q should be rank-1 (plus a
    rank-1 term from the row/column means -- we remove means first, so a
    perfectly factorizable field leaves rank 0 residual and the first
    singular value of the *centred* log field should be small).

    Returns singular values of the centred log-q field and the fraction of
    variance the leading component explains. Dominant first component =>
    factorization holds.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.log(q)
    L[~np.isfinite(L)] = np.nan

    # keep the largest fully-observed submatrix (SVD needs no NaNs)
    rows = ~np.all(np.isnan(L), axis=1)
    cols = ~np.all(np.isnan(L), axis=0)
    L = L[np.ix_(rows, cols)]
    keep_r = ~np.any(np.isnan(L), axis=1)
    keep_c = ~np.any(np.isnan(L), axis=0)
    if keep_r.sum() < 3 or keep_c.sum() < 3:
        # fall back: drop whichever axis loses less data
        keep_r = ~np.any(np.isnan(L), axis=1)
        L = L[keep_r, :]
        keep_c = ~np.any(np.isnan(L), axis=0)
        L = L[:, keep_c]
    else:
        L = L[np.ix_(keep_r, keep_c)]

    if L.size == 0 or min(L.shape) < 2:
        return {"ok": False, "reason": "not enough complete voxels"}

    sv = np.linalg.svd(L, compute_uv=False)
    return {
        "ok": True,
        "shape": L.shape,
        "singular_values": sv,
        "leading_fraction": float(sv[0] ** 2 / np.sum(sv**2)),
        "rank1_residual_fraction": float(np.sum(sv[1:] ** 2) / np.sum(sv**2)),
    }


def boltzmann_test(q, u_edges, tau_edges):
    """STZ predicts Lambda = exp(-1/chi), so with chi ~ u the u-dependence of
    log q should be linear in 1/u. Returns per-tau-column fits of
    log q vs 1/u; a consistent slope across columns supports the form.

    Caveat worth keeping in view: free-volume theory gives the same
    exp(-1/u) structure. This separates activated-in-1/u from power-law-in-u,
    not STZ from free volume.
    """
    u_c = 0.5 * (u_edges[:-1] + u_edges[1:])
    x = 1.0 / u_c

    rows = []
    with np.errstate(invalid="ignore", divide="ignore"):
        L = np.log(q)
    for j in range(q.shape[1]):
        y = L[:, j]
        m = np.isfinite(y) & np.isfinite(x)
        if m.sum() < 4:
            continue
        slope, intercept = np.polyfit(x[m], y[m], 1)
        pred = slope * x[m] + intercept
        ss = np.sum((y[m] - y[m].mean()) ** 2)
        r2 = 1 - np.sum((y[m] - pred) ** 2) / ss if ss > 0 else np.nan
        rows.append(
            {
                "tau_bin": j,
                "tau": 0.5 * (tau_edges[j] + tau_edges[j + 1]),
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
                "n": int(m.sum()),
            }
        )
    return pd.DataFrame(rows)


def effective_temperature_test(fields, u_edges, tau_edges):
    """STZ's chi equation:  du/dgamma = (tau * q / c0) * (1 - u / chi_hat).

    So (du/dgamma) / (tau * q) plotted against u must be a straight line, and
    its zero-crossing must equal the observed steady-state u. Two parameters,
    one of them independently measurable -- this is the sharpest single test
    in the set.
    """
    q, drift_u = fields["q"], fields["drift_u"]
    u_c = 0.5 * (u_edges[:-1] + u_edges[1:])
    tau_c = 0.5 * (tau_edges[:-1] + tau_edges[1:])
    U, T = np.meshgrid(u_c, tau_c, indexing="ij")

    with np.errstate(invalid="ignore", divide="ignore"):
        y = drift_u / (T * q)

    m = np.isfinite(y) & np.isfinite(U) & (q > 0.05)
    if m.sum() < 10:
        return {"ok": False, "reason": "too few valid voxels"}

    slope, intercept = np.polyfit(U[m], y[m], 1)
    chi_hat = -intercept / slope if slope != 0 else np.nan
    pred = slope * U[m] + intercept
    ss = np.sum((y[m] - y[m].mean()) ** 2)

    return {
        "ok": True,
        "c0_inv": float(intercept),
        "c0": float(1.0 / intercept) if intercept != 0 else np.nan,
        "chi_hat": float(chi_hat),
        "r2": float(1 - np.sum((y[m] - pred) ** 2) / ss) if ss > 0 else np.nan,
        "n_voxels": int(m.sum()),
    }


# ---------------------------------------------------------------------------
# Jump kernel
# ---------------------------------------------------------------------------

def jump_collapse(df, u_edges=None, tau_edges=None, per_voxel=False):
    """Test whether the 2D jump kernel collapses to a 1D avalanche size.

    If both du and dtau are proportional to the plastic strain of a single
    rearrangement, then dtau = -k * du with k set by the modulus and system
    size, and p(du, dtau | u, tau) reduces to p(s | u, tau) for a scalar s.
    That would replace the R^2=0.1 conditional-mean regression with a
    scaling law.
    """
    ev = df[df["event"]].dropna(subset=["du", "dtau"])

    def _fit(sub):
        if len(sub) < 50:
            return None
        x, y = sub["du"].to_numpy(), sub["dtau"].to_numpy()
        k = np.polyfit(x, y, 1)
        resid = y - (k[0] * x + k[1])
        ss = np.sum((y - y.mean()) ** 2)
        return {
            "slope": float(k[0]),
            "intercept": float(k[1]),
            "r2": float(1 - np.sum(resid**2) / ss) if ss > 0 else np.nan,
            "n": int(len(sub)),
        }

    if not per_voxel:
        return _fit(ev)

    ev = assign_bins(ev, u_edges, tau_edges)
    rows = []
    for (i, j), sub in ev.groupby(["u_bin", "tau_bin"]):
        r = _fit(sub)
        if r:
            rows.append({"u_bin": i, "tau_bin": j, **r})
    return pd.DataFrame(rows)


def hazard_field(df, u_edges, tau_edges, min_count=200):
    """Event hazard lambda(u, tau) per unit strain.

    In AQS this is set by the density of soft spots at threshold, so it is a
    measurement of pseudogap structure -- not a classification problem. The
    right validation is calibration and the inter-event strain distribution,
    not per-step AUC (which is bounded far below 1 at a 0.7% event rate even
    for a perfect rate model).
    """
    df = assign_bins(df, u_edges, tau_edges)
    df = df.dropna(subset=["dstrain"])
    grp = df.groupby(["u_bin", "tau_bin"])
    rate = grp["event"].mean()
    cnt = grp.size()
    dg = grp["dstrain"].median()

    n_u, n_tau = len(u_edges) - 1, len(tau_edges) - 1
    lam = np.full((n_u, n_tau), np.nan)
    for (i, j), v in rate.items():
        if cnt.loc[(i, j)] >= min_count:
            lam[i, j] = v / dg.loc[(i, j)]
    return lam


def interevent_strains(df):
    """Strain intervals between successive events, per trajectory. The
    survival-analysis target that replaces AUC."""
    out = []
    for _, sub in df.groupby("index", sort=False):
        s = sub.loc[sub["event"], "strain"].to_numpy()
        if len(s) > 1:
            out.append(np.diff(s))
    return np.concatenate(out) if out else np.array([])
