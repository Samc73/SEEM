import numpy as np


def build_SEEM_library(
    u,
    tau,
    include_poly=True,
    include_log_exp=True,
    include_rbf=True,
    poly_order=3,
    n_rbf_u=5,
    n_rbf_tau=5,
    eps=1e-8,
    rbf_centers=None,
    rbf_sigmas=None,
):
    """
    Build candidate function library Theta(u,tau) for SEEM/SINDy regression.

    State variables:
        u   = pe - u0
        tau = stress / 10000

    Returns
    -------
    Theta : ndarray, shape (N, n_features)
        Feature matrix.

    feature_names : list of str
        Names of each feature.

    metadata : dict
        Stores RBF centers/sigmas so the same library can be reused later.
    """

    u = np.asarray(u).ravel()
    tau = np.asarray(tau).ravel()

    features = []
    names = []

    features.append(np.ones_like(u))
    names.append("1")

    if include_poly:
        if poly_order >= 1:
            features += [u, tau]
            names += ["u", "tau"]

        if poly_order >= 2:
            features += [u**2, u * tau, tau**2]
            names += ["u^2", "u*tau", "tau^2"]

        if poly_order >= 3:
            features += [
                u**3,
                u**2 * tau,
                u * tau**2,
                tau**3,
            ]
            names += [
                "u^3",
                "u^2*tau",
                "u*tau^2",
                "tau^3",
            ]

        if poly_order >= 4:
            features += [
                u**4,
                u**3 * tau,
                u**2 * tau**2,
                u * tau**3,
                tau**4,
            ]
            names += [
                "u^4",
                "u^3*tau",
                "u^2*tau^2",
                "u*tau^3",
                "tau^4",
            ]

    # -------------------------
    # Log / exponential terms
    # -------------------------
    log_shift = None

    if include_log_exp:
        log_shift = np.nanmin(u) - eps
        u_safe = u - log_shift
        logu = np.log(u_safe)

        exp_tau = np.exp(np.clip(tau, -50, 50))
        exp_neg_tau = np.exp(np.clip(-tau, -50, 50))

        features += [
            logu,
            tau * logu,
            exp_tau,
            exp_neg_tau,
            u * exp_tau,
            u * exp_neg_tau,
        ]

        names += [
            "log(u)",
            "tau*log(u)",
            "exp(tau)",
            "exp(-tau)",
            "u*exp(tau)",
            "u*exp(-tau)",
        ]

    # -------------------------
    # Gaussian / RBF terms
    # -------------------------
    if include_rbf:
        if rbf_centers is None or rbf_sigmas is None:
            u_min, u_max = np.nanpercentile(u, [1, 99])
            tau_min, tau_max = np.nanpercentile(tau, [1, 99])

            u_centers = np.linspace(u_min, u_max, n_rbf_u)
            tau_centers = np.linspace(tau_min, tau_max, n_rbf_tau)

            sigma_u = (u_max - u_min) / max(n_rbf_u - 1, 1)
            sigma_tau = (tau_max - tau_min) / max(n_rbf_tau - 1, 1)

            sigma_u = max(sigma_u, eps)
            sigma_tau = max(sigma_tau, eps)

            rbf_centers = []
            for uc in u_centers:
                for tc in tau_centers:
                    rbf_centers.append((uc, tc))

            rbf_sigmas = (sigma_u, sigma_tau)

        sigma_u, sigma_tau = rbf_sigmas

        for uc, tc in rbf_centers:
            phi = np.exp(
                -0.5 * ((u - uc) / sigma_u) ** 2
                -0.5 * ((tau - tc) / sigma_tau) ** 2
            )

            features.append(phi)
            names.append(f"RBF(u={uc:.4g},tau={tc:.4g})")

    Theta = np.column_stack(features)

    metadata = {
        "include_poly": include_poly,
        "include_log_exp": include_log_exp,
        "include_rbf": include_rbf,
        "poly_order": poly_order,
        "n_rbf_u": n_rbf_u,
        "n_rbf_tau": n_rbf_tau,
        "eps": eps,
        "log_shift": log_shift,
        "rbf_centers": rbf_centers,
        "rbf_sigmas": rbf_sigmas,
    }

    return Theta, names, metadata


def build_SEEM_library_from_metadata(u, tau, metadata):
    """
    Rebuild the same library using saved metadata.

    Use this for test data, generated trajectories, or model evaluation.
    """

    return build_SEEM_library(
        u=u,
        tau=tau,
        include_poly=metadata["include_poly"],
        include_log_exp=metadata["include_log_exp"],
        include_rbf=metadata["include_rbf"],
        poly_order=metadata["poly_order"],
        n_rbf_u=metadata["n_rbf_u"],
        n_rbf_tau=metadata["n_rbf_tau"],
        eps=metadata["eps"],
        rbf_centers=metadata["rbf_centers"],
        rbf_sigmas=metadata["rbf_sigmas"],
    )


def print_equation(coefficients, feature_names, lhs_name="du/dgamma", threshold=1e-12):
    """
    Print readable equation from learned coefficients.
    """

    coefficients = np.asarray(coefficients).ravel()

    terms = []
    for c, name in zip(coefficients, feature_names):
        if abs(c) > threshold:
            terms.append(f"({c:.6e})*{name}")

    if len(terms) == 0:
        print(f"{lhs_name} = 0")
    else:
        equation = " + ".join(terms)
        equation = equation.replace("+ (-", "- (")
        print(f"{lhs_name} = {equation}")


def evaluate_library_model(u, tau, coefficients, metadata):
    """
    Evaluate a learned model:
        y_pred = Theta(u,tau) @ coefficients
    """

    Theta, feature_names, _ = build_SEEM_library_from_metadata(u, tau, metadata)
    y_pred = Theta @ coefficients

    return y_pred