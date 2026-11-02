"""Density-weighted orthogonal regression and paired bootstrap intervals."""

from pathlib import Path
import json
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist


def density_kernel(x, y, bandwidth_factor=1.0):
    """Gaussian Scott-rule kernel, with self-contributions removed.

    The covariance is estimated in 2D for density estimation only. The fitted
    line still minimises distances in the original equal-dex coordinates.
    """
    xy = np.column_stack([x, y])
    kde = gaussian_kde(xy.T, bw_method=lambda k: bandwidth_factor * k.scotts_factor())
    white = np.linalg.solve(np.linalg.cholesky(kde.covariance), xy.T).T
    kernel = np.exp(-0.5 * cdist(white, white, metric="sqeuclidean"))
    np.fill_diagonal(kernel, 0.0)
    return (kernel, kde.covariance)


def weighted_lines(x, y, weights):
    """Global weighted TLS solution for one or multiple sets of weights."""
    w = np.atleast_2d(weights)
    s = w.sum(axis=1)
    cx = w @ x / s
    cy = w @ y / s
    dx = x[None, :] - cx[:, None]
    dy = y[None, :] - cy[:, None]
    theta = 0.5 * np.arctan2(2 * np.sum(w * dx * dy, axis=1), np.sum(w * (dx * dx - dy * dy), axis=1))
    return np.column_stack([theta, cx, cy])


def fit_density_orthogonal(x, y, posterior_path=None, n_bootstrap=2000, seed=42, bandwidth_factor=1.0):
    """Leave-one-out KDE weights, with density re-estimated in each resample.

    Fix the full-sample kernel covariance during bootstrap, but recompute local
    densities using the resampled measurements, including multiplicities.
    This is conditional on the selected smoothing covariance. Weighting
    emphasises the populated ridge; it does not correct detection selection.
    """
    (x, y) = (np.asarray(x, float), np.asarray(y, float))
    n = len(x)
    if n < 3:
        from hiphases.resolvedks import orthogonal_line

        return orthogonal_line(x, y, posterior_path)
    (kernel, covariance) = density_kernel(x, y, bandwidth_factor)
    weights = np.maximum(kernel.sum(axis=1), 1e-12)
    (theta, cx, cy) = weighted_lines(x, y, weights)[0]
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, n_bootstrap, 200):
        ii = rng.integers(n, size=(min(200, n_bootstrap - start), n))
        counts = np.array([np.bincount(indices, minlength=n) for indices in ii])
        density = counts @ kernel.T + counts - 1.0
        w = counts * np.maximum(density, 1e-12)
        a = weighted_lines(x, y, w)
        variance = np.sum(w * ((x[None, :] - a[:, 1, None]) ** 2 + (y[None, :] - a[:, 2, None]) ** 2), axis=1)
        a = a[variance > 1e-20]
        a[:, 0] = theta + (a[:, 0] - theta + np.pi / 2) % np.pi - np.pi / 2
        draws.append(a)
    draws = np.concatenate(draws) if draws else np.empty((0, 3))
    slope = np.tan(theta)
    rr = -np.sin(theta) * (x - cx) + np.cos(theta) * (y - cy)
    row = dict(
        status="ok",
        fit_method="density-weighted orthogonal",
        slope=float(slope),
        intercept=float(cy - slope * cx),
        centroid_x=float(cx),
        centroid_y=float(cy),
        angle_deg=float(np.rad2deg(theta)),
        scatter_dex=float(np.sqrt(np.mean(rr**2))),
        weighted_perpendicular_rms=float(np.sqrt(np.average(rr**2, weights=weights))),
        density_bandwidth_factor=bandwidth_factor,
        density_weight_neff=float(weights.sum() ** 2 / (weights @ weights)),
        bootstrap_valid=len(draws),
        uncertainty_status="unavailable",
        slope_interval_unbounded=False,
        slope_p16=np.nan,
        slope_p84=np.nan,
        angle_p16=np.nan,
        angle_p84=np.nan,
        angle_p025=np.nan,
        angle_p975=np.nan,
    )
    if len(draws):
        (q025, lo, hi, q975) = np.percentile(draws[:, 0], [2.5, 16, 84, 97.5])
        cross = np.floor((lo + np.pi / 2) / np.pi) != np.floor((hi + np.pi / 2) / np.pi)
        row.update(
            angle_p16=float(np.rad2deg(lo)),
            angle_p84=float(np.rad2deg(hi)),
            angle_p025=float(np.rad2deg(q025)),
            angle_p975=float(np.rad2deg(q975)),
            slope_interval_unbounded=bool(cross),
            uncertainty_status="paired bootstrap; densities recomputed at fixed kernel covariance",
        )
        if not cross:
            row.update(slope_p16=float(np.tan(lo)), slope_p84=float(np.tan(hi)))
    if posterior_path is not None:
        path = Path(posterior_path)
        path = path.with_name(path.name.replace("_posterior", "_density_bootstrap"))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            x=x,
            y=y,
            weights=weights / weights.max(),
            kernel_covariance=covariance,
            samples=draws,
            parameters=np.array(["angle_rad", "centroid_x", "centroid_y"]),
            settings=np.array(
                json.dumps(
                    dict(
                        fit=row["fit_method"],
                        kernel="Gaussian 2D, leave one out",
                        bandwidth="Scott",
                        bandwidth_factor=bandwidth_factor,
                        bootstrap_draws=n_bootstrap,
                        seed=seed,
                        bootstrap="recompute density with resample multiplicities; fixed full-sample kernel covariance",
                        numerical_weight_floor=1e-12,
                        excluded_points=0,
                    )
                )
            ),
        )
    return row
