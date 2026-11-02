"""Beam-spaced sampling and the adopted gas–SFR fitting estimator."""

from __future__ import annotations
import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from pathlib import Path
import json

MIN_GAS = 0.0
H2_FIT_MIN = 0.0
FIT_SETTINGS = dict(
    fit="unweighted orthogonal distance regression in log gas and log SFR",
    solver="global total-least-squares solution using the covariance eigenvector",
    reference="https://arxiv.org/abs/2607.16467v1",
    coordinate_weights="equal weights per dex in both coordinates; no rescaling",
    measurement_errors="unavailable; weights are a geometric choice, not measured uncertainties",
    uncertainty="paired-measurement bootstrap, 2000 draws, seed 42",
    interval_percentiles=[16, 84],
    seed=42,
    bootstrap_draws=2000,
    vertical_slopes="report fitted orientation when the slope interval crosses infinity",
    minimum_points=2,
)


def orthogonal_line(x, y, posterior_path=None):
    """Global equal-weight ODR, retaining nearly vertical solutions.

    Fit in angle/centroid coordinates, avoiding an artificial bound on slope.
    Paired resampling estimates orientation uncertainty; an interval crossing
    a vertical line maps to an unbounded, disconnected slope confidence set.
    """
    (x, y) = (np.asarray(x, float), np.asarray(y, float))
    if len(x) < 2:
        return dict(status="undefined_fewer_than_two_points")
    (xm, ym) = (x.mean(), y.mean())
    (dx, dy) = (x - xm, y - ym)
    (sxx, sxy, syy) = (np.mean(dx * dx), np.mean(dx * dy), np.mean(dy * dy))
    if sxx + syy == 0:
        return dict(status="undefined_identical_points")
    if sxx == syy and sxy == 0:
        return dict(status="undefined_isotropic_orientation")
    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    (c, s) = (np.cos(theta), np.sin(theta))
    slope = s / c if abs(c) > 1e-12 else np.inf
    intercept = ym - slope * xm if np.isfinite(slope) else np.nan
    residual = -s * dx + c * dy
    row = dict(
        status="ok",
        slope=float(slope),
        intercept=float(intercept),
        centroid_x=float(xm),
        centroid_y=float(ym),
        angle_deg=float(np.rad2deg(theta)),
        scatter_dex=float(np.sqrt(np.mean(residual**2))),
        ols_yx=float(sxy / sxx) if sxx > 0 else np.nan,
        slope_p16=np.nan,
        slope_p84=np.nan,
        angle_p16=np.nan,
        angle_p84=np.nan,
        angle_p025=np.nan,
        angle_p975=np.nan,
        slope_interval_unbounded=not np.isfinite(slope),
        uncertainty_status="unavailable_two_points",
        bootstrap_valid=0,
        fit_method="ODR",
    )
    draws = np.empty((0, 3))
    if len(x) > 2:
        rng = np.random.default_rng(FIT_SETTINGS["seed"])
        ii = rng.integers(len(x), size=(FIT_SETTINGS["bootstrap_draws"], len(x)))
        (xx, yy) = (x[ii], y[ii])
        (mx, my) = (xx.mean(axis=1), yy.mean(axis=1))
        (xx, yy) = (xx - mx[:, None], yy - my[:, None])
        (vx, vy, cov) = (np.mean(xx * xx, axis=1), np.mean(yy * yy, axis=1), np.mean(xx * yy, axis=1))
        good = (vx + vy > 0) & ~((vx == vy) & (cov == 0))
        angles = 0.5 * np.arctan2(2 * cov[good], vx[good] - vy[good])
        angles = theta + (angles - theta + np.pi / 2) % np.pi - np.pi / 2
        if angles.size:
            (low, high) = np.percentile(angles, [16, 84])
            (low95, high95) = np.percentile(angles, [2.5, 97.5])
            crossing = np.floor((low + np.pi / 2) / np.pi) != np.floor((high + np.pi / 2) / np.pi)
            row.update(
                angle_p16=float(np.rad2deg(low)),
                angle_p84=float(np.rad2deg(high)),
                angle_p025=float(np.rad2deg(low95)),
                angle_p975=float(np.rad2deg(high95)),
                slope_interval_unbounded=bool(crossing or not np.isfinite(slope)),
                uncertainty_status="bootstrap",
                bootstrap_valid=int(angles.size),
            )
            if not row["slope_interval_unbounded"]:
                row.update(slope_p16=float(np.tan(low)), slope_p84=float(np.tan(high)))
            draws = np.column_stack([angles, mx[good], my[good]])
    if posterior_path is not None:
        path = Path(posterior_path).with_name(
            Path(posterior_path).name.replace("_posterior", "_odr_bootstrap")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            samples=draws,
            parameters=np.array(["angle_rad", "centroid_x", "centroid_y"]),
            x=x,
            y=y,
            settings=np.array(json.dumps(FIT_SETTINGS)),
        )
    return row


def fit_segment(row, x, y):
    """Parameterize a fitted line inside the supplied x/y bounding rectangle."""
    theta = np.deg2rad(row["angle_deg"])
    v = np.array([np.cos(theta), np.sin(theta)])
    centre = np.array([row["centroid_x"], row["centroid_y"]])
    (lower, upper) = (-np.inf, np.inf)
    for values, origin, direction in zip([x, y], centre, v):
        if abs(direction) > 1e-12:
            bounds = (np.array([np.min(values), np.max(values)]) - origin) / direction
            lower = max(lower, bounds.min())
            upper = min(upper, bounds.max())
    return centre + np.linspace(lower, upper, 100)[:, None] * v


def fit_label(row, show_angle=False):
    """Fitted orientation or molecular index, its interval, and Pearson r."""
    if row["status"] == "ok":
        if show_angle:
            value = f"$\\theta = {row['angle_deg']:.2f}^\\circ$"
            if np.isfinite(row["angle_p16"]):
                plus = row["angle_p84"] - row["angle_deg"]
                minus = row["angle_deg"] - row["angle_p16"]
                value = f"$\\theta = {row['angle_deg']:.2f}^{{+{plus:.2f}}}_{{-{minus:.2f}}}\\!^\\circ$"
            return value + f"\n$r = {row['r']:.2f}$"
        if row.get("slope_interval_unbounded", False):
            value = f"$\\theta = {row['angle_deg']:.2f}^\\circ$"
            if np.isfinite(row["angle_p16"]):
                plus = row["angle_p84"] - row["angle_deg"]
                minus = row["angle_deg"] - row["angle_p16"]
                value = f"$\\theta = {row['angle_deg']:.2f}^{{+{plus:.2f}}}_{{-{minus:.2f}}}\\!^\\circ$"
            return value + f"\n$N$: unbounded interval\n$r = {row['r']:.2f}$"
        if row.get("uncertainty_status") == "unavailable_two_points":
            return f"$N = {row['slope']:.2f}$\n2 measurements\n$r = {row['r']:.2f}$"
        (plus, minus) = (row["slope_p84"] - row["slope"], row["slope"] - row["slope_p16"])
        return f"$N = {row['slope']:.2f}^{{+{plus:.2f}}}_{{-{minus:.2f}}}$\n$r = {row['r']:.2f}$"
    return f"$r = {row['r']:.2f}$" if np.isfinite(row["r"]) else ""


def geometry(header, shape, galaxy):
    """Sky-correct disc radii, azimuths, and a beam-spaced sampling grid."""
    (y, x) = np.indices(shape)
    wcs = WCS(header).celestial
    (x0, y0) = (float(header["CRPIX1"]) - 1, float(header["CRPIX2"]) - 1)
    centre = wcs.pixel_to_world(x0, y0)
    sky = wcs.pixel_to_world(x, y)
    (east, north) = centre.spherical_offsets_to(sky)
    (east, north) = (east.to_value(u.arcsec), north.to_value(u.arcsec))
    pa = np.deg2rad(galaxy.position_angle)
    major = east * np.sin(pa) + north * np.cos(pa)
    minor = (east * np.cos(pa) - north * np.sin(pa)) / galaxy.axis_ratio
    radius = np.hypot(major, minor)
    beam = float(header["BMAJ"]) * 3600.0
    pixel = abs(float(header["CDELT2"])) * 3600.0
    r25 = galaxy.r25_pixels(abs(float(header["CDELT2"]))) * pixel
    rmax = max(r25 - 0.5 * beam, 0.0)
    inside = radius < rmax
    step = max(int(np.ceil(beam / pixel - 1e-09)), 1)
    sampling = ((x - round(x0)) % step == 0) & ((y - round(y0)) % step == 0)
    wedge = np.abs(major) >= 0.5 * radius
    return dict(
        radius=radius,
        inside=inside,
        sampled=inside & sampling,
        wedge=wedge,
        beam_arcsec=beam,
        pixel_arcsec=pixel,
        rmax_arcsec=rmax,
        r25_arcsec=r25,
        step_pixels=step,
    )


def numerical_zero_tolerance(values):
    """Float64 arithmetic floor, not an observational detection threshold.

    FFT smoothing can leave O(eps * map amplitude) residues where the input
    contributes exactly zero. A conservative64eps margin prevents taking
    logarithms of these artificial positive residues. All genuinely resolved
    faint values many orders above machine precision remain eligible.
    """
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    scale = float(np.max(np.abs(array[finite]))) if finite.any() else 0.0
    return 64 * np.finfo(np.float64).eps * scale


def positive_measurements(values):
    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array > numerical_zero_tolerance(array))


def sample_masks(gas, sfr, geom, molecular=False):
    """Return displayed and fitted beam-spaced measurements."""
    plot_good = geom["sampled"] & positive_measurements(gas) & positive_measurements(sfr)
    fit_good = plot_good
    return (plot_good, fit_good)


def fit_samples(gas, sfr, geom, molecular=False, posterior_path=None, estimator="density_orthogonal"):
    """Fit eligible samples with no extra range or correlation cuts."""
    (plot_good, fit_good) = sample_masks(gas, sfr, geom, molecular)
    (x, y) = (np.log10(gas[fit_good]), np.log10(sfr[fit_good]))
    row = dict(
        n=int(x.size),
        slope=np.nan,
        intercept=np.nan,
        r=np.nan,
        scatter_dex=np.nan,
        ols_yx=np.nan,
        fit_min_gas=H2_FIT_MIN if molecular else MIN_GAS,
        gas_numerical_zero_tolerance=numerical_zero_tolerance(gas),
        sfr_numerical_zero_tolerance=numerical_zero_tolerance(sfr),
        status="undefined_fewer_than_two_points",
    )
    if x.size >= 2 and np.ptp(x) > 0 and (np.ptp(y) > 0):
        row["r"] = float(np.corrcoef(x, y)[0, 1])
    if estimator != "density_orthogonal":
        raise ValueError("Only the adopted density-weighted orthogonal estimator is available")
    from hiphases.densityks import fit_density_orthogonal

    row.update(fit_density_orthogonal(x, y, posterior_path))
    return (row, plot_good)
