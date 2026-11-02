"""Stellar-arm templates and area-weighted narrow-fraction comparison."""

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import gaussian_filter
from hiphases import surfaces


def angular_coordinates(h, shape, g):
    (y, x) = np.indices(shape)
    w = WCS(h).celestial
    sky = w.pixel_to_world(x, y)
    centre = w.pixel_to_world(h["CRPIX1"] - 1, h["CRPIX2"] - 1)
    (east, north) = centre.spherical_offsets_to(sky)
    east = east.to_value("arcsec")
    north = north.to_value("arcsec")
    pa = np.deg2rad(g.position_angle)
    return np.arctan2(
        (east * np.cos(pa) - north * np.sin(pa)) / g.axis_ratio, east * np.sin(pa) + north * np.cos(pa)
    )


def design(theta):
    return np.column_stack(
        [np.ones(theta.size)] + [fn(m * theta) for m in range(1, 7) for fn in [np.cos, np.sin]]
    )


def robust_fit(a, y):
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    weights = np.ones_like(y)
    for _ in range(8):
        residual = y - a @ coef
        scale = 1.4826 * np.median(abs(residual - np.median(residual)))
        if not np.isfinite(scale) or scale <= 1e-20:
            break
        weights = np.minimum(1, 2.5 * scale / np.maximum(abs(residual), 1e-30))
        sw = np.sqrt(weights)
        new = np.linalg.lstsq(a * sw[:, None], y * sw, rcond=None)[0]
        if np.linalg.norm(new - coef) <= 1e-05 * max(np.linalg.norm(coef), 1e-20):
            coef = new
            break
        coef = new
    return (coef, float(np.mean(weights < 0.1)))


def suppress_compact_sources(raw, header):
    """Replace only high-contrast PSF-like sources, before beam matching.

    The extended image is not median-filtered. Point-like intrinsic stellar
    knots can also be removed; this is an old-stellar-pattern proxy, not a
    compact-source census or foreground-membership classification.
    """
    from photutils.detection import DAOStarFinder

    array = np.squeeze(raw).astype(float)
    valid = np.isfinite(array)
    array = np.where(valid, array, 0)
    pixel = float(abs(proj_plane_pixel_scales(WCS(header).celestial)[0]) * 3600)
    fwhm = surfaces.DEFAULT_IRAC_BEAM_ARCSEC / pixel
    background = gaussian_filter(array, 3 * fwhm)
    residual = array - background
    negative = -residual[valid & (residual < 0)]
    noise = 1.4826 * np.median(negative)
    noise = max(float(noise), 1e-12)
    finder = DAOStarFinder(threshold=10 * noise, fwhm=fwhm, roundness_range=(-0.6, 0.6), exclude_border=True)
    sources = finder(residual)
    mask = np.zeros(array.shape, dtype=bool)
    count = 0
    if sources is not None:
        for source in sources:
            x = float(source["x_centroid"])
            y = float(source["y_centroid"])
            iy = int(round(y))
            ix = int(round(x))
            if not (0 <= iy < array.shape[0] and 0 <= ix < array.shape[1]):
                continue
            if array[iy, ix] <= 4 * max(background[iy, ix], noise):
                continue
            radius = 2 * fwhm
            y0 = max(0, int(y - radius - 1))
            y1 = min(array.shape[0], int(y + radius + 2))
            x0 = max(0, int(x - radius - 1))
            x1 = min(array.shape[1], int(x + radius + 2))
            (yy, xx) = np.mgrid[y0:y1, x0:x1]
            mask[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
            count += 1
    return (np.where(mask, background, array), mask, count)


def area_comparison(cold, warm, arm, inter, pixel_area, block_pixels):
    total = cold + warm
    fraction = np.divide(cold, total, out=np.zeros_like(total), where=total > 0)
    area = np.full(cold.shape, pixel_area)
    (yy, xx) = np.indices(cold.shape)
    ids = yy // block_pixels * (cold.shape[1] // block_pixels + 1) + xx // block_pixels
    blocks = np.unique(ids[arm | inter])
    arr = []
    for k in blocks:
        a = arm & (ids == k)
        i = inter & (ids == k)
        arr.append(
            [np.sum(area[a] * fraction[a]), area[a].sum(), np.sum(area[i] * fraction[i]), area[i].sum()]
        )
    arr = np.asarray(arr)
    rng = np.random.default_rng(735)
    s = arr[rng.integers(len(arr), size=(4000, len(arr)))].sum(1)
    good = (s[:, 1] > 0) & (s[:, 3] > 0)
    delta = s[good, 0] / s[good, 1] - s[good, 2] / s[good, 3]
    ca = float(np.average(fraction[arm], weights=area[arm]))
    ci = float(np.average(fraction[inter], weights=area[inter]))
    return dict(
        n_arm=int(arm.sum()),
        n_interarm=int(inter.sum()),
        arm_area_pc2=float(area[arm].sum()),
        interarm_area_pc2=float(area[inter].sum()),
        cold_fraction_arm=ca,
        cold_fraction_interarm=ci,
        warm_fraction_arm=1 - ca,
        warm_fraction_interarm=1 - ci,
        contrast=ca - ci,
        ci_low=float(np.percentile(delta, 2.5)),
        ci_high=float(np.percentile(delta, 97.5)),
        n_spatial_blocks=len(blocks),
        cold_model_surface_density_ratio=float(cold[arm].mean() / cold[inter].mean()),
        warm_model_surface_density_ratio=float(warm[arm].mean() / warm[inter].mean()),
        mass_fraction_arm=float(cold[arm].sum() / total[arm].sum()),
        mass_fraction_interarm=float(cold[inter].sum() / total[inter].sum()),
    )
