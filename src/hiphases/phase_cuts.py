"""Position-dependent kinematic separation from locally averaged Gaussian widths."""

from __future__ import annotations
import numpy as np
from scipy import ndimage

KERNEL_BEAMS = 3.0
MIN_LOCAL_BEAMS = 1.0
RING_R25 = 0.2
MAX_R25 = 2.5


def cell_grids(data):
    """Flux-weighted narrow and broad width at every position.

    A moving Gaussian kernel of ``KERNEL_BEAMS`` FWHM, so the local mean is
    ``sum(sigma w K) / sum(w K)`` evaluated everywhere by two convolutions
    rather than over disjoint cells.
    """
    header = data["header"]
    beam = float(header["BMAJ"]) * 3600.0
    pixel = abs(float(header["CDELT2"])) * 3600.0
    step = max(int(round(beam / pixel)), 1)
    kernel_sigma = KERNEL_BEAMS * step / 2.355
    beam_area = max(step**2, 1)
    (sigma, flux) = (data["sigma"], data["flux"])
    keep = data["measured"] & data["source"][None, :, :]
    out = []
    for index in (0, -1):
        weight = np.where(keep[index] & np.isfinite(sigma[index]), flux[index], 0.0)
        value = np.where(keep[index] & np.isfinite(sigma[index]), sigma[index] * flux[index], 0.0)
        numerator = ndimage.gaussian_filter(value, kernel_sigma)
        denominator = ndimage.gaussian_filter(weight, kernel_sigma)
        counts = ndimage.gaussian_filter((weight > 0).astype(float), kernel_sigma) * (
            2 * np.pi * kernel_sigma**2
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            local = numerator / denominator
        out.append(np.where((denominator > 0) & (counts >= MIN_LOCAL_BEAMS * beam_area), local, np.nan))
    (narrow, broad) = out
    swap = np.isfinite(narrow) & np.isfinite(broad) & (broad <= narrow)
    narrow = np.where(swap, np.nan, narrow)
    broad = np.where(swap, np.nan, broad)
    return (narrow, broad, step)


def radial_fallback(data, narrow, broad):
    """The galaxy's own radial trend, to fill cells with no measurement."""
    edges = np.arange(0.0, MAX_R25 + RING_R25, RING_R25)
    (centres, n_values, b_values) = ([], [], [])
    for lo, hi in zip(edges[:-1], edges[1:]):
        ring = data["source"] & (data["radius"] >= lo) & (data["radius"] < hi)
        if ring.sum() < 5:
            continue
        n = np.nanmedian(narrow[ring])
        b = np.nanmedian(broad[ring])
        if np.isfinite(n) and np.isfinite(b):
            centres.append(0.5 * (lo + hi))
            n_values.append(n)
            b_values.append(b)
    if len(centres) < 2:
        return (None, None)
    return (
        np.interp(data["radius"], centres, n_values, left=n_values[0], right=n_values[-1]),
        np.interp(data["radius"], centres, b_values, left=b_values[0], right=b_values[-1]),
    )


def smooth_filling_gaps(image, sigma_pixels):
    """Gaussian smoothing that ignores NaN rather than spreading it."""
    valid = np.isfinite(image)
    filled = np.where(valid, image, 0.0)
    numerator = ndimage.gaussian_filter(filled, sigma_pixels)
    denominator = ndimage.gaussian_filter(valid.astype(float), sigma_pixels)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = numerator / denominator
    return np.where(denominator > 0.05, out, np.nan)
