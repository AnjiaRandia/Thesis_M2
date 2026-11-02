"""Surface-density conversions, Gaussian beam matching and reprojection."""

from __future__ import annotations
import re
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_exact
from .sample import Galaxy

HI_COEFFICIENT = 0.02
H2_COEFFICIENTS = {"CO10": 4.4, "CO21": 5.5}


def _cos_i(galaxy: Galaxy) -> float:
    return float(np.cos(galaxy.inclination.to(u.rad).value))


def moment0_to_hi(moment0_k_m_s: np.ndarray, galaxy: Galaxy) -> np.ndarray:
    """HI surface density from a moment-0 map in K m/s (Leroy+08 Eq. A1).

    SoFiA writes moment-0 maps in the cube's intensity unit times the spectral
    unit, i.e. K m/s here, hence the factor 1000 down to K km/s.  Negative
    values, which the masking can leave behind at the map edges, are clipped to
    zero: a negative column density is unphysical and would poison the logs
    taken later.
    """
    intensity = np.nan_to_num(moment0_k_m_s, nan=0.0) / 1000.0
    return np.clip(HI_COEFFICIENT * _cos_i(galaxy) * intensity, 0.0, None)


def co_to_h2(co_moment0_k_km_s: np.ndarray, galaxy: Galaxy) -> np.ndarray:
    """Molecular surface density from a CO moment-0 map in K km/s (Leroy+08).

    The coefficient is selected from the galaxy's ``co_line``, so a CO(1-0) map
    is not silently scaled as if it were CO(2-1) -- that mistake would inflate
    the molecular surface density by 25%.
    """
    try:
        coefficient = H2_COEFFICIENTS[galaxy.co_line]
    except KeyError:
        raise ValueError(
            f"{galaxy.name}: unknown CO transition {galaxy.co_line!r}; expected one of {sorted(H2_COEFFICIENTS)}"
        ) from None
    intensity = np.nan_to_num(co_moment0_k_km_s, nan=0.0)
    return np.clip(coefficient * _cos_i(galaxy) * intensity, 0.0, None)


DEFAULT_IRAC_BEAM_ARCSEC = 1.7


def native_beam_arcsec(header: fits.Header, default: float | None = None) -> float | None:
    """Resolution of a 2-D map, in arcsec, from its header.

    ``BMAJ`` if present (the CO and SFR maps carry it; the SFR maps of Leroy et
    al. 2008 are already at a fixed *physical* scale, which is why their BMAJ
    differs from galaxy to galaxy). Falls back to the AIPS ``HISTORY`` line the
    THINGS cubes use, then to ``default``.
    """
    if header.get("BMAJ"):
        return float(header["BMAJ"]) * 3600.0
    for card in [str(c) for c in header.get("HISTORY", [])]:
        match = re.search("BMAJ=\\s*([0-9.eE+-]+)", card)
        if match:
            return float(match.group(1)) * 3600.0
    return default


def convolve_map_to_beam(
    data: np.ndarray, header: fits.Header, target_arcsec: float, default_beam: float | None = None
) -> tuple[np.ndarray, float | None, bool]:
    """Smooth a 2-D map to a circular beam of ``target_arcsec``.

    Returns the map, its native resolution and whether the target was reached.
    A map already coarser than the target is returned untouched and flagged, so
    the caller can record that this galaxy's comparison is limited by that map
    rather than silently mixing resolutions -- which is what the pipeline did
    before: the CO, SFR and 3.6 um maps were reprojected onto the HI grid at
    whatever resolution they happened to have.
    """
    from astropy.convolution import Gaussian2DKernel, convolve_fft

    array = np.squeeze(np.asarray(data, dtype="float64"))
    native = native_beam_arcsec(header, default_beam)
    if native is None or native >= target_arcsec:
        return (array, native, native is not None and native <= target_arcsec)
    kernel_fwhm = np.sqrt(target_arcsec**2 - native**2)
    from astropy.wcs.utils import proj_plane_pixel_scales

    map_wcs = WCS(header)
    if map_wcs.naxis > 2:
        map_wcs = map_wcs.celestial
    pixel_arcsec = float(abs(proj_plane_pixel_scales(map_wcs)[0]) * 3600.0)
    fwhm_to_sigma = 1.0 / np.sqrt(8 * np.log(2))
    kernel = Gaussian2DKernel(kernel_fwhm / pixel_arcsec * fwhm_to_sigma)
    smoothed = convolve_fft(
        array, kernel, normalize_kernel=True, allow_huge=True, nan_treatment="interpolate"
    )
    return (smoothed, native, True)


def reproject_onto(
    data: np.ndarray, header: fits.Header, target_wcs: WCS, shape: tuple[int, int]
) -> np.ndarray:
    """Flux-conserving reprojection of a 2-D map onto the HI grid.

    ``reproject_exact`` is used rather than an interpolating scheme because the
    input maps (CO, 24um+FUV, 3.6um) have pixels comparable to or larger than
    the HI pixels, and their integrated signal must be preserved.
    """
    wcs = WCS(header)
    if wcs.naxis > 2:
        wcs = wcs.celestial
    array = np.squeeze(data)
    (reprojected, _) = reproject_exact((array, wcs), target_wcs, shape_out=shape)
    return np.nan_to_num(reprojected, nan=0.0)
