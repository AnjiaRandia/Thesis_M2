"""Preparation, spectral units and noise estimation for THINGS cubes."""

from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy.nddata import block_reduce
from radio_beam import Beam
from .sample import RMAX_R25, Galaxy


@dataclass(frozen=True)
class PreparedCube:
    """A cube ready for ROHSA, together with the header describing it."""

    data: np.ndarray
    header: fits.Header
    spatial_binning: int
    noise_channels: tuple[int, int]
    pedestal_pixels_corrected: float = 0.0

    def write(self, path) -> None:
        fits.writeto(path, self.data.astype("float32"), self.header, overwrite=True)


def load_things_cube(path) -> tuple[np.ndarray, fits.Header]:
    """Read a THINGS cube, dropping its degenerate Stokes axis.

    THINGS cubes are stored as (Stokes, velocity, dec, ra); everything
    downstream wants (velocity, dec, ra).
    """
    hdu = fits.open(path)[0]
    data = np.asarray(hdu.data, dtype="float64")
    while data.ndim > 3:
        data = data[0]
    data[~np.isfinite(data)] = 0.0
    return (data, hdu.header)


def crop_to_r25(
    data: np.ndarray, header: fits.Header, galaxy: Galaxy, factor: float = RMAX_R25
) -> tuple[np.ndarray, fits.Header]:
    """Crop a cube to a square of side ``2 * factor * r25`` about the map centre.

    The side is clipped to the available image, so galaxies whose ``1.25 r25``
    box is larger than the THINGS field (the nearby dwarfs) keep their full map.
    """
    r25_pix = galaxy.r25_pixels(abs(header["CDELT2"]))
    side = int(round(2 * factor * r25_pix))
    (ny, nx) = data.shape[-2:]
    side = min(side, ny, nx)
    y0 = max((ny - side) // 2, 0)
    x0 = max((nx - side) // 2, 0)
    cropped = data[:, y0 : y0 + side, x0 : x0 + side]
    out = header.copy()
    (out["NAXIS1"], out["NAXIS2"]) = (cropped.shape[2], cropped.shape[1])
    out["CRPIX1"] = header["CRPIX1"] - x0
    out["CRPIX2"] = header["CRPIX2"] - y0
    return (cropped, out)


def rebin_spatial(data: np.ndarray, header: fits.Header, factor: int) -> tuple[np.ndarray, fits.Header]:
    """Block-average the two spatial axes by an integer ``factor``."""
    if factor <= 1:
        return (data, header)
    binned = block_reduce(data, block_size=(1, factor, factor), func=np.mean)
    out = header.copy()
    (out["NAXIS1"], out["NAXIS2"]) = (binned.shape[2], binned.shape[1])
    for axis in (1, 2):
        out[f"CDELT{axis}"] = header[f"CDELT{axis}"] * factor
        out[f"CRPIX{axis}"] = (header[f"CRPIX{axis}"] - 0.5) / factor + 0.5
    return (binned, out)


COMMON_BEAM_ARCSEC = 15.0


def native_beam_arcsec(header: fits.Header) -> float:
    """The beam the cube already has, in arcsec.

    THINGS cubes do not carry BMAJ as a keyword; the beam is recorded in an AIPS
    HISTORY card of the form ``AIPS CLEAN BMAJ= 4.3345E-03 BMIN= ...`` in
    degrees. Read it once here rather than in each caller.
    """
    if header.get("BMAJ"):
        return float(header["BMAJ"]) * 3600.0
    for line in (str(x) for x in header.get("HISTORY", [])):
        match = re.search("BMAJ=\\s*([0-9.eE+-]+)", line)
        if match:
            return float(match.group(1)) * 3600.0
    raise KeyError("no beam in the header: neither BMAJ nor an AIPS CLEAN HISTORY card")


def target_beam_arcsec(galaxy, resolution_pc: float) -> float:
    """Beam that subtends ``resolution_pc`` at this galaxy's distance.

    Smoothing every galaxy to the same *linear* scale is what makes a sample
    comparison fair: the alternative, one angular beam for all, averages over
    four times more of a distant galaxy than a nearby one.
    """
    distance_pc = galaxy.distance.to(u.pc).value
    return float(np.degrees(resolution_pc / distance_pc) * 3600.0)


def convolve_to_common_beam(
    data: np.ndarray, header: fits.Header, target_arcsec: float = COMMON_BEAM_ARCSEC
) -> tuple[np.ndarray, fits.Header]:
    """Smooth a cube to a circular beam of ``target_arcsec``.

    The THINGS natural-weighted beams in this sample range from 5.3" to 15.9",
    so a common resolution is needed before the galaxies can be compared with
    each other and with the 13" CO maps.  Cubes already coarser than the target
    (only M81 dwA, at 15.9") are left untouched rather than sharpened.

    The convolution is done channel by channel with the Gaussian kernel that
    takes the current beam to the target.

    **The cube must already be in K.** A normalised kernel conserves surface
    brightness, which is what smoothing a brightness-temperature map should do;
    applied to a Jy/beam cube it leaves the values roughly unchanged when they
    should grow with the beam area, and converting to K afterwards then divides
    by the larger beam and suppresses T_B by that same ratio. That is what the
    first version of this pipeline did -- it smoothed in Jy/beam and converted
    afterwards -- which cost NGC 2976 a factor 4 in Sigma_HI at 15" and NGC 3184
    a factor 8, each galaxy by its own (target/native)^2.
    """
    from astropy.convolution import Gaussian2DKernel, convolve_fft

    current = Beam.from_fits_header(header)
    target = Beam(major=target_arcsec * u.arcsec, minor=target_arcsec * u.arcsec, pa=0 * u.deg)
    if current.major >= target.major:
        return (data, header)
    kernel_beam = target.deconvolve(current)
    pixel_deg = abs(header["CDELT2"])
    fwhm_to_sigma = 1.0 / np.sqrt(8 * np.log(2))
    kernel = Gaussian2DKernel(
        x_stddev=kernel_beam.major.to(u.deg).value / pixel_deg * fwhm_to_sigma,
        y_stddev=kernel_beam.minor.to(u.deg).value / pixel_deg * fwhm_to_sigma,
        theta=kernel_beam.pa.to(u.rad).value,
    )
    smoothed = np.stack(
        [convolve_fft(plane, kernel, normalize_kernel=True, allow_huge=True) for plane in data]
    )
    return (smoothed, target.attach_to_header(header.copy()))


def choose_binning(
    header: fits.Header, side: int, min_side: int = 32, beam_arcsec: float = COMMON_BEAM_ARCSEC
) -> int:
    """Pick the spatial block-averaging factor for a cropped cube of ``side`` pixels.

    THINGS oversamples its beam heavily, so the cubes can be shrunk a long way
    before ROHSA sees them.  The factor keeps two binned pixels across the
    common beam -- Nyquist sampling, so no spatial information is thrown away --
    while never leaving a map smaller than ``min_side`` pixels.  That floor
    matters for the most compact dwarfs (DDO 53, M81 dwA), whose ``1.25 r25``
    box is only a few tens of pixels wide to begin with.
    """
    pixel_arcsec = abs(header["CDELT2"]) * 3600.0
    factor = max(int(beam_arcsec / 2.0 / pixel_arcsec + 1e-06), 1)
    while factor > 1 and side // factor < min_side:
        factor -= 1
    return factor


def jy_per_beam_to_kelvin(data: np.ndarray, header: fits.Header) -> tuple[np.ndarray, fits.Header, float]:
    """Convert a Jy/beam cube to brightness temperature.

    Returns the converted cube, a header with ``BUNIT`` and the beam recorded,
    and the conversion factor itself (K per Jy/beam) for the provenance log.
    """
    beam = Beam.from_fits_header(header)
    frequency = header["RESTFREQ"] * u.Hz
    factor = (1 * u.Jy).to(u.K, equivalencies=u.brightness_temperature(frequency, beam))
    out = header.copy()
    out["BUNIT"] = "K"
    out = beam.attach_to_header(out)
    return (data * factor.value, out, float(factor.value))


MAX_NOISE_CHANNEL_FRACTION = 0.02


def find_noise_channels(data: np.ndarray, edge_fraction: float = 0.1) -> tuple[int, int]:
    """Find a line-free channel range at whichever end of the band has one.

    ROHSA estimates the per-pixel noise from a range of signal-free channels
    (``lstd``/``ustd`` in its parameter file).  The original notebooks read that
    range off a plot of the global profile.  Here it is found automatically: the
    range is grown inwards from an end of the band for as long as the global
    profile stays both within 3 MAD of the baseline and below
    :data:`MAX_NOISE_CHANNEL_FRACTION` of the peak.

    Both ends are tried and the longer clean run wins. An earlier version grew
    the range from channel zero alone, which put the window on the line for any
    cube whose emission starts at that end: NGC 7793's window carried 54 per
    cent of the peak flux, NGC 0925's 35 per cent and NGC 2403's 13 per cent.
    The MAD test by itself did not catch it, because a window that is mostly
    line has a large MAD and so accepts the next channel too; the flux ceiling
    is what stops it.

    The baseline is taken from the faintest tenth of the channels rather than
    from the edges, since in a cube whose line fills the band the edges are not
    the quiet part. At least three channels are always returned, since ROHSA
    needs a few, even where that means including some emission: a cube with no
    clean margin is better flagged by the diagnostics than silently trimmed.
    """
    profile = np.nansum(data, axis=(1, 2))
    nchan = profile.size
    peak = np.nanmax(profile)
    quiet = np.sort(profile)[: max(int(round(edge_fraction * nchan)), 3)]
    baseline = np.median(quiet)
    mad = np.median(np.abs(quiet - baseline)) * 1.4826
    ceiling = MAX_NOISE_CHANNEL_FRACTION * peak
    threshold = min(baseline + 3 * mad, ceiling) if mad > 0 else ceiling

    def clean_run(values: np.ndarray) -> int:
        k = 0
        while k < values.size and values[k] <= threshold:
            k += 1
        return k

    (lower, upper) = (clean_run(profile), clean_run(profile[::-1]))
    cap = max(nchan // 4, 3)
    if lower >= upper:
        return (0, int(np.clip(lower, 3, cap)))
    return (nchan - int(np.clip(upper, 3, cap)), nchan)


def channel_width_kms(header: fits.Header) -> float:
    """Width of one spectral channel in km/s, whatever the third axis is.

    THINGS cubes do not agree on how they describe their spectral axis: most use
    an optical-velocity axis (``FELO-HEL`` or ``VELO-HEL``) whose ``CDELT3`` is
    in m/s, but NGC 6946's is a frequency axis whose ``CDELT3`` is in Hz.  Taking
    ``CDELT3`` at face value there reads 12207 as 12.2 km/s when the cube is
    really sampled at 2.58 km/s -- a factor 4.7 that propagates straight into
    every fitted dispersion.

    For a frequency axis the width is converted through the radio velocity
    convention about the line rest frequency, which is what the FELO axes of the
    other cubes already encode.
    """
    delta = abs(header["CDELT3"])
    if str(header.get("CTYPE3", "")).upper().startswith("FREQ"):
        rest_frequency = header.get("RESTFREQ", header.get("RESTFRQ"))
        if rest_frequency is None:
            raise KeyError(
                "A frequency spectral axis needs RESTFREQ to be converted to km/s, but neither RESTFREQ nor RESTFRQ is present in the header."
            )
        return float(const.c.to(u.km / u.s).value * delta / rest_frequency)
    return float(delta / 1000.0)


def velocity_axis_kms(header: fits.Header, length: int | None = None) -> np.ndarray:
    """Velocity of every channel in km/s, whatever the third axis is.

    The companion to :func:`channel_width_kms`, and needed for the same reason.
    Building the axis inline as ``(CRVAL3 + (i + 1 - CRPIX3) * CDELT3) / 1000``
    is correct for the FELO-HEL and VELO-HEL cubes that make up the rest of the
    sample and silently wrong for NGC 6946, whose axis is in Hz: it returns
    frequencies in the millions where the others return velocities in the
    hundreds.  That mistake reached ``scripts/beam_smearing.py``, which reported
    20.3 km/s of rotation broadening for NGC 6946 -- more than the galaxy's own
    broad component, and the reason a median filter and a noise deconvolution
    were added there to explain a number that was never physical.

    A frequency axis is converted through the radio convention about the line
    rest frequency, so the returned velocities decrease with channel exactly as
    the optical-velocity axes of the other cubes do.
    """
    channels = np.arange(length if length is not None else header["NAXIS3"])
    world = header["CRVAL3"] + (channels + 1 - header["CRPIX3"]) * header["CDELT3"]
    if str(header.get("CTYPE3", "")).upper().startswith("FREQ"):
        rest_frequency = header.get("RESTFREQ", header.get("RESTFRQ"))
        if rest_frequency is None:
            raise KeyError(
                "A frequency spectral axis needs RESTFREQ to be converted to km/s, but neither RESTFREQ nor RESTFRQ is present in the header."
            )
        light = const.c.to(u.km / u.s).value
        return light * (rest_frequency - world) / rest_frequency
    return world / 1000.0


def normalise_spectral_axis(data: np.ndarray, header: fits.Header):
    """Rewrite a frequency third axis as an optical-velocity axis in m/s.

    Every cube the pipeline hands on is expected to describe its third axis the
    same way, and all but one of the THINGS cubes already do.  NGC 6946's is in
    Hz, and leaving it that way has two separate costs.  The first is arithmetic:
    any code that reads CDELT3 as a velocity is wrong for that galaxy alone,
    which is the hardest kind of error to notice because nineteen out of twenty
    galaxies are fine.  The second is that the header becomes internally
    inconsistent as it travels -- the exported cube reached BAYGAUD carrying
    ``CTYPE3 = 'FREQ'`` beside ``CUNIT3 = 'm/s'``, wcslib refused to set the WCS,
    and NGC 6946 was the one galaxy of the nineteen with no BAYGAUD fit at all.

    Converting once, here, means the rest of the pipeline never has to know.  The
    channels are reversed if needed so that velocity decreases with channel, the
    orientation the FELO-HEL cubes use, and the axis is written as ``VELO-HEL``
    in m/s with a linear CDELT3 -- exact, because the radio convention is linear
    in frequency.  Cubes that already carry a velocity axis are returned
    untouched.
    """
    if not str(header.get("CTYPE3", "")).upper().startswith("FREQ"):
        return (data, header)
    velocity = velocity_axis_kms(header) * 1000.0
    out = header.copy()
    if velocity[-1] < velocity[0]:
        step = velocity[1] - velocity[0]
        reference = velocity[0]
        flipped = data
    else:
        step = velocity[-2] - velocity[-1]
        reference = velocity[-1]
        flipped = data[::-1]
    out["CTYPE3"] = "VELO-HEL"
    out["CUNIT3"] = "m/s"
    out["CRVAL3"] = float(reference)
    out["CDELT3"] = float(step)
    out["CRPIX3"] = 1.0
    return (flipped, out)


def noise_map(data: np.ndarray, floor_percentile: float = 5.0) -> np.ndarray:
    """Per-sight-line noise from the median absolute deviation of the negatives.

    ROHSA's own estimator is the standard deviation over the line-free channels
    ``lstd`` to ``ustd``, and on these cubes it fails twice over.

    It is starved. Over the full disc the HI spans almost the whole band -- NGC
    3184's global profile is still at 26 per cent of its peak in the first and
    last channel -- so the search for line-free channels comes back with three.
    A standard deviation from three samples carries 50 per cent scatter per
    pixel by construction, and the measured scatter of that map is 55 per cent:
    it is not a noise map, it is noise. ROHSA divides the data term by it pixel
    by pixel, so that scatter becomes random per-pixel weighting in the cost
    function.

    It is also biased low -- 1.68 K against 2.86 K here -- because three
    adjacent channels of a Hanning-smoothed cube are correlated, and because a
    standard deviation over three samples is biased low anyway. Telling ROHSA
    the noise is a factor 1.7 smaller than it is inflates the data term against
    the regularisation, which is the wrong direction for a method whose whole
    contribution is spatial coherence.

    The negatives fix both. Emission is positive, so the negative half of the
    distribution is very nearly pure noise however much line fills the band,
    which leaves all ``nchan`` channels usable instead of three: the scatter
    falls to 23 per cent, and what remains is the real radial structure the
    primary-beam correction puts there. Folding about zero,

        sigma = 1.4826 * median(|x|) over x < 0,

    is the standard radio estimator and is what SoFiA's ``fluxRange: negative``
    uses on the same cubes.

    It assumes the noise has zero mean, which a residual negative bowl violates;
    where a bowl survives, this reads slightly high. That is the safe direction,
    and it is small -- 2.86 K against the 2.6 K measured from the faint sight
    lines of the band edges.
    """
    negative = np.where(data < 0, -data, np.nan)
    with np.errstate(invalid="ignore"):
        sigma = 1.4826 * np.nanmedian(negative, axis=0)
    sigma = np.asarray(sigma, dtype=float)
    good = np.isfinite(sigma) & (sigma > 0)
    if not good.any():
        raise ValueError("no sight line has a negative channel; cannot estimate the noise")
    sigma[~good] = float(np.percentile(sigma[good], floor_percentile))
    return sigma


def moment0(data: np.ndarray, header: fits.Header) -> np.ndarray:
    """Integrate a cube along velocity, giving a moment-0 map in K m/s.

    The phase cubes are analytic sums of the fitted Gaussians and carry no
    noise, so the integral is taken directly over every channel.  This is
    deliberately *not* done with a source finder: SoFiA and its relatives set
    their threshold from the measured noise, which is identically zero here, so
    source finding on a model cube is undefined.  ROHSA's fit has already done
    the job of separating signal from noise.

    The result is in K m/s, which is what :func:`surfaces.moment0_to_hi` wants.
    """
    return np.nansum(data, axis=0) * channel_width_kms(header) * 1000.0


NOISE_MARGIN_CHANNELS = 12


def trim_bad_channels(
    data: np.ndarray, header: fits.Header, fraction: float = 0.05
) -> tuple[np.ndarray, fits.Header, tuple[int, int]]:
    """Cut channels beyond the line whose global profile is strongly negative.

    NGC 5236's cube carries a band of deeply negative channels either side of
    the line -- its global profile reaches -6000 K where the line peaks at
    +10900 -- from imperfect continuum subtraction. ROHSA fits sums of positive
    Gaussians and cannot represent them: they leave residuals larger than the
    line itself, and no choice of hyper-parameters helps.

    The line itself (anything above a tenth of the peak) is never cut: negative
    sidelobes sit right against the line wings in interferometric data, and an
    earlier version of this that trimmed on the negative criterion alone took
    NGC 5055 from 46 channels down to 16, eating most of its line. Trimming only
    extends outwards from the line, keeping up to
    :data:`NOISE_MARGIN_CHANNELS` clean channels on each side so the noise
    estimate still has line-free channels to work with. A cube with no clean
    margin on either side is left alone, to be flagged rather than mangled.
    """
    profile = np.nansum(data, axis=(1, 2))
    peak_flux = np.nanmax(profile)
    if not np.isfinite(peak_flux) or peak_flux <= 0:
        return (data, header, (0, data.shape[0]))
    line = profile > 0.1 * peak_flux
    if not line.any():
        return (data, header, (0, data.shape[0]))
    first_line = int(np.argmax(line))
    last_line = len(line) - 1 - int(np.argmax(line[::-1]))
    clean = profile > -fraction * peak_flux
    low = first_line
    while low > 0 and clean[low - 1] and (first_line - low < NOISE_MARGIN_CHANNELS):
        low -= 1
    high = last_line
    while high < len(clean) - 1 and clean[high + 1] and (high - last_line < NOISE_MARGIN_CHANNELS):
        high += 1
    if first_line - low + (high - last_line) < 3 or (low, high + 1) == (0, data.shape[0]):
        return (data, header, (0, data.shape[0]))
    out = header.copy()
    out["CRPIX3"] = header["CRPIX3"] - low
    return (data[low : high + 1], out, (low, high + 1))


MIN_DETACHED_FRACTION = 0.05
MAX_DETACHED_FLUX_FRACTION = 0.15


def trim_detached_emission(
    data: np.ndarray, header: fits.Header
) -> tuple[np.ndarray, fits.Header, tuple[int, int]]:
    """Cut emission that sits beyond an empty channel, away from the line.

    IC 2574's cube carries a second emission feature centred at -50 km/s,
    peaking at 30 per cent of the galaxy's own peak and separated from it by a
    channel holding no flux at all. It is not IC 2574: the galaxy's rotation
    reaches only about -20 km/s from a systemic 48 km/s, and the feature is
    Galactic HI or a high-velocity cloud along the line of sight. ROHSA has no
    notion of which emission belongs to the galaxy, so it fits the feature with
    narrow Gaussians, and being genuinely cold and bright those dominate the
    dispersion PDF: the narrow phase came out holding 71 per cent of the flux
    against the 27 per cent of the thesis' own table.

    The cut is made at the bottom of the gap, so the line keeps its wings and
    whatever clean channels lie between it and the island stay available to the
    noise estimate. Emission that runs continuously into the line is never cut,
    however lopsided the profile: only an island beyond a channel that falls
    back to the baseline counts, which is what distinguishes a foreground source
    from a broad wing.
    """
    profile = np.nansum(data, axis=(1, 2))
    nchan = profile.size
    peak_flux = np.nanmax(profile)
    if not np.isfinite(peak_flux) or peak_flux <= 0:
        return (data, header, (0, nchan))
    floor = MAX_NOISE_CHANNEL_FRACTION * peak_flux
    quiet = np.sort(profile)[: max(int(round(0.1 * nchan)), 3)]
    baseline = np.median(quiet)
    mad = np.median(np.abs(quiet - baseline)) * 1.4826
    island_level = max(MIN_DETACHED_FRACTION * peak_flux, baseline + 5 * mad)
    peak_channel = int(np.nanargmax(profile))
    line_lo = peak_channel
    while line_lo > 0 and profile[line_lo - 1] > floor:
        line_lo -= 1
    line_hi = peak_channel
    while line_hi < nchan - 1 and profile[line_hi + 1] > floor:
        line_hi += 1
    (low, high) = (0, nchan - 1)
    above = profile > island_level
    outside = above & (np.r_[above[1:], False] | np.r_[False, above[:-1]])
    if line_lo > 0 and outside[:line_lo].any():
        island_end = int(np.max(np.flatnonzero(outside[:line_lo])))
        low = island_end + 1 + int(np.nanargmin(profile[island_end + 1 : line_lo]))
    if line_hi < nchan - 1 and outside[line_hi + 1 :].any():
        island_start = line_hi + 1 + int(np.min(np.flatnonzero(outside[line_hi + 1 :])))
        high = line_hi + 1 + int(np.nanargmin(profile[line_hi + 1 : island_start]))
    if (low, high) == (0, nchan - 1):
        return (data, header, (0, nchan))
    positive = np.where(profile > 0, profile, 0.0)
    total = positive.sum()
    kept = positive[low : high + 1].sum()
    if total <= 0 or (total - kept) / total > MAX_DETACHED_FLUX_FRACTION:
        return (data, header, (0, nchan))
    out = header.copy()
    out["CRPIX3"] = header["CRPIX3"] - low
    return (data[low : high + 1], out, (low, high + 1))


PEDESTAL_GUARD_KMS = 30.0
PEDESTAL_NSIGMA = 3.0


def subtract_negative_pedestal(
    data: np.ndarray,
    header: fits.Header,
    guard_kms: float = PEDESTAL_GUARD_KMS,
    nsigma: float = PEDESTAL_NSIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill in the negative bowl under each line of sight.

    An interferometer with no short spacings puts a negative pedestal around
    bright extended emission. NGC 2403 has a bad case: 30 per cent of the
    positive flux in its bright spectra is cancelled by negative flux, rising
    from 13 per cent at the centre to 35 per cent in the outskirts. ROHSA fits
    sums of positive Gaussians, so it cannot put anything there, and its model
    necessarily stands above the data -- the column densities of every one of
    its 64 tuning fits came out about 35 per cent high while the residual
    skewness, a shape statistic, stayed near zero and reported nothing wrong.

    Only pedestals that are negative at ``nsigma`` are touched, and only the
    negative part. Clipping an ungated estimate at zero looks harmless but is
    not: where the true baseline is zero, noise makes the estimate negative half
    the time, and subtracting those halves adds flux to every cube in the
    sample. With the gate, a clean galaxy is left alone -- DDO 53 comes through
    with no pixel corrected and its flux unchanged to three decimals -- while
    NGC 2403 has 54 per cent of its pixels corrected.

    This does not make NGC 2403 whole: the part of the bowl that sits at the
    line's own velocities cannot be separated from the line and stays behind,
    which is why its negative fraction falls to 19 rather than to a few per
    cent. Returns the corrected cube and the pedestal that was removed, so that
    what was done to a galaxy can be reported with its results.
    """
    dv = abs(channel_width_kms(header))
    guard = max(int(round(guard_kms / dv)), 3)
    nchan = data.shape[0]
    peak_channel = np.argmax(data, axis=0)
    channels = np.arange(nchan)[:, None, None]
    off_line = np.abs(channels - peak_channel[None]) > guard
    masked = np.where(off_line, data, np.nan)
    with np.errstate(all="ignore"):
        pedestal = np.nanmedian(masked, axis=0)
        scatter = np.nanmedian(np.abs(masked - pedestal[None]), axis=0) * 1.4826
    counted = np.maximum(off_line.sum(axis=0), 1)
    error = 1.2533 * scatter / np.sqrt(counted)
    pedestal = np.where(np.isfinite(pedestal), pedestal, 0.0)
    error = np.where(np.isfinite(error), error, np.inf)
    pedestal = np.where(pedestal < -nsigma * error, pedestal, 0.0)
    return (data - pedestal[None], pedestal)


def mask_bounding_box(mask_path, pad: int = 2):
    """The sky region a source mask occupies, as world coordinates.

    Returned in degrees as ``(ra_min, ra_max, dec_min, dec_max)`` rather than as
    pixel indices, because the caller's grid is not the mask's grid. The SoFiA
    masks were made on the cube binned for a 295 or 659 pc beam; a cube prepared
    at its native resolution is binned less and so is several times larger in
    pixels. Handing pixel indices across that change cuts a small off-centre
    patch instead of the galaxy -- NGC 2976's box, correct on its 204x204 grid at
    7.5 arcsec per pixel, selected a corner of the 512x512 native grid at 3.0
    arcsec per pixel, and every fit of it came back with a 30 to 65 per cent
    column-density error.

    ``pad`` is in pixels of the mask's own grid, converted to an angle here.
    """
    from astropy.wcs import WCS

    with fits.open(mask_path) as hdul:
        mask = np.squeeze(hdul[0].data) > 0
        wcs = WCS(hdul[0].header).celestial
    sky = mask.any(axis=0)
    (rows, cols) = np.where(sky)
    if rows.size == 0:
        raise ValueError(f"{mask_path}: the mask is empty")
    pad_deg = pad * abs(wcs.wcs.cdelt[1])
    corners_x = [cols.min(), cols.max(), cols.min(), cols.max()]
    corners_y = [rows.min(), rows.min(), rows.max(), rows.max()]
    (ra, dec) = wcs.all_pix2world(corners_x, corners_y, 0)
    return (
        float(np.min(ra) - pad_deg),
        float(np.max(ra) + pad_deg),
        float(np.min(dec) - pad_deg),
        float(np.max(dec) + pad_deg),
    )


def sky_box_to_pixels(header: fits.Header, sky_box) -> tuple[int, int, int, int]:
    """Where a sky region falls on this cube's grid, as ``(y0, y1, x0, x1)``.

    Clipped to the array, so a mask reaching past the edge of a field simply
    stops there.
    """
    from astropy.wcs import WCS

    (ra_min, ra_max, dec_min, dec_max) = sky_box
    wcs = WCS(header).celestial
    ra = [ra_min, ra_max, ra_min, ra_max]
    dec = [dec_min, dec_min, dec_max, dec_max]
    (x, y) = wcs.all_world2pix(ra, dec, 0)
    (ny, nx) = (header["NAXIS2"], header["NAXIS1"])
    return (
        int(max(np.floor(np.min(y)), 0)),
        int(min(np.ceil(np.max(y)), ny - 1)),
        int(max(np.floor(np.min(x)), 0)),
        int(min(np.ceil(np.max(x)), nx - 1)),
    )


def prepare(
    path,
    galaxy: Galaxy,
    binning: int | None = None,
    resolution_pc: float | None = None,
    crop_factor: float | None = RMAX_R25,
    crop_box: tuple[float, float, float, float] | None = None,
    native: bool = False,
) -> PreparedCube:
    """Run the full crop -> smooth -> rebin -> K conversion chain for one galaxy.

    With ``resolution_pc`` the cube is smoothed to the beam that subtends that
    physical scale at the galaxy's distance, so every galaxy in a run is
    averaged over the same region; without it the fixed angular
    :data:`COMMON_BEAM_ARCSEC` is used, as in the first version of the pipeline.

    ``binning`` defaults to :func:`choose_binning`; pass an integer to override.

    ``crop_box`` is a sky region in degrees, ``(ra_min, ra_max, dec_min, dec_max)``
    as :func:`mask_bounding_box` returns it. It is converted to pixels on this
    cube's own grid, so the same box works whatever resolution the cube is
    prepared at.

    ``crop_factor`` is the multiple of r25 the field is cut to, and ``None``
    keeps the whole THINGS field. The default exists because the star formation
    and CO maps only cover the optical disc, so there is nothing to compare the
    gas with further out. A study of the cold and warm gas alone has no such
    limit -- HI discs run well past r25, and the outer disc is where the warm
    phase dominates -- so that analysis should be given the full field rather
    than a box drawn for a different question.
    """
    (data, header) = load_things_cube(path)
    (data, header) = normalise_spectral_axis(data, header)
    if crop_factor is not None:
        (data, header) = crop_to_r25(data, header, galaxy, factor=crop_factor)
        (data, header, kept) = trim_bad_channels(data, header)
        (data, header, _) = trim_detached_emission(data, header)
    else:
        (inner, inner_header) = crop_to_r25(data, header, galaxy)
        (_, _, kept_bad) = trim_bad_channels(inner, inner_header)
        inner = inner[kept_bad[0] : kept_bad[1]]
        inner_header = inner_header.copy()
        inner_header["CRPIX3"] = inner_header["CRPIX3"] - kept_bad[0]
        (_, _, kept_island) = trim_detached_emission(inner, inner_header)
        low = kept_bad[0] + kept_island[0]
        high = kept_bad[0] + kept_island[1]
        header = header.copy()
        header["CRPIX3"] = header["CRPIX3"] - low
        data = data[low:high]
        kept = (low, high)
    header = header.copy()
    header["NAXIS3"] = data.shape[0]
    (data, header, _) = jy_per_beam_to_kelvin(data, header)
    if native:
        beam_arcsec = native_beam_arcsec(header)
    else:
        beam_arcsec = (
            COMMON_BEAM_ARCSEC if resolution_pc is None else target_beam_arcsec(galaxy, resolution_pc)
        )
        (data, header) = convolve_to_common_beam(data, header, beam_arcsec)
    if binning is None:
        binning = choose_binning(header, data.shape[-1], beam_arcsec=beam_arcsec)
    (data, header) = rebin_spatial(data, header, binning)
    if crop_box is not None:
        (y0, y1, x0, x1) = sky_box_to_pixels(header, crop_box)
        data = data[:, y0 : y1 + 1, x0 : x1 + 1]
        header = header.copy()
        header["CRPIX1"] = header["CRPIX1"] - x0
        header["CRPIX2"] = header["CRPIX2"] - y0
        (header["NAXIS1"], header["NAXIS2"]) = (data.shape[2], data.shape[1])
    (data, pedestal) = subtract_negative_pedestal(data, header)
    return PreparedCube(
        data=data,
        header=header,
        spatial_binning=binning,
        noise_channels=find_noise_channels(data),
        pedestal_pixels_corrected=float((pedestal < 0).mean()),
    )
