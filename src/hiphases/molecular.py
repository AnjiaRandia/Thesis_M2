"""Native-beam matching and cell-block statistics for the CO association comparison."""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, next_fast_len
from astropy.wcs import WCS
from hiphases import surfaces


def matched(image, header, target):
    """Gaussian beam covariance difference, including ellipticity and PA.

    Apply the Gaussian transfer function directly, allowing a zero kernel
    eigenvalue when the target equals an input major beam. Pad to avoid wrap.
    Interpolate missing data by normalising with the finite-pixel response.
    """
    major = surfaces.native_beam_arcsec(header)
    minor = header.get("BMIN", major / 3600) * 3600
    pa = np.deg2rad(header.get("BPA", 0))
    u = np.array([np.sin(pa), np.cos(pa)])
    v = np.array([np.cos(pa), -np.sin(pa)])
    native = (major**2 * np.outer(u, u) + minor**2 * np.outer(v, v)) / (8 * np.log(2))
    kernel = target**2 * np.eye(2) / (8 * np.log(2)) - native
    eigen = np.linalg.eigvalsh(kernel)
    assert eigen.min() > -1e-05, (major, minor, target, eigen)
    matrix = WCS(header).celestial.pixel_scale_matrix * 3600
    inv = np.linalg.inv(matrix)
    cov = inv @ kernel @ inv.T
    cov = (cov + cov.T) / 2
    eig = np.linalg.eigvalsh(cov)
    data = np.squeeze(np.asarray(image, float))
    valid = np.isfinite(data)
    if eig.max() < 1e-10:
        return (data.copy(), valid.astype(float), cov)
    pad = max(8, int(np.ceil(6 * np.sqrt(max(eig.max(), 0)))))
    shape = tuple((next_fast_len(n + 2 * pad) for n in data.shape))
    numerator = np.zeros(shape)
    denominator = np.zeros(shape)
    sl = (slice(pad, pad + data.shape[0]), slice(pad, pad + data.shape[1]))
    numerator[sl] = np.where(valid, data, 0)
    denominator[sl] = valid
    ky = fftfreq(shape[0])[:, None]
    kx = fftfreq(shape[1])[None, :]
    transfer = np.exp(
        -2 * np.pi**2 * np.maximum(cov[0, 0] * kx * kx + 2 * cov[0, 1] * kx * ky + cov[1, 1] * ky * ky, 0)
    )
    smooth = ifft2(fft2(numerator) * transfer).real[sl]
    weight = ifft2(fft2(denominator) * transfer).real[sl]
    result = np.divide(smooth, weight, out=np.full(data.shape, np.nan), where=weight > 0.01)
    return (result, np.clip(weight, 0, 1), cov)


def describe(p, meta):
    out = dict(
        **meta, n_pairs=len(p), n_cells=p[["cell_x", "cell_y"]].drop_duplicates().shape[0] if len(p) else 0
    )
    if p.empty:
        return out
    out.update(
        bright_area_pc2=float(len(p) * p.beam_area_pc2.iloc[0]),
        faint_area_pc2=float(len(p) * p.beam_area_pc2.iloc[0]),
        median_bright_co=float(p.bright_co.median()),
        median_faint_co=float(p.faint_co.median()),
        median_separation_beams=float(p.separation_beams.median()),
        median_radius_difference_beams=float(p.radius_difference_beams.median()),
        n_faint_nonpositive=int((p.faint_co <= 0).sum()),
    )
    for key in METRICS:
        q = p[np.isfinite(p[key])]
        out[key + "_n_valid"] = len(q)
        out[key + "_mean"] = float(q[key].mean())
        out[key + "_median"] = float(q[key].median())
        out[key + "_p025"] = out[key + "_p975"] = np.nan
        if len(q):
            groups = q.groupby(["cell_x", "cell_y"])
            n = groups.size().to_numpy()
            v = groups[key].sum().to_numpy()
            if len(n) >= 6:
                rng = np.random.default_rng(990)
                ii = rng.integers(len(n), size=(4000, len(n)))
                draw = v[ii].sum(1) / n[ii].sum(1)
                (out[key + "_p025"], out[key + "_p975"]) = np.percentile(draw, [2.5, 97.5])
    out["status"] = "local cell-block intervals" if out["n_cells"] >= 6 else "few cells; descriptive only"
    return out


METRICS = ["delta_log_narrow", "delta_log_broad", "delta_log_hi", "delta_fn"]
