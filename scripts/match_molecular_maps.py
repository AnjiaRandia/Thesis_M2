"""Match the native HI components and original CO maps to their common beam."""

import json
import _setup
from hiphases.molecular import matched
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from hiphases import surfaces, resolvedks
from hiphases.sample import load_galaxy
from hiphases.paths import co_path as _co_path

ROOT = _setup.PROJECT_ROOT
BASE = ROOT / "cache/native"
OUT = ROOT / "results/molecular"
NAMES = ["NGC0628", "NGC2903", "NGC2976", "NGC3184", "NGC3351", "NGC4736", "NGC5194", "NGC6946"]


def build(name):
    g = load_galaxy(name)
    folder = BASE / "inputs" / name
    info = json.loads((folder / "input.json").read_text())
    h = fits.getheader(folder / "prepared_cube.fits")
    coh = fits.getheader(_co_path(g))
    co = np.squeeze(fits.getdata(_co_path(g))).astype(float)
    hi_beam = info["native_beam_arcsec"]
    co_beam = surfaces.native_beam_arcsec(coh)
    target = max(hi_beam, co_beam)
    dest = OUT / "maps" / name
    dest.mkdir(parents=True, exist_ok=True)
    n = info["reference"]["n_gauss"]
    params = BASE / "fits" / name / f"n{n}_local1"
    sigma = np.load(params / "parameters.npz")["dispersion"].astype(float) * info["channel_width_kms"]
    flux = np.load(params / "component_integrals.npy").astype(float)
    cut = fits.getdata(BASE / "position_cuts" / f"{name}_position_cut.fits")
    narrow = np.where(sigma < cut[None], flux, 0).sum(0)
    broad = flux.sum(0) - narrow
    maps = {}
    covariance = None
    for key, image in [("narrow", narrow), ("broad", broad)]:
        surface = surfaces.moment0_to_hi(image * info["channel_width_kms"] * 1000, g)
        (smooth, coverage, covariance) = matched(surface, h, target)
        maps[key] = np.maximum(smooth, 0)
    maps["hi"] = maps["narrow"] + maps["broad"]
    (signed, co_cover, co_covariance) = matched(co, coh, target)
    good = (co_cover >= 0.95) & np.isfinite(signed)
    floor = 64 * np.finfo(float).eps * np.nanmax(abs(signed))
    negative = -signed[good & (signed < -floor)]
    assert len(negative) >= 64
    sigma50 = np.median(negative) / 0.67448975
    sigma84 = np.percentile(negative, 84) / 1.40507156
    noise = float(max(sigma50, sigma84))
    wcs = WCS(h).celestial
    shape = maps["hi"].shape
    signed = surfaces.reproject_onto(signed, coh, wcs, shape)
    coverage_co = surfaces.reproject_onto(co_cover, coh, wcs, shape)
    source = fits.getdata(folder / "source_mask.fits") > 0
    valid = source & (coverage_co >= 0.95) & np.isfinite(cut) & resolvedks.positive_measurements(maps["hi"])
    maps.update(
        h2=np.maximum(signed, 0) * surfaces.H2_COEFFICIENTS[g.co_line] * g.axis_ratio,
        co_signed=signed,
        co_background_ratio=signed / noise,
        co_coverage=coverage_co,
        valid_coverage=valid.astype(float),
    )
    cube = fits.getdata(folder / "prepared_cube.fits").astype(float)
    observed = surfaces.moment0_to_hi(np.nansum(cube, axis=0) * info["channel_width_kms"] * 1000, g)
    (maps["observed_hi"], _, _) = matched(observed, h, target)
    header = wcs.to_header()
    header["BMAJ"] = header["BMIN"] = target / 3600
    header["RESPC"] = np.deg2rad(target / 3600) * g.distance.to_value("pc")
    for key, image in maps.items():
        oh = header.copy()
        oh["BUNIT"] = (
            "K km/s"
            if key == "co_signed"
            else "1"
            if key in ["co_background_ratio", "co_coverage", "valid_coverage"]
            else "Msun pc-2"
        )
        fits.writeto(dest / f"{key}.fits", np.asarray(image, float), oh, overwrite=True)
    note = dict(
        galaxy=name,
        resolution_pc=header["RESPC"],
        native_hi_pc=info["native_resolution_pc"],
        hi_beam_arcsec=hi_beam,
        hi_minor_arcsec=info["native_beam_minor_arcsec"],
        co_beam_arcsec=co_beam,
        target_beam_arcsec=target,
        co_background_sigma=noise,
        n_negative_background=len(negative),
        hi_kernel_covariance_pix2=covariance.tolist(),
        co_kernel_covariance_pix2=co_covariance.tolist(),
        n_source_pixels=int(source.sum()),
        n_valid_pixels=int(valid.sum()),
        n_zero_cold=int((valid & (maps["narrow"] <= 0)).sum()),
    )
    (dest / "build.json").write_text(json.dumps(note, indent=2))
    return note


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--galaxies", nargs="+", choices=NAMES, default=NAMES)
    args = parser.parse_args()
    for name in args.galaxies:
        build(name)
