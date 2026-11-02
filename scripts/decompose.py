#!/usr/bin/env python3
"""Explicit preparation, ROHSA fitting and local classification stages."""

import _setup  # noqa: F401
import argparse
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from hiphases import cubes, decomposition, surfaces, resolvedks
from hiphases.paths import ROOT, artifact
from hiphases.sample import load_galaxy
from hiphases.phase_cuts import cell_grids, radial_fallback, smooth_filling_gaps


def configuration(name):
    return json.loads((ROOT / "config/rohsa.json").read_text())[name]


def prepare(name):
    galaxy = load_galaxy(name)
    cfg = configuration(name)
    mask = ROOT / "data/support/sofia_masks" / f"{name}.fits"
    raw = ROOT / "data/raw/things" / f"{galaxy.things_stem}.FITS"
    cube = cubes.prepare(raw, galaxy, native=True, crop_factor=None, crop_box=cubes.mask_bounding_box(mask))
    cube.header["BMAJ"] = cubes.native_beam_arcsec(cube.header) / 3600.0
    destination = ROOT / "results/native/inputs" / name
    destination.mkdir(parents=True, exist_ok=True)
    cube.write(destination / "prepared_cube.fits")
    spatial = WCS(cube.header).celestial.to_header()
    mask_data, mask_header = fits.getdata(mask, header=True)
    coverage = surfaces.reproject_onto(
        np.any(mask_data > 0, axis=0).astype(float),
        mask_header,
        WCS(cube.header).celestial,
        cube.data.shape[1:],
    )
    fits.writeto(destination / "source_mask.fits", (coverage >= 0.5).astype("uint8"), spatial, overwrite=True)
    fits.writeto(
        destination / "noise_K.fits", cubes.noise_map(cube.data).astype("float32"), spatial, overwrite=True
    )
    info = dict(
        galaxy=name,
        n_gauss=cfg["n_gauss"],
        reference={"n_gauss": cfg["n_gauss"]},
        spatial_binning=cube.spatial_binning,
        noise_channels=list(cube.noise_channels),
        channel_width_kms=cubes.channel_width_kms(cube.header),
    )
    (destination / "input.json").write_text(json.dumps(info, indent=2) + "\n")


def fit(name):
    cfg = configuration(name)
    source = artifact(f"native/inputs/{name}")
    info = json.loads((source / "input.json").read_text())
    image, header = fits.getdata(source / "prepared_cube.fits", header=True)
    dv = info["channel_width_kms"]
    floor = cfg["minimum_fwhm_channels"] * dv / 2.355
    cube = cubes.PreparedCube(
        image.astype(float), header, info["spatial_binning"], tuple(info["noise_channels"])
    )
    settings = decomposition.RohsaSettings(
        n_gauss=cfg["n_gauss"],
        lambda_amp=0.0,
        lambda_mu=1.0,
        lambda_sig=1.0,
        lambda_var_sig=0.0,
        lb_sig_kms=floor,
        ub_sig_kms=40.0,
        lb_sig_init_kms=floor,
        sig_init_kms=max(8.0, 1.5 * floor),
        ub_sig_init_kms=max(20.0, 3 * floor),
        noise_from_negatives=True,
    )
    destination = ROOT / "results/native/fits" / name / f"n{cfg['n_gauss']}_local1"
    destination.mkdir(parents=True, exist_ok=True)
    fitted = decomposition.run(cube, destination / "work", settings)
    np.savez_compressed(
        destination / "parameters.npz",
        amplitude=fitted.amplitude.astype("float32"),
        centroid=fitted.centroid.astype("float32"),
        dispersion=fitted.dispersion.astype("float32"),
        model=fitted.model.astype("float32"),
    )
    integrals = np.zeros_like(fitted.dispersion, dtype=float)
    for component in range(cfg["n_gauss"]):
        for start in range(0, image.shape[0], 8):
            channel = np.arange(start, min(start + 8, image.shape[0]))[:, None, None]
            integrals[component] += (
                fitted.amplitude[component][None]
                * np.exp(
                    -0.5
                    * (
                        (channel - fitted.centroid[component][None])
                        / np.maximum(fitted.dispersion[component][None], 1.0e-10)
                    )
                    ** 2
                )
            ).sum(0)
    np.save(destination / "component_integrals.npy", integrals.astype("float32"))
    (destination / "diagnostics.json").write_text(json.dumps(dict(lb_sig_kms=floor, ub_sig_kms=40.0)) + "\n")


def classify(name):
    cfg = configuration(name)
    source = artifact(f"native/inputs/{name}")
    parameters = artifact(f"native/fits/{name}/n{cfg['n_gauss']}_local1")
    info = json.loads((source / "input.json").read_text())
    store = np.load(parameters / "parameters.npz")
    flux = np.load(parameters / "component_integrals.npy").astype(float)
    sigma = store["dispersion"].astype(float)
    amplitude = store["amplitude"].astype(float)
    order = np.argsort(sigma, axis=0)
    sigma, amplitude, flux = [np.take_along_axis(v, order, axis=0) for v in [sigma, amplitude, flux]]
    data, header = fits.getdata(source / "prepared_cube.fits", header=True)
    source_mask = fits.getdata(source / "source_mask.fits").astype(bool)
    noise = fits.getdata(source / "noise_K.fits").astype(float)
    dv = info["channel_width_kms"]
    bounds = json.loads((parameters / "diagnostics.json").read_text())
    lower, upper = max(bounds["lb_sig_kms"] / dv, 1.0), max(bounds["ub_sig_kms"] / dv, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = amplitude / noise[None] * np.sqrt(np.maximum(sigma, 0.0) * np.sqrt(np.pi))
        residual = (
            np.sqrt(np.nanmean((data.astype(float) - store["model"].astype(float)) ** 2, axis=0)) / noise
        )
    measured = (
        np.isfinite(sigma)
        & (flux > 0)
        & (snr >= 3.0)
        & np.isfinite(residual)[None]
        & (residual[None] <= 2.0)
        & (sigma > lower * 1.001)
        & (sigma < upper * 0.999)
        & (sigma >= 3.0 / 2.355)
    )
    geo = resolvedks.geometry(header, source_mask.shape, load_galaxy(name))
    sample = dict(
        galaxy=name,
        header=header,
        sigma=sigma * dv,
        flux=flux,
        measured=measured,
        source=source_mask,
        radius=geo["radius"] / geo["r25_arcsec"],
        floor_kms=3.0 * dv / 2.355,
    )
    narrow, broad, step = cell_grids(sample)
    fallback_n, fallback_b = radial_fallback(sample, narrow, broad)
    if fallback_n is not None:
        narrow = np.where(np.isfinite(narrow), narrow, fallback_n)
        broad = np.where(np.isfinite(broad), broad, fallback_b)
    narrow = smooth_filling_gaps(narrow, step / 2.355)
    broad = smooth_filling_gaps(broad, step / 2.355)
    cut = np.where(source_mask, np.maximum(np.sqrt(narrow * broad), sample["floor_kms"]), np.nan)
    destination = ROOT / "results/native/position_cuts"
    destination.mkdir(parents=True, exist_ok=True)
    spatial = WCS(header).celestial.to_header()
    spatial["BMAJ"], spatial["BMIN"] = header["BMAJ"], header["BMIN"]
    fits.writeto(destination / f"{name}_position_cut.fits", cut.astype("float32"), spatial, overwrite=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "fit", "classify"])
    parser.add_argument("galaxy")
    arguments = parser.parse_args()
    {"prepare": prepare, "fit": fit, "classify": classify}[arguments.stage](arguments.galaxy)
