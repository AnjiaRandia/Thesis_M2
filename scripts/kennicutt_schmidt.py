#!/usr/bin/env python3
"""Fit and plot the retained gas–SFR relations on explicitly requested cached maps."""

import _setup
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from hiphases import resolvedks
from hiphases.sample import load_galaxy
from hiphases.ks_plotting import density_cloud, depletion_lines, XLIM, YLIM


def panel(ax, maps, header, name, tracer, output):
    galaxy = load_galaxy(name)
    gas, sfr = maps[tracer], maps["sfr"]
    geometry = resolvedks.geometry(header, gas.shape, galaxy)
    record, sampled = resolvedks.fit_samples(gas, sfr, geometry, posterior_path=output)
    all_pixels = np.isfinite(gas) & np.isfinite(sfr) & (gas > 0) & (sfr > 0)
    density_cloud(ax, gas, sfr, all_pixels)
    depletion_lines(ax)
    ax.plot(
        np.log10(gas[sampled]),
        np.log10(sfr[sampled]),
        "o",
        color="#0072B2",
        markersize=3.2,
        markeredgewidth=0,
        zorder=3,
    )
    if record["status"] == "ok":
        line = resolvedks.fit_segment(record, XLIM, YLIM)
        ax.plot(line[:, 0], line[:, 1], "--", color="black", lw=1.4)
    ax.text(
        0.05,
        0.94,
        resolvedks.fit_label(record, show_angle=tracer != "h2"),
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.set(xlim=XLIM, ylim=YLIM)
    return dict(galaxy=name, tracer=tracer, **record)


def individual(name):
    root = _setup.PROJECT_ROOT
    source = root / "cache/maps/individual" / name
    destination = root / "results/ks/individual" / name
    destination.mkdir(parents=True, exist_ok=True)
    galaxy = load_galaxy(name)
    header = fits.getheader(source / "hi.fits")
    keys = ["hi", "narrow", "broad"] + (["h2"] if galaxy.has_co else [])
    maps = {key: fits.getdata(source / f"{key}.fits").astype(float) for key in keys + ["sfr"]}
    fig, axes = plt.subplots(
        1, len(keys), figsize=(2.35 * len(keys), 3.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0}
    )
    labels = dict(
        hi="Total H I", narrow="Kinematically cold H I", broad="Kinematically warm H I", h2=r"H$_2$"
    )
    rows = []
    for ax, key in zip(np.atleast_1d(axes), keys):
        rows.append(panel(ax, maps, header, name, key, destination / f"{key}_bootstrap.npz"))
        ax.set_title(labels[key], fontsize=12, pad=14)
        ax.tick_params(pad=7)
        ax.label_outer()
    fig.supxlabel(r"$\log\Sigma_{\rm gas}$ [$M_\odot$ pc$^{-2}$]")
    fig.supylabel(r"$\log\Sigma_{\rm SFR}$ [$M_\odot$ yr$^{-1}$ kpc$^{-2}$]")
    fig.savefig(destination / "relations.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(destination / "fits.csv", index=False)


def combined():
    from hiphases.densityks import fit_density_orthogonal

    root = _setup.PROJECT_ROOT
    destination = root / "results/ks/combined"
    destination.mkdir(parents=True, exist_ok=True)
    samples = []
    for source in sorted((root / "cache/maps/combined").iterdir()):
        galaxy = load_galaxy(source.name)
        header = fits.getheader(source / "hi.fits")
        sfr = fits.getdata(source / "sfr.fits").astype(float)
        geo = resolvedks.geometry(header, sfr.shape, galaxy)
        for key in ["hi", "narrow", "broad"] + (["h2"] if galaxy.has_co else []):
            gas = fits.getdata(source / f"{key}.fits").astype(float)
            keep, _ = resolvedks.sample_masks(gas, sfr, geo)
            samples.extend(
                dict(galaxy=galaxy.name, group=galaxy.galaxy_class, tracer=key, x=float(x), y=float(y))
                for x, y in zip(np.log10(gas[keep]), np.log10(sfr[keep]))
            )
    measurements = pd.DataFrame(samples)
    rows = []
    for group in ["dwarf", "spiral", "all"]:
        selected = measurements if group == "all" else measurements[measurements.group == group]
        for key, data in selected.groupby("tracer"):
            result = fit_density_orthogonal(
                data.x.to_numpy(), data.y.to_numpy(), destination / f"{group}_{key}_bootstrap.npz"
            )
            result.update(n=len(data), r=float(np.corrcoef(data.x, data.y)[0, 1]), group=group, tracer=key)
            rows.append(result)
    measurements.to_csv(destination / "samples.csv", index=False)
    pd.DataFrame(rows).to_csv(destination / "fits.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("galaxy", nargs="?")
    parser.add_argument("--combined", action="store_true")
    args = parser.parse_args()
    if args.combined:
        combined()
    elif args.galaxy:
        individual(args.galaxy)
    else:
        parser.error("Specify a galaxy or --combined.")
