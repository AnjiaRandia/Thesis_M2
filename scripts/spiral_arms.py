"""Native-resolution stellar-arm analysis using the adopted component maps."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse, json, warnings
import _setup
from hiphases.spiral_arms import (
    angular_coordinates,
    area_comparison,
    design,
    robust_fit,
    suppress_compact_sources,
)
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_area
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
from matplotlib.backends.backend_pdf import PdfPages
from hiphases import surfaces, resolvedks
from hiphases.sample import load_galaxy

ROOT = _setup.PROJECT_ROOT
CAMPAIGN = ROOT / "cache/native"
OUT = ROOT / "results/arms"
REFIT = CAMPAIGN / "fits"
CANDIDATES = [
    "NGC0628",
    "NGC0925",
    "NGC2403",
    "NGC2903",
    "NGC2976",
    "NGC3184",
    "NGC3351",
    "NGC4736",
    "NGC5194",
    "NGC6946",
    "NGC7793",
]
warnings.filterwarnings("ignore", category=FITSFixedWarning)


def context(name):
    folder = CAMPAIGN / "inputs" / name
    info = json.loads((folder / "input.json").read_text())
    h = fits.getheader(folder / "prepared_cube.fits")
    shape = tuple(info["shape"][1:])
    g = load_galaxy(name)
    geo = resolvedks.geometry(h, shape, g)
    support = fits.getdata(folder / "source_mask.fits") > 0
    return (info, h, shape, g, geo, support)


def prepare():
    (OUT / "morphology").mkdir(parents=True, exist_ok=True)
    records = []
    with PdfPages(OUT / "morphology_screening.pdf") as pdf:
        (contact, contact_axes) = plt.subplots(3, 4, figsize=(15, 11), layout="constrained")
        for index, name in enumerate(CANDIDATES):
            (info, h, shape, g, geo, support) = context(name)
            w = WCS(h).celestial
            (raw, rh) = fits.getdata(ROOT / "data/raw/stellar_mass" / f"{g.stellar_stem}.fits", header=True)
            (raw, point_mask, n_points) = suppress_compact_sources(raw, rh)
            fits.writeto(
                OUT / "morphology" / f"{name}_compact_source_mask_raw.fits",
                point_mask.astype("uint8"),
                rh,
                overwrite=True,
            )
            mh = rh.copy()
            mh["BMAJ"] = surfaces.DEFAULT_IRAC_BEAM_ARCSEC / 3600
            mh["BMIN"] = mh["BMAJ"]
            (star, _, reached) = surfaces.convolve_map_to_beam(raw, mh, geo["beam_arcsec"])
            assert reached
            star = surfaces.reproject_onto(star, mh, w, shape)
            theta = angular_coordinates(h, shape, g)
            ring = np.floor(geo["radius"] / (0.5 * geo["beam_arcsec"])).astype(int)
            radial = (geo["radius"] >= 0.15 * geo["r25_arcsec"]) & (geo["radius"] < 0.95 * geo["r25_arcsec"])
            domain = support & radial & np.isfinite(star) & (star > 0)
            score = np.full(shape, np.nan)
            m0 = np.full(shape, np.nan)
            outliers = []
            for k in np.unique(ring[domain]):
                rc = (k + 0.5) * 0.5 * geo["beam_arcsec"]
                fitting = (
                    (abs(geo["radius"] - rc) < 0.5 * geo["beam_arcsec"]) & np.isfinite(star) & (star > 0)
                )
                if fitting.sum() < 30:
                    continue
                (coef, discard) = robust_fit(design(theta[fitting]), star[fitting])
                outliers.append(discard)
                if coef[0] <= 0:
                    continue
                target = domain & (ring == k)
                score[target] = design(theta[target]) @ coef / coef[0]
                m0[target] = coef[0]
            for key, image in [("stellar", star), ("score", score), ("azimuthal_mean", m0)]:
                fits.writeto(
                    OUT / "morphology" / f"{name}_{key}.fits", image.astype("float32"), h, overwrite=True
                )
            valid = domain & np.isfinite(score)
            stretch = np.percentile(star[valid], 99)
            (fig, axes) = plt.subplots(1, 3, figsize=(13, 4.7), layout="constrained")
            views = [
                (
                    star,
                    "Observed stellar morphology",
                    dict(
                        cmap="gray_r",
                        norm=AsinhNorm(linear_width=max(0.03 * stretch, 1e-10), vmin=0, vmax=stretch),
                    ),
                ),
                (
                    np.divide(star, m0, out=np.full(shape, np.nan), where=m0 > 0),
                    "Observed / azimuthal mean",
                    dict(cmap="magma", vmin=0.5, vmax=1.8),
                ),
                (score, "Robust Fourier stellar template", dict(cmap="magma", vmin=0.7, vmax=1.4)),
            ]
            for ax, (a, title, kw) in zip(axes, views):
                image = ax.imshow(np.where(radial, a, np.nan), origin="lower", **kw)
                for r0 in [0.2, 0.4, 0.8]:
                    ax.contour(
                        geo["radius"] / geo["r25_arcsec"],
                        levels=[r0],
                        colors=["cyan"],
                        linewidths=0.5,
                        alpha=0.7,
                    )
                (yy, xx) = np.where(radial & support)
                ax.set_xlim(xx.min() - 2, xx.max() + 2)
                ax.set_ylim(yy.min() - 2, yy.max() + 2)
                ax.set_title(title, fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(image, ax=ax, shrink=0.7)
            fig.suptitle(
                f"{name}: native HI beam {info['native_resolution_pc']:.0f} pc; sample class={g.galaxy_class} is a mass classification\nMorphology screening only: no cold/warm fractions used. Cyan radii: 0.2, 0.4, 0.8 r25.",
                fontsize=11,
            )
            pdf.savefig(fig)
            fig.savefig(OUT / "morphology" / f"{name}_screen.png", dpi=130)
            plt.close(fig)
            ax = contact_axes.ravel()[index]
            ax.imshow(
                np.where(radial, star, np.nan),
                origin="lower",
                cmap="gray_r",
                norm=AsinhNorm(linear_width=max(0.03 * stretch, 1e-10), vmin=0, vmax=stretch),
            )
            (yy, xx) = np.where(radial & support)
            ax.set_xlim(xx.min() - 2, xx.max() + 2)
            ax.set_ylim(yy.min() - 2, yy.max() + 2)
            ax.set_title(f"{name} ({info['native_resolution_pc']:.0f} pc)", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            records.append(
                dict(
                    galaxy=name,
                    sample_class=g.galaxy_class,
                    hubble_T=g.hubble_T,
                    native_resolution_pc=info["native_resolution_pc"],
                    native_beam_major_arcsec=info["native_beam_arcsec"],
                    native_beam_minor_arcsec=info["native_beam_minor_arcsec"],
                    compact_sources_suppressed=n_points,
                    raw_point_mask_area_fraction=float(np.mean(point_mask)),
                    fit_outlier_fraction=float(np.mean(outliers)),
                    score_p10=float(np.nanpercentile(score, 10)),
                    score_p90=float(np.nanpercentile(score, 90)),
                )
            )
            print(name, "morphology ready", flush=True)
        contact_axes.ravel()[-1].axis("off")
        contact.suptitle(
            "Native-resolution stellar morphology: candidate spiral galaxies (arms not assumed from mass class)",
            fontsize=14,
        )
        contact.savefig(OUT / "morphology_contact.png", dpi=140)
        contact.savefig(OUT / "morphology_contact.pdf")
        plt.close(contact)
    pd.DataFrame(records).to_csv(OUT / "morphology_screening.csv", index=False)


def analyse():
    selection = json.loads((ROOT / "config/arm_selection.json").read_text())
    rows = []
    (OUT / "maps").mkdir(exist_ok=True)
    selected = [r for r in selection if r["include"]]
    with PdfPages(OUT / "native_arm_interarm_atlas.pdf") as pdf:
        for entry in selected:
            name = entry["galaxy"]
            (info, h, shape, g, geo, support) = context(name)
            count = info["reference"]["n_gauss"]
            folder = REFIT / name / f"n{count}_local1"
            z = np.load(folder / "parameters.npz")
            width = z["dispersion"] * info["channel_width_kms"]
            flux = np.load(folder / "component_integrals.npy").astype(float)
            cut = fits.getdata(CAMPAIGN / "position_cuts" / f"{name}_position_cut.fits")
            cold = np.where(width < cut[None], flux, 0).sum(0)
            warm = np.where(width < cut[None], 0, flux).sum(0)
            assert np.allclose(cold + warm, flux.sum(0), rtol=1e-12, atol=1e-10)
            total = cold + warm
            fraction = np.divide(cold, total, out=np.full(shape, np.nan), where=total > 0)
            score = fits.getdata(OUT / "morphology" / f"{name}_score.fits")
            star = fits.getdata(OUT / "morphology" / f"{name}_stellar.fits")
            ring = np.floor(geo["radius"] / (0.5 * geo["beam_arcsec"])).astype(int)
            domain = (
                support
                & np.isfinite(score)
                & np.isfinite(cut)
                & np.isfinite(fraction)
                & (total > 0)
                & (geo["radius"] >= entry["rmin"] * geo["r25_arcsec"])
                & (geo["radius"] < entry["rmax"] * geo["r25_arcsec"])
            )
            pixel_area = (
                float(proj_plane_pixel_area(WCS(h).celestial))
                * (np.pi / 180) ** 2
                * g.distance.to_value("pc") ** 2
                / g.axis_ratio
            )
            block = max(1, int(np.ceil(4 * geo["beam_arcsec"] / geo["pixel_arcsec"])))
            for q in [0.2, 0.25, 0.3, 0.4]:
                arm = np.zeros(shape, dtype=bool)
                inter = arm.copy()
                for k in np.unique(ring[domain]):
                    ix = np.flatnonzero(domain & (ring == k))
                    order = ix[np.argsort(score.flat[ix])]
                    n = int(np.floor(q * len(order)))
                    if n < 2:
                        continue
                    arm.flat[order[-n:]] = True
                    inter.flat[order[:n]] = True
                assert arm.sum() == inter.sum() and arm.any() and (not np.any(arm & inter))
                result = area_comparison(cold, warm, arm, inter, pixel_area, block)
                if q == 0.25:
                    for beams in [2, 3]:
                        check = area_comparison(
                            cold,
                            warm,
                            arm,
                            inter,
                            pixel_area,
                            max(1, int(np.ceil(beams * geo["beam_arcsec"] / geo["pixel_arcsec"]))),
                        )
                        assert np.isclose(check["contrast"], result["contrast"])
                        for key in ["ci_low", "ci_high", "n_spatial_blocks"]:
                            result[f"{key}_block{beams}"] = check[key]
                assert np.isclose(result["arm_area_pc2"], result["interarm_area_pc2"])
                assert np.isclose(result["cold_fraction_arm"], np.mean(fraction[arm]))
                rows.append(
                    dict(
                        galaxy=name,
                        area_fraction_per_ring=q,
                        native_resolution_pc=info["native_resolution_pc"],
                        sample_class=g.galaxy_class,
                        arm_morphology=entry["morphology"],
                        arm_mask_quality=entry.get("mask_quality", "not_recorded"),
                        rmin_r25=entry["rmin"],
                        rmax_r25=entry["rmax"],
                        **result,
                    )
                )
                if q == 0.25:
                    label = np.where(arm, 1, np.where(inter, -1, np.nan))
                    for key, a in [("cold_fraction", fraction), ("arm_interarm", label)]:
                        fits.writeto(
                            OUT / "maps" / f"{name}_{key}.fits", a.astype("float32"), h, overwrite=True
                        )
                    (fig, axs) = plt.subplots(1, 4, figsize=(15, 4.7), layout="constrained")
                    vmax = np.percentile(star[domain], 99)
                    views = [
                        (
                            star,
                            "Stellar morphology at native HI beam",
                            dict(
                                cmap="gray_r",
                                norm=AsinhNorm(linear_width=max(0.03 * vmax, 1e-10), vmin=0, vmax=vmax),
                            ),
                        ),
                        (score, "Stellar arm template", dict(cmap="magma", vmin=0.7, vmax=1.4)),
                        (label, "Arm (red) / interarm (blue)", dict(cmap="RdBu_r", vmin=-1, vmax=1)),
                        (fraction, "Local cold fraction (native)", dict(cmap="viridis", vmin=0, vmax=1)),
                    ]
                    for ax, (a, title, kw) in zip(axs, views):
                        a = (
                            np.where(support & (geo["radius"] < entry["rmax"] * geo["r25_arcsec"]), a, np.nan)
                            if title.startswith("Stellar morphology")
                            else np.where(domain, a, np.nan)
                        )
                        im = ax.imshow(a, origin="lower", **kw)
                        (yy, xx) = np.where(domain)
                        ax.set_xlim(xx.min() - 2, xx.max() + 2)
                        ax.set_ylim(yy.min() - 2, yy.max() + 2)
                        ax.set_title(title, fontsize=10)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        fig.colorbar(im, ax=ax, shrink=0.7)
                    fig.suptitle(
                        f"{name}: native HI {info['native_resolution_pc']:.0f} pc; cold fraction arms/interarm={100 * result['cold_fraction_arm']:.1f}/{100 * result['cold_fraction_interarm']:.1f}%\nArea-weighted contrast {100 * result['contrast']:+.1f} pp; conditional95% [{100 * result['ci_low']:+.1f},{100 * result['ci_high']:+.1f}]. No HI smoothing.",
                        fontsize=12,
                    )
                    pdf.savefig(fig)
                    fig.savefig(OUT / f"{name}_native_arms.png", dpi=130)
                    plt.close(fig)
            print(name, "native comparison ready", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "native_arm_interarm_results.csv", index=False)
    main = d[np.isclose(d.area_fraction_per_ring, 0.25)].copy()
    main.to_csv(OUT / "native_arm_interarm_summary.csv", index=False)
    (fig, ax) = plt.subplots(figsize=(10, 7), layout="constrained")
    y = np.arange(len(main))
    val = 100 * main.contrast.to_numpy()
    lo = 100 * main.ci_low.to_numpy()
    hi = 100 * main.ci_high.to_numpy()
    ax.errorbar(val, y, xerr=[val - lo, hi - val], fmt="o", color="#0072B2")
    ax.axvline(0, color=".5", ls=":")
    ax.invert_yaxis()
    ax.set_yticks(
        y,
        [
            f"{r.galaxy}"
            + ("†" if r.arm_mask_quality == "tentative_outer_barred" else "")
            + f" ({r.native_resolution_pc:.0f} pc)"
            for r in main.itertuples()
        ],
    )
    ax.set_xlabel("Area-weighted local cold fraction: arms minus interarm (percentage points)")
    ax.set_title(
        "Visible-arm spirals at their native HI resolution\nEqual areas at matched radii; conditional95% intervals (4-native-beam blocks)",
        fontsize=13,
    )
    fig.supxlabel(
        "† Outer-arm templates in these barred systems are less secure; mask uncertainty is not in the error bars.",
        fontsize=9,
    )
    fig.savefig(OUT / "native_arm_interarm_comparison.pdf")
    fig.savefig(OUT / "native_arm_interarm_comparison.png", dpi=150)
    plt.close(fig)
    (OUT / "method.json").write_text(
        json.dumps(
            dict(
                resolution="Each input native HI beam retained; no HI spatial convolution or common-resolution grid.",
                fit=str(REFIT),
                selection=selection,
                stellar_processing="High-contrast PSF-like compact sources replaced locally before IRAC PSF matching to native HI major beam and reprojection. Extended stellar image not median-filtered. Robust Fourier modes0..6 in one-native-beam windows; no fixed-kpc median smoothing. Template masks are stellar-enhancement proxies, visually screened.",
                fraction="SUM(A_i*f_cold_i)/SUM(A_i), constant deprojected area per galaxy; equal arm/interarm areas in every half-native-beam radial bin.",
                classification="Unchanged adopted position cut; its existing3-native-beam scale is retained, not regenerated or tested here.",
                bootstrap="4000 resamples of projected blocks4native beams across, covering existing3beam cut averaging; conditional on fixed fitted fractions and stellar masks. 2/3beam block sensitivities at primary25% selection; no gas smoothing.",
                systematic_errors="No decomposition or classification uncertainty propagation. Native physical resolutions differ across galaxies.",
                thesis_modified=False,
            ),
            indent=2,
        )
    )
    print(
        main[
            [
                "galaxy",
                "native_resolution_pc",
                "cold_fraction_arm",
                "cold_fraction_interarm",
                "contrast",
                "ci_low",
                "ci_high",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "analyse"], required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        analyse()
