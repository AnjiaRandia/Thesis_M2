"""Matched-aperture and radial star formation efficiencies."""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
import _setup
from hiphases.efficiency import region_sfe
import numpy as np
import pandas as pd
from astropy.io import fits
from hiphases import resolvedks
from hiphases.sample import load_galaxy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = _setup.PROJECT_ROOT
BASE = ROOT
MAPS = ROOT / "cache/maps/individual"
OUT = ROOT / "results/sfe"
FIG = OUT / "figures"
LABELS = {
    "hi": "Total H I",
    "narrow": "Narrow H I",
    "broad": "Broad H I",
    "h2": "H$_2$",
    "h2_narrow": "H$_2$ + narrow H I",
    "h2_broad": "H$_2$ + broad H I",
    "gas": "H$_2$ + total H I",
}
COLORS = dict(
    hi="black",
    narrow="#0072B2",
    broad="#D55E00",
    h2="#009E73",
    h2_narrow="#0072B2",
    h2_broad="#D55E00",
    gas="black",
)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    aperture = []
    radial = []
    inventory = []
    for folder in sorted(MAPS.iterdir()):
        if not (folder / "hi.fits").exists():
            continue
        name = folder.name
        g = load_galaxy(name)
        h = fits.getheader(folder / "hi.fits")
        m = {k: fits.getdata(folder / f"{k}.fits").astype(float) for k in ["hi", "narrow", "broad", "sfr"]}
        geo = resolvedks.geometry(h, m["hi"].shape, g)
        finite = np.isfinite(m["hi"]) & np.isfinite(m["narrow"] + m["broad"])
        assert np.allclose(m["hi"][finite], (m["narrow"] + m["broad"])[finite], atol=2e-05, rtol=2e-06)
        atomic = geo["inside"].copy()
        for key in m:
            atomic &= resolvedks.positive_measurements(m[key])
        scopes = [("atomic", atomic, ["hi", "narrow", "broad"])]
        if g.has_co:
            m["h2"] = fits.getdata(folder / "h2.fits").astype(float)
            m.update(h2_narrow=m["h2"] + m["narrow"], h2_broad=m["h2"] + m["broad"], gas=m["h2"] + m["hi"])
            scopes.append(
                (
                    "co_common",
                    atomic & resolvedks.positive_measurements(m["h2"]),
                    ["h2", "h2_narrow", "h2_broad", "gas"],
                )
            )
        for scope, valid, keys in scopes:
            inventory.append(
                dict(
                    galaxy=name,
                    galaxy_class=g.galaxy_class,
                    scope=scope,
                    n_pixels=int(valid.sum()),
                    beam_pc=np.deg2rad(h["BMAJ"]) * g.distance.to_value("pc"),
                )
            )
            if not valid.any():
                continue
            for key, value in region_sfe(m, valid, keys).items():
                aperture.append(
                    dict(
                        galaxy=name,
                        galaxy_class=g.galaxy_class,
                        scope=scope,
                        tracer=key,
                        log_sfe=value,
                        n_pixels=int(valid.sum()),
                    )
                )
            beam_area = (
                np.pi
                / (4 * np.log(2))
                * (h["BMAJ"] * 3600 / geo["pixel_arcsec"])
                * (h.get("BMIN", h["BMAJ"]) * 3600 / geo["pixel_arcsec"])
            )
            for index in range(10):
                region = (
                    valid
                    & (geo["radius"] / geo["r25_arcsec"] >= 0.1 * index)
                    & (geo["radius"] / geo["r25_arcsec"] < 0.1 * (index + 1))
                )
                if region.sum() < beam_area:
                    continue
                for key, value in region_sfe(m, region, keys).items():
                    radial.append(
                        dict(
                            galaxy=name,
                            galaxy_class=g.galaxy_class,
                            scope=scope,
                            tracer=key,
                            radius=0.1 * index + 0.05,
                            log_sfe=value,
                            n_pixels=int(region.sum()),
                        )
                    )
        print(name, "SFE measured", flush=True)
    ap = pd.DataFrame(aperture)
    rad = pd.DataFrame(radial)
    ap.to_csv(OUT / "sfe_per_galaxy.csv", index=False)
    rad.to_csv(OUT / "sfe_radial_per_galaxy.csv", index=False)
    pd.DataFrame(inventory).to_csv(OUT / "sfe_inventory.csv", index=False)
    summaries = []
    rng = np.random.default_rng(42)
    for group in ["dwarf", "spiral", "all"]:
        d = ap if group == "all" else ap[ap.galaxy_class == group]
        for (scope, key), part in d.groupby(["scope", "tracer"]):
            v = part.log_sfe.to_numpy()
            draw = np.median(rng.choice(v, size=(10000, len(v))), axis=1)
            summaries.append(
                dict(
                    group=group,
                    scope=scope,
                    tracer=key,
                    n_galaxies=len(v),
                    median_log_sfe=np.median(v),
                    p16=np.percentile(draw, 16),
                    p84=np.percentile(draw, 84),
                    galaxy_scatter_dex=np.std(v, ddof=1),
                )
            )
    pd.DataFrame(summaries).to_csv(OUT / "sfe_summary.csv", index=False)
    (fig, axes) = plt.subplots(
        2, 2, figsize=(10, 7.6), sharex=True, sharey=True, gridspec_kw=dict(wspace=0, hspace=0)
    )
    profile = []
    for row, (group, label) in enumerate([("dwarf", "Dwarfs"), ("spiral", "Spirals")]):
        for col, (scope, keys) in enumerate(
            [("atomic", ["hi", "narrow", "broad"]), ("co_common", ["h2", "h2_narrow", "h2_broad", "gas"])]
        ):
            ax = axes[row, col]
            d = rad[(rad.galaxy_class == group) & (rad.scope == scope)]
            for key in keys:
                p = d[d.tracer == key]
                x = []
                y = []
                lo = []
                hi = []
                for radius, part in p.groupby("radius"):
                    if len(part) < 3:
                        continue
                    v = part.log_sfe.to_numpy()
                    x.append(radius)
                    y.append(np.median(v))
                    lo.append(np.percentile(v, 16))
                    hi.append(np.percentile(v, 84))
                    profile.append(
                        dict(
                            group=group,
                            scope=scope,
                            tracer=key,
                            radius=radius,
                            n_galaxies=len(part),
                            median_log_sfe=y[-1],
                            p16_galaxies=lo[-1],
                            p84_galaxies=hi[-1],
                        )
                    )
                ax.plot(
                    x,
                    y,
                    "o",
                    color=COLORS[key],
                    ms=4,
                    ls="--" if key == "h2_broad" else "-",
                    lw=1.8,
                    label=LABELS[key],
                )
                if key in ["hi", "h2"]:
                    ax.fill_between(x, lo, hi, color=COLORS[key], alpha=0.12, lw=0)
            ax.text(0.95, 0.05, label, transform=ax.transAxes, ha="right", fontsize=12)
            ax.set(
                xlim=(0, 1), ylim=(-11.5, -7.4), xticks=np.arange(0.2, 1.0, 0.2), yticks=[-11, -10, -9, -8]
            )
            ax.tick_params(pad=7, labelsize=11)
            ax.label_outer()
            if row == 0:
                ax.set_title("Atomic components" if col == 0 else "CO-covered regions", fontsize=13, pad=12)
                ax.legend(fontsize=11, loc="upper right", frameon=False)
    fig.supxlabel("Galactocentric radius $r/r_{25}$", y=0.025, fontsize=13)
    fig.supylabel("$\\log_{10}\\mathrm{SFE}$ [yr$^{-1}$]", x=0.025, fontsize=13)
    fig.subplots_adjust(left=0.105, right=0.98, bottom=0.1, top=0.94)
    fig.savefig(FIG / "adopted_sfe_profiles.pdf")
    fig.savefig(FIG / "adopted_sfe_profiles.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(profile).to_csv(OUT / "sfe_radial_summary.csv", index=False)
    w = ap[ap.scope == "atomic"].pivot(index="galaxy", columns="tracer", values="log_sfe")
    assert (w.narrow >= w.hi - 1e-10).all() and (w.broad >= w.hi - 1e-10).all()
    w = ap[ap.scope == "co_common"].pivot(index="galaxy", columns="tracer", values="log_sfe")
    assert (w.h2 >= w.h2_narrow - 1e-10).all() and (w.h2_narrow >= w.gas - 1e-10).all()
    assert (w.h2 >= w.h2_broad - 1e-10).all() and (w.h2_broad >= w.gas - 1e-10).all()
    (OUT / "sfe_method.json").write_text(
        json.dumps(
            dict(
                source=str(MAPS),
                resolution="Per-galaxy matched tracer beams used in individual KS plots; no extra convolution.",
                region="r<r25-halfbeam; positive HI,narrow,broad,SFR; CO comparisons additionally positive H2 on identical pixels.",
                sfe="1e-6 SUM(SigmaSFR)/SUM(SigmaGas),yr^-1; constant pixel area cancels",
                radial="0.1r25 bins with at least one projected beam area per galaxy/bin; median galaxy logSFE shown only for >=3galaxies",
                summary="median per-galaxy aperture logSFE;10000whole-galaxy bootstrap,16/84percentiles",
                uncertainty="No calibration or decomposition systematics; shaded radial bands are galaxy scatter, not errors on medians.",
                interpretation="Same observed SFR assigned to every denominator; not independently measured phase-specific star formation rates.",
                validation="Narrow+broad=HI; identical apertures; nested-denominator inequalities verified.",
            ),
            indent=2,
        )
    )
    print(pd.DataFrame(summaries).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
