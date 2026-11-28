#!/usr/bin/env python3
"""Plot total, kinematically cold and kinematically warm HI surface-density maps.

Each figure shows three panels on shared WCS axes with a common linear colour
scale, shared footprint mask and the circular resolution beam in the lower-left
corner of each panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.io import fits  # noqa: E402

import _setup  # noqa: F401, E402
from hiphases import style  # noqa: E402
from hiphases.paths import ROOT  # noqa: E402

CLIP_PERCENTILE = 99.9
BLANK_BELOW = 1.37
DEFAULT_CMAP = "RdYlBu_r"

FIGURE_GALAXIES = (
    "HO_II",
    "IC2574",
    "NGC0925",
    "NGC2403",
    "NGC2976",
    "NGC4214",
    "NGC7793",
    "NGC0628",
    "NGC2903",
    "NGC3184",
    "NGC3351",
    "NGC4736",
    "NGC5194",
    "NGC6946",
)

PANELS = (
    ("total", "Total H I"),
    ("narrow", "Kinematically cold H I"),
    ("broad", "Kinematically warm H I"),
)


def _beam_from_header(header: fits.Header) -> tuple[float, float, float] | None:
    """Return (bmaj, bmin, bpa) in degrees from a FITS header, if present."""
    bmaj = header.get("BMAJ")
    if bmaj is None or float(bmaj) <= 0:
        return None
    bmin = header.get("BMIN", bmaj)
    bpa = header.get("BPA", 0.0)
    return float(bmaj), float(bmin), float(bpa)


def _panel(
    fig,
    hdu: fits.PrimaryHDU,
    subplot: list[float],
    title: str,
    vmax: float,
    cmap: str,
    stretch: str,
    show_y: bool,
    beam: tuple[float, float, float] | None,
):
    """Draw one sky-map panel with WCS axes and resolution beam."""
    import aplpy

    panel = aplpy.FITSFigure(hdu, figure=fig, subplot=subplot)
    panel.show_colorscale(vmin=0.0, vmax=vmax, stretch=stretch, cmap=cmap)
    panel.set_nan_color("white")

    panel.ticks.set_tick_direction("in")
    panel.ticks.set_color("black")
    panel.ticks.set_length(7)
    panel.ticks.set_linewidth(1.0)
    panel.ticks.set_minor_frequency(5)
    panel.ax.coords[0].display_minor_ticks(True)
    panel.ax.coords[1].display_minor_ticks(True)

    panel.tick_labels.set_font(size=12)
    panel.tick_labels.set_xformat("hh:mm:ss")
    panel.tick_labels.set_yformat("dd:mm")
    panel.axis_labels.set_font(size=14)

    ny, nx = np.squeeze(hdu.data).shape
    ra = panel.ax.wcs.pixel_to_world_values(
        np.array([0.2, 0.8]) * (nx - 1), np.full(2, (ny - 1) / 2)
    )[0]
    lower, upper = np.min(ra), np.max(ra)
    choices = []
    for seconds in (1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600):
        step = seconds * 15 / 3600
        values = np.arange(np.ceil(lower / step), np.floor(upper / step) + 1) * step
        if 2 <= len(values) <= 3:
            choices.append((abs(len(values) - 2), -step, values))
    if choices:
        values = min(choices, key=lambda item: item[:2])[2]
        panel.ax.coords[0].set_ticks(values=values * u.deg)
    else:
        panel.ax.coords[0].set_ticks(number=2)
    panel.ax.coords[1].set_ticks(number=3)
    panel.ax.coords[0].set_ticklabel(exclude_overlapping=True, pad=9)
    panel.ax.coords[1].set_ticklabel(exclude_overlapping=True, pad=9)
    panel.axis_labels.set_xpad(1.2)
    panel.axis_labels.set_ypad(1.4)
    panel.ax.tick_params(which="both", pad=9)

    if not show_y:
        panel.axis_labels.hide_y()
        panel.tick_labels.hide_y()

    if beam is not None:
        bmaj, bmin, bpa = beam
        panel.add_beam(
            major=bmaj,
            minor=bmin,
            angle=bpa,
            corner="bottom left",
            frame=True,
            pad=0.35,
            borderpad=0.65,
            facecolor="black",
            edgecolor="black",
            linewidth=0.8,
        )
        panel.beam._beam.patch.set_facecolor("white")
        panel.beam._beam.patch.set_edgecolor("black")
        panel.beam._beam.patch.set_linewidth(0.7)
        panel.beam._beam.patch.set_alpha(0.92)

    panel.ax.set_title(title, fontsize=14, pad=16)
    return panel


def plot_galaxy(
    galaxy: str,
    maps_dir: Path,
    output: Path,
    cmap: str = DEFAULT_CMAP,
    stretch: str = "linear",
    blank_below: float = BLANK_BELOW,
) -> None:
    """Render the three-panel HI surface-density map for one galaxy."""
    data_dir = maps_dir / galaxy
    paths = {phase: data_dir / f"{phase}.fits" for phase in ("narrow", "broad")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing input map(s): {', '.join(missing)}")

    images: dict[str, np.ndarray] = {}
    header = None
    for phase, path in paths.items():
        with fits.open(path) as opened:
            images[phase] = np.squeeze(opened[0].data).astype(float)
            header = opened[0].header.copy()

    assert header is not None
    beam = _beam_from_header(header)

    total = images["narrow"] + images["broad"]
    images["total"] = total
    footprint = np.isfinite(total) & (total >= blank_below)
    hdus = {
        phase: fits.PrimaryHDU(np.where(footprint, image, np.nan), header)
        for phase, image in images.items()
    }

    ceilings = []
    for image in images.values():
        finite = image[footprint & np.isfinite(image) & (image > 0)]
        if finite.size:
            ceilings.append(float(np.percentile(finite, CLIP_PERCENTILE)))
    vmax = max(ceilings) if ceilings else 1.0

    fig = plt.figure(figsize=(9.8, 4.9))
    left, panel_width = 0.12, 0.285
    panel = None
    for index, (phase, title) in enumerate(PANELS):
        panel = _panel(
            fig,
            hdus[phase],
            subplot=[left + panel_width * index, 0.19, panel_width, 0.55],
            title=title,
            vmax=vmax,
            cmap=cmap,
            stretch=stretch,
            show_y=(index == 0),
            beam=beam,
        )

    assert panel is not None
    color_axis = fig.add_axes([left, 0.94, 3 * panel_width, 0.035])
    colorbar = fig.colorbar(panel.image, cax=color_axis, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=12, pad=8, direction="in")
    colorbar.set_label(
        r"$\Sigma_{\mathrm{H\,I}}\ (M_\odot\ \mathrm{pc}^{-2})$",
        fontsize=14,
        labelpad=10,
    )
    colorbar.ax.xaxis.set_label_position("top")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    beam_str = f"{beam[0] * 3600.0:.2f}\"" if beam is not None else "none"
    print(f"wrote {output} (vmax={vmax:.2f} Msun/pc2, beam={beam_str})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "galaxies",
        nargs="*",
        help="Galaxies to plot (default: the 14 galaxies shown in Figure 5).",
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=ROOT / "cache/maps/individual",
        help="Directory containing per-galaxy narrow.fits and broad.fits maps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/surface_maps",
        help="Destination directory for <galaxy>-cnm_wnm.pdf files.",
    )
    parser.add_argument("--cmap", default=DEFAULT_CMAP)
    parser.add_argument(
        "--stretch",
        default="linear",
        choices=["linear", "sqrt", "arcsinh", "log"],
    )
    parser.add_argument("--blank-below", type=float, default=BLANK_BELOW)
    args = parser.parse_args()

    style.apply()
    plt.rcParams.update(
        {
            "axes.linewidth": 1.0,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.minor.size": 3.5,
            "ytick.minor.size": 3.5,
        }
    )

    galaxies = args.galaxies or list(FIGURE_GALAXIES)
    for galaxy in galaxies:
        plot_galaxy(
            galaxy=galaxy,
            maps_dir=args.maps_dir,
            output=args.output_dir / f"{galaxy}-cnm_wnm.pdf",
            cmap=args.cmap,
            stretch=args.stretch,
            blank_below=args.blank_below,
        )


if __name__ == "__main__":
    main()
