"""Context density and constant-consumption-time guides for the publication gas–SFR plots."""

from __future__ import annotations
import numpy as np
from matplotlib.colors import PowerNorm
from hiphases import resolvedks

XLIM = (-1.5, 2.5)
YLIM = (-6.5, 0.0)


def density_cloud(ax, gas, sfr, keep):
    """Restore the original full-field, positive-pixel grey distribution.

    This is a context layer. The blue beam-spaced sample and its fitted line
    retain their own optical-disc selection.
    """
    keep = keep & resolvedks.positive_measurements(gas) & resolvedks.positive_measurements(sfr)
    (x, y) = (np.log10(gas[keep]), np.log10(sfr[keep]))
    if x.size < 50:
        return
    ax.hist2d(
        x,
        y,
        bins=90,
        range=[XLIM, YLIM],
        cmap="Greys",
        norm=PowerNorm(gamma=0.45),
        cmin=1,
        rasterized=True,
        zorder=0,
    )


def depletion_lines(ax) -> None:
    """Lines of constant gas depletion time, the reference grid of these plots."""
    grid = np.logspace(XLIM[0], XLIM[1], 50)
    for depletion_gyr in (0.1, 1, 10, 100):
        sfr = grid * 1000000.0 / (depletion_gyr * 1000000000.0)
        ax.plot(np.log10(grid), np.log10(sfr), "-", color="0.85", linewidth=0.8, zorder=1)
        exit_x = YLIM[1] + 3.0 + np.log10(depletion_gyr)
        if XLIM[0] < exit_x < XLIM[1]:
            ax.annotate(
                f"{depletion_gyr:g} Gyr",
                xy=(exit_x, YLIM[1]),
                xytext=(exit_x, YLIM[1] - 0.12),
                fontsize=6,
                color="0.55",
                rotation=52,
                ha="center",
                va="top",
            )
