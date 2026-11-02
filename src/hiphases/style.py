"""Shared publication plotting style."""

from __future__ import annotations
import glob
import os
import matplotlib as mpl
from matplotlib import font_manager

_FONT_NAME = "TeX Gyre Heros"
_FONT_DIRS = (
    os.environ.get("HIPHASES_FONT_DIR", ""),
    "/usr/share/texmf/fonts/opentype/public/tex-gyre",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre",
    "/usr/local/texlive/texmf-dist/fonts/opentype/public/tex-gyre",
)


def _register_font() -> bool:
    """Register the non-condensed TeX Gyre Heros faces; True if any was found."""
    found = False
    for directory in _FONT_DIRS:
        if not directory or not os.path.isdir(directory):
            continue
        for path in glob.glob(os.path.join(directory, "texgyreheros-*.otf")):
            try:
                font_manager.fontManager.addfont(path)
                found = True
            except Exception:
                pass
    return found


def apply() -> None:
    """Apply the shared publication style to the global Matplotlib rcParams."""
    _register_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [_FONT_NAME, "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": _FONT_NAME,
            "mathtext.it": f"{_FONT_NAME}:italic",
            "mathtext.bf": f"{_FONT_NAME}:bold",
            "mathtext.cal": f"{_FONT_NAME}:italic",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.pad": 7,
            "ytick.major.pad": 7,
            "axes.labelpad": 12,
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
