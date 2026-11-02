"""Portable paths for raw data, frozen products and regenerated results."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def artifact(relative):
    """Prefer an explicitly regenerated result, otherwise use its frozen counterpart."""
    relative = Path(relative)
    generated = ROOT / "results" / relative
    return generated if generated.exists() else ROOT / "cache" / relative


def co_path(galaxy):
    for suffix in [".fits.gz", ".FITS", ".fits", ".fits.GZ"]:
        path = ROOT / "data/raw/co" / (galaxy.co_stem + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(galaxy.co_stem)
