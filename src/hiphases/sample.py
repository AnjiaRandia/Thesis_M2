"""The retained 16-galaxy sample and its geometry."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from astropy import units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = PROJECT_ROOT / "config" / "galaxies.csv"
RMAX_R25 = 1.25


@dataclass(frozen=True)
class Galaxy:
    """One row of the sample table, with the geometry helpers used everywhere."""

    name: str
    galaxy_class: str
    distance: u.Quantity
    inclination: u.Quantity
    r25: u.Quantity
    position_angle: float
    hubble_T: float
    hubble_T_err: float
    log_mstar: float
    abs_B_mag: float
    sfr: float
    metallicity: float
    things_stem: str
    sfr_stem: str
    stellar_stem: str
    co_stem: str | None
    co_line: str | None

    @property
    def has_co(self) -> bool:
        """True when the galaxy has CO coverage, i.e. an H2 term in Sigma_gas."""
        return bool(self.co_stem)

    @property
    def axis_ratio(self) -> float:
        """Projected minor/major axis ratio, cos(i), used for elliptical annuli."""
        return float(np.cos(self.inclination.to(u.rad).value))

    def pixel_scale(self, cdelt_deg: float) -> u.Quantity:
        """Physical size of one pixel, given the map's ``CDELT2`` in degrees."""
        return (self.distance * np.sin(cdelt_deg * u.deg)).to(u.kpc)

    def r25_pixels(self, cdelt_deg: float) -> float:
        """Optical radius expressed in pixels of a map with this ``CDELT2``."""
        return float(self.r25 / self.pixel_scale(cdelt_deg))


def read_table(path: Path | str = SAMPLE_CSV) -> pd.DataFrame:
    """Read the sample CSV, skipping its leading comment block."""
    return pd.read_csv(path, comment="#")


def load_sample(path: Path | str = SAMPLE_CSV) -> dict[str, Galaxy]:
    """Return the whole sample keyed by galaxy name, in table order."""
    table = read_table(path)
    return {row["name"]: _from_row(row) for (_, row) in table.iterrows()}


def load_galaxy(name: str, path: Path | str = SAMPLE_CSV) -> Galaxy:
    """Return a single galaxy by name."""
    try:
        return load_sample(path)[name]
    except KeyError:
        raise KeyError(f"{name!r} is not in the sample table {path}") from None


def _blank_to_none(value) -> str | None:
    """Empty CSV cells arrive as NaN; everything downstream wants ``None``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def _from_row(row: pd.Series) -> Galaxy:
    return Galaxy(
        name=row["name"],
        galaxy_class=row["class"],
        distance=float(row["D_Mpc"]) * u.Mpc,
        inclination=float(row["incl_deg"]) * u.deg,
        r25=float(row["r25_kpc"]) * u.kpc,
        position_angle=float(row["PA_deg"]),
        hubble_T=float(row["T"]),
        hubble_T_err=float(row["T_err"]),
        log_mstar=float(row["logMstar"]),
        abs_B_mag=float(row["MB_mag"]),
        sfr=float(row["SFR_Msun_yr"]),
        metallicity=float(row["metallicity"]),
        things_stem=row["things_stem"],
        sfr_stem=row["sfr_stem"],
        stellar_stem=row["stellar_stem"],
        co_stem=_blank_to_none(row["co_stem"]),
        co_line=_blank_to_none(row["co_line"]),
    )
