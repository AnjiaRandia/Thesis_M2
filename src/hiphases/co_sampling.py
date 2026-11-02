"""Beam-spaced samples from matched native HI and CO maps."""

import numpy as np
import pandas as pd
from astropy.io import fits
from hiphases import resolvedks
from hiphases.sample import load_galaxy


def samples(name, maps_root):
    folder = maps_root / name
    g = load_galaxy(name)
    data = {
        k: np.asarray(fits.getdata(folder / f"{k}.fits"), dtype=float)
        for k in ["hi", "narrow", "broad", "h2", "co_signed", "co_background_ratio", "valid_coverage"]
    }
    h = fits.getheader(folder / "hi.fits")
    geom = resolvedks.geometry(h, data["hi"].shape, g)
    (yy, xx) = np.indices(data["hi"].shape)
    x0 = round(h["CRPIX1"] - 1)
    y0 = round(h["CRPIX2"] - 1)
    step = geom["step_pixels"]
    grid = ((xx - x0) % step == 0) & ((yy - y0) % step == 0)
    (y, x) = np.where(grid)
    scale = np.deg2rad(abs(h["CDELT2"])) * g.distance.to_value("kpc")
    d = pd.DataFrame(
        dict(
            galaxy=name,
            galaxy_class=g.galaxy_class,
            y=y,
            x=x,
            kx=((x - x0) // step).astype(int),
            ky=((y - y0) // step).astype(int),
            offset_x=(x - h["CRPIX1"] + 1) * scale,
            offset_y=(y - h["CRPIX2"] + 1) * scale,
            radius=geom["radius"][y, x] / geom["r25_arcsec"],
        )
    )
    for key, a in data.items():
        d[key] = a[y, x]
    d["fn"] = np.divide(d.narrow, d.hi, out=np.full(len(d), np.nan), where=d.hi > 0)
    d["fmol"] = np.divide(d.h2, d.hi + d.h2, out=np.full(len(d), np.nan), where=d.hi + d.h2 > 0)
    d["spacing_kpc"] = step * scale
    return (d, h, data)
