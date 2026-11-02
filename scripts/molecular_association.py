#!/usr/bin/env python3
"""Repeat the retained equal-area CO comparison on the frozen matched native maps."""

import _setup
import json
import numpy as np
import pandas as pd
from hiphases.co_pairs import pair_samples, choose, SHIFTS
from hiphases.co_sampling import samples
from hiphases.molecular import describe
from hiphases.sample import load_galaxy


def main():
    root = _setup.PROJECT_ROOT
    out = root / "results/molecular"
    out.mkdir(parents=True, exist_ok=True)
    maps = root / "cache/maps/molecular"
    rows, paired, decisions = [], [], []
    for folder in sorted(maps.iterdir()):
        name = folder.name
        d, header, _ = samples(name, maps)
        galaxy = load_galaxy(name)
        d["radius_kpc"] = d.radius * galaxy.r25.to_value("kpc")
        beam = np.deg2rad(header["BMAJ"]) * galaxy.distance.to_value("pc")
        d["beam_kpc"] = beam / 1000.0
        candidates = []
        for nside in [3, 4, 6]:
            for shift_id, shift in enumerate(SHIFTS):
                with np.errstate(divide="ignore", invalid="ignore"):
                    p = pair_samples(d, nside, shift)
                candidates.append(describe(p, dict(galaxy=name, nside=nside, shift_id=shift_id)))
        selection = choose(candidates)
        decisions.append(dict(galaxy=name, selection=selection))
        if selection is None:
            rows.append(dict(galaxy=name, n_pairs=0, n_cells=0, resolution_pc=beam))
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            p = pair_samples(d, selection["nside"], SHIFTS[selection["shift_id"]])
        row = describe(
            p, dict(galaxy=name, resolution_pc=beam, nside=selection["nside"], shift_id=selection["shift_id"])
        )
        row.update(
            cold_fraction_bright=float(p.bright_fn.mean()), cold_fraction_faint=float(p.faint_fn.mean())
        )
        rows.append(row)
        paired.append(p)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    if paired:
        pd.concat(paired).to_csv(out / "pairs.csv", index=False)
    (out / "selection.json").write_text(json.dumps(decisions, indent=2) + "\n")


if __name__ == "__main__":
    main()
