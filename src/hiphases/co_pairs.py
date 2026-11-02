"""Equal-area CO-bright and CO-faint pairing at matched galactocentric radii."""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SHIFTS = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]


def pair_samples(d, nside, shift, bright_cut=8.0, faint_cut=2.0, radius_beams=1.0):
    d = d[d.valid_coverage.astype(bool)].copy()
    d["cell_x"] = np.floor((d.kx + shift[0] * nside) / nside).astype(int)
    d["cell_y"] = np.floor((d.ky + shift[1] * nside) / nside).astype(int)
    records = []
    for (cx, cy), cell in d.groupby(["cell_x", "cell_y"]):
        b = cell[cell.co_background_ratio >= bright_cut]
        f = cell[np.abs(cell.co_background_ratio) <= faint_cut]
        if b.empty or f.empty:
            continue
        (bx, by) = (b.offset_x.to_numpy(), b.offset_y.to_numpy())
        (fx, fy) = (f.offset_x.to_numpy(), f.offset_y.to_numpy())
        distance = np.hypot(bx[:, None] - fx, by[:, None] - fy)
        radial = np.abs(b.radius_kpc.to_numpy()[:, None] - f.radius_kpc.to_numpy())
        beam = d.beam_kpc.iloc[0]
        valid = (distance <= 3.0 * beam) & (radial <= radius_beams * beam)
        cost = np.where(valid, distance / beam + radial / beam, 1000000.0)
        (ii, jj) = linear_sum_assignment(cost)
        for i, j in zip(ii, jj):
            if not valid[i, j]:
                continue
            (a, c) = (b.iloc[i], f.iloc[j])
            row = dict(
                galaxy=a.galaxy,
                cell_x=int(cx),
                cell_y=int(cy),
                bright_index=int(b.index[i]),
                faint_index=int(f.index[j]),
                bright_x=a.offset_x,
                bright_y=a.offset_y,
                faint_x=c.offset_x,
                faint_y=c.offset_y,
                bright_co=a.co_signed,
                faint_co=c.co_signed,
                bright_background_ratio=a.co_background_ratio,
                faint_background_ratio=c.co_background_ratio,
                separation_beams=float(distance[i, j] / beam),
                radius_difference_beams=float(radial[i, j] / beam),
                resolution_pc=beam * 1000,
                beam_area_pc2=float(np.pi / (4 * np.log(2)) * (beam * 1000) ** 2),
                bright_threshold=bright_cut,
                faint_threshold=faint_cut,
                nside=nside,
                shift_id=SHIFTS.index(shift),
                radius_limit_beams=radius_beams,
            )
            for t in ["narrow", "broad", "hi"]:
                row["bright_" + t] = float(a[t])
                row["faint_" + t] = float(c[t])
                row["delta_log_" + t] = float(np.log10(a[t] / c[t]))
            row["bright_fn"] = a.fn
            row["faint_fn"] = c.fn
            row["delta_fn"] = a.fn - c.fn
            records.append(row)
    return pd.DataFrame(records)


def choose(configs):
    for minimum_cells, minimum_pairs in [(6, 10), (3, 5)]:
        eligible = [c for c in configs if c["n_cells"] >= minimum_cells and c["n_pairs"] >= minimum_pairs]
        if eligible:
            return sorted(eligible, key=lambda c: (c["nside"], -c["n_pairs"], c["shift_id"]))[0]
    eligible = [c for c in configs if c["n_pairs"] > 0]
    return sorted(eligible, key=lambda c: (-c["n_pairs"], c["nside"], c["shift_id"]))[0] if eligible else None
