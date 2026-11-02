"""Ratio of summed SFR to gas over identical spatial samples."""

import numpy as np


def region_sfe(m, mask, keys):
    sfr = m["sfr"][mask].sum() * 1e-06
    return {k: float(np.log10(sfr / m[k][mask].sum())) for k in keys}
