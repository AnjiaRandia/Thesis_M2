"""ROHSA execution and the fitted Gaussian parameter arrays."""

from __future__ import annotations
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from . import cubes
from .cubes import PreparedCube

ROHSA_BINARY_ENV = "HIPHASES_ROHSA"


@dataclass(frozen=True)
class RohsaSettings:
    """Hyper-parameters of one ROHSA run (see the ROHSA paper, Section 3).

    The dispersion bounds are given here in **km/s** and converted to channels
    for ROHSA at run time.  ROHSA itself works in channels, but the THINGS
    channel width in this sample ranges from 1.3 to 12.2 km/s, so a bound fixed
    in channels would impose a wildly different physical prior on each galaxy.
    The defaults bracket the neutral medium: a few km/s for the cold phase, a
    few tens for the warm one.
    """

    n_gauss: int = 2
    lambda_amp: float = 0.0
    lambda_mu: float = 1.0
    lambda_sig: float = 1.0
    lambda_var_sig: float = 0.0
    sig_init_kms: float = 8.0
    lb_sig_init_kms: float = 2.0
    ub_sig_init_kms: float = 20.0
    lb_sig_kms: float = 1.5
    ub_sig_kms: float = 40.0
    init_option: str = "mean"
    maxiter: int = 800
    maxiter_init: int = 15000
    noise_from_negatives: bool = False

    def in_channels(self, channel_kms: float) -> dict[str, float]:
        """The dispersion hyper-parameters expressed in channels.

        Each bound is floored at one channel: ROHSA cannot resolve a line
        narrower than the spectral sampling, and asking it to try makes the
        fit unstable.
        """
        return {
            "sig_init": max(self.sig_init_kms / channel_kms, 1.0),
            "lb_sig_init": max(self.lb_sig_init_kms / channel_kms, 1.0),
            "ub_sig_init": max(self.ub_sig_init_kms / channel_kms, 1.0),
            "lb_sig": max(self.lb_sig_kms / channel_kms, 1.0),
            "ub_sig": max(self.ub_sig_kms / channel_kms, 1.0),
        }


@dataclass(frozen=True)
class Decomposition:
    """The Gaussian parameter maps returned by one ROHSA run."""

    amplitude: np.ndarray
    centroid: np.ndarray
    dispersion: np.ndarray
    model: np.ndarray


def rohsa_binary() -> str:
    """Locate the ROHSA executable, preferring ``$HIPHASES_ROHSA``."""
    candidate = os.environ.get(ROHSA_BINARY_ENV)
    if candidate:
        return candidate
    found = shutil.which("ROHSA")
    if found:
        return found
    raise FileNotFoundError(
        f"ROHSA executable not found. Build it from https://github.com/antoinemarchal/ROHSA and point ${ROHSA_BINARY_ENV} at the binary."
    )


def run(cube: PreparedCube, workdir: Path, settings: RohsaSettings) -> Decomposition:
    """Run ROHSA on a prepared cube inside ``workdir`` and read the result back."""
    import matplotlib

    matplotlib.use("Agg")
    from ROHSApy import ROHSA

    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    data_dat = workdir / "cube.dat"
    params_txt = workdir / "parameters.txt"
    result_dat = workdir / "gaussians.dat"
    core = ROHSA(cube.data, hdr=cube.header)
    core.cube2dat(filename=str(data_dat))
    (lstd, ustd) = cube.noise_channels
    sigma_bounds = settings.in_channels(cubes.channel_width_kms(cube.header))
    noise_file = ""
    noise_flag = ".false."
    if settings.noise_from_negatives:
        sigma = cubes.noise_map(cube.data)
        noise_file = str(workdir / "rms_map.dat")
        core.rms_map(sigma.astype("float64"), filename=noise_file)
        noise_flag = ".true."
    core.gen_parameters(
        filename=str(data_dat),
        filename_parameters=str(params_txt),
        fileout=str(result_dat),
        timeout=str(workdir / "timesteps.dat"),
        filename_noise=noise_file,
        n_gauss=settings.n_gauss,
        lambda_amp=settings.lambda_amp,
        lambda_mu=settings.lambda_mu,
        lambda_sig=settings.lambda_sig,
        lambda_var_sig=settings.lambda_var_sig,
        **sigma_bounds,
        init_option=settings.init_option,
        maxiter=settings.maxiter,
        maxiter_init=settings.maxiter_init,
        noise=noise_flag,
        lstd=lstd,
        ustd=ustd,
        save_grid=".false.",
    )
    completed = subprocess.run(
        [rohsa_binary(), str(params_txt)],
        cwd=workdir,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    (workdir / "rohsa.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0 or not result_dat.exists():
        raise RuntimeError(f"ROHSA failed (exit {completed.returncode}); see {workdir / 'rohsa.log'}")
    gaussian = core.read_gaussian(str(result_dat))
    return Decomposition(
        amplitude=gaussian[0::3],
        centroid=gaussian[1::3],
        dispersion=gaussian[2::3],
        model=core.return_result_cube(gaussian=gaussian),
    )
