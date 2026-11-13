"""Synthetic data for the VBLDS demo scripts."""
import numpy as np

from model.vblds import VBLDS
from utils.utils import get_phasor


def load_data(N=100, rng=None):
    """Sample a synthetic sequence from a known 2-D oscillator LDS.

    The target model has a 2-D latent state that rotates about 6.55 cycles
    per 100 samples and decays slightly (A = 0.999 * rotation), seen through a
    random 1 x 2 matrix C with observation noise variance 0.05. The initial
    phase and C are random on each call; pass a seed or
    ``numpy.random.Generator`` as ``rng`` for repeatable data.

    Args:
        N: number of time steps.
        rng: seed or ``numpy.random.Generator``.

    Returns:
        y: observations, shape (1, N).
        x: true latent states, shape (2, N).
    """
    rng = np.random.default_rng(rng)
    dimx, dimy = 2, 1
    model = VBLDS(dimy=dimy, dimx=dimx, rng=rng)
    model.A = get_phasor(2 * np.pi * 6.55 / 100) * 0.999
    model.Q = 1e-3 * np.eye(dimx)
    model.C = rng.random((dimy, dimx)) * 2 - 1
    model.R = 0.05 * np.eye(dimy)
    omega = 2 * np.pi * rng.standard_normal()
    model.m0 = np.array([np.cos(omega), np.sin(omega)])
    model.P0 = 1e-9 * np.eye(dimx)
    return model.sample(N=N)
