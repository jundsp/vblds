"""Small helpers for the VBLDS scripts."""
import numpy as np


def get_phasor(omega):
    """Return the 2 x 2 rotation matrix for angle omega (radians)."""
    c, s = np.cos(omega), np.sin(omega)
    return np.array([[c, -s], [s, c]])


def plot_matrix(plt, A):
    """Show matrix A as an image with a color scale centered on zero."""
    amax = np.abs(A).max()
    plt.imshow(A, vmin=-amax, vmax=amax)
