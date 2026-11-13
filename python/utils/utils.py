"""Small helpers for the VBLDS scripts."""
import torch
from torch import nn
import numpy as np

def get_phasor(omega):
	"""Return the 2 x 2 rotation matrix for angle omega (a 0-dim tensor, radians)."""
	cos_omega = omega.cos()
	sin_omega = omega.sin()
	A = torch.tensor([[cos_omega, -sin_omega], [sin_omega, cos_omega]])
	return A

def plot_matrix(plt,A):
	"""Show matrix A as an image with a color scale centered on zero."""
	amax = A.abs().max()
	plt.imshow(A,vmin=-amax,vmax=amax)

def symmetric(A):
	"""Return (A + A^T) / 2."""
	return 0.5*(A + A.T)