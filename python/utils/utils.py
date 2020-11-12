import torch
from torch import nn
import numpy as np

def get_phasor(omega):
	cos_omega = omega.cos()
	sin_omega = omega.sin()
	A = torch.tensor([[cos_omega, -sin_omega], [sin_omega, cos_omega]])
	return A

def plot_matrix(plt,A):
	amax = A.abs().max()
	plt.imshow(A,vmin=-amax,vmax=amax)

def symmetric(A):
	return 0.5*(A + A.T)