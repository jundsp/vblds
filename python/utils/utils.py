from __future__ import print_function
import torch
from torch import nn
import numpy as np

def get_phasor(omega):
	cos_omega = omega.cos()
	sin_omega = omega.sin()
	A = torch.tensor([[cos_omega, -sin_omega], [sin_omega, cos_omega]])
	return A