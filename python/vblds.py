from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim, distributions
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def get_phasor(omega):
	cos_omega = omega.cos()
	sin_omega = omega.sin()
	A = torch.tensor([[cos_omega, -sin_omega], [sin_omega, cos_omega]])
	return A


class VBLDS:
	def __init__(self, A=None,C=None,Q=None,R=None,m0=None,P0=None):
		super(VBLDS, self).__init__()
		self.A = A
		self.C = C
		self.Q = Q
		self.R = R
		self.m0 = m0
		self.P0 = P0
		self.dimx = C.size(0)
		self.dimz = A.size(0)

	def __repr__(self):
		to_print = ('VBLDS object'  +  '\n'
		'	A = ' + str(self.A) + '\n'
		'	Q = ' + str(self.Q)) + '\n'
		return to_print

	def sample(self,N=100):
		z = torch.zeros(self.dimz,N)
		x = torch.zeros(self.dimx,N)

		initial = torch.distributions.MultivariateNormal(self.m0, self.P0)
		transition = torch.distributions.MultivariateNormal(torch.zeros(self.dimz), self.Q)
		emission = torch.distributions.MultivariateNormal(torch.zeros(self.dimx), self.R)

		z_n = initial.sample()
		for n in range(N):
			z[:,n] = z_n
			x[:,n] = self.C @ z_n + emission.sample()
			# Update state
			z_n = self.A @ z_n + transition.sample()

		return x, z