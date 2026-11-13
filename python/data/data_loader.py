"""Synthetic data for the VBLDS demo scripts."""
from __future__ import print_function
import torch
from utils.utils import *
from model.vblds import *


def load_data(N=100):
	"""Sample a synthetic sequence from a known 2-D oscillator LDS.

	The target model has a 2-D latent state that rotates about 6.55 cycles
	per 100 samples and decays slightly (A = 0.999 * rotation), seen through a
	random 1 x 2 matrix C with observation noise variance 0.05. The initial
	phase and C are random on each call, so seed torch for repeatable data.

	Args:
	    N: number of time steps.

	Returns:
	    x: observations, shape (1, N).
	    z: true latent states, shape (2, N).
	"""
	# Create target model to sample from
	dimz = 2
	dimx = 1
	model = VBLDS(dimx=dimx,dimz=dimz)
	#  Set parameters of target model
	model.A.data = get_phasor(torch.tensor(2*np.pi*6.55/100))*.999
	model.Q.data = torch.eye(dimz)*1e-3
	model.C.data = torch.rand(dimx,dimz)*2-1
	model.R.data = torch.eye(dimx)*.05
	omega = 2*np.pi*torch.randn(1)
	model.m0.data = torch.tensor([torch.cos(omega), torch.sin(omega)])
	model.P0.data = torch.eye(dimz)*1e-9

	# Sample from target model
	x, z = model.sample(N=N)
	return x, z