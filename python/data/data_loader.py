from __future__ import print_function
import torch
from utils.utils import *
from model.vblds import *


def load_data(N=100):
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