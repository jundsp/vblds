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

def symmetric(A):
	return 0.5*(A + A.T)

class VBLDS(nn.Module):
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
		self.Sigma_AQA = 0*torch.ones(self.dimz,self.dimz)
		self.Sigma_CRC = 0*torch.ones(self.dimz,self.dimz)

	def __repr__(self):
		to_print = ('VBLDS object'  +  '\n'
		'	A = ' + str(self.A) + '\n'
		'	Q = ' + str(self.Q)) + '\n'
		return to_print

	def woodbury(self,A,B):
		I = torch.eye(self.dimz)
		return I - A @ torch.inverse(I + B @ A) @ B

	def predict(self,mu,V):
		G = self.woodbury(V,self.Sigma_AQA)
		m = self.A @ G @ mu
		P = self.A @ G @ V @ self.A.T + self.Q
		return m, symmetric(P), G

	def update(self,x,m,P):
		L = self.woodbury(P,self.Sigma_CRC)
		I = torch.eye(self.dimz)

		x_hat = self.C @ L @ m
		S_hat = self.C @ L @ P @ self.C.T + self.R

		K = L @ P @ self.C.T @ S_hat.inverse()
		mu = L @ m + K @ (x - x_hat)
		V = (I - K @ self.C) @ L @ P
		return mu.squeeze(-1), symmetric(V)

	def filter(self,x):
		N = x.size(-1)
		mu, V = torch.zeros(self.dimz,N), torch.zeros(self.dimz,self.dimz,N)
		m, P = self.m0.clone(), self.P0.clone()
		for n in range(N):
			mu[:,n], V[:,:,n] = self.update(x[:,n].unsqueeze(-1),m,P)
			m, P, _ = self.predict(mu[:,n].unsqueeze(-1),V[:,:,n])
		return mu, V

	def smooth(self,muf,Vf,mu_plus,V_plus):
		m,P,G = self.predict(muf,Vf)
		J = G @ Vf @ self.A.T @ P.inverse()

		mu = G @ muf + J @ (mu_plus - m)
		V = G @ Vf + J @ (V_plus - P) @ J.T
		V12 = J @ V_plus
		return mu, symmetric(V), V12

	def smoother(self,muf,Vf):
		N = muf.size(-1)
		mus, Vs = torch.zeros(self.dimz,N), torch.zeros(self.dimz,self.dimz,N)
		V12 = torch.zeros(self.dimz,self.dimz,N-1)
		mus[:,-1], Vs[:,:,-1] = muf[:,-1].clone(), Vf[:,:,-1].clone()
		for n in range(N-2,-1,-1):
			mus[:,n], Vs[:,:,n], V12[:,:,n] = self.smooth(muf[:,n],Vf[:,:,n],mus[:,n+1],Vs[:,:,n+1])
		return mus, Vs, V12


	def learn(self,x,mu,V,V12):
		N = mu.size(-1)
		zz = mu @ mu.T + V.sum(-1)
		z1z1 = mu[:,:-1] @ mu[:,:-1].T + V[:,:,:-1].sum(-1)
		z1z2 = mu[:,:-1] @ mu[:,1:].T + V12.sum(-1)
		z2z2 = mu[:,1:] @ mu[:,1:].T + V[:,:,1:].sum(-1)
		zx = mu @ x.T
		xx = x @ x.T

		self.m0 = mu[:,0].unsqueeze(-1)
		self.P0 = symmetric(V[:,:,0])

		m0m0 = self.m0 @ self.m0.T
		z0m0 = mu[:,0].unsqueeze(-1) @ self.m0.T
		z0z0 = mu[:,0].unsqueeze(-1) @ mu[:,0].unsqueeze(-1).T + V[:,:,0]
		quad = symmetric(z0z0 - (z0m0 + z0m0.T) + m0m0)
		ell_z0 = -1/2*torch.logdet(2*np.pi*self.P0) - 1/2*torch.trace(self.P0.inverse() @ quad)

		self.A = z1z2.T @ z1z1.inverse()
		AzzA = self.A @ z1z1 @ self.A.T
		Az1z2 = self.A @ z1z2
		quad = symmetric(z2z2 - (Az1z2 + Az1z2.T) + AzzA)
		self.Q = quad/(N-1)
		ell_z = -(N-1)/2*torch.logdet(2*np.pi*self.Q) - 1/2*torch.trace(self.Q.inverse() @ quad)

		self.C = zx.T @ zz.inverse()
		Czx = self.C @ zx
		CzzC = self.C @ zz @ self.C.T
		quad = symmetric(xx - (Czx + Czx.T) + CzzC)
		self.R = quad/N
		ell_x = -N/2*torch.logdet(2*np.pi*self.R) - 1/2*torch.trace(self.R.inverse() @ quad)

		ell = ell_z + ell_x + ell_z0
		return ell

	def forward(self,x):
		muf, Vf = self.filter(x)
		mus, Vs, V12 = self.smoother(muf,Vf)
		ell = self.learn(x,mus,Vs,V12)
		return ell,muf,Vf,mus,Vs,V12

	def sample(self,N=100):
		z = torch.zeros(self.dimz,N)
		x = torch.zeros(self.dimx,N)

		initial = torch.distributions.MultivariateNormal(self.m0.squeeze(-1), self.P0)
		transition = torch.distributions.MultivariateNormal(torch.zeros(self.dimz), self.Q)
		emission = torch.distributions.MultivariateNormal(torch.zeros(self.dimx), self.R)

		z_n = initial.sample()
		for n in range(N):
			z[:,n] = z_n
			x[:,n] = self.C @ z_n + emission.sample()
			# Update state
			z_n = self.A @ z_n + transition.sample()

		return x, z