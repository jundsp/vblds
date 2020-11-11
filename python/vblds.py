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


class GSSM:
	def __init__(self, A=None,C=None,Q=None,R=None,m0=None,P0=None):
		super(GSSM, self).__init__()
		self.A = A
		self.C = C
		self.Q = Q
		self.R = R
		self.m0 = m0
		self.P0 = P0

	def __repr__(self):
		to_print = ('GSSM object'  +  '\n'
		'	A = ' + str(self.A) + '\n'
		'	Q = ' + str(self.Q)) + '\n'
		return to_print

	def __call__(self,input):
		return self.infer(input)

	def infer(self,x):
		mu,V = self.filter(x)
		mu,V,V12 = self.smoother(mu,V)
		return mu, V, V12

	def filter(self,x):
		dimx, N = x.size()
		dimz = self.A.size(0)
		mu = torch.zeros(dimz,N)
		V = torch.zeros(dimz,dimz,N)
		m = self.m0.clone()
		P = self.P0.clone()
		for n in range(N):
			mu[:,n], V[:,:,n] = self.update(x[:,n],m,P)
			m, P = self.predict(mu[:,n], V[:,:,n])
		return mu, V

	def update(self,x,m,P):
		I = torch.eye(m.size(0))
		x_hat = self.C @ m
		S = self.C @ P @ self.C.T + self.R
		K = P @ self.C.T @ S.inverse()
		mu = m + K @ (x - x_hat)
		V = (I - K @ self.C) @ P
		return mu, V

	def predict(self,mu,V):
		m = self.A @ mu
		P = self.A @ V @ self.A.T + self.Q
		return m, P

	def smoother(self,muf,Vf):
		dimz, N = muf.size()

		mu = torch.zeros(dimz,N)
		V = torch.zeros(dimz,dimz,N)
		V12 = torch.zeros(dimz,dimz,N)
		
		mu[:,-1] = muf[:,-1]
		V[:,:,-1] = Vf[:,:,-1]
		for n in range(N-2,-1,-1):
			mu[:,n], V[:,:,n], V12[:,:,n] = self.smooth(mu[:,n+1], V[:,:,n+1], muf[:,n], Vf[:,:,n])

		return mu, V, V12

	def smooth(self,mu,V,muf,Vf):
		m, P = self.predict(muf,Vf)
		J = Vf @ self.A.T @ P.inverse()
		mu = muf + J @ (mu - m)
		V = Vf + J @ (V - P) @ J.T
		V12 = J @ V
		return mu, V, V12

	def learn(self,x,mu,V,V12):
		dimz, N = mu.size()

		z2z2 = mu[:,1:] @ mu[:,1:].T + V[:,:,1:].sum(2)
		z1z1 = mu[:,:-1] @ mu[:,:-1].T + V[:,:,:-1].sum(2)
		z1z2 = mu[:,:-1] @ mu[:,1:].T + V12.sum(2)

		xx = x @ x.T
		xz = x @ mu.T 
		zz = mu @ mu.T + V.sum(2)

		self.A = z1z2.T @ z1z1.inverse()
		self.Q = 1/(N-1)*(z2z2 - self.A @ z1z2)

		self.m0 = mu[:,0]
		self.P0 = V[:,:,0]

		self.C = xz @ zz.inverse()
		self.R = 1/N*(xx - self.C @ xz.T)

		likelihood = -N/2*self.R.logdet() - 1/2*(self.R.inverse() @ (xx - (self.C @ xz.T + xz @ self.C.T) + self.C @ zz @ self.C.T)).trace()
		return likelihood

	def sample(self,N):
		dimz = self.A.size(0)
		dimx = self.C.size(0)

		z = torch.zeros(dimz,N)
		x = torch.zeros(dimx,N)

		initial = torch.distributions.MultivariateNormal(self.m0, self.P0)
		transition = torch.distributions.MultivariateNormal(torch.zeros(dimz), self.Q)
		emission = torch.distributions.MultivariateNormal(torch.zeros(dimx), self.R)

		z_ = initial.sample()
		for n in range(N):
			z[:,n] = z_
			x[:,n] = self.C @ z_ + emission.sample()
			# Update state
			z_ = self.A @ z_ + transition.sample()

		return x, z