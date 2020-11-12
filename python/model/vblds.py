import torch
import torch.utils.data
from torch import nn, optim, distributions
from torch.nn import functional as F
import numpy as np

def symmetric(A):
	return 0.5*(A + A.T)
	
class VBLDS(nn.Module):
	def __init__(self, dimx, dimz):
		super(VBLDS, self).__init__()
		self.dimx, self.dimz = dimx, dimz

		m0 = torch.zeros(dimz,1)
		P0 = torch.eye(dimz)
		A = torch.eye(dimz) + 1e-3*torch.randn(dimz,dimz)
		R = torch.eye(dimx)*1e-1
		Q = torch.eye(dimz)*1e-2
		C = torch.rand(dimx,dimz)
		Sigma_AQA = 1e-9*torch.ones(self.dimz,self.dimz)
		Sigma_CRC = 1e-9*torch.ones(self.dimz,self.dimz)

		# Model Parameters
		self.register_parameter('A',nn.Parameter(A,requires_grad=False))
		self.register_parameter('C',nn.Parameter(C,requires_grad=False))
		self.register_parameter('Q',nn.Parameter(Q,requires_grad=False))
		self.register_parameter('R',nn.Parameter(R,requires_grad=False))
		self.register_parameter('m0',nn.Parameter(m0,requires_grad=False))
		self.register_parameter('P0',nn.Parameter(P0,requires_grad=False))
		self.register_parameter('Sigma_AQA',nn.Parameter(Sigma_AQA,requires_grad=False))
		self.register_parameter('Sigma_CRC',nn.Parameter(Sigma_CRC,requires_grad=False))
		
		# Model Hyperparameters (constant)
		self.alpha = 1e-6*torch.ones(self.dimz,self.dimz)
		self.beta = 1e-6*torch.ones(self.dimx,self.dimz)
		self.a, self.b, self.c, self.d = 1e-9, 1e-9, 1e-9, 1e-9

	def __repr__(self):
		to_print = ['::: VBLDS object :::'  +  '\n']
		for name, param in self.named_parameters():
			to_print.append(name + ' = ' + str(param.data) + '\n')
		return ''.join(to_print)

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


	def learn_AQ(self,z2z2,z1z2,z1z1,N):
		self.Sigma_AQA.fill_(0.0)
		self.Q.fill_(0.0)
		for i in range(self.dimz):
			ell = z1z2[:,i].T
			Lam = z1z1 + self.alpha[i].diag()
			Sigma_A = Lam.inverse()
			self.A[i] = ell @ Sigma_A

			self.Sigma_AQA += Sigma_A

			c_hat = self.c + 1/2*(N-1)
			d_hat = self.d + 1/2*(z2z2[i,i] - self.A[i] @ ell.T)
			self.Q[i,i] = d_hat/c_hat

		AzzA = self.A @ z1z1 @ self.A.T
		Az1z2 = self.A @ z1z2
		quad = symmetric(z2z2 - (Az1z2 + Az1z2.T) + AzzA)
		ell = -(N-1)/2*torch.logdet(2*np.pi*self.Q) - 1/2*torch.trace(self.Q.inverse() @ quad)
		return ell

	def learn_CR(self,xx,zx,zz,N):
		self.Sigma_CRC.fill_(0.0)
		self.R.fill_(0.0)
		for i in range(self.dimx):
			ell = zx[:,i].T
			Lam = zz + self.beta[i].diag()
			Sigma_C = Lam.inverse()
			self.C[i] = ell @ Sigma_C

			self.Sigma_CRC += Sigma_C

			a_hat = self.a + 1/2*N
			b_hat = self.b + 1/2*(xx[i,i] - self.C[i] @ ell.T)
			self.R[i,i] = b_hat/a_hat
		Czx = self.C @ zx
		CzzC = self.C @ zz @ self.C.T
		quad = symmetric(xx - (Czx + Czx.T) + CzzC)
		ell = -N/2*torch.logdet(2*np.pi*self.R) - 1/2*torch.trace(self.R.inverse() @ quad)
		return ell

	def learn_mP(self,mu,V):
		self.m0.data = mu
		self.P0.data = symmetric(V)

		m0m0 = self.m0 @ self.m0.T
		z0m0 = mu @ self.m0.T
		z0z0 = mu @ mu.T + V
		quad = symmetric(z0z0 - (z0m0 + z0m0.T) + m0m0)
		ell = -1/2*torch.logdet(2*np.pi*self.P0) - 1/2*torch.trace(self.P0.inverse() @ quad)
		return ell


	def learn(self,x,mu,V,V12):
		N = mu.size(-1)
		zz = mu @ mu.T + V.sum(-1)
		z1z1 = mu[:,:-1] @ mu[:,:-1].T + V[:,:,:-1].sum(-1)
		z1z2 = mu[:,:-1] @ mu[:,1:].T + V12.sum(-1)
		z2z2 = mu[:,1:] @ mu[:,1:].T + V[:,:,1:].sum(-1)
		zx = mu @ x.T
		xx = x @ x.T

		ell_z0 = self.learn_mP(mu[:,0].unsqueeze(-1),V[:,:,0])
		ell_z = self.learn_AQ(z2z2,z1z2,z1z1,N)
		ell_x = self.learn_CR(xx,zx,zz,N)

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
			z_n = self.A @ z_n + transition.sample()
		return x, z