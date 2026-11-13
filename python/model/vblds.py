"""Variational Bayesian linear dynamical system (VBLDS) in PyTorch.

Reference: J. Neri, R. Badeau, P. Depalle, "Probabilistic Filter and Smoother
for Variational Inference of Bayesian Linear Dynamical Systems," ICASSP 2020.
"""
import torch
import torch.utils.data
from torch import nn, optim, distributions
from torch.nn import functional as F
import numpy as np

def symmetric(A):
	"""Return (A + A^T) / 2, removing round-off asymmetry."""
	return 0.5*(A + A.T)
	
class VBLDS(nn.Module):
	"""Variational Bayesian linear dynamical system.

	Model (x = observations, z = latent states):
	    z_1 ~ N(m0, P0)
	    z_n = A z_{n-1} + w_n,  w_n ~ N(0, Q)
	    x_n = C z_n + v_n,      v_n ~ N(0, R)

	A, Q, C, R are learned by variational Bayesian EM. The rows of A and C
	have Normal priors (ARD precisions ``alpha``, ``beta``), and the diagonal
	precisions of Q and R have Gamma priors (``c, d`` and ``a, b``). m0 and P0
	are point estimates.

	The attributes A, Q, C, R hold the expected parameters. Sigma_AQA and
	Sigma_CRC hold the parameter-uncertainty corrections
	E[A^T Q^-1 A] - <A>^T <Q^-1> <A> (and the same for C, R), which the
	filter and smoother use. All are ``nn.Parameter`` with
	``requires_grad=False``, so they are saved in ``state_dict()`` and
	updated in closed form, not by gradient descent.

	Args:
	    dimx: observation dimension.
	    dimz: latent state dimension.

	Calling ``model(x)`` runs one VB-EM iteration; see ``forward``.
	"""
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
		"""Return (I + A B)^-1, computed as I - A (I + B A)^-1 B."""
		I = torch.eye(self.dimz)
		return I - A @ torch.inverse(I + B @ A) @ B

	def predict(self,mu,V):
		"""Prediction step of the Bayesian Kalman filter.

		Corrects the posterior N(mu, V) for uncertainty in A, Q with
		G = (I + V Sigma_AQA)^-1, then propagates it one step.

		Returns:
		    m, P: predictive mean (dimz, 1) and covariance (dimz, dimz).
		    G: the correction matrix, which ``smooth`` reuses.
		"""
		G = self.woodbury(V,self.Sigma_AQA)
		m = self.A @ G @ mu
		P = self.A @ G @ V @ self.A.T + self.Q
		return m, symmetric(P), G

	def update(self,x,m,P):
		"""Update step of the Bayesian Kalman filter.

		Conditions the predictive N(m, P) on the observation x (dimx, 1),
		after correcting for uncertainty in C, R with
		L = (I + P Sigma_CRC)^-1.

		Returns:
		    mu: filtered mean (dimz,).
		    V: filtered covariance (dimz, dimz).
		"""
		L = self.woodbury(P,self.Sigma_CRC)
		I = torch.eye(self.dimz)

		x_hat = self.C @ L @ m
		S_hat = self.C @ L @ P @ self.C.T + self.R

		K = L @ P @ self.C.T @ S_hat.inverse()
		mu = L @ m + K @ (x - x_hat)
		V = (I - K @ self.C) @ L @ P
		return mu.squeeze(-1), symmetric(V)

	def filter(self,x):
		"""Run the Bayesian Kalman filter over x of shape (dimx, N).

		Returns:
		    mu: filtered means E[z_n | x_{1:n}], shape (dimz, N).
		    V: filtered covariances, shape (dimz, dimz, N).
		"""
		N = x.size(-1)
		mu, V = torch.zeros(self.dimz,N), torch.zeros(self.dimz,self.dimz,N)
		m, P = self.m0.clone(), self.P0.clone()
		for n in range(N):
			mu[:,n], V[:,:,n] = self.update(x[:,n].unsqueeze(-1),m,P)
			m, P, _ = self.predict(mu[:,n].unsqueeze(-1),V[:,:,n])
		return mu, V

	def smooth(self,muf,Vf,mu_plus,V_plus):
		"""One backward step of the Bayesian RTS smoother.

		Combines the filtered (muf, Vf) at step n with the smoothed
		(mu_plus, V_plus) at step n+1.

		Returns:
		    mu, V: smoothed mean and covariance at step n.
		    V12: smoothed cross-covariance Cov(z_n, z_{n+1}).
		"""
		m,P,G = self.predict(muf,Vf)
		J = G @ Vf @ self.A.T @ P.inverse()

		mu = G @ muf + J @ (mu_plus - m)
		V = G @ Vf + J @ (V_plus - P) @ J.T
		V12 = J @ V_plus
		return mu, symmetric(V), V12

	def smoother(self,muf,Vf):
		"""Run the backward smoother over the filter's output.

		Returns:
		    mus: smoothed means E[z_n | x_{1:N}], shape (dimz, N).
		    Vs: smoothed covariances, shape (dimz, dimz, N).
		    V12: cross-covariances Cov(z_n, z_{n+1}), shape (dimz, dimz, N-1).
		"""
		N = muf.size(-1)
		mus, Vs = torch.zeros(self.dimz,N), torch.zeros(self.dimz,self.dimz,N)
		V12 = torch.zeros(self.dimz,self.dimz,N-1)
		mus[:,-1], Vs[:,:,-1] = muf[:,-1].clone(), Vf[:,:,-1].clone()
		for n in range(N-2,-1,-1):
			mus[:,n], Vs[:,:,n], V12[:,:,n] = self.smooth(muf[:,n],Vf[:,:,n],mus[:,n+1],Vs[:,:,n+1])
		return mus, Vs, V12


	def learn_AQ(self,z2z2,z1z2,z1z1,N):
		"""Update A, Q and Sigma_AQA in place from the transition moments.

		Returns the transition term of the expected log-likelihood.
		"""
		self.Sigma_AQA.fill_(0.0)
		self.Q.fill_(0.0)
		for i in range(self.dimz):
			ell = z1z2[:,i]
			Lam = z1z1 + self.alpha[i].diag()
			Sigma_A = Lam.inverse()
			self.A[i] = ell @ Sigma_A

			self.Sigma_AQA += Sigma_A

			c_hat = self.c + 1/2*(N-1)
			d_hat = self.d + 1/2*(z2z2[i,i] - self.A[i] @ ell)
			self.Q[i,i] = d_hat/c_hat

		AzzA = self.A @ z1z1 @ self.A.T
		Az1z2 = self.A @ z1z2
		quad = symmetric(z2z2 - (Az1z2 + Az1z2.T) + AzzA)
		ell = -(N-1)/2*torch.logdet(2*np.pi*self.Q) - 1/2*torch.trace(self.Q.inverse() @ quad)
		return ell

	def learn_CR(self,xx,zx,zz,N):
		"""Update C, R and Sigma_CRC in place from the observation moments.

		Returns the observation term of the expected log-likelihood.
		"""
		self.Sigma_CRC.fill_(0.0)
		self.R.fill_(0.0)
		for i in range(self.dimx):
			ell = zx[:,i]
			Lam = zz + self.beta[i].diag()
			Sigma_C = Lam.inverse()
			self.C[i] = ell @ Sigma_C

			self.Sigma_CRC += Sigma_C

			a_hat = self.a + 1/2*N
			b_hat = self.b + 1/2*(xx[i,i] - self.C[i] @ ell)
			self.R[i,i] = b_hat/a_hat
		Czx = self.C @ zx
		CzzC = self.C @ zz @ self.C.T
		quad = symmetric(xx - (Czx + Czx.T) + CzzC)
		ell = -N/2*torch.logdet(2*np.pi*self.R) - 1/2*torch.trace(self.R.inverse() @ quad)
		return ell

	def learn_mP(self,mu,V):
		"""Set m0, P0 to the smoothed moments of the first state.

		Returns the initial-state term of the expected log-likelihood.
		"""
		self.m0.data = mu
		self.P0.data = symmetric(V)

		m0m0 = self.m0 @ self.m0.T
		z0m0 = mu @ self.m0.T
		z0z0 = mu @ mu.T + V
		quad = symmetric(z0z0 - (z0m0 + z0m0.T) + m0m0)
		ell = -1/2*torch.logdet(2*np.pi*self.P0) - 1/2*torch.trace(self.P0.inverse() @ quad)
		return ell


	def learn(self,x,mu,V,V12):
		"""M-step: update all parameters from the smoothed moments.

		Returns:
		    The expected complete-data log-likelihood E_q[log p(x, z)] at
		    the updated parameters. It does not include the prior and
		    entropy terms of the full variational lower bound.
		"""
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
		"""Run one VB-EM iteration on x of shape (dimx, N).

		Runs the filter, the smoother, then the parameter update (in place).

		Returns:
		    ell: expected log-likelihood after the update.
		    muf, Vf: filtered means (dimz, N) and covariances (dimz, dimz, N).
		    mus, Vs: smoothed means and covariances, same shapes.
		    V12: smoothed cross-covariances (dimz, dimz, N-1).
		"""
		muf, Vf = self.filter(x)
		mus, Vs, V12 = self.smoother(muf,Vf)
		ell = self.learn(x,mus,Vs,V12)
		return ell,muf,Vf,mus,Vs,V12

	def sample(self,N=100):
		"""Draw a length-N sequence from the model with the current parameters.

		Returns:
		    x: observations, shape (dimx, N).
		    z: latent states, shape (dimz, N).
		"""
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