from __future__ import print_function
import sys
sys.path.append('/utils')
from vblds import *
from utils import *
import torch
from torch import nn
import numpy as np 
import matplotlib.pyplot as plt
import os
import seaborn as sns

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(1)

dimz = 2
dimx = 1
N = 100

#  Sample Data from an LDS
A = get_phasor(torch.tensor(2*np.pi*4.5/N))
Q = torch.eye(dimz)*1e-3
C = torch.rand(dimx,dimz)*2-1
R = torch.eye(dimx)*1e-2
omega = 2*np.pi*torch.randn(1)
m0 = torch.zeros(dimz,1)
m0[0], m0[1] = torch.cos(omega), torch.sin(omega)
P0 = torch.eye(dimz)*1e-9

model = VBLDS(A=A,Q=Q,C=C,R=R,m0=m0,P0=P0)
x, z = model.sample(N=N)


dimz = 4
m0 = torch.zeros(dimz,1)
P0 = torch.eye(dimz)
A = torch.eye(dimz) + 1e-3*torch.randn(dimz,dimz)
R = torch.eye(dimx)*1e-1
Q = torch.eye(dimz)*1e-2
C = torch.rand(dimx,dimz)*2-1

model = VBLDS(A=A,Q=Q,C=C,R=R,m0=m0,P0=P0)

ELL = []
epochs = 200
for epoch in range(epochs):
	ell, muf, Vf, mus, Vs, V12 = model(x)
	ELL.append(ell)

	print(('Epoch {:0.0f} / {:0.0f} ==> ELL = {:4.2f}').format(epoch,epochs,ell))

xf = model.C @ muf
xs = model.C @ mus


# Plots
t = torch.tensor(range(N))
plt.figure(figsize=(5,5))
plt.subplot(211)
plt.scatter(t,x[0],c='b',s=10)
plt.plot(t,xf[0],c='r')
plt.plot(t,xs[0],c='g')
plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel('Time')

plt.subplot(212)
plt.plot(ELL)
plt.xlabel('Iteration')
plt.autoscale(enable=True, axis='both', tight=True)
plt.tight_layout()
plt.show()

def plot_matrix(A):
	amax = A.abs().max()
	plt.imshow(A,vmin=-amax,vmax=amax)

plt.set_cmap('RdBu') # a good start: blue to white to red colormap
plt.subplot(221)
plot_matrix(model.A)
plt.subplot(222)
plot_matrix(model.Q)
plt.subplot(223)
plot_matrix(model.C)
plt.subplot(224)
plot_matrix(model.R)
plt.show()
