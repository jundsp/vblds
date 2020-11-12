from __future__ import print_function
import sys
from vblds import *
from utils.utils import *
import torch
from torch import nn
import numpy as np 
import matplotlib.pyplot as plt
import os

torch.set_default_tensor_type(torch.DoubleTensor)
# torch.manual_seed(1)

dimz = 2
dimx = 1
N = 200

#  Sample Data from an LDS
A = get_phasor(torch.tensor(2*np.pi*8.55/100))*.99
Q = torch.eye(dimz)*1e-2
C = torch.rand(dimx,dimz)*2-1
R = torch.eye(dimx)*.1
omega = 2*np.pi*torch.randn(1)
m0 = torch.zeros(dimz,1)
m0[0], m0[1] = torch.cos(omega), torch.sin(omega)
P0 = torch.eye(dimz)*1e-9

model = VBLDS(A=A,Q=Q,C=C,R=R,m0=m0,P0=P0)
x, z = model.sample(N=N)


dimz = 8
m0 = torch.zeros(dimz,1)
P0 = torch.eye(dimz)
A = torch.eye(dimz) + 1e-3*torch.randn(dimz,dimz)
R = torch.eye(dimx)*1e-1
Q = torch.eye(dimz)*1e-2
C = torch.rand(dimx,dimz)

model = VBLDS(A=A,Q=Q,C=C,R=R,m0=m0,P0=P0)

# model.load_state_dict(torch.load('saves/model.pt',map_location=torch.device('cpu')))
print(model)

ELL = []
epochs = 10
for epoch in range(epochs):
	ell, muf, Vf, mus, Vs, V12 = model(x)
	ELL.append(ell)

	print(('Epoch {:0.0f} / {:0.0f} ==> ELL = {:4.2f}').format(epoch,epochs,ell))

xf = model.C @ muf
xs = model.C @ mus

x_sample,z_sample = model.sample(N=N)

torch.save(model.state_dict(),'saves/model.pt')

# Plots
t = torch.tensor(range(N))
plt.figure(figsize=(5,5))
plt.subplot(311)
plt.scatter(t,x[0],c='b',s=10)
plt.plot(t,xf[0],c='r')
plt.plot(t,xs[0],c='g')
plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel('Time')

plt.subplot(312)
t = torch.tensor(range(x_sample.size(-1)))
plt.scatter(t,x_sample[0],c='g',s=10)
plt.plot(t,(model.C@z_sample)[0],c='m')

plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel('Time')

plt.subplot(313)
plt.plot(ELL)
plt.xlabel('Iteration')
plt.autoscale(enable=True, axis='both', tight=True)
plt.tight_layout()
plt.show()

plt.set_cmap('RdBu') # a good start: blue to white to red colormap
plt.subplot(221)
plot_matrix(plt, model.A)
plt.subplot(222)
plot_matrix(plt,model.Q)
plt.subplot(223)
plot_matrix(plt,model.C)
plt.subplot(224)
plot_matrix(plt,model.R)
plt.show()

plt.plot(mus.T)
plt.show()
