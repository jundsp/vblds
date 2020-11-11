from __future__ import print_function
import sys
sys.path.append('/utils')
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
from vblds import *
from utils import *

torch.set_default_tensor_type(torch.DoubleTensor)


dimz = 2
dimx = 1

A = get_phasor(torch.tensor(2*np.pi*.02))
Q = torch.eye(dimz)*1e-6
C = torch.zeros(dimx,dimz)
C[0,0] = 1
R = torch.eye(dimx)*1e-2
m0 = torch.randn(dimz)
P0 = torch.eye(dimz)*1e-9

model_true = GSSM(A=A,Q=Q,C=C,R=R,m0=m0,P0=P0)

N = 200
x, z = model_true.sample(N)

dimz = 4

A = torch.eye(dimz)
Q = torch.eye(dimz)*1e-2
C = torch.zeros(dimx,dimz)
C[0,0] = 1
R = torch.eye(dimx)*1e-2
m0 = torch.zeros(dimz)
P0 = torch.eye(dimz)

model = GSSM(A=A,Q=Q,C=C,R=R,m0=m0,P0=P0)

epochs = 100
likelihood = torch.zeros(epochs)
for epoch in range(epochs):
	mu_hat, V_hat, V12 = model.infer(x)
	likelihood[epoch] = model.learn(x,mu_hat,V_hat,V12)
	print(('Epoch {:0.0f} / {:0.0f} ==> Likelihood = {:4.2f}').format(epoch,epochs,likelihood[epoch].detach()))


mu, V = model.filter(x)
x_filt = model.C @ mu
x_smooth = model.C @ mu_hat

plt.plot(x.detach().numpy().T,'b.',label='Data')
plt.plot(x_filt.detach().numpy().T,'r--',label='Filtered')
plt.plot(x_smooth.detach().numpy().T,'k',label='Smoothed')
plt.legend()
plt.xlabel('Time')
plt.xlim((1,N))
plt.grid()
plt.show()

plt.plot(likelihood.numpy())
plt.show()