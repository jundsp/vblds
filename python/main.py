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

dimz = 2
dimx = 1
N = 200

A = get_phasor(torch.tensor(2*np.pi*4.5/N))
Q = torch.eye(dimz)*1e-3
C = torch.rand(dimx,dimz)*2-1
R = torch.eye(dimx)*1e-2
omega = 2*np.pi*torch.randn(1)
m0 = torch.zeros(dimz)
m0[0], m0[1] = torch.cos(omega), torch.sin(omega)
P0 = torch.eye(dimz)*1e-9

model = VBLDS(A=A,Q=Q,C=C,R=R,m0=m0,P0=P0)

x, z = model.sample(N=N)

# Plots
t = torch.tensor(range(N))
plt.figure(figsize=(6,2))
plt.scatter(t,x,c='b',s=10)
plt.autoscale(enable=True, axis='both', tight=True)
plt.show()
