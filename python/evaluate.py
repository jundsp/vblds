from __future__ import print_function
import sys
from data.data_loader import *
from utils.utils import *
from model.vblds import *
import torch
from torch import nn
import numpy as np 
import matplotlib.pyplot as plt
import os

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(1)

# Load the data
x,z = load_data(N=200)

dimx, N = x.size()

# Create VBLDS model 
dimz = 8 # Dimension of latent space
model = VBLDS(dimx=dimx,dimz=dimz)
# Load Saved Model
model.load_state_dict(torch.load('saves/model.pt',map_location=torch.device('cpu')))

ell, muf, Vf, mus, Vs, V12 = model(x)

# Filtered and Smoothed data
xf = model.C @ muf
xs = model.C @ mus

# Sample from learned model
x_sample,z_sample = model.sample(N=N)

# PLOT
print('Plotting results.')
t = torch.tensor(range(N))
plt.figure(figsize=(6,4))
plt.subplot(211)
plt.scatter(t,x[0],c='b',s=10,label='Data x')
plt.plot(t,xf[0],c='r',label='Filtered')
plt.plot(t,xs[0],c='g',label='Smoothed')
plt.legend()
plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel('Time')

plt.subplot(212)
t = torch.tensor(range(x_sample.size(-1)))
plt.scatter(t,x_sample[0],c='g',s=10)
plt.plot(t,(model.C@z_sample)[0],c='m')
plt.title('Sampled from Learned Model')
plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel('Time')
plt.tight_layout()
plt.show()

plt.set_cmap('RdBu') # a good start: blue to white to red colormap
plt.subplot(221)
plot_matrix(plt, model.A), plt.title('A'), plt.axis('off')
plt.subplot(222)
plot_matrix(plt,model.Q), plt.title('Q'), plt.axis('off')
plt.subplot(223)
plot_matrix(plt,model.C), plt.title('C'), plt.axis('off')
plt.subplot(224)
plot_matrix(plt,model.R), plt.title('R'), plt.axis('off')
plt.tight_layout()
plt.show()
