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
import shutil

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0)

# Load the data
x,z = load_data(N=100)

dimx, N = x.size()

# Create VBLDS model 
dimz = 8 # Dimension of latent space
model = VBLDS(dimx=dimx,dimz=dimz)

# Train
epochs = 100
for epoch in range(epochs):
	ell, muf, Vf, mus, Vs, V12 = model(x)

	print(('Epoch {:0.0f} / {:0.0f} ==> ELL = {:4.2f}').format(epoch,epochs,ell))

# Save the model.
root = 'saves'
if os.path.exists(root):
    shutil.rmtree(root)
os.mkdir(root)
torch.save(model.state_dict(),os.path.join(root,'model.pt'))
print('Saved trained model.')

