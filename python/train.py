"""Train a VBLDS on a synthetic sequence and save it to saves/model.pt.

Samples 100 steps from a known 2-D oscillator LDS (data/data_loader.py), fits
an 8-dimensional VBLDS with 200 VB-EM iterations, and prints the expected
log-likelihood (ELL) after each iteration. Run evaluate.py afterwards.
"""
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

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# Load the data
x,z = load_data(N=100)

dimx, N = x.size()

# Create VBLDS model 
dimz = 8 # Dimension of latent space
model = VBLDS(dimx=dimx,dimz=dimz)

# Train
epochs = 200
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

