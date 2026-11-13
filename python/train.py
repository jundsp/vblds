"""Train a VBLDS on a synthetic sequence and save it to saves/model.npz.

Samples 100 steps from a known 2-D oscillator LDS (data/data_loader.py), fits
an 8-dimensional VBLDS with 300 VB-EM iterations, and prints the expected
log-likelihood (ELL) and the filter's predictive log-likelihood after each
iteration. Saves both curves to saves/history.npz. Run evaluate.py afterwards.
"""
import os

import numpy as np

from data.data_loader import load_data
from model.vblds import VBLDS

COVARIANCE = 'diagonal'  # form of Q and R: 'diagonal' or 'full'
DIMX = 8                 # latent state dimension
ITERATIONS = 300

rng = np.random.default_rng(0)

# Load the data
y, x = load_data(N=100, rng=rng)
dimy, N = y.shape

# Create and train the model
model = VBLDS(dimy=dimy, dimx=DIMX, covariance=COVARIANCE, rng=rng)
history = model.fit(y, iterations=ITERATIONS, verbose=True)

# Save the model and the training curves (plotted by evaluate.py)
os.makedirs('saves', exist_ok=True)
model.save(os.path.join('saves', 'model.npz'))
np.savez(os.path.join('saves', 'history.npz'), **history._asdict())
print('Saved the model to saves/model.npz and training curves to saves/history.npz.')
