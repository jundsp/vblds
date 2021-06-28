# VBLDS: with PyTorch 

Variational inference of a Bayesian linear dynamical system. Written with PyTorch for fast learning over multiple data sequences.

# File descriptions

train.py - runs the variational EM algorithm to learn the Bayesian LDS given data sequence(s).

evaluate.py - evaluates the trained model on unseen test data. Also generates new data from the learned model. Plots results.

# Easy usage:
* Change the working directory to vblds/python
* Execute the following command in Unix: ./run.sh

This will make a virtual env in the directory, install all packages, then train and evaluate.
