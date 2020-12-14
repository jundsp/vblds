# VBLDS: with PyTorch 

Variational inference of a Bayesian linear dynamical system, with PyTorch.

train.py - runs the variational EM algorithm to learn the Bayesian LDS given data sequence(s).
evalual.py - evaluates the trained model on unseen test data. Also generates new data from the learned model. Plots results.

Easy usage:
* Change the working directory to vblds/python
* Execute the following command in Unix: ./run.sh

This will make a virtual env in the directory, install all packages, then train and evaluate.
