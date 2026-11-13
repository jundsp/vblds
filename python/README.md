# VBLDS in PyTorch

A PyTorch implementation of variational Bayesian learning for linear dynamical
systems. See the [top-level README](../README.md) for the model and the
algorithm.

> **Notation:** this code uses `x` for **observations** and `z` for **latent
> states**, the reverse of the paper and the MATLAB code. `dimx` is the
> observation dimension and `dimz` is the latent dimension.

## Requirements

Python 3.10 or newer with `torch`, `numpy` and `matplotlib`. Runs on CPU in
double precision. No GPU is needed.

## Quick start

From this folder (`vblds/python`), with [uv](https://docs.astral.sh/uv/):

```bash
uv run train.py      # fits the model, saves saves/model.pt
uv run evaluate.py   # loads it, writes figures to results/
```

The first `uv run` creates a `.venv/` here and installs the versions pinned in
`uv.lock`. Later runs reuse it. This folder is not an installable package;
`pyproject.toml` only lists the dependencies.

Without uv, install the three dependencies in any environment and run the
scripts with Python:

```bash
pip install torch numpy matplotlib
python train.py
python evaluate.py
```

| Script        | What it does |
| ------------- | ------------ |
| `train.py`    | Samples a 100-step training sequence from a known 2-D oscillator LDS (`data/data_loader.py`), fits an 8-dimensional VBLDS with 200 VB-EM iterations, and saves the parameters to `saves/model.pt`. |
| `evaluate.py` | Samples a new 200-step sequence from the same kind of system, runs the trained model's filter and smoother on it, draws a new sequence from the learned model, and saves `results/results.png` (data, filtered and smoothed reconstructions, generated sample) and `results/learned_parameters.png` (A, Q, C, R). |

Both scripts delete and recreate their output folder each time they run.

## Using the model

```python
import torch
from model.vblds import VBLDS

torch.set_default_dtype(torch.float64)

x = ...                       # observations, shape (dimx, N)
model = VBLDS(dimx=x.size(0), dimz=8)

for it in range(200):
    ell, muf, Vf, mus, Vs, V12 = model(x)   # one VB-EM iteration

x_hat = model.C @ mus                       # smoothed reconstruction
x_new, z_new = model.sample(N=500)          # generate from the learned model
```

Each call to `model(x)` runs **one full VB-EM iteration**: the Bayesian Kalman
filter, the smoother, then the variational parameter update. The parameters are
updated in place. They are stored as `nn.Parameter`s with
`requires_grad=False`, so they are saved in `state_dict()` but no gradients
are used.

| `forward(x)` returns | Shape | Meaning |
| -------------------- | ----- | ------- |
| `ell`  | scalar          | Expected log-likelihood after the update |
| `muf`  | (dimz, N)       | Filtered means E[z_n \| x_{1:n}] |
| `Vf`   | (dimz, dimz, N) | Filtered covariances |
| `mus`  | (dimz, N)       | Smoothed means E[z_n \| x_{1:N}] |
| `Vs`   | (dimz, dimz, N) | Smoothed covariances |
| `V12`  | (dimz, dimz, N-1) | Smoothed cross-covariances Cov(z_n, z_{n+1}) |

Model attributes: `A`, `Q`, `C`, `R`, `m0`, `P0` (the expected parameters)
and `Sigma_AQA`, `Sigma_CRC` (corrections for parameter uncertainty). The
prior hyperparameters are `alpha`, `beta` (ARD precisions on the rows of A and
C) and `a, b, c, d` (Gamma priors on the noise precisions). They are set in
`__init__`.

## Layout

```
train.py              Training script
evaluate.py           Evaluation and plotting script
pyproject.toml        Dependencies, for uv (not a package definition)
uv.lock               Exact dependency versions used by uv run
model/vblds.py        VBLDS class
data/data_loader.py   load_data(N): samples from a known 2-D oscillator LDS
utils/utils.py        get_phasor, plot_matrix, symmetric
```
