"""Evaluate the trained VBLDS on a new synthetic sequence and plot the results.

Loads saves/model.npz (run train.py first), runs the filter and smoother on a
new 200-step sequence, samples a new sequence from the learned model, and
writes three figures:

- results/results.png: the test data with the marginal distribution of each
  observation (mean C mu, covariance C V C^T + R) under the filtered and
  smoothed posteriors, and a sequence sampled from the learned model.
- results/training.png: the training curves from saves/history.npz (ELL and
  predictive log-likelihood per iteration).
- results/learned_parameters.png: the learned A, Q, C, R.

Note: calling model(y) runs one VB-EM iteration, so the loaded parameters are
also updated once on the test sequence.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from data.data_loader import load_data
from model.vblds import VBLDS
from utils.utils import plot_matrix

print('Evaluating on test data...')
rng = np.random.default_rng(1)

# Load the data and the trained model
y, x = load_data(N=200, rng=rng)
N = y.shape[1]
model = VBLDS.load(os.path.join('saves', 'model.npz'), rng=rng)

result = model(y)

# Marginal distribution of each observation under the posterior over x_n:
#   mean C mu_n,  covariance C V_n C^T + R
# for the filtered (mu_n, V_n given y_{1:n}) and smoothed (given y_{1:N})
# posteriors. Plotted for the first observation dimension.
def observation_marginal(mu, V):
    mean = model.C @ mu
    cov = np.einsum('ij,jkn,lk->iln', model.C, V, model.C) + model.R[:, :, None]
    return mean, np.sqrt(np.einsum('iin->in', cov))


yf, yf_std = observation_marginal(result.muf, result.Vf)
ys, ys_std = observation_marginal(result.mus, result.Vs)

# Sample from the learned model
y_sample, x_sample = model.sample(N=N)

# Training curves saved by train.py
history_path = os.path.join('saves', 'history.npz')
history = np.load(history_path) if os.path.exists(history_path) else None

os.makedirs('results', exist_ok=True)

# Test data with the observation marginals, and a generated sequence
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 7.5))

t = np.arange(N)
ax.fill_between(t, ys[0] - 2 * ys_std[0], ys[0] + 2 * ys_std[0],
                color='g', alpha=0.15, linewidth=0,
                label=r'Smoothed $\pm 2\sigma$, $C V C^T + R$')
ax.scatter(t, y[0], c='b', s=10, label='Data y')
ax.plot(t, yf[0], c='r', lw=1, label=r'Filtered $C\mu$')
ax.plot(t, ys[0], c='g', lw=1.5, label=r'Smoothed $C\hat\mu$')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4,
          fontsize='small', frameon=False,
          title='Test data with the marginal distribution of each observation')
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_xlabel('Time')

ax = ax2
t = np.arange(y_sample.shape[-1])
ax.scatter(t, y_sample[0], c='g', s=10, label='Sampled y')
ax.plot(t, (model.C @ x_sample)[0], c='m', label='Sampled C x')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
          fontsize='small', frameon=False,
          title='Sequence sampled from the learned model')
ax.autoscale(enable=True, axis='x', tight=True)
ax.set_xlabel('Time')

fig.tight_layout()
fig.savefig(os.path.join('results', 'results.png'), dpi=120)

# Training curves, one value per VB-EM iteration
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
curves = [
    ('loglik', 'Predictive log-likelihood\n$\\sum_n \\log N(y_n;\\, \\hat y_n, S_n)$'),
    ('ell', 'Expected log-likelihood (ELL)\n$E_q[\\log p(y, x)]$'),
]
for k, (key, title) in enumerate(curves):
    ax = axes[k]
    if history is not None:
        ax.plot(np.arange(1, len(history[key]) + 1), history[key], c='k')
        ax.autoscale(enable=True, axis='x', tight=True)
    else:
        ax.text(0.5, 0.5, 'Run train.py first', ha='center', transform=ax.transAxes)
    ax.set_title(title, fontsize='medium')
    ax.set_xlabel('VB-EM iteration')

fig.tight_layout()
fig.savefig(os.path.join('results', 'training.png'), dpi=120)

# Learned parameters
plt.figure(figsize=(6, 4))
plt.set_cmap('RdBu')  # blue-white-red colormap, centered on zero by plot_matrix
for k, name in enumerate(['A', 'Q', 'C', 'R']):
    plt.subplot(2, 2, k + 1)
    plot_matrix(plt, getattr(model, name))
    plt.title(name)
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join('results', 'learned_parameters.png'))

print('Images saved in "results" folder.')
print('Done!')
