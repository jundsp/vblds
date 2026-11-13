"""Variational Bayesian linear dynamical system (VBLDS) in NumPy.

Reference: J. Neri, R. Badeau, P. Depalle, "Probabilistic Filter and Smoother
for Variational Inference of Bayesian Linear Dynamical Systems," ICASSP 2020.
"""
from collections import namedtuple

import numpy as np

StepResult = namedtuple(
    'StepResult', 'ell loglik muf Vf mus Vs V12 y_pred S')
StepResult.__doc__ = """Output of one VB-EM iteration (``VBLDS.step``).

    ell: expected log-likelihood after the parameter update.
    loglik: log-likelihood of the data under the filter's predictive
        distribution, sum_n log N(y_n; y_pred_n, S_n), with the parameters
        from before the update.
    muf, Vf: filtered means (dimx, N) and covariances (dimx, dimx, N).
    mus, Vs: smoothed means and covariances, same shapes.
    V12: smoothed cross-covariances Cov(x_n, x_{n+1}), (dimx, dimx, N-1).
    y_pred: one-step predicted means E[y_n | y_{1:n-1}], (dimy, N).
    S: one-step predicted covariances, (dimy, dimy, N).
"""

FitHistory = namedtuple('FitHistory', 'ell loglik')
FitHistory.__doc__ = """Per-iteration ``ell`` and ``loglik`` from ``VBLDS.fit``."""


def symmetric(A):
    """Return (A + A^T) / 2, removing round-off asymmetry."""
    return 0.5 * (A + A.T)


def woodbury(A, B):
    """Return (I + A B)^-1, computed as I - A (I + B A)^-1 B."""
    eye = np.eye(A.shape[0])
    return eye - A @ np.linalg.solve(eye + B @ A, B)


class VBLDS:
    """Variational Bayesian linear dynamical system.

    Model (y = observations, x = latent states):
        x_1 ~ N(m0, P0)
        x_n = A x_{n-1} + w_n,  w_n ~ N(0, Q)
        y_n = C x_n + v_n,      v_n ~ N(0, R)

    A, Q, C, R are learned by variational Bayesian EM. The E-step is the
    Bayesian Kalman filter and RTS smoother from the paper. The M-step depends
    on ``covariance``:

    - ``'diagonal'``: each row of A (C) has a Normal prior with ARD precisions
      ``alpha`` (``beta``), and each diagonal precision of Q (R) has a Gamma
      prior with shape and rate ``c, d`` (``a, b``). Q and R are diagonal.
    - ``'full'``: matrix-normal-Wishart priors on (A, Q) and (C, R), so Q and
      R are full covariance matrices. The ARD precisions are shared by all
      rows (the first row of ``alpha``/``beta`` is used), and the Wishart
      priors are matched to the Gamma priors (degrees of freedom ``2c``,
      inverse scale ``2d I``; ``2a``, ``2b I`` for R). With equal ARD
      precisions, the diagonal of a ``'full'`` update equals the
      ``'diagonal'`` update.

    m0 and P0 are point estimates.

    The attributes A, Q, C, R hold the expected parameters (Q and R are the
    inverse expected precisions). Sigma_AQA and Sigma_CRC hold the
    parameter-uncertainty corrections E[A^T Q^-1 A] - <A>^T <Q^-1> <A> (and
    the same for C, R) that the filter and smoother use.

    Args:
        dimy: observation dimension.
        dimx: latent state dimension.
        covariance: ``'diagonal'`` (default) or ``'full'``.
        rng: ``numpy.random.Generator`` or seed used for the random initial
            A and C, and as the default for ``sample``.

    Calling ``model(y)`` runs one VB-EM iteration and returns a
    ``StepResult``; see ``step``.
    """

    PARAMETERS = ('A', 'Q', 'C', 'R', 'm0', 'P0', 'Sigma_AQA', 'Sigma_CRC')

    def __init__(self, dimy, dimx, covariance='diagonal', rng=None):
        if covariance not in ('diagonal', 'full'):
            raise ValueError("covariance must be 'diagonal' or 'full'")
        self.dimy, self.dimx, self.covariance = dimy, dimx, covariance
        self.rng = np.random.default_rng(rng)

        # Model parameters
        self.A = np.eye(dimx) + 1e-3 * self.rng.standard_normal((dimx, dimx))
        self.Q = 1e-2 * np.eye(dimx)
        self.C = self.rng.random((dimy, dimx))
        self.R = 1e-1 * np.eye(dimy)
        self.m0 = np.zeros(dimx)
        self.P0 = np.eye(dimx)
        self.Sigma_AQA = 1e-9 * np.eye(dimx)
        self.Sigma_CRC = 1e-9 * np.eye(dimx)

        # Prior hyperparameters (constant)
        self.alpha = 1e-6 * np.ones((dimx, dimx))
        self.beta = 1e-6 * np.ones((dimy, dimx))
        self.a, self.b, self.c, self.d = 1e-9, 1e-9, 1e-9, 1e-9

    def __repr__(self):
        lines = [(f'VBLDS(dimy={self.dimy}, dimx={self.dimx}, '
                  f"covariance='{self.covariance}')")]
        for name in self.PARAMETERS:
            lines.append(f'{name} =\n{getattr(self, name)}')
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # E-step: Bayesian Kalman filter and RTS smoother
    # ------------------------------------------------------------------

    def predict(self, mu, V):
        """Prediction step of the Bayesian Kalman filter.

        Corrects the posterior N(mu, V) for uncertainty in A, Q with
        G = (I + V Sigma_AQA)^-1, then propagates it one step.

        Returns:
            m, P: predictive mean (dimx,) and covariance (dimx, dimx).
            G: the correction matrix, which ``smooth`` reuses.
        """
        G = woodbury(V, self.Sigma_AQA)
        m = self.A @ G @ mu
        P = self.A @ G @ V @ self.A.T + self.Q
        return m, symmetric(P), G

    def update(self, y, m, P):
        """Update step of the Bayesian Kalman filter.

        Conditions the predictive N(m, P) on the observation y (dimy,),
        after correcting for uncertainty in C, R with
        L = (I + P Sigma_CRC)^-1.

        Returns:
            mu: filtered mean (dimx,).
            V: filtered covariance (dimx, dimx).
            y_hat: predicted observation mean E[y_n | y_{1:n-1}] (dimy,).
            S: predicted observation covariance (dimy, dimy). Together,
                N(y_hat, S) is the paper's q(y_n | y_{1:n-1}) (Eq. 19-22).
        """
        L = woodbury(P, self.Sigma_CRC)
        y_hat = self.C @ L @ m
        S = self.C @ L @ P @ self.C.T + self.R
        K = np.linalg.solve(S, self.C @ L @ P).T  # = L P C^T S^-1 (S symmetric)
        mu = L @ m + K @ (y - y_hat)
        V = (np.eye(self.dimx) - K @ self.C) @ L @ P
        return mu, symmetric(V), y_hat, symmetric(S)

    def filter(self, y):
        """Run the Bayesian Kalman filter over y of shape (dimy, N).

        Returns:
            mu: filtered means E[x_n | y_{1:n}], shape (dimx, N).
            V: filtered covariances, shape (dimx, dimx, N).
            y_pred: one-step predicted means E[y_n | y_{1:n-1}], (dimy, N).
            S: one-step predicted covariances, (dimy, dimy, N).
            loglik: log N(y_n; y_pred_n, S_n) for each n, shape (N,).
        """
        N = y.shape[-1]
        mu, V = np.zeros((self.dimx, N)), np.zeros((self.dimx, self.dimx, N))
        y_pred = np.zeros((self.dimy, N))
        S = np.zeros((self.dimy, self.dimy, N))
        loglik = np.zeros(N)
        m, P = self.m0.copy(), self.P0.copy()
        for n in range(N):
            mu[:, n], V[:, :, n], y_pred[:, n], S[:, :, n] = self.update(y[:, n], m, P)
            r = y[:, n] - y_pred[:, n]
            _, logdet = np.linalg.slogdet(2 * np.pi * S[:, :, n])
            loglik[n] = -0.5 * (logdet + r @ np.linalg.solve(S[:, :, n], r))
            m, P, _ = self.predict(mu[:, n], V[:, :, n])
        return mu, V, y_pred, S, loglik

    def smooth(self, muf, Vf, mu_plus, V_plus):
        """One backward step of the Bayesian RTS smoother.

        Combines the filtered (muf, Vf) at step n with the smoothed
        (mu_plus, V_plus) at step n+1.

        Returns:
            mu, V: smoothed mean and covariance at step n.
            V12: smoothed cross-covariance Cov(x_n, x_{n+1}).
        """
        m, P, G = self.predict(muf, Vf)
        J = np.linalg.solve(P, self.A @ Vf.T @ G.T).T  # = G Vf A^T P^-1
        mu = G @ muf + J @ (mu_plus - m)
        V = G @ Vf + J @ (V_plus - P) @ J.T
        V12 = J @ V_plus
        return mu, symmetric(V), V12

    def smoother(self, muf, Vf):
        """Run the backward smoother over the filter's output.

        Returns:
            mus: smoothed means E[x_n | y_{1:N}], shape (dimx, N).
            Vs: smoothed covariances, shape (dimx, dimx, N).
            V12: cross-covariances Cov(x_n, x_{n+1}), shape (dimx, dimx, N-1).
        """
        N = muf.shape[-1]
        mus, Vs = muf.copy(), Vf.copy()
        V12 = np.zeros((self.dimx, self.dimx, N - 1))
        for n in range(N - 2, -1, -1):
            mus[:, n], Vs[:, :, n], V12[:, :, n] = self.smooth(
                muf[:, n], Vf[:, :, n], mus[:, n + 1], Vs[:, :, n + 1])
        return mus, Vs, V12

    # ------------------------------------------------------------------
    # M-step: variational parameter updates
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_ell(n, cov, quad):
        """-n/2 log|2 pi cov| - 1/2 tr(cov^-1 quad)."""
        _, logdet = np.linalg.slogdet(2 * np.pi * cov)
        return -n / 2 * logdet - 0.5 * np.trace(np.linalg.solve(cov, quad))

    def learn_AQ(self, x2x2, x1x2, x1x1, N):
        """Update A, Q and Sigma_AQA from the transition moments.

        Returns the transition term of the expected log-likelihood.
        """
        if self.covariance == 'full':
            Lam = x1x1 + np.diag(self.alpha[0])
            Lam_inv = np.linalg.inv(Lam)
            self.A = x1x2.T @ Lam_inv
            Winv = 2 * self.d * np.eye(self.dimx) + x2x2 - self.A @ Lam @ self.A.T
            self.Q = symmetric(Winv / (2 * self.c + N - 1))
            self.Sigma_AQA = self.dimx * Lam_inv
        else:
            self.Sigma_AQA = np.zeros((self.dimx, self.dimx))
            self.Q = np.zeros((self.dimx, self.dimx))
            for i in range(self.dimx):
                ell = x1x2[:, i]
                Sigma_A = np.linalg.inv(x1x1 + np.diag(self.alpha[i]))
                self.A[i] = ell @ Sigma_A
                self.Sigma_AQA += Sigma_A
                c_hat = self.c + 0.5 * (N - 1)
                d_hat = self.d + 0.5 * (x2x2[i, i] - self.A[i] @ ell)
                self.Q[i, i] = d_hat / c_hat

        Ax1x2 = self.A @ x1x2
        quad = symmetric(x2x2 - (Ax1x2 + Ax1x2.T) + self.A @ x1x1 @ self.A.T)
        return self._gaussian_ell(N - 1, self.Q, quad)

    def learn_CR(self, yy, xy, xx, N):
        """Update C, R and Sigma_CRC from the observation moments.

        Returns the observation term of the expected log-likelihood.
        """
        if self.covariance == 'full':
            Lam = xx + np.diag(self.beta[0])
            Lam_inv = np.linalg.inv(Lam)
            self.C = xy.T @ Lam_inv
            Winv = 2 * self.b * np.eye(self.dimy) + yy - self.C @ Lam @ self.C.T
            self.R = symmetric(Winv / (2 * self.a + N))
            self.Sigma_CRC = self.dimy * Lam_inv
        else:
            self.Sigma_CRC = np.zeros((self.dimx, self.dimx))
            self.R = np.zeros((self.dimy, self.dimy))
            for i in range(self.dimy):
                ell = xy[:, i]
                Sigma_C = np.linalg.inv(xx + np.diag(self.beta[i]))
                self.C[i] = ell @ Sigma_C
                self.Sigma_CRC += Sigma_C
                a_hat = self.a + 0.5 * N
                b_hat = self.b + 0.5 * (yy[i, i] - self.C[i] @ ell)
                self.R[i, i] = b_hat / a_hat

        Cxy = self.C @ xy
        quad = symmetric(yy - (Cxy + Cxy.T) + self.C @ xx @ self.C.T)
        return self._gaussian_ell(N, self.R, quad)

    def learn_mP(self, mu, V):
        """Set m0, P0 to the smoothed moments of the first state.

        Returns the initial-state term of the expected log-likelihood.
        """
        self.m0 = mu.copy()
        self.P0 = symmetric(V)
        d = mu - self.m0
        quad = symmetric(np.outer(d, d) + V)
        return self._gaussian_ell(1, self.P0, quad)

    def learn(self, y, mu, V, V12):
        """M-step: update all parameters from the smoothed moments.

        Returns:
            The expected complete-data log-likelihood E_q[log p(y, x)] at
            the updated parameters. It does not include the prior and
            entropy terms of the full variational lower bound.
        """
        N = mu.shape[-1]
        xx = mu @ mu.T + V.sum(-1)
        x1x1 = mu[:, :-1] @ mu[:, :-1].T + V[:, :, :-1].sum(-1)
        x1x2 = mu[:, :-1] @ mu[:, 1:].T + V12.sum(-1)
        x2x2 = mu[:, 1:] @ mu[:, 1:].T + V[:, :, 1:].sum(-1)
        xy = mu @ y.T
        yy = y @ y.T

        ell_x0 = self.learn_mP(mu[:, 0], V[:, :, 0])
        ell_x = self.learn_AQ(x2x2, x1x2, x1x1, N)
        ell_y = self.learn_CR(yy, xy, xx, N)
        return ell_x + ell_y + ell_x0

    # ------------------------------------------------------------------
    # Training, sampling, saving
    # ------------------------------------------------------------------

    def step(self, y):
        """Run one VB-EM iteration on y of shape (dimy, N).

        Runs the filter, the smoother, then the parameter update (in place).

        Returns:
            A ``StepResult`` with fields ell, loglik, muf, Vf, mus, Vs, V12,
            y_pred, S (see ``StepResult``).
        """
        y = np.asarray(y, dtype=float).reshape(self.dimy, -1)
        muf, Vf, y_pred, S, loglik = self.filter(y)
        mus, Vs, V12 = self.smoother(muf, Vf)
        ell = self.learn(y, mus, Vs, V12)
        return StepResult(ell, loglik.sum(), muf, Vf, mus, Vs, V12, y_pred, S)

    __call__ = step

    def fit(self, y, iterations=100, verbose=False):
        """Run ``iterations`` VB-EM iterations.

        Returns:
            A ``FitHistory`` of arrays ``ell`` and ``loglik``, one value per
            iteration (see ``StepResult``).
        """
        ell, loglik = np.zeros(iterations), np.zeros(iterations)
        for it in range(iterations):
            result = self.step(y)
            ell[it], loglik[it] = result.ell, result.loglik
            if verbose:
                print(f'Iteration {it + 1} / {iterations} ==> '
                      f'ELL = {ell[it]:.2f}, log-likelihood = {loglik[it]:.2f}')
        return FitHistory(ell, loglik)

    def sample(self, N=100, rng=None):
        """Draw a length-N sequence from the model with the current parameters.

        Uses ``rng`` if given, otherwise the model's own generator.

        Returns:
            y: observations, shape (dimy, N).
            x: latent states, shape (dimx, N).
        """
        rng = self.rng if rng is None else np.random.default_rng(rng)
        x = np.zeros((self.dimx, N))
        y = np.zeros((self.dimy, N))
        x_n = rng.multivariate_normal(self.m0, self.P0, method='eigh')
        for n in range(N):
            x[:, n] = x_n
            y[:, n] = rng.multivariate_normal(self.C @ x_n, self.R, method='eigh')
            x_n = rng.multivariate_normal(self.A @ x_n, self.Q, method='eigh')
        return y, x

    def save(self, path):
        """Save the parameters and settings to a NumPy ``.npz`` file."""
        np.savez(path, covariance=self.covariance,
                 **{name: getattr(self, name) for name in self.PARAMETERS})

    @classmethod
    def load(cls, path, rng=None):
        """Create a model from a file written by ``save``."""
        with np.load(path) as f:
            dimy, dimx = f['C'].shape
            model = cls(dimy, dimx, covariance=str(f['covariance']), rng=rng)
            for name in cls.PARAMETERS:
                setattr(model, name, f[name].copy())
        return model
