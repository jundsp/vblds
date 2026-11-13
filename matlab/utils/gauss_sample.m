% GAUSS_SAMPLE  Draw one sample from a multivariate Gaussian.
%
%   x = GAUSS_SAMPLE(mu, S) returns a column vector drawn from N(mu, S).
%   It uses only base MATLAB, so the Statistics and Machine Learning Toolbox
%   (mvnrnd) is not needed.
%
%   S must be symmetric positive semidefinite. A Cholesky factor is used
%   when S is positive definite. Otherwise the code falls back to an
%   eigendecomposition and sets small negative eigenvalues (round-off) to
%   zero.
%
% Inputs:
%   mu    mean                                               (d x 1 or 1 x d)
%   S     covariance                                         (d x d)
%
% Output:
%   x     sample                                             (d x 1)

function x = gauss_sample(mu, S)

    d = numel(mu);
    [L, p] = chol(S, 'lower');
    if p ~= 0
        [U, D] = eig((S + S')/2);
        L = U*diag(sqrt(max(diag(D), 0)));
    end
    x = mu(:) + L*randn(d, 1);

end
