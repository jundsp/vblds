% VBLDS_INITIALIZE  Create an initial VBLDS parameter struct.
%
%   params = VBLDS_INITIALIZE(dimy, dimx) returns a struct for a model with
%   dimy-dimensional observations and a dimx-dimensional latent state:
%     A          identity plus small random noise            (dimx x dimx)
%     Q          0.1 * I                                     (dimx x dimx)
%     C          uniform random entries in [0, 1]            (dimy x dimx)
%     R          0.1 * I                                     (dimy x dimy)
%     m0, P0     zeros(dimx,1), I
%     Sigma_AQA, Sigma_CRC   near-zero parameter-uncertainty corrections
%
%   A and C are random, so call rng() first for repeatable results.
%
% See also VBLDS_INFER, VBLDS_LEARN.
%
% Citation:
% J. Neri, R. Badeau and P. Depalle, "Probabilistic Filter and Smoother for
% Variational Inference of Bayesian Linear Dynamical Systems," 
% IEEE International Conference on Acoustics, Speech and Signal Processing 
% (ICASSP 2020), Barcelona, Spain, 2020, pp. 5885-5889.
%
% Author: Julian Neri
% Affil: McGill University
% Date: May 1, 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [parameters] = vblds_initialize(dimy,dimx)

    
    parameters.A = eye(dimx) + 1e-3*randn(dimx);
    parameters.Q = 1e-1*eye(dimx);
    parameters.C = rand(dimy,dimx);
    parameters.R = 1e-1*eye(dimy);
    parameters.m0 = zeros(dimx,1);
    parameters.P0 = eye(dimx);
    % Uncertainty over the model parameters
    parameters.Sigma_AQA = 1e-9*eye(dimx);
    parameters.Sigma_CRC = 1e-9*eye(dimx);
end

