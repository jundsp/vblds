%%% Initialize the VBLDS model paramters %%%%%%%%%%%%%%%%%%%%%%%
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
    parameters.Sigma_CRC = 1e-9*eye(dimy);
end

