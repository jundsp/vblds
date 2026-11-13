% VBLDS_SAMPLE  Draw a sequence from the linear dynamical system.
%
%   [y, x] = VBLDS_SAMPLE(params, N) samples N steps using the expected
%   parameters in params (A, Q, C, R, m0, P0). It does not sample the
%   parameters themselves.
%
% Outputs:
%   y   sampled observations                                 (dimy x N)
%   x   sampled latent states                                (dimx x N)
%
% Uses only base MATLAB (GAUSS_SAMPLE, in utils/, must be on the path).
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

function [y,x] = vblds_sample(parameters,N)

dimy = size(parameters.R,1);
dimx = size(parameters.A,1);
x = zeros(dimx,N);
y = zeros(dimy,N);
x(:,1) = gauss_sample(parameters.m0,parameters.P0);
y(:,1) = gauss_sample(parameters.C*x(:,1),parameters.R);
for n = 2:N
    x(:,n) = gauss_sample(parameters.A*x(:,n-1),parameters.Q);
    y(:,n) = gauss_sample(parameters.C*x(:,n),parameters.R);
end
        
end


