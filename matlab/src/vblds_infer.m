% VBLDS_INFER  E-step: Bayesian Kalman filter and smoother over a sequence.
%
%   [mu, V, V12] = VBLDS_INFER(y, params) runs the forward filter
%   (VBLDS_UPDATE, VBLDS_PREDICT), then the backward smoother (VBLDS_SMOOTH),
%   and returns the smoothed posterior moments of the latent states.
%
% Inputs:
%   y       observations                                     (dimy x N)
%   params  parameter struct from VBLDS_INITIALIZE or VBLDS_LEARN
%
% Outputs:
%   mu      smoothed means, E[x_n | y_{1:N}]                 (dimx x N)
%   V       smoothed covariances                             (dimx x dimx x N)
%   V12     smoothed cross-covariances Cov(x_n, x_{n+1})     (dimx x dimx x N-1)
%
% See also VBLDS_LEARN, VBLDS_UPDATE, VBLDS_PREDICT, VBLDS_SMOOTH.
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

function [mu,V,V12] = vblds_infer(y,params)

    N = size(y,2);
    dimx = size(params.m0,1);
    mu = zeros(dimx,N);
    V = zeros(dimx,dimx,N);

    % Filter
    m = params.m0;
    P = params.P0;
    for n = 1:N
        [mu(:,n),V(:,:,n)] = vblds_update(m,P,y(:,n),params.C,params.R,params.Sigma_CRC);
        [m,P] = vblds_predict(mu(:,n),V(:,:,n),params.A,params.Q,params.Sigma_AQA);
    end

    % Smooth
    for n = N-1:-1:1
        [mu(:,n), V(:,:,n), V12(:,:,n)] = vblds_smooth(mu(:,n+1),V(:,:,n+1),mu(:,n),V(:,:,n),params.A,params.Q,params.Sigma_AQA);
    end
        
end


