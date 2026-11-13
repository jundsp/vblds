% VBLDS_UPDATE  Filter update step of the Bayesian Kalman filter.
%
%   [mu, V] = VBLDS_UPDATE(m, P, y, C, R, Sigma_CRC) conditions the
%   predictive distribution N(m, P) on the observation y. First the
%   predictive distribution is corrected for uncertainty in C and R with
%   L = (I + P*Sigma_CRC)^-1, then a Kalman update is applied using the
%   expected C and R. With Sigma_CRC = 0 this is the standard Kalman update.
%
% Inputs:
%   m          predictive mean                               (dimx x 1)
%   P          predictive covariance                         (dimx x dimx)
%   y          observation y_n                               (dimy x 1)
%   C          expected observation matrix                   (dimy x dimx)
%   R          expected observation noise covariance         (dimy x dimy)
%   Sigma_CRC  E[C'R^-1C] - <C>'<R^-1><C>                    (dimx x dimx)
%
% Outputs:
%   mu         filtered mean                                 (dimx x 1)
%   V          filtered covariance                           (dimx x dimx)
%
% See also VBLDS_PREDICT, VBLDS_INFER, WOODBURY_INVERSION.
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

function [mu,V] = vblds_update(m,P,y,C,R,Sigma_CRC)

    I = eye(size(m,1));

    L = woodbury_inversion(P,Sigma_CRC);
    y_hat = C*L*m;
    S = C*L*P*C' + R;
    
    % Bayesian Kalman Gain
    K = L*P*C'/S;
    
    mu = L*m + K*(y-y_hat);
    V = (I-K*C)*L*P;
        
end