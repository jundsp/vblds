% VBLDS_PREDICT  Filter prediction step of the Bayesian Kalman filter.
%
%   [m, P] = VBLDS_PREDICT(mu, V, A, Q, Sigma_AQA) propagates the filtered
%   distribution N(mu, V) to the next time step. The posterior is first
%   corrected for uncertainty in A and Q with G = (I + V*Sigma_AQA)^-1, then
%   propagated with the expected A and Q. With Sigma_AQA = 0 this is the
%   standard Kalman prediction.
%
% Inputs:
%   mu         filtered mean at step n                       (dimx x 1)
%   V          filtered covariance at step n                 (dimx x dimx)
%   A          expected state transition matrix              (dimx x dimx)
%   Q          expected state noise covariance               (dimx x dimx)
%   Sigma_AQA  E[A'Q^-1A] - <A>'<Q^-1><A>                    (dimx x dimx)
%
% Outputs:
%   m          predictive mean for step n+1                  (dimx x 1)
%   P          predictive covariance for step n+1            (dimx x dimx)
%
% See also VBLDS_UPDATE, VBLDS_SMOOTH, WOODBURY_INVERSION.
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

function [m,P] = vblds_predict(mu,V,A,Q,Sigma_AQA)

    G = woodbury_inversion(V,Sigma_AQA);
    m = A*G*mu;
    P = A*G*V*A' + Q;
        
end