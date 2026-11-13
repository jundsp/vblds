% VBLDS_SMOOTH  One backward step of the Bayesian RTS smoother.
%
%   [mu_hat, V_hat, V12] = VBLDS_SMOOTH(mu_plus, V_plus, mu, V, A, Q, Sigma_AQA)
%   combines the filtered distribution at step n, N(mu, V), with the
%   smoothed distribution at step n+1, N(mu_plus, V_plus), to give the
%   smoothed distribution at step n. With Sigma_AQA = 0 this is the
%   standard Rauch-Tung-Striebel step.
%
% Inputs:
%   mu_plus, V_plus  smoothed mean/covariance at step n+1
%   mu, V            filtered mean/covariance at step n
%   A, Q             expected transition matrix and state noise covariance
%   Sigma_AQA        E[A'Q^-1A] - <A>'<Q^-1><A>
%
% Outputs:
%   mu_hat, V_hat    smoothed mean/covariance at step n
%   V12              smoothed cross-covariance Cov(x_n, x_{n+1})
%
% See also VBLDS_INFER, VBLDS_PREDICT.
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

function [mu_hat,V_hat, V12] = vblds_smooth(mu_plus,V_plus,mu,V,A,Q,Sigma_AQA)

    G = woodbury_inversion(V,Sigma_AQA);
    m = A*G*mu;
    P = A*G*V*A' + Q;
    
    J = G*V*A'/P;
    mu_hat = G*mu + J*(mu_plus - m);
    V_hat = G*V + J*(V_plus - P)*J';
    V12 = J*V_plus;
        
end


