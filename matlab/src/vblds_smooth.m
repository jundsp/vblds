%%% Smoothing Step for VBLDS %%%%%%%%%%%%%%%%%%%%%%%
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


