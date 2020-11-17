%%% Prediction Step for VBLDS %%%%%%%%%%%%%%%%%%%%%%%
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