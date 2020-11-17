%%% Update Step for VBLDS %%%%%%%%%%%%%%%%%%%%%%%
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