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