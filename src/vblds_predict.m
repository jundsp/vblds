function [m,P] = vblds_predict(mu,V,A,Q,Sigma_AQA)

    G = woodbury_inversion(V,Sigma_AQA);
    m = A*G*mu;
    P = A*G*V*A' + Q;
        
end