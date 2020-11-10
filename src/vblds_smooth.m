function [mu_hat,V_hat, V12] = vblds_smooth(mu_plus,V_plus,mu,V,A,Q,Sigma_AQA)

	G = woodbury_inversion(V,Sigma_AQA);
    m = A*G*mu;
    P = A*G*V*A' + Q;
    
    J = G*V*A'/P;
    mu_hat = G*mu + J*(mu_plus - m);
    V_hat = G*V + J*(V_plus - P)*J';
    V12 = J*V_plus;
        
end


