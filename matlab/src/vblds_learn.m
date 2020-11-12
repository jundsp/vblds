function [params] = vblds_learn(y,mu,V,V12,params)

    [dimx,N] = size(mu);
    dimy = size(y,1);
    
    % Hyperparameters
    a0 = 1e-12; b0 = 1e-12;
    c0 = 1e-12; d0 = 1e-12;
    alpha = 1e-9*ones(dimx,dimx);
    beta = 1e-9*ones(dimy,dimx);

    xx = mu*mu' + sum(V,3);
    x1x2 = mu(:,1:N-1)*mu(:,2:N)' + sum(V12,3);
    x1x1 = xx - (mu(:,N)*mu(:,N)' + V(:,:,N));
    x2x2 = xx - (mu(:,1)*mu(:,1)' + V(:,:,1));
    
    xx = make_symmetric(xx);
    x1x1 = make_symmetric(x1x1);
    x2x2 = make_symmetric(x2x2);
    xy = mu*y';
    yy = y*y';
        
    %% m0, P0
    m0 = mu(:,1);
    P0 = V(:,:,1);
    P0 = make_symmetric(P0);
    
    %% A & Q
    A = zeros(dimx,dimx);
    Sigma_A = zeros(dimx,dimx,dimx);
    c_hat = zeros(dimx,1);
    d_hat = zeros(dimx,1);
    for i = 1:dimx
        Lam = x1x1 + diag(alpha(i,:));
        ell = x1x2(:,i)';
        A(i,:) = ell / Lam;
        Sigma_A(:,:,i) = inv(Lam);
        
        c_hat(i) = c0 + 1/2*(N-1);
        d_hat(i) = d0 + 1/2*(x2x2(i,i) - ell/Lam*ell');
        
    end
    tau = c_hat./d_hat;
    Q = diag(1./tau);
    
    %% Sigma AQA
    Sigma_AQA = sum(Sigma_A,3);
    
    %% C & R
    C = xy'/xx;
    Sigma_C = zeros(dimx,dimx,dimy);
    a_hat = zeros(dimy,1);
    b_hat = zeros(dimy,1);
    for i = 1:dimy
        ell = xy(:,i)';
        Lam = xx + diag(beta(i,:));
        C(i,:) = ell / Lam;
        Sigma_C(:,:,i) = inv(Lam);
        
        a_hat(i) = a0 + 1/2*N;
        b_hat(i) = b0 + 1/2*(yy(i,i) - ell/Lam*ell');
    end
    rho = a_hat./b_hat;
    R = diag(1./rho);
    
    %% Sigma CRC
    Sigma_CRC = sum(Sigma_C,3);
    
    %% Save Params

    params.A = A;
    params.Q = Q;
    params.C = C;
    params.R = R;
    params.m0 = m0;
    params.P0 = P0;
    params.Sigma_AQA = Sigma_AQA;
    params.Sigma_CRC = Sigma_CRC;
        
end


