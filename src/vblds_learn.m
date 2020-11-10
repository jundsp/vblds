function [params] = vblds_learn(y,mu,V,V12,params)

    [dimx,N] = size(mu);
    dimy = size(y,1);
    
    nu0_Q = 1e-12;
    W0_Q = 1e-12*eye(dimx);
    nu0_R = 1e-12;
    W0_R = 1e-12*eye(dimy);

    xx = mu*mu' + sum(V,3);
    x1x2 = mu(:,1:N-1)*mu(:,2:N)' + sum(V12,3);
    x1x1 = xx - (mu(:,N)*mu(:,N)' + V(:,:,N));
    x2x2 = xx - (mu(:,1)*mu(:,1)' + V(:,:,1));
    
    xx = 1/2*(xx+xx');
    x1x1 = 1/2*(x1x1+x1x1');
    x2x2 = 1/2*(x2x2+x2x2');
    xy = mu*y';
    yy = y*y';
        
    %% m0, P0
    m0 = mu(:,1);
    P0 = V(:,:,1);
    P0 = 1/2*(P0+P0');
    
    %% A
    A = x1x2'/x1x1;
    
    %% Q
    nuM_Q = nu0_Q + (N-1);
    WM_Q = W0_Q + x2x2 - A*x1x2;
    Q = WM_Q/nuM_Q;
    Q = 1/2*(Q + Q');
    
    %% C
    C = xy'/xx;
    
    %% R
    nuM_R = nu0_R + N;
    WM_R = W0_R + yy - C*xy;
    R = WM_R/nuM_R;
    R = 1/2*(R+R');

    params.A = A;
    params.Q = Q;
    params.C = C;
    params.R = R;
    params.m0 = m0;
    params.P0 = P0;
        
end

