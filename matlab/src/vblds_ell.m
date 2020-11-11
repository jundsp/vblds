function [ell] = vblds_ell(y,mu,V,V12,params)

    N = size(y,2);
    ell = 0;
    for n = 1:N
        yy = y(:,n)*y(:,n)';
        Cxy = params.C*mu(:,n)*y(:,n)';
        CxxC = params.C*(mu(:,n)*mu(:,n)' + V(:,:,n))*params.C';
        quad = yy - (Cxy + Cxy') + CxxC;
        loss_n = -1/2*log(det(2*pi*params.R)) - 1/2*trace(params.R\quad);
        ell = ell + loss_n;
    end
    
    xx = mu(:,1)*mu(:,1)' + V(:,:,1);
    xm = mu(:,1)*params.m0';
    mm = params.m0*params.m0';
    quad = xx - (xm + xm') + mm;
    loss_n = -1/2*log(det(2*pi*params.P0)) - 1/2*trace(params.P0\quad);
    ell = ell + loss_n;
    for n = 2:N
        xx = mu(:,n)*mu(:,n)' + V(:,:,n);
        Ax1x2 = params.A*(mu(:,n-1)*mu(:,n)' + V12(:,:,n-1));
        Ax1x1A = params.A*(mu(:,n-1)*mu(:,n-1)' + V(:,:,n-1))*params.A';
        quad = xx - (Ax1x2 + Ax1x2') + Ax1x1A;
        loss_n = -1/2*log(det(2*pi*params.Q)) - 1/2*trace(params.Q\quad);
        ell = ell + loss_n;
    end
        
end


