%%% Expected Log-Likelihood for VBLDS %%%%%%%%%%%%%%%%%%%%%%%
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

function [ell] = vblds_ell(y,mu,V,V12,params)

    N = size(y,2);
    ell = 0;
    yy = y*y';
    Cxy = params.C*(mu*y');
    CxxC = params.C*(mu*mu' + sum(V,3))*params.C';
    quad = yy - (Cxy + Cxy') + CxxC;
    loss_n = -N/2*log(det(2*pi*params.R)) - 1/2*trace(params.R\quad);
    ell = ell + loss_n;
    
    xx = mu(:,1)*mu(:,1)' + V(:,:,1);
    xm = mu(:,1)*params.m0';
    mm = params.m0*params.m0';
    quad = xx - (xm + xm') + mm;
    loss_n = -1/2*log(det(2*pi*params.P0)) - 1/2*trace(params.P0\quad);
    ell = ell + loss_n;
    
    x2x2 = mu(:,2:end)*mu(:,2:end)' + sum(V(:,:,2:end),3);
    x1x1 = mu(:,1:end-1)*mu(:,1:end-1)' + sum(V(:,:,1:end-1),3);
    x1x2 = mu(:,1:end-1)*mu(:,2:end)' + sum(V12,3);
    Ax1x2 = params.A*x1x2;
    Ax1x1A = params.A*x1x1*params.A';
    quad = x2x2 - (Ax1x2 + Ax1x2') + Ax1x1A;
    loss_n = -(N-1)/2*log(det(2*pi*params.Q)) - 1/2*trace(params.Q\quad);
    ell = ell + loss_n;
    
        
end


