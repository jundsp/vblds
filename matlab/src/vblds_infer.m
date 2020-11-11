function [mu,V,V12] = vblds_infer(y,params)

N = size(y,2);
dimx = size(params.m0,1);
mu = zeros(dimx,N);
V = zeros(dimx,dimx,N);

% Filter
m = params.m0;
P = params.P0;
for n = 1:N
    [mu(:,n),V(:,:,n)] = vblds_update(m,P,y(:,n),params.C,params.R,params.Sigma_CRC);
    [m,P] = vblds_predict(mu(:,n),V(:,:,n),params.A,params.Q,params.Sigma_AQA);
end

% Smooth
for n = N-1:-1:1
    [mu(:,n), V(:,:,n), V12(:,:,n)] = vblds_smooth(mu(:,n+1),V(:,:,n+1),mu(:,n),V(:,:,n),params.A,params.Q,params.Sigma_AQA);
end
        
end


