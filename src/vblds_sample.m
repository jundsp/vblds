function [y,x] = vblds_sample(parameters,N)

dimy = size(parameters.R,1);
dimx = size(parameters.A,1);
x = zeros(dimx,N);
y = zeros(dimy,N);
x(:,1) = mvnrnd(parameters.m0,parameters.P0);
y(:,1) = mvnrnd(parameters.C*x(:,1),parameters.R);
for n = 2:N
    x(:,n) = mvnrnd(parameters.A*x(:,n-1),parameters.Q);
    y(:,n) = mvnrnd(parameters.C*x(:,n),parameters.R);
end
        
end


