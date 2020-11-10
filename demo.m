clear;clc;close all;

addpath('src','data','utils');

[y,x,m0,P0,A,Q,C,R] = data_loader('gaussian noise');

% Initialize parameters
dimy = size(y,1);
dimx = 4;
parameters = vblds_initialize(dimy,dimx);

epochs = 200;
for epoch = 1:epochs
    % Bayesian Kalman filter and smoother
    [mu,V,V12] = vblds_infer(y,parameters);
    % Learn Parameters
    parameters = vblds_learn(y,mu,V,V12,parameters);
end

t = 1:length(y);
y_hat = parameters.C*mu;
[y_sampled,x_sampled] = vblds_sample(parameters,100);

% Figures
figure('pos',[0 0 500 500])
subplot(3,2,[1 2]);
hold on;
scatter(t,y,'filled'); 
plot(t,y_hat,'linewidth',2); box on; axis tight; title('Data');
subplot(3,2,3);
imagesc(parameters.A); title('A'); axis image;
subplot(3,2,4);
imagesc(parameters.Q); title('Q'); axis image;
subplot(3,2,[5 6]);
hold on;
t = 1:length(y_sampled)
scatter(t,y_sampled,'k','filled'); 
plot(t,parameters.C*x_sampled,'g','linewidth',2); box on; axis tight; title('Sampled from Learned Model');