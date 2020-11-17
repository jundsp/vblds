%%% Demo Script for Variational Bayesian Linear Dynamical System (VBLDS) %%%
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

clear;clc;close all;
rng(1);

addpath('src','data','utils');

[y,x,m0,P0,A,Q,C,R] = data_loader('gaussian noise');

% Initialize parameters
dimy = size(y,1);
dimx = 8;
parameters = vblds_initialize(dimy,dimx);

epochs = 1000;
ell = zeros(1,epochs);
for epoch = 1:epochs
    % Bayesian Kalman filter and smoother4
    [mu,V,V12] = vblds_infer(y,parameters);
    % Learn Parameters
    parameters = vblds_learn(y,mu,V,V12,parameters);
    ell(epoch) = vblds_ell(y,mu,V,V12,parameters);
    fprintf('Epoch %4.0f/%4.0f ==> ELL = %4.2f \n',epoch,epochs,ell(epoch));
end

t = 1:length(y);
y_hat = parameters.C*mu;
[y_sampled,x_sampled] = vblds_sample(parameters,100);

% Figures
figure('pos',[0 1000 500 500])
subplot(4,2,[1 2]);
hold on;
scatter(t,y,'filled'); 
plot(t,y_hat,'linewidth',2); box on; axis tight; title('Data'); 
subplot(4,2,5);
imagesc(parameters.A); title('A'); axis image;
subplot(4,2,6);
imagesc(parameters.Q); title('Q'); axis image;
subplot(4,2,7);
imagesc(parameters.C); title('C'); axis image;
subplot(4,2,8);
imagesc(parameters.R); title('R'); axis image;
subplot(4,2,[3 4]);
hold on;
t = 1:length(y_sampled);
scatter(t,y_sampled,'g','filled'); 
plot(t,parameters.C*x_sampled,'m','linewidth',2); box on; axis tight; title('Sampled from Learned Model');
xlabel('Time');

figure('pos',[400 0 500 200])
plot(t,mu'); axis tight; title('Latent State');

figure('pos',[0 0 400 200])
plot(ell,'m','linewidth',2); 
axis tight; xlabel('Iteration'); title('Expected Log-Likelihood');