% DATA_LOADER  Generate a synthetic test sequence for the VBLDS demo.
%
%   [y, x, m0, P0, A, Q, C, R] = DATA_LOADER(trial) samples N = 100 steps
%   of a 2-D latent oscillator (a slowly decaying rotation, see PHASOR)
%   observed through a random 1 x 2 matrix C. The parameters are drawn at
%   random on each call, so use rng() for repeatable data.
%
%   trial selects the observation noise:
%     'gaussian noise' Gaussian noise, variance 0.05 (used by demo.m)
%     'outliers'       Gaussian noise (variance 0.01); each sample has a
%                      10% chance of being replaced by a draw from N(0, 1)
%     'noise switch'   Gaussian noise (variance 0.001), except variance 1
%                      for 30 < n < 60
%
% Outputs:
%   y     observations                                       (1 x N)
%   x     true latent states                                 (2 x N)
%   m0,P0,A,Q,C,R   parameters used to generate the data
%
% Uses only base MATLAB (Gaussian samples come from GAUSS_SAMPLE).
%
% Author: Julian Neri
% Affil: McGill University
% Date: May 1, 2020

function [y,x,m0,P0,A,Q,C,R] = data_loader(trial)

    % Observations (Time samples)
    N =  100;
    dimx = 2;
    dimy = 1;

    x = zeros(dimx,N);
    y = zeros(dimy,N);
    P0 = 1e-10*eye(dimx);
    % Random state noise covariance: a Wishart draw with dimx degrees of
    % freedom and scale 1e-5*I/dimx, i.e. the sum of dimx outer products of
    % Gaussian vectors.
    W = sqrt(1e-5/dimx)*randn(dimx,dimx);
    Q = W*W';

    phase = 2*pi*rand;
    m0 = [cos(phase); sin(phase)];
    A = phasor(2*pi*6.55/N)*.99;

    C = randn(dimy,dimx);
    C = bsxfun(@rdivide, C, sum(abs(C),2));
    
    % Sample
    % Latent variable
    n = 1;
    x(:,n) = gauss_sample(m0,P0);
    for n = 2:N
        x(:,n) = gauss_sample(A*x(:,n-1),Q);
    end
    
    switch trial
        % Gaussian noise.
        case 'gaussian noise'
            R = .05*eye(dimy);
            for n = 1:N
                y(:,n) = gauss_sample(C*x(:,n),R);
            end
            
        % Gaussian noise with outliers occuring randomly every 1 in 10 samples.
        case 'outliers'
            R = .01*eye(dimy);
            for n = 1:N
                if rand < .1
                    y(:,n) = gauss_sample(0,1);
                else
                    y(:,n) = gauss_sample(C*x(:,n),R);
                end
            end
            
        % Gaussian noise that is much 'louder' when 30 < n < 60.
        case 'noise switch'
            R = .001*eye(dimy);
            for n = 1:N
                if n > 30 && n < 60
                    y(:,n) = gauss_sample(C*x(:,n),1);
                else
                    y(:,n) = gauss_sample(C*x(:,n),R);
                end
            end
    end

end
