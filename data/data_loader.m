% Loads test data for the Laplace state space filter.
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
    Q = wishrnd(1e-4*eye(dimx)/dimx,dimx);

    phase = 2*pi*rand;
    m0 = [cos(phase); sin(phase)];
    A = phasor(2*pi*5.55/N)*.999;

    C = randn(dimy,dimx);
    C = bsxfun(@rdivide, C, sum(abs(C),2));
    
    % Sample
    % Latent variable
    n = 1;
    x(:,n) = mvnrnd(m0,P0);
    for n = 2:N
        x(:,n) = mvnrnd(A*x(:,n-1),Q);
    end
    
    switch trial
        % Laplace-distributed noise.
        case 'laplace noise'
            R = .1*eye(dimy);
            for n = 1:N
                y(:,n) = laprnd(C*x(:,n),R,1);
            end
            
        % Gaussian noise with outliers occuring randomly every 1 in 10 samples.
        case 'outliers'
            R = .01*eye(dimy);
            for n = 1:N
                if rand < .1
                    y(:,n) = mvnrnd(0,1);
                else
                    y(:,n) = mvnrnd(C*x(:,n),R);
                end
            end
            
        % Gaussian noise that is much 'louder' when 30 < n < 60.
        case 'noise switch'
            R = .001*eye(dimy);
            for n = 1:N
                if n > 30 && n < 60
                    y(:,n) = mvnrnd(C*x(:,n),1);
                else
                    y(:,n) = mvnrnd(C*x(:,n),R);
                end
            end
    end

end
