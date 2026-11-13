% VBLDS_LEARN  M-step: variational update of the VBLDS parameters.
%
%   params = VBLDS_LEARN(y, mu, V, V12, params) updates the parameters
%   using the smoothed moments from VBLDS_INFER. The form of the noise
%   covariances is set by params.covariance (see VBLDS_INITIALIZE):
%
%   'diagonal' (default): Normal-Gamma posterior for each row of A and the
%       matching diagonal precision of Q, and the same for C and R.
%   'full': matrix-normal-Wishart posterior for (A, Q) and for (C, R),
%       so Q and R are full covariance matrices.
%
%   In both cases Q and R are set to the inverse expected precision,
%   <Q^-1>^-1 and <R^-1>^-1, m0 and P0 are point estimates (the smoothed
%   moments of x_1), and Sigma_AQA, Sigma_CRC are the corrections
%   E[A'Q^-1A] - <A>'<Q^-1><A> (and the same for C, R) that the filter and
%   smoother use to account for parameter uncertainty.
%
%   Hyperparameters (set below), all close to zero for near-non-informative
%   priors:
%     alpha, beta  ARD prior precisions on the entries of A and C. In
%                  'full' mode they must be shared by all rows, so only
%                  the first row of each is used.
%     c0, d0       Gamma shape and rate for the precisions of Q ('diagonal').
%     a0, b0       the same for R.
%   In 'full' mode the Wishart priors are matched to these: degrees of
%   freedom 2*c0 (2*a0) and inverse scale 2*d0*I (2*b0*I). With equal ARD
%   precisions, the diagonal of a 'full' update then equals the
%   'diagonal' update.
%
% Inputs:
%   y        observations                                    (dimy x N)
%   mu, V, V12   smoothed moments from VBLDS_INFER
%   params   current parameter struct (only its fields are replaced)
%
% Output:
%   params   updated struct with fields A, Q, C, R, m0, P0, Sigma_AQA, Sigma_CRC
%
% See also VBLDS_INFER, VBLDS_ELL.
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
    
    if isfield(params,'covariance')
        covariance = params.covariance;
    else
        covariance = 'diagonal';
    end

    if strcmp(covariance,'full')
        % A & Q: matrix-normal-Wishart posterior
        % A | Q ~ MN(<A>, Q, inv(Lam)),  Q^-1 ~ Wishart(nu, inv(Winv))
        Lam = x1x1 + diag(alpha(1,:));
        A = x1x2' / Lam;
        Winv = 2*d0*eye(dimx) + x2x2 - A*Lam*A';
        Q = make_symmetric(Winv / (2*c0 + N - 1));
        Sigma_AQA = dimx * inv(Lam);

        % C & R: matrix-normal-Wishart posterior
        Lam = xx + diag(beta(1,:));
        C = xy' / Lam;
        Winv = 2*b0*eye(dimy) + yy - C*Lam*C';
        R = make_symmetric(Winv / (2*a0 + N));
        Sigma_CRC = dimy * inv(Lam);
    else
        % A & Q
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
    
        % Sigma AQA
        Sigma_AQA = sum(Sigma_A,3);
    
        % C & R
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
    
        % Sigma CRC
        Sigma_CRC = sum(Sigma_C,3);
    
    end

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


