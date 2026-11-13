% LAPRND  Draw samples from a Laplace distribution.
%
%   out = LAPRND(a, b, N) draws N samples per row from
%   Laplace(a, b), with density exp(-|y - a|/b) / (2*b), by inverse-CDF
%   sampling.
%
% Inputs:
%   a     location (mean); one row per dimension             (d x 1)
%   b     scale (scalar); variance is 2*b^2
%   N     number of samples per dimension (default 1)
%
% Output:
%   out   samples                                            (d x N)
%
% Author: Julian Neri
% Affil: McGill University
% Date: May 1, 2020

function out = laprnd(a,b,N)

dimy = size(a,1);

if nargin < 3
    N = 1;
end

U = rand(dimy,N)-1/2;
out = a - b*sign(U).*log(1-2*abs(U));