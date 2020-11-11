% Sample from a univariate Laplace distribution
%
% a: mean (location)
% b: scale
% N: number of samples to draw
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