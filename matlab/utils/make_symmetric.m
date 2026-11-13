% MAKE_SYMMETRIC  Symmetrize a square matrix.
%
%   B = MAKE_SYMMETRIC(A) returns (A + A')/2. It removes round-off asymmetry
%   from covariance and second-moment matrices.

function [B] = make_symmetric(A)
    B = 0.5*(A + A');
end
