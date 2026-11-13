% WOODBURY_INVERSION  Compute (I + A*B)^-1 with the push-through identity.
%
%   G = WOODBURY_INVERSION(A, B) returns inv(I + A*B), computed as
%   I - A*inv(I + B*A)*B. The filter and smoother use it to form the
%   parameter-uncertainty corrections, for example G = (I + V*Sigma_AQA)^-1.
%
%   A and B must be square and the same size.
%
% See also VBLDS_PREDICT, VBLDS_UPDATE, VBLDS_SMOOTH.

function [C] = woodbury_inversion(A,B)

I = eye(size(A,1));
C = I - A/(I+B*A)*B;

end
