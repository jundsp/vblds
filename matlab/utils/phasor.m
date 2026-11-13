% PHASOR  2-D rotation matrix.
%
%   R = PHASOR(omega) returns [cos(omega) -sin(omega); sin(omega) cos(omega)].
%   As a state transition matrix, it makes the latent state rotate by
%   omega radians per step, a sinusoid at frequency omega/(2*pi) cycles per
%   sample.
%
%   R = PHASOR(omega, D) puts the rotation in the top-left 2 x 2 block of a
%   D x D matrix of zeros. If D < 2, it prints a warning and returns
%   cos(omega).
%
% Author: Julian Neri
% Affil: McGill University
% Date: May 1, 2020

function R = phasor(omega, D)

if nargin < 2
    D = 2;
end

if D < 2
    disp('Dimension of phasor must be at least 2.');
    R = cos(omega);
    return;
end

R = zeros(D);

R(1:2,1:2) = [cos(omega) -sin(omega)
              sin(omega)  cos(omega)];

end

