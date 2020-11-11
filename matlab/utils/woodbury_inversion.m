function [C] = woodbury_inversion(A,B)

I = eye(size(A,1));
C = I - A/(I+B*A)*B;

end

