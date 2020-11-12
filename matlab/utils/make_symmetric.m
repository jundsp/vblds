function [B] = make_symmetric(A)
    B = 0.5*(A + A');
end

