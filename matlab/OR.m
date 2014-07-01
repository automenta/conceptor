function [RorQ, varargout] = OR(R, Q)


RorQ = NOT(AND(NOT(R), NOT(Q)));

nout = max(nargout,1)-1;

if nout > 0
    
    [Ux Sx Vx] = svd(RorQ);
    
    
    
    for k = 1:nout
        if k == 1
            varargout(k) = {Ux};
        elseif k == 2
            varargout(k) = {Sx};
        end
    end
    
end
