function [notR, varargout] = NOT(R)
% NOT definiert durch I - R

dim = size(R,1);

notR = eye(dim) - R;
[U S V] = svd(notR);

nout = max(nargout,1)-1;
for k = 1:nout
    if k == 1
        varargout(k) = {U};
    elseif k == 2
        varargout(k) = {S};
    end
end

