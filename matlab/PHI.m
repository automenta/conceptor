function Cnew = PHI(C, gamma)
% aperture adaptation of conceptor C by factor gamma, 
% where 0 <= gamma <= Inf

dim = size(C,1);

if gamma == 0
    
    [U S V] = svd(C);
    Sdiag = diag(S);
    Sdiag(Sdiag < 1) = zeros(sum(Sdiag < 1),1);
    Cnew = U * diag(Sdiag) * U';    
    
elseif gamma == Inf
    
    [U S V] = svd(C);
    Sdiag = diag(S);
    Sdiag(Sdiag > 0) = ones(sum(Sdiag > 0),1);
    Cnew = U * diag(Sdiag) * U'; 
    
    
else

    Cnew = C * inv(C + gamma^(-2) * (eye(dim) - C));

end