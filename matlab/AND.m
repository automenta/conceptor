function [CandB, varargout] = AND(C, B)

dim = size(C,1);
tol = 1e-14;

[UC SC UtC] = svd(C); [UB SB UtB] = svd(B);

dSC = diag(SC);
dSB = diag(SB);

numRankC =  sum(1.0 * (dSC > tol));
numRankB =  sum(1.0 * (dSB > tol));


UC0 = UC(:, numRankC + 1:end);
UB0 = UB(:, numRankB + 1:end);
[W Sigma Wt] = svd(UC0 * UC0' + UB0 * UB0');
numRankSigma =  sum(1.0 * (diag(Sigma) > tol));
Wgk = W(:,numRankSigma+1:end);
CandB = ...
  Wgk *inv(Wgk' *  ...
  ( pinv(C, tol) + pinv(B, tol) - eye(dim)) * Wgk) *Wgk';

nout = max(nargout,1)-1;

if nout > 0
    [Ux Sx Vx] = svd(CandB);
    for k = 1:nout
        if k == 1
            varargout(k) = {Ux};
        elseif k == 2
            varargout(k) = {Sx};
        end
    end
end
