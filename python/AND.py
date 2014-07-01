
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def AND(C, B):

    # Local Variables: UB0, varargout, numRankSigma, nout, k, Ux, UC0, Wt, tol, C, B, numRankB, numRankC, UtC, Vx, Wgk, W, Sigma, UtB, dim, Sx, CandB, SC, SB, dSB, dSC, UC, UB
    # Function calls: AND, eye, diag, svd, nargout, inv, pinv, max, sum, size
    dim = matcompat.size(C, 1.)
    tol = 1e-14
    [UC, SC, UtC] = plt.svd(C)
    [UB, SB, UtB] = plt.svd(B)
    dSC = np.diag(SC)
    dSB = np.diag(SB)
    numRankC = np.sum(np.dot(1.0, dSC > tol))
    numRankB = np.sum(np.dot(1.0, dSB > tol))
    UC0 = UC[:,int(numRankC+1.)-1:]
    UB0 = UB[:,int(numRankB+1.)-1:]
    [W, Sigma, Wt] = plt.svd((np.dot(UC0, UC0.conj().T)+np.dot(UB0, UB0.conj().T)))
    numRankSigma = np.sum(np.dot(1.0, np.diag(Sigma) > tol))
    Wgk = W[:,int(numRankSigma+1.)-1:]
    CandB = np.dot(np.dot(Wgk, linalg.inv(np.dot(np.dot(Wgk.conj().T, linalg.pinv(C, tol)+linalg.pinv(B, tol)-np.eye(dim)), Wgk))), Wgk.conj().T)
    nout = matcompat.max(nargout, 1.)-1.
    if nout > 0.:
        [Ux, Sx, Vx] = plt.svd(CandB)
        for k in np.arange(1., (nout)+1):
            if k == 1.:
                varargout[int(k)-1] = cellarray(np.hstack((Ux)))
            elif k == 2.:
                varargout[int(k)-1] = cellarray(np.hstack((Sx)))
                
            
            
    
    
    return [CandB, varargout]