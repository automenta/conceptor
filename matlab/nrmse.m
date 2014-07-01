function NRMSE = nrmse( output, target)
% Computes normalized root mean square error.
%
% Input arguments:
% - output, target: two time series of equal dim and length, format
%                   dim x timepoints
%
% Outputs:
% - nmse: normalized mean root square error (in column vector) 
%
% Created by Mantas Lukosevicius
% Rev 1, April 21 2008, HJaeger: changed ./ targVar in computation of nmse
%                           to / mean(targVar) to avoid Inf in case 
%                           of all-zero rows in target
% Rev 2, Aug 23, 2008 HJaeger: input format of output and input has
%                               now time in rows
% Rev 3, May 02 2009 HJaeger: division by target variance is now again
%                             dim-wise, Inf outputs are now possible again 

combinedVar = 0.5 * (var( target, 0, 2 ) + var( output, 0, 2 )); 


errorSignal = ( output - target );
NRMSE = sqrt( mean( errorSignal .^ 2, 2 ) ./ combinedVar );
