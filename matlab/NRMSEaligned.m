function nrmseVal = NRMSEaligned(reference, signal, variance, intRate)
% compute nrmse of signal w.r.t. reference, assuming that there is a shift
% between the two. This shift is compensated by (i) interpolation reference
% and signal with interpolation rate intRate, (ii) shifting the
% interpolated signal to optimal fit position, (iii) computing the nrmse
% using the input variance as variance.
%
% reference and signal must be row vectors, the reference being longer than
% the signal to allow trying shifts
%%
siglength = length(signal);
reflength = length(reference);

        sigInt = interp1((1:siglength)',signal',...
            (1:(1/intRate):siglength)', 'spline')';
        refInt = interp1((1:reflength)', reference',...
            (1:(1/intRate):reflength)', 'spline')';
        
        L = size(refInt,2); M = size(sigInt,2);
        phasematches = zeros(1,L - M);
        for phaseshift = 1:(L - M)
            phasematches(1,phaseshift) = ...
                norm(sigInt - ...
                refInt(1,phaseshift:phaseshift+M-1));
        end
        [maxVal maxInd] = max(-phasematches);
        refAligned = ...
            refInt(1,maxInd:intRate:...
            (maxInd+intRate*siglength-1));
       nrmseVal = sqrt(mean((signal - refAligned).^2) / variance);