function internalWeights = generate_internal_weights(nInternalUnits, ...
    connectivity)
% Create a random sparse reservoir for an ESN. Nonzero weights are normal
% distributed.
%
% inputs:
% nInternalUnits = the number of internal units in the ESN
% connectivity: a real in [0,1], the (rough) proportion of nonzero weights
%
% output:
% internalWeights = matrix of size nInternalUnits x nInternalUnits


success = 0;
while not(success)
    try
internalWeights = sprandn(nInternalUnits, nInternalUnits, connectivity);
opts.disp = 0;
specRad = abs(eigs(internalWeights,1, 'lm', opts));
internalWeights = internalWeights/specRad;
success = 1;
    catch
        ;
    end
end
