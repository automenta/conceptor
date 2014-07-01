function [normData, shifts, scales] = normalizeJapCellData(data)
% Normalizes Japanese vowel training input data. Returns shifts and
% scales of first 12 columns as row vectors (size 12), where 
% normData = scales * (data + shift).
% The last two columns of data are just preserved.

% assemble all samples in one big matrix
totalLength = 0;
for s = 1:270
    totalLength = totalLength + size(data{s},1);
end
allData = zeros(totalLength, 12);
currentStartIndex = 0;
for s = 1:270
    L = size(data{s},1);
    allData(currentStartIndex+1:currentStartIndex+L,:) = ...
        data{s}(:,1:12);
    currentStartIndex = currentStartIndex + L;
end

maxVals = max(allData);
minVals = min(allData);
shifts = - minVals;
scales = 1./(maxVals - minVals);
normData = cell(270,1);
for s = 1:270
    normData{s} = data{s}(:,1:12) + repmat(shifts, size(data{s},1),1);
    normData{s} = normData{s} * diag(scales);
    % add last two original columns
    normData{s} = [normData{s} data{s}(:,13:14)];
end


   