%%%%%% Import the Japanese vouwels data to Matlab

% This script is called from the main script
% of this package, JapVow_MainScript.

% For terms of use see notification in JapVow_MainScript.
% Created and copyright: Herbert Jaeger, Feb 23, 2007


% Inputs to this script: original ASCII files ae.train and ae.test
% downloaded from the UCI benchmark repository at http://kdd.ics.uci.edu.
% Data have been donated by M. Kudo, J. Toyama, and M. Shimbo

% This script assigns global Matlab cell arrays trainInputs, testInputs,
% trainOutputs, testOutputs with cell sizes (270,1), (370,1), (270,1),
% (370,1) respectively. Each cell contains the corresponding
% multidimensional time series (inputs 14 dim, outputs 9 dim)



load ae.train -ascii;
aetrain = ae;
load ae.test -ascii;
aetest = ae;

% aetrain and aetest contain the 12-dim time series, which have
% different lengthes, concatenated vertically and separated by ones(1,12)
% rows. We now sort them into cell arrays, such that each cell represents
% one time series
trainInputs = cell(270,1);
readindex = 0;
for c = 1:270
    readindex = readindex + 1;
    l = 0;    
    while aetrain(readindex, 1) ~= 1.0
        l = l+1;
        readindex = readindex + 1;
    end
    trainInputs{c,1} = aetrain(readindex-l:readindex-1,:);
    % add rampö and lenght indicator intput channels
    trainInputs{c,1} = [trainInputs{c,1}  (1:l)'/l (l/30) * ones(l,1)];
end

testInputs = cell(370,1);
readindex = 0;
for c = 1:370
    readindex = readindex + 1;
    l = 0;    
    while aetest(readindex, 1) ~= 1.0
        l = l+1;
        readindex = readindex + 1;
    end
    testInputs{c,1} = aetest(readindex-l:readindex-1,:);
    % add bias and lenght indicator intput channels
    testInputs{c,1} = [testInputs{c,1} (1:l)'/l (l/30) * ones(l,1)];
end

% produce teacher signals. For each input time series of size N x 12 this
% is a time series of size N x 9, all zeros except in the column indicating
% the speaker, where it is 1.
trainOutputs = cell(270,1);
for c = 1:270
    l = size(trainInputs{c,1},1);
    teacher = zeros(l,9);
    speakerIndex = ceil(c/30);
    teacher(:,speakerIndex) = ones(l,1);
    trainOutputs{c,1} = teacher;
end

testOutputs = cell(370,1);
speakerIndex = 1;
blockCounter = 0;
blockLengthes = [31 35 88 44 29 24 40 50 29];
for c = 1:370
    blockCounter = blockCounter + 1;
    if blockCounter == blockLengthes(speakerIndex)+ 1
        speakerIndex = speakerIndex + 1;
        blockCounter = 1;
    end
    l = size(testInputs{c,1},1);
    teacher = zeros(l,9);    
    teacher(:,speakerIndex) = ones(l,1);
    testOutputs{c,1} = teacher;
end


