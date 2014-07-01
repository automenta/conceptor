%%%% trying to classify the Jap vowels
%%%% here: with nonstationary C sweep, with signals normalized to
%%%% same duration for all speakers.
%%%% Here: with monitoring training classification error
%%%% Here: include input data in x
%%%% Here: using raw input data, interpolating network states
%%%% Here: use eigenvectors of evidence distribution matrices
%%%% Here: employ several apertures
%%%% Here: include xLarge' * xLarge in Cpos in testing
%%%% Here: variable training set size


set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment control
randstate = 1;
loadData = 1;
plotData = 1; % whether to produce illustrative plots of dataset
normalizePlots = 1;
trialsN = 1; % Nr of repetitions of experiment, with
% freshly sampled reservoir weights
plot4paper = 1; % whether to generate plots as used in the report

%%% Setting system params for C experiments
N = 10;
SR = 1.2;
WinScalings = .2 * ones(1,12);
biasScaling = 1;
xStartScaling = 1;

%%% Setting learning params
apsExploreExponents = 0:8; % exponents base 2 for apertures
trainN = 30; % how many samples per class are used for training
useExtendedC = 1; % whether in classification test we extend C's by
% current x' * x information
apSweepPlotFlag = 0; % whether we compute and plot classification
% evidences for different apertures from last trial (expensive)

% linear classifier for comparison
linAlphaSearchRange = 2.^(-10:0);

%%% input preprocessing
polyOrder = 3;
virtualLength = 4;

%%% Init rand generator
randn('state', randstate);
rand('twister', randstate);

M = N + 12;
apN = length(apsExploreExponents);
I = eye(M * virtualLength);

%% import data
if loadData
    % After this, we will have trainInputs = cell(270,1); testInputs =
    % cell(370,1); trainOutputs = cell(270,1); testOutputs = cell(370,1)
    % where each input cell contains the 12-dim signal (size time x 12)
    % and each output cell the 9-dim indicator signal of same size as
    % the corresponding input data cell.
    JapVouwels_dataPrep;
    trainInputsRaw = trainInputs;
    testInputsRaw = testInputs;
    [trainInputs,  shifts, scales] = normalizeJapCellData(trainInputs);
    % Normalizes Japanese vowel training input data. Returns shifts and
    % scales as row vectors (size 12), where
    % normData = scales * (data + shift).
    testInputs = transformJapData(testInputs,  shifts, scales); % submit the test Inputs to
    % the same transformation, discard last two cols
    
    % smoothing, plus we discard the last two channels which contain
    % a unit ramp (channel 13) and code sequence
    % length (channel 14)
    
    for i = 1:270
        p = trainInputs{i};
        pNew = zeros(virtualLength,12);
        l = size(p,1);
        for s = 1:12
            polyCoeffs = polyfit((1:l)', p(:,s), polyOrder);
            newS = polyval(polyCoeffs, (1:l)');
            newSNormalLength = ...
                interp1((1:l)', newS, 1:(l-1)/(virtualLength-1):l);
            pNew(:,s) = newSNormalLength';
        end
        trainInputs{i} = pNew ;
    end
    for i = 1:370
        p = testInputs{i};
        pNew = zeros(virtualLength,12);
        l = size(p,1);
        for s = 1:12
            polyCoeffs = polyfit((1:l)', p(:,s), polyOrder);
            newS = polyval(polyCoeffs, (1:l)');
            newSNormalLength = ...
                interp1((1:l)', newS, 1:(l-1)/(virtualLength-1):l);
            pNew(:,s) = newSNormalLength';
        end
        testInputs{i} = pNew ;
    end
    
end
%% plot figure for report
if plotData
     figure(23); clf;
     fs = 18;
     set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [800 300 800 400]);
     speakers = [4 6 8];
    currentPlot = 0;
    for speaker = speakers           
            currentPlot = currentPlot +1;
            subplot(2,3,currentPlot)
            utterancedata = trainInputsRaw{(speaker-1)*30+1,1};
            l = size(utterancedata,1);
            
            plot(1:l,utterancedata(:,1)', 'k','LineWidth', 2); hold on;
            plot(1:l,utterancedata(:,2)', 'k','LineWidth', 1);
            plot(1:l,utterancedata(:,3)', 'g','LineWidth', 2);
            plot(1:l,utterancedata(:,4)', 'g','LineWidth', 1);
            plot(1:l,utterancedata(:,5)', 'k:','LineWidth', 2);
            plot(1:l,utterancedata(:,6)', 'k:','LineWidth', 1);
            plot(1:l,utterancedata(:,7)', 'k:','LineWidth', 0.5);
            plot(1:l,utterancedata(:,8)', 'g','LineWidth', 0.5);
            plot(1:l,utterancedata(:,9)', 'k','LineWidth', 0.5);
            plot(1:l,utterancedata(:,10)', 'k--','LineWidth', 2);
            plot(1:l,utterancedata(:,11)', 'k--','LineWidth', 1);
            plot(1:l,utterancedata(:,12)', 'k--','LineWidth', 0.5);
            hold off;
            axis([1 25 -1.5 2]);
            set(gca, 'FontSize', fs);
            title(sprintf('speaker %g', speakers(currentPlot)));
        
    end
    for speaker = speakers       
            currentPlot = currentPlot +1;
            subplot(2,3,currentPlot)
            utterancedata = trainInputs{(speaker-1)*30+1,1};
            l = size(utterancedata,1);
            
            plot(1:l,utterancedata(:,1)', 'k','LineWidth', 2); hold on;
            plot(1:l,utterancedata(:,2)', 'k','LineWidth', 1);
            plot(1:l,utterancedata(:,3)', 'g','LineWidth', 2);
            plot(1:l,utterancedata(:,4)', 'g','LineWidth', 1);
            plot(1:l,utterancedata(:,5)', 'k:','LineWidth', 2);
            plot(1:l,utterancedata(:,6)', 'k:','LineWidth', 1);
            plot(1:l,utterancedata(:,7)', 'k:','LineWidth', 0.5);
            plot(1:l,utterancedata(:,8)', 'g','LineWidth', 0.5);
            plot(1:l,utterancedata(:,9)', 'k','LineWidth', 0.5);
            plot(1:l,utterancedata(:,10)', 'k--','LineWidth', 2);
            plot(1:l,utterancedata(:,11)', 'k--','LineWidth', 1);
            plot(1:l,utterancedata(:,12)', 'k--','LineWidth', 0.5);
            hold off;
            axis([1 4 0 1]);
            set(gca, 'FontSize', fs);
    end
end

%% do training and testing trials
% init of collectors for nrs of test errors and double errors
errPosCollector = zeros(1, trialsN);
errNegCollector = zeros(1, trialsN);
errCombCollector = zeros(1, trialsN);
errLinCollector = zeros(1, trialsN);
errPosDoubleCollector = zeros(1, trialsN);
errNegDoubleCollector = zeros(1, trialsN);
errCombDoubleCollector = zeros(1, trialsN);
errTrainCollector = zeros(1, trialsN);
% init of collectors of misclassificed patter indices
errPosIndCollector = zeros(1,370);
errNegIndCollector = zeros(1, 370);
errCombIndCollector = zeros(1, 370);
errLinIndCollector = zeros(1, 370);
% aperture and regularizer collectors
bestApCollector = zeros(2, trialsN);
bestRegularizerCollector = zeros(1, trialsN);

for triali = 1:trialsN
    disp(sprintf('***** trial Nr %g ******', triali));
    tic;
    % sample reservoir weights
    if N <= 20
        Cconnectivity = 1;
    else
        Cconnectivity = 10/N;
    end
    W = SR * generate_internal_weights(N, Cconnectivity);
    Win = randn(N, 12) * diag(WinScalings);
    bias = biasScaling * randn(N,1);
    xStart = xStartScaling *  randn(N,1);
    
    % collect all training states
    allStatesTrain = zeros(M*virtualLength, trainN, 9);
    for i = 1:9
        sampleInds = ((i-1)*30+1):(((i-1)*30+1) + trainN - 1);
        for n = 1:trainN
            p = trainInputs{sampleInds(n)};
            xSeq = zeros(M,virtualLength);
            x = xStart;
            for t = 1:virtualLength
                x = tanh(W * x + Win * p(t,:)' + bias);
                xSeq(:,t) = [x;p(t,:)'];
            end
            xLarge = reshape(xSeq, M * virtualLength, 1);
            allStatesTrain(:,n,i) = xLarge;
        end
    end
    
    
    % find best regularizer for linear comparison by 5-fold crossvalidation
    
    if mod(trainN,5) ~= 0
        error('linear comparison with Xval optimization needs trainN to be a multiple of 5');
    end
    candN = length(linAlphaSearchRange);
    mseCollector = zeros(1,candN);
    L = trainN / 5; % length of Xval data blocks (within each class)
    
    % construct targets
    trainTargs = zeros(9, 4*9*trainN / 5);
    testTargs = zeros(9, 1*9*trainN / 5);
    for i = 1:9
        trainTargs(i, (i-1)*4*L+1:i*4*L) = ones(1,4*L);
        testTargs(i, (i-1)*L+1:i*L) = ones(1,L);
    end
    for testi = 1:candN
        testAlpha = linAlphaSearchRange(testi);
        mseSum = 0;
        for f = 1:5 % the folds
            % collect train and test states
            testInds = (f-1)*L+1:f*L;
            trainInds = [1:(f-1)*L, f*L+1:trainN];
            trainx = reshape(allStatesTrain(:,trainInds,:),...
                M*virtualLength, 4 * 9 * trainN / 5);
            testx = reshape(allStatesTrain(:,testInds,:),...
                M*virtualLength, 1 * 9 * trainN / 5);
            % compute weights
            Wtest = (inv(trainx * trainx' + ...
                testAlpha * I) * ...
                trainx * trainTargs')';
            % compute test outputs
            testOut = Wtest * testx;
            % compute mse
            mse = mean(mean((testOut - testTargs).^2));
            mseSum = mseSum + mse;
        end
        mseCollector(1,testi) = mseSum;
    end
    % find best regularizer alpha
    [dummy alphaBestInd] = min(mseCollector);
    alpha = linAlphaSearchRange(alphaBestInd);
    disp(sprintf('best regularizer %g', alpha));     
    bestRegularizerCollector(1,triali) = alpha;
    
    %%
    % compute conceptors
    
    % get conceptors for different aps for exploration
    CPoss = cell(9, apN);
    RPoss = cell(9, 1);
    ROthers = cell(9, 1);
    CNegs = cell(9, apN);
    statesAllSpeakers = ...
        reshape(allStatesTrain, ...
        M * virtualLength, trainN  * 9);
    Rall = statesAllSpeakers * statesAllSpeakers';
    for i = 1:9
        R = allStatesTrain(:,:,i) * allStatesTrain(:,:,i)';
        RPoss{i} = R;
        Rnorm = R / trainN;
        ROther = Rall - R;
        ROthers{i} = ROther;
        ROthersNorm = ROther / (8 * trainN);
        for api = 1:apN
            CPoss{i, api} = Rnorm * inv(Rnorm + ...
                (2^apsExploreExponents(api))^(-2) * I);
            COther = ROthersNorm * inv(ROthersNorm + ...
                (2^apsExploreExponents(api))^(-2) * I);
            CNegs{i, api} = I - COther ;
        end
    end
    % find best apertures 

    
    bestApsPos = zeros(1,9); bestApsNeg = zeros(1,9);
    for i = 1:9
       normsPos = zeros(1,apN); normsNeg = zeros(1,apN); 
       for api = 1:apN
           normsPos(api) = norm(CPoss{i,api} ,'fro')^2 ;
           normsNeg(api) = norm(I - CNegs{i,api} ,'fro')^2 ;
       end
       intPts = apsExploreExponents(1):0.01:apsExploreExponents(end);
       normsPosIntpl = ...
           interp1(apsExploreExponents, normsPos, intPts, 'spline');
       normsNegIntpl = ...
           interp1(apsExploreExponents, normsNeg, intPts, 'spline');
       normsPosIntplGrad = ...
           (normsPosIntpl(1,2:end) - normsPosIntpl(1,1:end-1))/0.01;
       normsPosIntplGrad = [normsPosIntplGrad normsPosIntplGrad(end)];
       normsNegIntplGrad = ...
           (normsNegIntpl(1,2:end) - normsNegIntpl(1,1:end-1))/0.01;
       normsNegIntplGrad = [normsNegIntplGrad normsNegIntplGrad(end)];
       [maxVal maxIndPos] = max(abs(normsPosIntplGrad));
       [maxVal maxIndNeg] = max(abs(normsNegIntplGrad));
       bestApsPos(i) = 2^intPts(maxIndPos);
       bestApsNeg(i) = 2^intPts(maxIndNeg);
    end
    bestApPos = mean(bestApsPos);
    bestApNeg = mean(bestApsNeg);
    
    disp(sprintf('best Ap pos / neg  %0.3g  %0.3g', ...
        bestApPos, bestApNeg));
    
timeLearn = toc
    
    
   %%

    bestApCollector(:, triali) = [bestApPos; bestApNeg];
    % compute best-aperture conceptors
    CPosBest = cell(1,9);
    CNegBest = cell(1,9);
    for i = 1:9
        Rnorm = RPoss{i} / trainN;
        ROthersNorm = ROthers{i} / (8 * trainN);
        CPosBest{i} = Rnorm * inv(Rnorm + ...
            bestApPos^(-2) * I);
        COther = ROthersNorm * inv(ROthersNorm + ...
            bestApNeg^(-2) * I);
        CNegBest{i} = I - COther ;
    end    
    
    % compute weights of linear classifier    
    targets = zeros(9,trainN  * 9);
    args = zeros(M * virtualLength, trainN * 9);
    for i = 1:9
        targets(i,(i-1)*trainN+1:i*trainN) = ones(1, trainN);
        args(:,(i-1)*trainN+1:i*trainN) = ...
            allStatesTrain(:,1:trainN,i);
    end
    Wclass = (inv(args * args' + alpha * I)*...
        args * targets')'; 
    
    % performance on training data
     combEvTrain = zeros(9, 9*trainN);
    for iBlock = 1:9
        for n = 1:trainN
            x = allStatesTrain(:,n,iBlock);
            combEvVec = zeros(9,1);
            for i = 1:9
                combEvVec(i) = x' * CPosBest{i} * x + ...
                    x' * CNegBest{i} * x;                
            end
            k = (iBlock-1)*trainN + n;
            combEvTrain(:,k) = combEvVec;            
        end
    end
    correctMatTrain = zeros(9,9*trainN);
    for j = 1:9
        correctMatTrain(j,((j-1)*trainN+1):(j*trainN)) = ...
            ones(1,trainN);
    end
    
    % performance on test data
    posEv = zeros(9, 370);
    negEv = zeros(9, 370);
    combEv = zeros(9, 370);
    xsLargeTest = zeros(M * virtualLength, 370);
    if apSweepPlotFlag && triali == trialsN
        posEvApiPlots = cell(1,370);
        negEvApiPlots = cell(1,370);
        combEvApiPlots = cell(1,370);
    end
    tic
    for j = 1:370
        % drive reservoir with this pattern
        p = testInputs{j};
        xSeq = zeros(M,virtualLength);
        x = xStart;
        for t = 1:virtualLength
            x = tanh(W * x + Win * p(t,:)' + bias);
            xSeq(:,t) = [x;p(t,:)'];
        end
        xLarge = reshape(xSeq, M * virtualLength, 1);
        xsLargeTest(:,j) = xLarge;
        xTx = xLarge' * xLarge;
        
        % optionally collect evidences for the various apertures
        if apSweepPlotFlag && triali == trialsN
            posEvVecs = zeros(9, apN);
            negEvVecs = zeros(9, apN);
            for i = 1:9
                for api = 1:apN                    
                    Rpos =  (RPoss{i} + xLarge * xLarge') / (trainN+1);
                    Cpos = Rpos * inv(Rpos +...
                        (2^apsExploreExponents(api))^(-2) * I);
                    posEvVecs(i, api) = ...
                        xLarge' * Cpos * xLarge / xTx ;                    
                    Rneg = (ROthers{i} + 8*xLarge * xLarge') /...
                        (8 * trainN + 8);
                    Cneg = I - Rneg * inv(Rneg +...
                        (2^apsExploreExponents(api))^(-2) * I);
                    negEvVecs(i, api) = ...
                        xLarge' * Cneg * xLarge / xTx ;
                end
            end
            minValsPos = min(posEvVecs); maxValsPos = max(posEvVecs);
            rangeValsPos = maxValsPos - minValsPos;
            minValsNeg = min(negEvVecs); maxValsNeg = max(negEvVecs);
            rangeValsNeg = maxValsNeg - minValsNeg;
            meanValsPos = mean(posEvVecs);
            meanValsNeg = mean(negEvVecs);
            posEvVecsNorm = (posEvVecs - repmat(minValsPos, 9,1)) * ...
                diag(1./ rangeValsPos);
            negEvVecsNorm = (negEvVecs - repmat(minValsNeg, 9,1)) * ...
                diag(1./ rangeValsNeg);
            posEvApiPlots{1,j} = posEvVecsNorm;
            negEvApiPlots{1,j} = negEvVecsNorm;
            combEvApiPlots{1,j} = posEvVecsNorm + negEvVecsNorm;
        end
        
        % evidences
        posEvVec = zeros(9, 1);
        negEvVec = zeros(9, 1);
        
        for i = 1:9
            if useExtendedC
                R =  (RPoss{i} + xLarge * xLarge') / (trainN+1);
                ROther = (ROthers{i} + 8*xLarge * xLarge') /...
                    (8 * trainN + 8);
                Cpos = R * inv(R +...
                    bestApPos^(-2) * I);
                
                posEvVec(i) = ...
                    xLarge' * Cpos * xLarge / xTx ;
                Cneg = I - ROther * inv(ROther +...
                    bestApNeg^(-2)*I);
                negEvVec(i) = ...
                    xLarge' * Cneg * xLarge / xTx;
            else
                posEvVec(i) = ...
                    xLarge' * CPosBest{i} * xLarge / xTx;
                negEvVec(i) = ...
                    xLarge' * CNegBest{i} * xLarge / xTx;
            end
        end
        minValPos = min(posEvVec); maxValPos = max(posEvVec);
        rangePos = maxValPos - minValPos;
        minValNeg = min(negEvVec); maxValNeg = max(negEvVec);
        rangeNeg = maxValNeg - minValNeg;
        
        posEvVecNorm = (posEvVec - minValPos) / rangePos;
        negEvVecNorm = (negEvVec - minValNeg) / rangeNeg;
        
        posEv(:,j) = posEvVecNorm;
        negEv(:,j) = negEvVecNorm;
        
        combEv(:,j) = posEvVecNorm + negEvVecNorm;
        
        speakerIndicatorVec = testOutputs{j}(1,:);
        [xx speakerInd] = max(speakerIndicatorVec);
        correctMatTest(speakerInd, j) = 1;
    end
    timeTest = toc / 370
    % evidences from linear classifier
    compareEv = Wclass * xsLargeTest;
    
    
    % normalize evidence mats for column range 0-1 for nicer plots
    if normalizePlots
        [posEv, scalings, shifts] = normalizeData01(posEv);
        [negEv, scalings, shifts] = normalizeData01(negEv);
        [combEv, scalings, shifts] = normalizeData01(combEv);
        [compareEv, scalings, shifts] = normalizeData01(compareEv);
    end
    
    % compute Nr of incorrect classifications
    [maxVals maxIndsCorrect] = max(correctMatTest);
    [maxVals maxIndsCorrectTrain] = max(correctMatTrain);
    [maxVals maxIndsTest_comb] = max(combEv);
    [maxVals maxIndsTest_neg] = max(negEv);
    [maxVals maxIndsTest_pos] = max(posEv);
    [maxVals maxIndsTestCompare] = max(compareEv);
    [maxVals maxIndsTrain] = max(combEvTrain);
    
    % patterns where pos and neg evidence disagree
    disagreeLogical = maxIndsTest_neg - maxIndsTest_pos ~= 0;
    disagreeInds = [];
    for i = 1:370
        if disagreeLogical(i)
            disagreeInds = [disagreeInds i];
        end
    end
    
    % compute second places
    [sortVals sortIndsTest_pos] = sort(posEv, 'descend');
    secondIndsTest_pos = sortIndsTest_pos(2,:);
    [sortVals sortIndsTest_neg] = sort(negEv, 'descend');
    secondIndsTest_neg = sortIndsTest_neg(2,:);
    [sortVals sortIndsTest_comb] = sort(combEv, 'descend');
    secondIndsTest_comb = sortIndsTest_comb(2,:);
    [sortVals sortIndsTestCompare] = sort(compareEv, 'descend');
    secondIndsTestCompare = sortIndsTestCompare(2,:);
    
    
    % compute for how many patters both first and second placed are wrong
    errPattDoubleLogical_comb = ...
        (maxIndsCorrect - maxIndsTest_comb) .* ...
        (maxIndsCorrect - secondIndsTest_comb) ~= 0;
    errPattDoubleLogical_neg = ...
        (maxIndsCorrect - maxIndsTest_neg) .* ...
        (maxIndsCorrect - secondIndsTest_neg) ~= 0;
    errPattDoubleLogical_pos = ...
        (maxIndsCorrect - maxIndsTest_pos) .* ...
        (maxIndsCorrect - secondIndsTest_pos) ~= 0;
    errPattDoubleLogicalCompare = ...
        (maxIndsCorrect - maxIndsTestCompare) .* ...
        (maxIndsCorrect - secondIndsTestCompare) ~= 0;
    
    errorsDoubleTest_comb = ...
        sum(+errPattDoubleLogical_comb );
    errorsDoubleTest_neg = ...
        sum(+errPattDoubleLogical_neg);
    errorsDoubleTest_pos = ...
        sum(+errPattDoubleLogical_pos);
    errorsDoubleTestCompare = ...
        sum(+errPattDoubleLogicalCompare);
    
    errorsTest_comb = ...
        sum(+(maxIndsCorrect - maxIndsTest_comb ~= 0));
    errorsTest_neg = ...
        sum(+(maxIndsCorrect - maxIndsTest_neg ~= 0));
    errorsTest_pos = ...
        sum(+(maxIndsCorrect - maxIndsTest_pos ~= 0));
    errorsTestCompare = ...
        sum(+(maxIndsCorrect - maxIndsTestCompare ~= 0));
    errorsTrain = ...
        sum(+(maxIndsCorrectTrain - maxIndsTrain ~= 0));
    
    errPattLogical_comb = maxIndsCorrect - maxIndsTest_comb ~= 0;
    errPattInds_comb = [];
    errPattLogical_neg = maxIndsCorrect - maxIndsTest_neg ~= 0;
    errPattInds_neg = [];
    errPattLogical_pos = maxIndsCorrect - maxIndsTest_pos ~= 0;
    errPattInds_pos = [];
    errPattDoubleInds_comb = [];
    errPattDoubleInds_neg = [];
    errPattDoubleInds_pos = [];
    errPattLogicalCompare = maxIndsCorrect - maxIndsTestCompare ~= 0;
    errPattIndsCompare = [];
    errPattDoubleIndsCompare = [];
    
    for i = 1:370
        if errPattLogical_comb(i)
            errPattInds_comb = [errPattInds_comb i];
        end
        if errPattLogical_neg(i)
            errPattInds_neg = [errPattInds_neg i];
        end
        if errPattLogical_pos(i)
            errPattInds_pos = [errPattInds_pos i];
        end
        if errPattDoubleLogical_comb(i)
            errPattDoubleInds_comb = [errPattDoubleInds_comb i];
        end
        if errPattDoubleLogical_neg(i)
            errPattDoubleInds_neg = [errPattDoubleInds_neg i];
        end
        if errPattDoubleLogical_pos(i)
            errPattDoubleInds_pos = [errPattDoubleInds_pos i];
        end
        if errPattLogicalCompare(i)
            errPattIndsCompare = [errPattIndsCompare i];
        end
        if errPattDoubleLogicalCompare(i)
            errPattDoubleIndsCompare = [errPattDoubleIndsCompare i];
        end
        
    end
    
    disp(sprintf('PosErrs  %s', num2str(errPattInds_pos)));
    disp(sprintf('NegErrs  %s', num2str(errPattInds_neg)));
    disp(sprintf('CombErrs %s', num2str(errPattInds_comb)));
    disp(sprintf('PosErrs Double %s', num2str(errPattDoubleInds_pos)));
    disp(sprintf('NegErrs Double %s', num2str(errPattDoubleInds_neg)));
    disp(sprintf('CombErrs Double %s', num2str(errPattDoubleInds_comb)));
    disp(sprintf('disagree %s', num2str(disagreeInds)));
    disp(sprintf('Lin comparison %s', ...
        num2str(errPattIndsCompare)));
    
    errPosCollector(1,triali) = errorsTest_pos;
    errNegCollector(1,triali) = errorsTest_neg;
    errCombCollector(1,triali) = errorsTest_comb;
    errLinCollector(1,triali) = errorsTestCompare;
    errPosDoubleCollector(1,triali) = errorsDoubleTest_pos;
    errNegDoubleCollector(1,triali) = errorsDoubleTest_neg;
    errCombDoubleCollector(1,triali) = errorsDoubleTest_comb;
    errTrainCollector(1,triali) = errorsTrain;
    
    errPosIndCollector(1,errPattInds_pos) = ...
        errPosIndCollector(1,errPattInds_pos) + ...
        ones(1,size(errPattInds_pos,2));
    errNegIndCollector(1,errPattInds_neg) = ...
        errNegIndCollector(1,errPattInds_neg) + ...
        ones(1,size(errPattInds_neg,2));
    errCombIndCollector(1,errPattInds_comb) = ...
        errCombIndCollector(1,errPattInds_comb) + ...
        ones(1,size(errPattInds_comb,2));
    errLinIndCollector(1,errPattIndsCompare) = ...
        errLinIndCollector(1,errPattIndsCompare) + ...
        ones(1,size(errPattIndsCompare,2));
end

disp('****** collected error information *******');
disp(sprintf('mean pos err %0.2g  neg err %0.2g  comb err %0.2g ',...
    mean(errPosCollector), mean(errNegCollector), ...
    mean(errCombCollector)));
disp(sprintf('stdev pos err %0.2g  neg err %0.2g  comb err %0.2g ',...
    var(errPosCollector)^0.5, var(errNegCollector)^0.5, ...
    var(errCombCollector)^0.5));
disp(sprintf('mean train err %0.2g', mean(errTrainCollector)));
disp(sprintf('mean bestApPos %0.2g  stddev %0.2g', ...
    mean(bestApCollector(1,:)), sqrt(var(bestApCollector(1,:)))));
disp(sprintf('mean bestApNeg %0.2g  stddev %0.2g', ...
    mean(bestApCollector(2,:)), sqrt(var(bestApCollector(2,:)))));
disp(sprintf('mean linear comparison error %0.2g',...
sum(errLinIndCollector)/trialsN));
disp(sprintf('stdev linear comparison error %0.2g',...
var(errLinCollector)^0.5));

%% plotting details from last trial
% compute plot matrix of made classifications from last trial
classMat_comb = zeros(9, 370);
classMat_pos = zeros(9, 370);
classMat_neg = zeros(9, 370);
classMat_compare = zeros(9, 370);
for p = 1:370
    classMat_comb(maxIndsTest_comb(p),p) = 1;
    classMat_pos(maxIndsTest_pos(p),p) = 1;
    classMat_neg(maxIndsTest_neg(p),p) = 1;
    classMat_compare(maxIndsTestCompare(p),p) = 1;
end

%%
figure(1); clf;
[minVal maxVal] = plotmatrix(-posEv, 'g');
title(sprintf('pos evidence'), 'FontSize', 14);
figure(2); clf;
[minVal maxVal] = plotmatrix(-classMat_pos, 'g');
title(sprintf('errors: %g %g', errorsTest_pos, errorsDoubleTest_pos), ...
    'FontSize', 14);
%%

figure(3); clf;
[minVal maxVal] = plotmatrix(-negEv, 'g');
title(sprintf('neg evidence'), 'FontSize', 14);
figure(4); clf;
[minVal maxVal] = plotmatrix(-classMat_neg, 'g');
title(sprintf('errors: %g %g', errorsTest_neg, errorsDoubleTest_neg), ...
    'FontSize', 14);
%%

figure(5); clf;
[minVal maxVal] = plotmatrix(-combEv, 'g');
title(sprintf('comb evidence'), 'FontSize', 14);
figure(6); clf;
[minVal maxVal] = plotmatrix(-classMat_comb, 'g');
title(sprintf('errors: %g %g', errorsTest_comb, errorsDoubleTest_comb), ...
    'FontSize', 14);

figure(7); clf;
[minVal maxVal] = plotmatrix(-compareEv, 'g');
title(sprintf('compare evidence'), 'FontSize', 14);
figure(8); clf;
[minVal maxVal] = plotmatrix(-classMat_compare, 'g');
title(sprintf('errors: %g %g',...
    errorsTestCompare, errorsDoubleTestCompare), ...
    'FontSize', 14);

%% plotting for paper
if plot4paper
    figure(20); clf;
    fstitle = 32; fsaxes = 24; fsaxlabel = 28;
    set(gcf, 'WindowStyle','normal');
    set(gcf,'Position', [10 300 1600 450]);
    subplot(1,3,1);
    [minVal maxVal] = plotmatrix(-posEv, 'g');
    title(sprintf('positive evidences'), 'FontSize', fstitle);
    ylabel('speaker', 'FontSize', fsaxlabel);
    xlabel('test cases', 'FontSize', fsaxlabel);
    set(gca, 'FontSize', fsaxes, 'YTick', 1:9);
    subplot(1,3,2);
    [minVal maxVal] = plotmatrix(-negEv, 'g');
    title(sprintf('negative evidences'), 'FontSize', fstitle);
    set(gca, 'FontSize', fsaxes, 'YTick', 1:9);
    subplot(1,3,3);
    [minVal maxVal] = plotmatrix(-combEv, 'g');
    title(sprintf('combined evidences'), 'FontSize', fstitle);
    set(gca, 'FontSize', fsaxes, 'YTick', 1:9);
    
end

%%
if apSweepPlotFlag
    apPlotInds = [10 21 75 115 196 231 232 340 341] + 0;
    figure(9); clf;
    for i = 1:9
        subplot(3,3,i);
        correctInd = maxIndsCorrect(apPlotInds(i));
        decidedInd = maxIndsTest_pos(apPlotInds(i));
        PL = negEvApiPlots{1,apPlotInds(i)};
        
        maxVals = max(PL); minVals = min(PL); ranges = maxVals - minVals;
        snr = ranges ./ maxVals;
        PL = (PL - repmat(minVals,9,1))* diag(1./ ranges);
        hold on;
        plot(PL');
        plot(PL(correctInd,:), 'k', 'LineWidth',2);
        plot(PL(decidedInd,:), 'r', 'LineWidth',2);
        line([log2(bestApPos) log2(bestApPos)]', [1 0]',...
            'Color', 'k', 'LineStyle', ':');
        hold off;
        title(sprintf('PattNr %g',apPlotInds(i)));
        set(gca, 'XLim', [1,apN], 'Box', 'on');
    end
end
%%
if plot4paper
figure(10); clf;
fs = 18;
     set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1000 300 400 200]);
plot(intPts,normsPosIntplGrad, 'LineWidth',2);
set(gca, 'FontSize', fs, 'XTick', 0:8, ...
    'OuterPosition', [0 0.085 1 0.902],...
    'Position', [0.13 0.255 0.775 0.67]);
xlabel('$$\log_2 \gamma$$', 'Interpreter', 'latex', 'FontSize', fs);
%title('aperture selection criterion', 'FontSize', fs);
end
%%

figure(11); clf;
subplot(1,2,1);
bar(errCombIndCollector);
title(sprintf('all misclass C-based, mean ErrNo %0.2g',...
    mean(errCombCollector)));
subplot(1,2,2);
bar(errLinIndCollector);
title(sprintf('all misclass linear, mean ErrNo %0.2g',...
mean(errLinCollector)));



%%
figure(12); clf;
for i = 1:9
    subplot(3,3,i);
    [U Spos V] = svd(CPosBest{i});
    [U Sneg V] = svd(CNegBest{i});
    plot(diag(Spos), 'b'); hold on;
    plot(diag(Sneg), 'r'); hold off;
end


