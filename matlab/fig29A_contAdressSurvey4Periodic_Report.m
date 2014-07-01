
%addpath('./ESN_Toolbox');
set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment control
randstate = 1; newNets = 1; newSystemScalings = 1;
expN = 5; % Nr of experiments over which to average

%%% Setting system params
N = 100; % set to 100 | 200 for patternType 1 | 2
SR = 1.5;
WinScaling = 1.5; % scaling of pattern feeding weights
biasScaling = .5;

%%% Initial C adaptation in driven runs
trainWashoutLength = 100;
learnWLength = 50; % set to 50 | 500 for patternType 1 | 2

%%% D learning
incrementalLoad = 0;
TychonovAlphaD = .001;

%%% pattern readout learning
TychonovAlphaWout = 0.01;

%%% C learning and testing
aperture = 100;
initialWashout = 100;
cueLength = 15;
cueNL = 0.0;

CadaptRateCue = 0.02;
CadaptRateAfterCue = 0.01; % C adaptation rate
SNRTest = Inf ; % state signal-to-noise ratio
SNRCue = Inf;
RL = 300; % test runlength after cue

measureWashout = 50;
measureTemplateLength = 20;
measureRL = 500; % runlength for measuring output NRMSE.
% Must be at least twice measureTemplateLength
signalPlotLength = 10;
singValPlotLength = 20;

%%% Setting patterns
patternType = 1; % 1 random same period
% 2 two irrat sines mix
NpLoads = [2 3 5 8 12 16 25 50 100 200]; % for patternType = 1
%NpLoads = [2 3 5 8 12 16 25 50 100]; % for patternType = 2
NpTest = 10;


%%% Initializations

randn('state', randstate);
rand('twister', randstate);
I = eye(N);

 % Create raw weights
    if newNets
        if N <= 20
            Cconnectivity = 1;
        else
            Cconnectivity = 10/N;
        end
        WRaw = generate_internal_weights(N, Cconnectivity);
        WinRaw = randn(N, 1);
        biasRaw = randn(N, 1);
    end
    
    % Scale raw weights and initialize trainable weights
    if newSystemScalings
        W = SR * WRaw;
        Win = WinScaling * WinRaw;
        bias = biasScaling * biasRaw;
    end

allNRMSEOwns = cell(length(NpLoads), expN);
allNRMSEOthers = cell(length(NpLoads), expN);

for expi = 1:expN
    disp(sprintf('******* experiment Nr %g ********', expi));
   
    
    % Set pattern handles
    NpLoadMax = NpLoads(end);
    patts = cell(1,NpLoadMax);
    pattsOther = cell(1,NpTest);
    
    if patternType == 1
        pLength = 4;
        for p = 1:NpLoadMax
            rp = rand(1,pLength);
            maxVal = max(rp); minVal = min(rp);
            rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
            patts{p} = @(n) (rp(mod(n,pLength)+1));
        end
        for p = 1:NpTest
            rp = rand(1,pLength);
            maxVal = max(rp); minVal = min(rp);
            rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
            pattsOther{p} = @(n) (rp(mod(n,pLength)+1));
        end
    elseif patternType == 2
        stretch = sqrt(30);
        for p = 1:NpLoadMax
            relativePhase = rand;
            firstAmp = rand;
            patts{p} = @(n) ...
                firstAmp * sin(2 * pi * n / stretch) + ...
                (1 - firstAmp) * ...
                sin(4 * pi * (n / stretch + relativePhase));
        end
        for p = 1:NpTest
            relativePhase = rand;
            firstAmp = rand;
            pattsOther{p} = @(n) ...
                firstAmp * sin(2 * pi * n / stretch) + ...
                (1 - firstAmp) * ...
                sin(4 * pi * (n / stretch + relativePhase));
        end
    end
    
    loadIndex = 0;
    for NpLoad = NpLoads
        loadIndex = loadIndex + 1;
        %%% pattern loading
        
        % init collectors needed for computing Wout
        allTrainArgs = zeros(N, NpLoad * learnWLength);
        allTrainYTargs = zeros(1, NpLoad * learnWLength);
        pTemplates = zeros(measureTemplateLength, NpLoad);
        
        if incrementalLoad
            Call = zeros(N, N);
            D = zeros(N, N);
            for p = 1:NpLoad
                patt = patts{p}; % current pattern generator
                pTemplate = zeros(learnWLength,1);
                % drive reservoir with current pattern
                XCue = zeros(N,learnWLength);
                xOldCollector = zeros(N, learnWLength );
                pCollector = zeros(1, learnWLength );
                x = zeros(N, 1);
                for n = 1:trainWashoutLength
                    u = patt(n);
                    x =  tanh(W * x + Win * u + bias);
                end
                for n = 1:learnWLength
                    u = patt(n + trainWashoutLength);
                    xOld = x;
                    x =  tanh(W * x +  Win * u + bias);
                    XCue(:,n) = x;
                    xOldCollector(:, n ) = xOld;
                    pTemplate(n,1) = u;
                end
                pTemplates(:,p) = pTemplate(end-measureTemplateLength+1:end);
                allTrainArgs(:, (p-1)*learnWLength+1:p*learnWLength) = ...
                    XCue;
                allTrainYTargs(1, (p-1)*learnWLength+1:p*learnWLength) = ...
                    pTemplate;
                
                % compute D increment
                Dtargs = Win*pTemplate' - D * xOldCollector;
                F = NOT(Call);
                Dargs = F * xOldCollector ;
                Dinc = (pinv(Dargs * Dargs' / learnWLength + ...
                    aperture^(-2) * I) * Dargs * Dtargs' / learnWLength)' ;
                
                % update D and Call
                D = D  + Dinc ;
                R = xOldCollector * xOldCollector' / (learnWLength + 1);
                Cnative = R * inv(R + I);
                Cap = PHI(Cnative, aperture);
                Call = OR(Call, Cap);
                
            end
        else
            allTrainOldArgs = zeros(N, NpLoad * learnWLength);
            allTrainDTargs = zeros(N, NpLoad * learnWLength);
            
            for p = 1:NpLoad
                XCue = zeros(N,learnWLength);
                XOldCue = zeros(N,learnWLength);
                pTemplate = zeros(learnWLength,1);
                patt = patts{p};
                signValCollectCounter = 1;
                x = zeros(N, 1);
                for n = 1:trainWashoutLength
                    u = patt(n);
                    x =  tanh(W * x + Win * u + bias);
                end
                for n = 1: learnWLength
                    u = patt(n + trainWashoutLength);
                    XOldCue(:,n) = x;
                    x = tanh(W * x + Win * u + bias);
                    XCue(:,n) = x;
                    pTemplate(n,1) = u;
                end
                pTemplates(:,p) = pTemplate(end-measureTemplateLength+1:end);
                allTrainArgs(:, (p-1)*learnWLength+1:p*learnWLength) = ...
                    XCue;
                allTrainOldArgs(:, (p-1)*learnWLength+1:p*learnWLength) = ...
                    XOldCue;
                allTrainDTargs(:, (p-1)*learnWLength+1:p*learnWLength) = ...
                    Win * pTemplate';
                allTrainYTargs(1, (p-1)*learnWLength+1:p*learnWLength) = ...
                    pTemplate;
            end
            
            % % learn D
            D = (inv(allTrainOldArgs * allTrainOldArgs' + TychonovAlphaD * I) ...
                * allTrainOldArgs * allTrainDTargs')';
            NRMSED = mean(nrmse(allTrainDTargs, D * allTrainOldArgs));
        end
        
        % % compute mean variance of x
        varx = mean(var(allTrainArgs'));
        
        
        % % learn readouts
        Wout = (inv(allTrainArgs * allTrainArgs' + TychonovAlphaWout * I) ...
            * allTrainArgs * allTrainYTargs')';
        NRMSEWout = nrmse(allTrainYTargs, Wout * allTrainArgs);
        disp(sprintf('NRMSE Wout = %0.2g  D = %0.2g', NRMSEWout, NRMSED));
        
        %%% test retrieval with training patterns (maximally NpTest many)
        
        NpTestEff = min(NpLoad, NpTest);
        yTest_PL = zeros(measureRL, NpTestEff, 2);
        for p = 1:NpTestEff
            fprintf('%g ',p);
            patt = patts{p};
            x = zeros(N,1);
            % initial preliminary C estimation from driven run
            for n = 1:initialWashout
                u = patt(n);
                x =  tanh(W * x + Win * u + bias);
            end
            C = zeros(N,N);
            b = sqrt(varx /  SNRCue);
            for n = 1:cueLength
                u = patt(n + initialWashout) + cueNL * randn;
                x =  tanh(W * x + Win * u + bias ) + b * randn(N,1);
                C = C + CadaptRateCue * ((x - C*x)*x' - aperture^(-2)*C);
            end
            
            
            % measure quality
            xBeforeMeasure = x;
            for n = 1:measureWashout
                x = C * tanh(W *  x + D * x + bias);
            end
            for n = 1:measureRL
                r = tanh(W *  x + D * x + bias);
                x = C * r;
                yTest_PL(n,p,1) = Wout * r;
            end
            x = xBeforeMeasure;
            
            % run with autoadaptation
            
            % state noise scaling factor b
            
            b = sqrt(varx /  SNRTest);
            for n = 1:RL
                x = C * (tanh(W *  x + D * x + bias )+ b*randn(N,1));
                C = C + CadaptRateAfterCue * ((x - C*x)*x' - aperture^(-2)*C);
            end
            
            for n = 1:measureWashout
                x = C * tanh(W *  x + D * x + bias);
            end
            for n = 1:measureRL
                r = tanh(W *  x + D * x + bias);
                x = C * r;
                yTest_PL(n,p,2) = Wout * r;
            end
            
        end
        fprintf('\n ');
        %%
        % optimally align C-reconstructed readouts with drivers
        intRate = 2;
        pAligned_PL = zeros(NpTestEff, measureTemplateLength, 2);
        NRMSEsAlignedOwn = zeros(NpTestEff, 2);
        pTemplatesUsedInTest = pTemplates;
        
        for i = 1:2
            for p = 1:NpTestEff
                thisDriver = pTemplatesUsedInTest(:,p);
                thisOut = yTest_PL(:,p,i);
                thisDriverInt = interp1((1:measureTemplateLength)',thisDriver,...
                    (1:(1/intRate):measureTemplateLength)', 'spline')';
                thisOutInt = interp1((1:measureRL)', thisOut,...
                    (1:(1/intRate):measureRL)', 'spline')';
                
                L = size(thisOutInt,2); M = size(thisDriverInt,2);
                phasematches = zeros(1,L - M);
                for phaseshift = 1:(L - M)
                    phasematches(1,phaseshift) = ...
                        norm(thisDriverInt - ...
                        thisOutInt(1,phaseshift:phaseshift+M-1));
                end
                [maxVal maxInd] = max(-phasematches);
                pAligned_PL(p,:,i) = ...
                    thisOutInt(maxInd:intRate:...
                    (maxInd+intRate*measureTemplateLength-1))';
                
                NRMSEsAlignedOwn(p,i) = ...
                    nrmse(pAligned_PL(p,:,i),thisDriver');
            end
        end
        
        
        disp(sprintf('log10 mean NRMSEs test fromTrain %s',...
            num2str(log10(mean(NRMSEsAlignedOwn)),2)));
        allNRMSEOwns{loadIndex, expi} = NRMSEsAlignedOwn;
        
        %%% test retrieval with other patterns
        
        NpTestEff = NpTest;
        yTest_PL = zeros(measureRL, NpTestEff, 2);
        for p = 1:NpTestEff
            fprintf('%g ',p);
            patt = pattsOther{p};
            x = zeros(N,1);
            % initial preliminary C estimation from driven run
            for n = 1:initialWashout
                u = patt(n);
                x =  tanh(W * x + Win * u + bias);
            end
            C = zeros(N,N);
            b = sqrt(varx /  SNRCue);
            for n = 1:cueLength
                u = patt(n + initialWashout) + cueNL * randn;
                x =  tanh(W * x + Win * u + bias ) + b * randn(N,1);
                C = C + CadaptRateCue * ((x - C*x)*x' - aperture^(-2)*C);
            end
            
            
            % measure quality
            xBeforeMeasure = x;
            for n = 1:measureWashout
                x = C * tanh(W *  x + D * x + bias);
            end
            for n = 1:measureRL
                x = C * tanh(W *  x + D * x + bias);
                yTest_PL(n,p,1) = Wout * x;
            end
            x = xBeforeMeasure;
            
            % run with autoadaptation
            
            % state noise scaling factor b
            
            b = sqrt(varx /  SNRTest);
            for n = 1:RL
                x = C * (tanh(W *  x + D * x + bias )+ b*randn(N,1));
                C = C + CadaptRateAfterCue * ((x - C*x)*x' - aperture^(-2)*C);
            end
            
            for n = 1:measureWashout
                x = C * tanh(W *  x + D * x + bias);
            end
            for n = 1:measureRL
                x = C * tanh(W *  x + D * x + bias);
                yTest_PL(n,p,2) = Wout * x;
            end
            
        end
        fprintf('\n ');
        %%
        % optimally align C-reconstructed readouts with drivers
        intRate = 2;
        pAligned_PL = zeros(NpTestEff, measureTemplateLength, 2);
        NRMSEsAlignedOther = zeros(NpTestEff, 2);
        pTemplatesUsedInTest = zeros(measureTemplateLength, NpTestEff);
        for p = 1:NpTestEff
            patt = pattsOther{p};
            pTemplatesUsedInTest(:,p) = patt(1:measureTemplateLength)';
        end
        
        for i = 1:2
            for p = 1:NpTestEff
                thisDriver = pTemplatesUsedInTest(:,p);
                thisOut = yTest_PL(:,p,i);
                thisDriverInt = interp1((1:measureTemplateLength)',thisDriver,...
                    (1:(1/intRate):measureTemplateLength)', 'spline')';
                thisOutInt = interp1((1:measureRL)', thisOut,...
                    (1:(1/intRate):measureRL)', 'spline')';
                
                L = size(thisOutInt,2); M = size(thisDriverInt,2);
                phasematches = zeros(1,L - M);
                for phaseshift = 1:(L - M)
                    phasematches(1,phaseshift) = ...
                        norm(thisDriverInt - ...
                        thisOutInt(1,phaseshift:phaseshift+M-1));
                end
                [maxVal maxInd] = max(-phasematches);
                pAligned_PL(p,:,i) = ...
                    thisOutInt(maxInd:intRate:...
                    (maxInd+intRate*measureTemplateLength-1))';
                
                NRMSEsAlignedOther(p,i) = ...
                    nrmse(pAligned_PL(p,:,i),thisDriver');
            end
        end
        
        disp(sprintf('log10 mean NRMSEs test fromOther %s',...
            num2str(log10(mean(NRMSEsAlignedOther)),2)));
        allNRMSEOthers{loadIndex, expi} = NRMSEsAlignedOther;
    end
end

%%

% % plotting

% concatenate across experiments and take log10
allNRMSEconcatOwns = cell(1,length(NpLoads));
allNRMSEconcatOthers = cell(1,length(NpLoads));
for expi = 1:expN
    for loadi = 1:length(NpLoads)
        allNRMSEconcatOwns{1,loadi} = ...
            [allNRMSEconcatOwns{1,loadi}; ...
            log10(allNRMSEOwns{loadi, expi})];
        allNRMSEconcatOthers{1,loadi} = ...
            [allNRMSEconcatOthers{1,loadi}; ...
            log10(allNRMSEOthers{loadi, expi})];
    end
end

% compute Nrs of samples
NSampleOwn = zeros(1,length(NpLoads));
NSampleOther = zeros(1,length(NpLoads));
for loadi = 1:length(NpLoads)
    NSampleOwn(loadi) = size(allNRMSEconcatOwns{loadi},1);
    NSampleOther(loadi) = size(allNRMSEconcatOthers{loadi},1);
end

% compute means and variances
meanNRMSECueOwn = zeros(1,length(NpLoads));
meanNRMSECueOther = zeros(1,length(NpLoads));
meanNRMSETestOwn = zeros(1,length(NpLoads));
meanNRMSETestOther = zeros(1,length(NpLoads));
for loadi = 1:length(NpLoads)
    meanNRMSECueOwn(1,loadi) = mean(allNRMSEconcatOwns{loadi}(:,1));
    meanNRMSECueOther(1,loadi) = mean(allNRMSEconcatOthers{loadi}(:,1));
    meanNRMSETestOwn(1,loadi) = mean(allNRMSEconcatOwns{loadi}(:,2));
    meanNRMSETestOther(1,loadi) = mean(allNRMSEconcatOthers{loadi}(:,2));
end
varNRMSECueOwn = zeros(1,length(NpLoads));
varNRMSECueOther = zeros(1,length(NpLoads));
varNRMSETestOwn = zeros(1,length(NpLoads));
varNRMSETestOther = zeros(1,length(NpLoads));
for loadi = 1:length(NpLoads)
    varNRMSECueOwn(1,loadi) = var(allNRMSEconcatOwns{loadi}(:,1));
    varNRMSECueOther(1,loadi) = var(allNRMSEconcatOthers{loadi}(:,1));
    varNRMSETestOwn(1,loadi) = var(allNRMSEconcatOwns{loadi}(:,2));
    varNRMSETestOther(1,loadi) = var(allNRMSEconcatOthers{loadi}(:,2));
end
%%
figure(1); clf;
fs = 20; fstext = 20;
minColWeight = 0.1; maxColWeight = 0.7;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1000 250 600 600]); 
hold on;
errorbar(NpLoads, meanNRMSECueOther, ...
    1.96*sqrt(varNRMSECueOther) ./ sqrt(NSampleOther),...
    'Color', 0.7*[1 1 1], 'LineWidth', 6);
errorbar(NpLoads, meanNRMSETestOther, ...
    1.96*sqrt(varNRMSETestOther) ./ sqrt(NSampleOther), '--',...
    'Color', 0.7*[1 1 1], 'LineWidth', 6);
errorbar(NpLoads, meanNRMSECueOwn, ...
    1.96*sqrt(varNRMSECueOwn) ./ sqrt(NSampleOwn), 'k', 'LineWidth', 2);
errorbar(NpLoads, meanNRMSETestOwn, ...
    1.96*sqrt(varNRMSETestOwn) ./ sqrt(NSampleOwn), ...
    'k--', 'LineWidth', 2);
hold off;
set(gca, 'XLim', [1.5 NpLoads(end)+35], 'XTick', NpLoads, ...
    'XScale', 'log', 'FontSize', fs);
xlabel('Nr of loaded patterns', 'FontSize',fstext);
ylabel('log10 NRMSE', 'FontSize',fstext);
%%
