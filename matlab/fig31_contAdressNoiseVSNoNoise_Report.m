% creating Figure 31 "A staged re-run of..."
%addpath('./ESN_Toolbox');
set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment control
randstate = 1; newNets = 1; newSystemScalings = 1;
expN = 1; % Nr of experiments over which to average

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
cueNL = 0.0; % set to 0 or to 0.05 for the two panels

CadaptRateCue = 0.02;
CadaptRateAfterCue = 0.01; % C adaptation rate
SNRTests = Inf * [1 1 1]; % state signal-to-noise ratio
SNRCue = Inf;
RLs = 20*[10 90 900 ]; % test runlengthes after cue

measureWashout = 50;
measureTemplateLength = 20;
measureRL = 500; % runlength for measuring output NRMSE.
% Must be at least twice measureTemplateLength
signalPlotLength = 10;
singValPlotLength = 20;

%%% Setting patterns
% NpLoads = [2 3 5 8 12 16 25 50 100 200]; % for patternType = 1
% NpTest = 10;
NpLoads = [2 5 12 25 50 200 ]; % for patternType = 1
NpTest = 5;

%NpLoads = [2 3 5 8 12 16 25 50 100]; % for patternType = 2



%%% Initializations

randn('state', randstate);
rand('twister', randstate);
I = eye(N); 

NpLoad = NpLoads(end);

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

for expi = 1:expN
    disp(sprintf('********** experiment Nr %g ********', expi));
    
    % Set pattern handles
    patts = cell(1,NpLoad);
    for p = 1:NpLoad
        period = 4;
        rp = rand(1,period);
        maxVal = max(rp); minVal = min(rp);
        rp = 1.8 * (rp - minVal) / (maxVal - minVal) - 0.9;
        patts{p} = @(n) (rp(mod(n,period)+1));
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
        yTest_PL = zeros(measureRL, NpTestEff, size(SNRTests,2)+1);
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
             for i = 1:size(RLs,2)
            % state noise scaling factor b
            
            b = sqrt(varx /  SNRTests(i));
            RL = RLs(i);
            for n = 1:RL
                x = C * (tanh(W *  x + D * x + bias )+ b*randn(N,1));
                C = C + CadaptRateAfterCue * ((x - C*x)*x' - aperture^(-2)*C);
            end
            xBeforeMeasure = x;
            for n = 1:measureWashout
                x = C * tanh(W *  x + D * x + bias);
            end
            for n = 1:measureRL
                r = tanh(W *  x + D * x + bias);
                x = C * r;
                yTest_PL(n,p,i+1) = Wout * r;
            end
            x = xBeforeMeasure;
           end 
        end
        fprintf('\n ');
        %%
        % optimally align C-reconstructed readouts with drivers
        intRate = 2;
        pAligned_PL = zeros(NpTestEff, measureTemplateLength, 2);
        NRMSEsAlignedOwn = zeros(NpTestEff, size(SNRTests,2)+1);
        pTemplatesUsedInTest = pTemplates;
        
        for i = 1:size(SNRTests,2)+1
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
        
        
    end
end

%%

% % plotting

% concatenate across experiments and take log10
allNRMSEconcatOwns = cell(1,length(NpLoads));
for expi = 1:expN
    for loadi = 1:length(NpLoads)
        allNRMSEconcatOwns{1,loadi} = ...
            [allNRMSEconcatOwns{1,loadi}; ...
            log10(allNRMSEOwns{loadi, expi})];
       
    end
end

% compute Nrs of samples
NSampleOwn = zeros(1,length(NpLoads));
for loadi = 1:length(NpLoads)
    NSampleOwn(loadi) = size(allNRMSEconcatOwns{loadi},1);
   
end

% compute means and variances
meanNRMSEs = zeros(size(SNRTests,2)+1, length(NpLoads));
for loadi = 1:length(NpLoads)
    meanNRMSEs(:,loadi) = mean(allNRMSEconcatOwns{loadi});
    
end
varNRMSEs = zeros(size(SNRTests,2)+1, length(NpLoads));

for loadi = 1:length(NpLoads)
    varNRMSEs(:,loadi) = var(allNRMSEconcatOwns{loadi});
    
end
%%
figure(2); clf;
fs = 24; fstext = 24; ms = 18;
minColWeight = 0.1; maxColWeight = 0.7;
ColInc = (maxColWeight - minColWeight) / size(RLs,2);
colWeights = minColWeight:ColInc:maxColWeight;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1000 250 600 600]); 
hold on;
errorbar(NpLoads, meanNRMSEs(1,:), ...
    1.96*sqrt(varNRMSEs(1,:)) ./ sqrt(NSampleOwn),...
    's--','LineWidth',3,...
    'Color', colWeights(1)*[1 1 1], 'MarkerSize',ms);
errorbar(NpLoads, meanNRMSEs(2,:), ...
    1.96*sqrt(varNRMSEs(2,:)) ./ sqrt(NSampleOwn),...
    'x--','LineWidth',3,...
    'Color', colWeights(2)*[1 1 1], 'MarkerSize',ms);
errorbar(NpLoads, meanNRMSEs(3,:), ...
    1.96*sqrt(varNRMSEs(3,:)) ./ sqrt(NSampleOwn),...
    'd--','LineWidth',3,...
    'Color', colWeights(3)*[1 1 1], 'MarkerSize',ms);
errorbar(NpLoads, meanNRMSEs(4,:), ...
    1.96*sqrt(varNRMSEs(4,:)) ./ sqrt(NSampleOwn),...
    'p--','LineWidth',3,...
    'Color', colWeights(4)*[1 1 1], 'MarkerSize',ms);
set(gca, 'XLim', [1.5 NpLoads(end)+35], 'XTick', NpLoads, ...
    'XScale', 'log', 'FontSize', fs, 'Box', 'on');
xlabel('Nr of loaded patterns', 'FontSize',fstext);
ylabel('log10 NRMSE', 'FontSize',fstext);
%%
