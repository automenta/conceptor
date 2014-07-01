
%addpath('./ESN_Toolbox');
set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment control
randstate = 1; newNets = 1; newSystemScalings = 1;

%%% Setting system params
N = 200; SR = 1.5;
WinScaling = 1.5; % scaling of pattern feeding weights
biasScaling = .5;

%%% Initial C adaptation in driven runs
trainWashoutLength = 100;
learnWLength = 500;

%%% D learning
incrementalLoad = 0;
TychonovAlphaD = .001;

%%% pattern readout learning
TychonovAlphaWout = 0.01;

%%% C learning and testing
aperture = 100;
initialWashout = 100;
cueLength = 30;
cueNL = 0.0;

CadaptRateCue = 0.01;
CadaptRateAfterCue = 0.01; % C adaptation rate
SNRTests = [1]; % state signal-to-noise ratio
SNRCue = Inf;
RLs = [10000];
testWithOther = 0; %wether we use test cues not from training set

measureWashout = 50;
measureTemplateLength = 20;
measureRL = 500; % runlength for measuring output NRMSE.
% Must be at least twice measureTemplateLength
signalPlotLength = 10;
singValPlotLength = 20;

%%% Setting patterns
patternType = 4; % 1 from intPatternList 
                 % 2 random same period 3 irr. sine sweep
                 % 4 two irrat sines mix
NpLoad = 5;
NpTest = 5;


%%% Initializations

randn('state', randstate);
rand('twister', randstate);
I = eye(N);

if testWithOther
    NpTestEff = NpTest;
else
NpTestEff = min(NpLoad, NpTest);
end

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

% reset randstate to find nice-looking patterns
randstate = 2;
randn('state', randstate);
rand('twister', randstate);

% Set pattern handles
patts = cell(1,NpLoad); pattsOther = cell(1,NpTest);
if patternType == 1
    intPatterns = [1 2 9 11 12  44 39 40 13  34 16 60 ];

% 1 = sine10  2 = sine15  3 = sine20  4 = spike20
% 5 = spike10 6 = spike7  7 = 0   8 = 1
% 9 = rand4; 10 = rand5  11 = rand6 12 = rand7
% 13 = rand8 14 = sine10range01 15 = sine10rangept5pt9
% 16 = rand3 17 = rand9 18 = rand10 19 = 0.8 20 = sineroot27
% 21 = sineroot19 22 = sineroot50 23 = sineroot75
% 24 = sineroot10 25 = sineroot110 26 = sineroot75tenth
% 27 = sineroots20plus40  28 = sineroot75third
% 29 = sineroot243  30 = sineroot150  31 = sineroot200
% 32 = sine10.587352723 33 = sine10.387352723
% 34 = rand7  35 = sine12  36 = 10+perturb  37 = sine11
% 38 = sine10.17352723  39 = sine5 40 = sine6
% 41 = sine7 42 = sine8  43 = sine9 44 = sine12
% 45 = sine13  46 = sine14  47 = sine10.8342522
% 48 = sine11.8342522  49 = sine12.8342522  50 = sine13.1900453
% 51 = sine7.1900453  52 = sine7.8342522  53 = sine8.8342522
% 54 = sine9.8342522 55 = sine5.19004  56 = sine5.8045
% 57 = sine6.49004 58 = sine6.9004 59 = sine13.9004
% 60 = 18+perturb
    pattHandles;
    patternsAll = patts;
    for p = 1:NpLoad
        patts{p} = patternsAll{intPatterns(p)};
    end
elseif patternType == 2
    pLength = 4;
    for p = 1:NpLoad
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
elseif patternType == 3
    maxPeriod = 12; minPeriod = 4;
    pRange = maxPeriod - minPeriod;
    periods = (0:1/(Np-1):1) * pRange + minPeriod + 0.1 * sqrt(2);
    periodsOther = (0:1/(Np-1):1) * pRange + ...
        0.5 / (Np-1) + minPeriod + 0.1 * sqrt(2);
    for p = 1:NpLoad
        period = periods(p);
        patts{p} = @(n) sin(2 * pi * n / period);
    end
    for p = 1:NpTest
        periodOther = periodsOther(p);
        pattsOther{p} = @(n) sin(2 * pi * n / periodOther);
    end
elseif patternType == 4
    stretch = sqrt(30);
    for p = 1:NpLoad
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

%%% test retrieval

yTest_PL = zeros(measureRL, NpTest, size(SNRTests,2)+1);
SV_PL = zeros(N,NpTest, size(SNRTests,2)+1);
for p = 1:NpTestEff
    fprintf('%g ',p);
    if testWithOther
        patt = pattsOther{p};
    else
        patt = patts{p};
    end
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
    [U S V] = svd(C);
    SV_PL(:,p,1) = diag(S);
    
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
        
        % state and noise scaling factors a and b
        
        b = sqrt(varx /  SNRTests(i));
        RL = RLs(i);
        for n = 1:RL
            x = C * (tanh(W *  x + D * x + bias )+ b*randn(N,1));
            C = C + CadaptRateAfterCue * ((x - C*x)*x' - aperture^(-2)*C);
            %x = a * x + b * randn(N,1);
        end
        [U S V] = svd(C);
        SV_PL(:,p,i+1) = diag(S);
        
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
pAligned_PL = zeros(NpTest, measureTemplateLength, size(SNRTests,2)+1);
NRMSEsAligned = zeros(NpTest, size(SNRTests,2)+1);
if testWithOther
   pTemplatesUsedInTest = zeros(measureTemplateLength, NpTest);
   for p = 1:NpTest
       patt = pattsOther{p};
       pTemplatesUsedInTest(:,p) = patt(1:measureTemplateLength)';
   end
else
   pTemplatesUsedInTest = pTemplates;
end
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
        
        NRMSEsAligned(p,i) = ...
            nrmse(pAligned_PL(p,:,i),thisDriver');
    end
end

disp(sprintf('log10 mean NRMSEs test %s',...
    num2str(log10(mean(NRMSEsAligned)),2)));
%%

figure(1); clf;
fs = 24; fstext = 24;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [800 350  400 400]);

for p = 1:min(3,NpTestEff)
    
    subplot(3,2,2*(p-1)+1);
    hold on;
    plot(ones(1,singValPlotLength),'k--');
    plot(squeeze(SV_PL(1:singValPlotLength,p,:)), 'LineWidth',3);
    hold off;
    
    if p == 1
        title(sprintf...
            ('Singular Values'),...
            'FontSize',fstext);
    end
    if p < 3
        set(gca, 'XTickLabel',[]);
    end
    set(gca, 'YLim',[0,1.2],'YTick',[0 1], 'FontSize',fs, 'Box', 'on');
    
    
    
    
    subplot(3,2,2*(p-1)+2);
    hold on;
    plot(squeeze(pAligned_PL(p,1:signalPlotLength,end)),...
        'Color', 0.8 * [1  1 1], 'LineWidth',8);
    plot(pTemplatesUsedInTest(1:signalPlotLength,p), 'k', ...
        'LineWidth',2);
    hold off;
    if p < 3
        set(gca, 'XTickLabel',[]);
    end
    if p == 1 
        title(sprintf('y and p'),...
            'FontSize',fstext);
    end
    set(gca, 'YLim',[-1 1],'YTick',[-1 0 1], 'FontSize',fs, 'Box', 'on');
end 
%%
figure(2); clf;
  fs = 24; fstext = 24;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [800 250  400 400]);
    colWeights = [0 0.6];
    hold on;
plot(log10(NRMSEsAligned(:,1)),'s--','LineWidth',2,...
    'Color', colWeights(1)*[1 1 1], 'MarkerSize',12);
plot(log10(NRMSEsAligned(:,2)),'x--','LineWidth',2,...
    'Color', colWeights(2)*[1 1 1], 'MarkerSize',12);
% plot(log10(NRMSEsAligned(:,3)),'d--','LineWidth',3, ...
%     'Color', colWeights(3)*[1 1 1], 'MarkerSize',12);
% plot(log10(NRMSEsAligned(:,4)),'p--','LineWidth',3, ...
%     'Color', colWeights(4)*[1 1 1], 'MarkerSize',12);

hold off;
set(gca,  'FontSize',fs, 'Box', 'on',...
    'XLim',[0 NpTest+1], 'XTick', 1:NpTest);
xlabel('Pattern index', 'FontSize', fstext);
ylabel('log10 NRMSE', 'FontSize', fstext);
title('Reconstruction Error','FontSize',fstext);

