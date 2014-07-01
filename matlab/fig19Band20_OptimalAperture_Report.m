%%%% plain demo that when a RNN is driven by different signals, the induced
%%%% internal signals will inhabit different subspaces of the signal space.


% set figure window to 1 x 2 panels


set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment control
randstate = 8; newNets = 1; newSystemScalings = 1;
linearMorphing = 1;

%%% Setting system params
Netsize = 100; % network size
NetSR = 1.5; % spectral radius
NetinpScaling = 1.5; % scaling of pattern feeding weights
BiasScaling = 0.2; % size of bias


%%% loading learning
TychonovAlpha = .0001; % regularizer for  W training
washoutLength = 500;
learnLength = 1000;
signalPlotLength = 20;

%%% pattern readout learning
TychonovAlphaReadout = 0.00001;


%%% C learning and testing
alpha = 1;
CtestLength = 200;
CtestWashout = 100;
SplotLength = 50;

% %%% Autoadapt testing
% cueLength = 50; postCueLength = 300;
% deviationPlotInterval = 100;
% TalphaAuto = 0.02;
% startAlpha = .02; % starting value for cueing phase
% TautoLR = 0.02;
% TcueLR = 0.02;
% SNR_cue = Inf; SNR_freeRun = Inf; % can be Inf for zero noise

%%% Setting patterns

patterns = [53 54 10 36];
%patterns = [23 6];
%patterns = [1 2 21 20 22 8 19 6  16 9 10 11 12];

% 1 = sine10  2 = sine15  3 = sine20  4 = spike20
% 5 = spike10 6 = spike7  7 = 0   8 = 1
% 9 = rand5; 10 = rand5  11 = rand6 12 = rand7
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

%%% Initializations

randn('state', randstate);
rand('twister', randstate);

% Create raw weights
if newNets
    if Netsize <= 20
        Netconnectivity = 1;
    else
        Netconnectivity = 10/Netsize;
    end
    WstarRaw = generate_internal_weights(Netsize, Netconnectivity);
    WinRaw = randn(Netsize, 1);
    WbiasRaw = randn(Netsize, 1);
end

% Scale raw weights and initialize weights
if newSystemScalings
    Wstar = NetSR * WstarRaw;
    Win = NetinpScaling * WinRaw;
    Wbias = BiasScaling * WbiasRaw;
end

% Set pattern handles
pattHandles;

I = eye(Netsize);

% % learn equi weights

% harvest data from network externally driven by patterns
Np = length(patterns);
allTrainArgs = zeros(Netsize, Np * learnLength);
allTrainOldArgs = zeros(Netsize, Np * learnLength);
allTrainTargs = zeros(Netsize, Np * learnLength);
allTrainOuts = zeros(1, Np * learnLength);
readoutWeights = cell(1,Np);
patternCollectors = cell(1,Np);
xCollectorsCentered = cell(1,Np);
xCollectors = cell(1,Np);
SRCollectors = cell(1,Np);
URCollectors = cell(1,Np);
patternRs = cell(1,Np);
train_xPL = cell(1,Np);
train_pPL = cell(1,Np);
startXs = zeros(Netsize, Np);
% collect data from driving native reservoir with different drivers
for p = 1:Np
    patt = patts{patterns(p)}; % current pattern generator
    xCollector = zeros(Netsize, learnLength );
    xOldCollector = zeros(Netsize, learnLength );
    pCollector = zeros(1, learnLength );
    x = zeros(Netsize, 1);
    for n = 1:(washoutLength + learnLength)
        u = patt(n); % pattern input
        xOld = x;
        x = tanh(Wstar * x + Win * u + Wbias);
        if n > washoutLength
            xCollector(:, n - washoutLength ) = x;
            xOldCollector(:, n - washoutLength ) = xOld;
            pCollector(1, n - washoutLength) = u;
        end
    end
    
    xCollectorCentered = xCollector - ...
        repmat( mean(xCollector,2),1,learnLength);
    xCollectorsCentered{1,p} = xCollectorCentered;
    xCollectors{1,p} = xCollector;
    R = xCollector * xCollector' / learnLength;
    [Ux Sx Vx] = svd(R);
    SRCollectors{1,p} = Sx;
    URCollectors{1,p} = Ux;
    patternRs{p} = R;
    
    
    startXs(:,p) = x;
    train_xPL{1,p} = xCollector(1:5,1:signalPlotLength);
    train_pPL{1,p} = pCollector(1,1:signalPlotLength);
    
    patternCollectors{1,p} = pCollector;
    allTrainArgs(:, (p-1)*learnLength+1:p*learnLength) = ...
        xCollector;
    allTrainOldArgs(:, (p-1)*learnLength+1:p*learnLength) = ...
        xOldCollector;
    allTrainOuts(1, (p-1)*learnLength+1:p*learnLength) = ...
        pCollector;
    allTrainTargs(:, (p-1)*learnLength+1:p*learnLength) = ...
        Win * pCollector;
end

%%% compute readout

Wout = (inv(allTrainArgs * allTrainArgs' + ...
    TychonovAlphaReadout * eye(Netsize)) ...
    * allTrainArgs * allTrainOuts')';
% training error
NRMSE_readout = nrmse(Wout*allTrainArgs, allTrainOuts);
disp(sprintf('NRMSE readout: %g', NRMSE_readout));

%%% compute W
Wtargets = (atanh(allTrainArgs) - repmat(Wbias,1,Np*learnLength));
W = (inv(allTrainOldArgs * allTrainOldArgs' + ...
    TychonovAlpha * eye(Netsize)) * allTrainOldArgs * Wtargets')';
% training errors per neuron
NRMSE_W = nrmse(W*allTrainOldArgs, Wtargets);
disp(sprintf('mean NRMSE W: %g', mean(NRMSE_W)));

%%% run loaded reservoir to observe a messy output. Do this with starting
%%% from four states originally obtained in the four driving conditions
%%
% figure(10); clf;
% % initialize network state
% for p = 1:4
%     x = startXs(:,p);
%     messyOutPL = zeros(1,CtestLength);
%     % run
%     for n = 1:CtestLength
%         x = tanh(W*x + Wbias);
%         y = Wout * x;
%         messyOutPL(1,n) = y;
%     end
%     subplot(2,2,p);
%     plot(messyOutPL(1,end-19:end));
% end




% % compute projectors
Cs = cell(4, Np);
for p = 1:Np
    R = patternRs{p};
    [U S V] = svd(R);
    Snew = (S * inv(S + alpha^(-2) * eye(Netsize)));
    
    C = U * Snew * U';
    Cs{1, p} = C;
    Cs{2, p} = U;
    Cs{3, p} = diag(Snew);
    Cs{4, p} = diag(S);
end




bestPhis = 100 * [1 1 1 1];
factors = 10^(1/4) *  [1 1 1 1];

halfPlotNumber = 17;
exponents = -halfPlotNumber:halfPlotNumber;
Nphis = 2*halfPlotNumber+1;
allPhis = zeros(4,Nphis);
attenuationPL = zeros(4,Nphis);
for i=1:Nphis
    allPhis(:,i) = (bestPhis').*(factors'.^exponents(i));
end

allNRMSEs = zeros(4,Nphis);
allAttenuations = zeros(4,Nphis);
allDiffs  = zeros(4,Nphis);
allQuotas = zeros(4,Nphis);
allZengys = zeros(4,Nphis);
CnormPL = zeros(4,Nphis);

for k = 1:Nphis
    
    
    %%% run all patterns with conceptor aperture-adapted by phi, compute
    %%% attenuation
    
    p_CTestPL = zeros(Np, CtestLength);
    attenuations = zeros(1, Np);
    diffs = zeros(1, Np);
    quotas = zeros(1, Np);
    zengys = zeros(1, Np);
    for p = 1:Np
        C = PHI(Cs{1, p}, allPhis(p,k));
        CnormPL(p,k) = norm(C,'fro')^2;
        quotas(1,p) = trace(C) / Netsize;
        x = startXs(:,p);
        for n = 1:CtestWashout
            x = C * tanh(W *  x + Wbias);
        end
        att = 0; diff = 0; zengy = 0;
        for n = 1:CtestLength
            z = tanh(W *  x + Wbias);
            x = C * z;
            att = att + norm(x-z)^2 / norm(z)^2;
            diff = diff + norm(x-z)^2;
            zengy = zengy + norm(z)^2;
            p_CTestPL(p,n) = Wout * x;
        end
        attMean = att / CtestLength;
        attenuations(1,p) = - log(1-attMean);
        diffMean = diff / CtestLength;
        diffs(1,p) = diffMean;
        zengys(1,p) = zengy / CtestLength;
    end
    allAttenuations(:,k) = attenuations';
    allDiffs(:,k) = diffs';
    allQuotas(:,k) = quotas';
    allZengys(:,k) = zengys';
    
    %%% align optimally and compute NRMSEs
    
    
    test_pAligned_PL = cell(1,Np);
    NRMSEsAligned = zeros(1,Np);
    
    for p = 1:Np
        intRate = 20;
        thisDriver = train_pPL{1,p};
        thisOut = p_CTestPL(p,:);
        thisDriverInt = interp1((1:signalPlotLength)',thisDriver',...
            (1:(1/intRate):signalPlotLength)', 'spline')';
        thisOutInt = interp1((1:CtestLength)', thisOut',...
            (1:(1/intRate):CtestLength)', 'spline')';
        
        L = size(thisOutInt,2); M = size(thisDriverInt,2);
        phasematches = zeros(1,L - M);
        for phaseshift = 1:(L - M)
            phasematches(1,phaseshift) = ...
                norm(thisDriverInt - ...
                thisOutInt(1,phaseshift:phaseshift+M-1));
        end
        [maxVal maxInd] = max(-phasematches);
        test_pAligned_PL{1,p} = ...
            thisOutInt(1,maxInd:intRate:...
            (maxInd+intRate*signalPlotLength-1));
        coarseMaxInd = ceil(maxInd / intRate);
        NRMSEsAligned(1,p) = ...
            nrmse(test_pAligned_PL{1,p},train_pPL{1,p});
    end
    allNRMSEs(:,k) = NRMSEsAligned';
    
end

%% best apertures based on norm gradient
normGrads = CnormPL(:,2:end) - CnormPL(:,1:end-1);
normGrads = [normGrads(:,1) normGrads];
figure(2); clf;
set(gcf, 'WindowStyle','normal');
set(gcf, 'Position',[900 200 600 400]);
fs = 18;
for p = 1:Np
    subplot(2,2,p);
    line(log10(allPhis(p,:)), ...
        (normGrads(p,:)),...
        'Color','k', 'LineWidth',2);
    
    set(gca, 'FontSize',fs, 'XTickLabel',[], 'XLim',[-2.5 6.5]);
    if p == 3
        xlabel('log10 aperture', 'FontSize',fs);
        ylabel('norm^2 gradient', 'FontSize',fs);
    end
    if p > 2
        set(gca, 'YLim', [-.5 1.5]);
    end
    
    ax1 = gca;
    ax2 = axes('Position',get(ax1,'Position'),...
        'XAxisLocation','top',...
        'YAxisLocation','right',...
        'Color','none',...
        'YLim',[floor(min(log10(allNRMSEs(p,:)))) 1],...
        'XLim',[-2.5 6.5],...
        'XColor','k','YColor',0.5*[1 1 1],...
        'FontSize',fs, 'Box', 'on');
    
    line(log10(allPhis(p,:)), log10(allNRMSEs(p,:)), ...
        'Color',0.6*[1 1 1],'LineWidth',6,'Parent', ax2);
    if p == 4
    ylabel('log10 NRMSE');
    end
    
    
end
%%

allDZengys = [allZengys(:,2) - allZengys(:,1), ...
    0.5*(allZengys(:,3:end) - allZengys(:,1:end-2)),...
    allZengys(:,end) - allZengys(:,end-1)];
%%
figure(1); clf;
set(gcf, 'WindowStyle','normal');
set(gcf, 'Position',[900 200 600 400]);
fs = 18;
for p = 1:Np
    subplot(2,2,p);
    line(log10(allPhis(p,:)), ...
        log10(allDiffs(p,:) ./ allZengys(p,:)),...
        'Color','k', 'LineWidth',2);
    
    set(gca, 'FontSize',fs, 'XTickLabel',[], 'XLim',[-2.5 6.5]);
    if p == 3
        xlabel('log10 aperture', 'FontSize',fs);
        ylabel('log10 attenuation', 'FontSize',fs);
    end
    
    ax1 = gca;
    ax2 = axes('Position',get(ax1,'Position'),...
        'XAxisLocation','top',...
        'YAxisLocation','right',...
        'Color','none',...
        'YLim',[floor(min(log10(allNRMSEs(p,:)))) 1],...
        'XLim',[-2.5 6.5],...
        'XColor','k','YColor',0.5*[1 1 1],...
        'FontSize',fs, 'Box', 'on');
    
    line(log10(allPhis(p,:)), log10(allNRMSEs(p,:)), ...
        'Color',0.6*[1 1 1],'LineWidth',6,'Parent', ax2);
    if p == 4
    ylabel('log10 NRMSE');
    end
    
    
end
%%
%
% figure(5); clf;
% plot(log10(allPhis'));


% figure(1); clf;
% fs = 18;
% set(gcf,'DefaultAxesColorOrder',[0  0.4 0.65 0.8]'*[1 1 1]);
% set(gcf, 'WindowStyle','normal');
%
% set(gcf,'Position', [600 400 1000 500]);
% for p = 1:Np
%     subplot(Np,4,(p-1)*4+1);
%     plot(test_pAligned_PL{1,p}, 'LineWidth',6,'Color',0.85*[1 1 1]); hold on;
%     plot(train_pPL{1,p},'LineWidth',1); hold off;
%     if p == 1
%         title('driver and y','FontSize',fs);
%     end
%     if p ~= Np
%         set(gca, 'XTickLabel',[]);
%     end
%     set(gca, 'YLim',[-1,1], 'FontSize',fs);
%     rectangle('Position', [0.5,-0.95,8,0.5],'FaceColor','w',...
%         'LineWidth',1);
%     text(1,-0.7,num2str(NRMSEsAligned(1,p),2),...
%         'Color','k','FontSize',fs, 'FontWeight', 'bold');
%
%     subplot(Np,4,(p-1)*4+2);
%     plot(train_xPL{1,p}(1:3,:)','LineWidth',2);
%
%     if p == 1
%         title('reservoir states','FontSize',fs);
%     end
%     if p ~= Np
%         set(gca, 'XTickLabel',[]);
%     end
%     set(gca,'YLim',[-1,1],'YTickLabel',[], 'FontSize',fs);
%
%     subplot(Np,4,(p-1)*4+3);
%     %diagNormalized = sDiagCollectors{1,p} / sum(sDiagCollectors{1,p});
%     plot(log10(diag(SRCollectors{1,p})),'LineWidth',2);
%
%     set(gca,'YLim',[-20,10], 'FontSize',fs);
%     if p == 1
%         title('log10 PC energy','FontSize',fs);
%     end
%     subplot(Np,4,(p-1)*4+4);
%     plot(diag(SRCollectors{1,p}(1:10,1:10)),'LineWidth',2);
%     if p == 1
%         title('leading PC energy','FontSize',fs);
%     end
%     set(gca,'YLim',[0,40], 'FontSize',fs);
% end
%
