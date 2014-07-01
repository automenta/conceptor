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
TychonovAlpha = .01; % regularizer for  W training
washoutLength = 500;
learnLength = 1000;
signalPlotLength = 20;

%%% pattern readout learning
TychonovAlphaReadout = 0.01;


%%% C learning and testing
alpha = 1000;
CtestLength = 200;
SplotLength = 50;

%%% morphing
morphRange = [-2 3];
morphTime = 95; morphWashout = 500; preMorphRecordLength = 0;
delayMorphTime = 500; delayPlotPoints = 25;
tN = 8;


%%% Setting patterns


patterns = [53 54 10 36];
%patterns = [54 48  18 60];
%patterns = [23 6];
%patterns = [1 2 21 20 22 8 19 6  16 9 10 11 12];

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



%% linear morphing

if linearMorphing
    
    ms = morphRange(1):(morphRange(2)-morphRange(1))/morphTime:...
        morphRange(2);
    morphPL = zeros(1, morphTime);
    % sinewave morphing
    C1 = Cs{1,3}; C2 = Cs{1,4};
    x = randn(Netsize,1);
    % washing out
    m = ms(1);
    for i = 1:morphWashout
        x = ((1-m)*C1 + m*C2) * tanh(W * x + Wbias);
    end
    % morphing and recording
    
    preMorphPL = zeros(1,preMorphRecordLength);
    m = ms(1);
    for i = 1:preMorphRecordLength
        x = ((1-m)*C1 + m*C2) * tanh(W * x + Wbias);
        preMorphPL(1,i) = Wout * x;
    end
    
    
    for i = 1:morphTime
        m = ms(i);
        x = ((1-m)*C1 + m*C2) * tanh(W * x + Wbias);
        morphPL(1,i) = Wout * x;
    end
    % post morphem
    postMorphRecordLenght = preMorphRecordLength;
    postMorphPL = zeros(1,postMorphRecordLenght);
    m = ms(end);
    for i = 1:postMorphRecordLenght
        x = ((1-m)*C1 + m*C2) * tanh(W * x + Wbias);
        postMorphPL(1,i) = Wout * x;
        
    end
    
    
    % % transform to period length plotlist
    L = preMorphRecordLength+morphTime+postMorphRecordLenght;
    totalMorphPL = [preMorphPL morphPL postMorphPL];
    learnPoint1 = preMorphRecordLength + ...
        morphTime * (- morphRange(1) / ...
        (morphRange(2) - morphRange(1)));
    learnPoint2 = preMorphRecordLength + ...
        morphTime * (- (morphRange(1)-1) / ...
        (morphRange(2) - morphRange(1)));
    
     % delay plot fingerprints computations
    delayplotMs = morphRange(1):...
        (morphRange(2)-morphRange(1))/(tN-1) :morphRange(2);
    delayData = zeros(tN,delayMorphTime);
    x0 = rand(Netsize,1);
    for i = 1:tN
        x = x0;
        Cmix = (1-delayplotMs(i))*C1 + delayplotMs(i)*C2;
         for n = 1:morphWashout
            x = Cmix * tanh(W * x + Wbias);
        end
        % collect x
        for n = 1:delayMorphTime
            x = Cmix * tanh(W * x + Wbias);
            delayData(i,n) = Wout * x;
        end
    end
    
    fingerPrintPoints = preMorphRecordLength + ...
        (0:tN-1)*morphTime/(tN-1);
    fingerPrintPoints(1) = 1;
 %%   
    figure(1); clf;
    fs = 18;
    set(gcf, 'WindowStyle','normal');
    set(gcf,'Position', [700 400 800 266]);
    
    for i = 1:tN
        panelWidth = (1/(tN + 1)) * (1-0.08) ;
        panelHight = 1/(3.2);
        panelx = (1-0.08)*(i-1)*(1/tN) + ...
            (1-0.08)*(i-1)*(1/tN - panelWidth)/tN...
            + 0.04;
        panely = 1/2 + 1.5/10 ;
        
        subplot('Position', [panelx, panely, panelWidth, panelHight]);
        thisdata = delayData(i,1:delayPlotPoints+1);
        plot(delayData(i,1:end-1), delayData(i,2:end),'k.',...
            'MarkerSize',1);
        hold on;
        plot(thisdata(1,1:end-1), thisdata(1,2:end), 'k.',...
         'MarkerSize',20);
     hold off;
        set(gca, 'XTickLabel',[],'YTickLabel',[],...
            'XLim',[-1.4 1.4],'YLim',[-1.4 1.4],'Box','on');
    end
    
    subplot('Position',[0.04 0.15 1-0.08 1/2-0.05]);
    
    plot(totalMorphPL, 'k-', 'LineWidth',2);
    hold on;
    plot([learnPoint1, learnPoint2], [-1, -1], 'k.', 'MarkerSize',35);
    plot(fingerPrintPoints, 1.1*ones(1,tN), 'kv', ...
        'MarkerSize',10, 'MarkerFaceColor','k');
    hold off;
    set(gca,  'YLim',[-1.2 1.2], 'XLim', [1 95],'FontSize',fs);
    
%     figure(2); clf;
%     set(gcf,'DefaultAxesColorOrder',...
%         [0 0.2 0.4 0.6 0.7 0.8 0.9]'*[1 1 1]);
%     set(gcf, 'WindowStyle','normal');
%     set(gcf,'Position', [700 200 200 200]);
%     
%     pattPL = zeros(7,5);
%     for i = 1:7
%         pattPL(i,:) = totalMorphPL(1, ((i-1)*15+1):((i-1)*15+5));
%     end
%     plot(pattPL', 'LineWidth',4); 
%     set(gca,'XTick',[1 2 3 4 5], 'FontSize',fs);
%     
end
% Cend = ((1-m)*C1 + m*C2);
% [Uend Send Vend] = svd(Cend);
% figure(8); clf;
% plot(diag(Send));
% Zend = Uend' * Vend;
% Zend(1:5,1:5)

