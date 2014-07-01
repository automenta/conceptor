%%%% plain demo that when a RNN is driven by different signals, the induced
%%%% internal signals will inhabit different subspaces of the signal space.


% set figure window to 1 x 2 panels


set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment control
randstate = 8; newNets = 1; newSystemScalings = 1;
linearMorphing = 1;
unitaryMorphing = 0;

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
alpha = 10;
CtestLength = 200;
SplotLength = 50;

%%% morphing
morphRange = [-2 3];
morphTime = 200; morphWashout = 500; preMorphRecordLength = 50;
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
    C1 = Cs{1,1}; C2 = Cs{1,2};
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
    % interpolate
    interpolInc = 0.1;
    interpolPoints = 1:interpolInc:L;
    interpolL = length(interpolPoints);
    totalMorphPLInt = ...
        interp1((1:L)', totalMorphPL', interpolPoints', 'spline');
    
    downcrossingDistcounts = zeros(1,interpolL);
    oldVal = 1;
    counter = 0;
    for i = 1:interpolL-1
        if totalMorphPLInt(i) < 0 && totalMorphPLInt(i+1) >= 0
            counter = counter + 1;
            downcrossingDistcounts(i) = counter;
            oldVal = counter;
            counter = 0;
        else
            downcrossingDistcounts(i) = oldVal;
            counter = counter + 1;
        end
    end
    %subsample
    downcrossingDistcounts = ...
        downcrossingDistcounts(1,interpolInc^(-1):interpolInc^(-1):interpolL);
    downcrossingDistcounts = downcrossingDistcounts * interpolInc;
    downcrossingDistcounts(1,1:20) = ...
        ones(1,20) * downcrossingDistcounts(20);
    
    % reference period lengthes
    p1 = 8.8342522; p2 = 9.8342522;
    pdiff = p2 - p1;
    pstart = p1 + morphRange(1)*pdiff;
    pend = p1 + morphRange(2)*pdiff;
    refPL = [pstart * ones(1,preMorphRecordLength), ...
        (0:(morphTime-1)) / (morphTime-1) * (pend-pstart)+pstart,...
        pend * ones(1,preMorphRecordLength)];
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
    
    
 %%   
    figure(1); clf;
    fs = 18;
    set(gcf, 'WindowStyle','normal');
    set(gcf,'Position', [700 400 800 400]);
    
    for i = 1:tN
        panelWidth = (1/(tN + 1)) * (1-0.08) ;
        panelHight = 1/(4.5);
        panelx = (1-0.08)*(i-1)*(1/tN) + ...
            (1-0.08)*(i-1)*(1/tN - panelWidth)/tN...
            + 0.04;
        panely = 2/3 + 1/20 ;
        
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
    
    
    subplot('Position',[0.04 1/3+0.05 1-0.08 1/3-0.05]);
    plot(totalMorphPL,'k-', 'LineWidth',2); hold on;
    plot([learnPoint1, learnPoint2], [-1.1 -1.1], 'k.',...
        'MarkerSize',35);
    plot(fingerPrintPoints, 1.1*ones(1,tN), 'kv', ...
        'MarkerSize',10, 'MarkerFaceColor','k');
    hold off;
    set(gca,  'YLim',[-1.3 1.3],'FontSize',fs);
    
    
    subplot('Position',[0.04 0.03 1-0.08 1/3-0.05]);
    hold on;
    plot(downcrossingDistcounts, 'LineWidth',6,'Color',0.75*[1 1 1]);
    plot(refPL,'k');
    plot([learnPoint1, learnPoint2], [7,7], 'k.', 'MarkerSize',35);
    hold off;
    set(gca, 'Box','on', 'XTickLabel',[],'FontSize',fs);
    
end
% Cend = ((1-m)*C1 + m*C2);
% [Uend Send Vend] = svd(Cend);
% figure(8); clf;
% plot(diag(Send));
% Zend = Uend' * Vend;
% Zend(1:5,1:5)

% %%
% % % unitary morphing
% if unitaryMorphing
%     C1 = Cs{1,1}; C2 = Cs{1,2};
%     [U1 S1 V1] = svd(C1); [U2 S2 V2] = svd(C2);
%     Urot = U2 * U1';
%     Sdiag1 = diag(S1); Sdiag2 = diag(S2);
%     eps = 0.0000001;
%     Sdiag1 = max(Sdiag1, eps); Sdiag2 = max(Sdiag2, eps);
%     z1 = - log(1./Sdiag1 - 1); z2 = - log(1./Sdiag2 - 1);
%     
%     ms = morphRange(1):(morphRange(2)-morphRange(1))/morphTime:...
%         morphRange(2);
%     morphPL = zeros(1, morphTime);
%     % sinewave morphing
%     
%     x = randn(Netsize,1);
%     % washing out
%     m = ms(1);
%     thisS = diag(1./ (1 + exp(-((1-m)*z1 + m*z2))));
%     thisU = real(Urot^m) * U1;
%     thisC = thisU * thisS * thisU';
%     for i = 1:morphWashout
%         x = thisC * tanh(W * x + Wbias);
%     end
%     % morphing and recording
%     preMorphRecordLength = 100;
%     preMorphPL = zeros(1,preMorphRecordLength);
%     
%     for i = 1:preMorphRecordLength
%         x = thisC * tanh(W * x + Wbias);
%         preMorphPL(1,i) = Wout * x;
%     end
%     
%     
%     for i = 1:morphTime
%         m = ms(i);
%         thisS = diag(1./ (1 + exp(-((1-m)*z1 + m*z2))));
%         thisU = real(Urot^m) * U1;
%         thisC = thisU * thisS * thisU';
%         x = thisC * tanh(W * x + Wbias);
%         morphPL(1,i) = Wout * x;
%     end
%     % post morphem
%     postMorphRecordLenght = 100;
%     postMorphPL = zeros(1,postMorphRecordLenght);
%     m = ms(end);
%     thisS = diag(1./ (1 + exp(-((1-m)*z1 + m*z2))));
%     thisU = real(Urot^m) * U1;
%     thisC = thisU * thisS * thisU';
%     for i = 1:postMorphRecordLenght
%         x = thisC * tanh(W * x + Wbias);
%         postMorphPL(1,i) = Wout * x;
%         
%     end
%     
%     % % transform to period length plotlist
%     L = preMorphRecordLength+morphTime+postMorphRecordLenght;
%     totalMorphPL = [preMorphPL morphPL postMorphPL];
%     % interpolate
%     interpolInc = 0.1;
%     interpolPoints = 1:interpolInc:L;
%     interpolL = length(interpolPoints);
%     totalMorphPLInt = ...
%         interp1((1:L)', totalMorphPL', interpolPoints', 'spline');
%     
%     downcrossingDistcounts = zeros(1,interpolL);
%     oldVal = 1;
%     counter = 0;
%     for i = 1:interpolL-1
%         if totalMorphPLInt(i) < 0 && totalMorphPLInt(i+1) >= 0
%             counter = counter + 1;
%             downcrossingDistcounts(i) = counter;
%             oldVal = counter;
%             counter = 0;
%         else
%             downcrossingDistcounts(i) = oldVal;
%             counter = counter + 1;
%         end
%     end
%     %subsample
%     downcrossingDistcounts = ...
%         downcrossingDistcounts(1,interpolInc^(-1):interpolInc^(-1):interpolL);
%     downcrossingDistcounts = downcrossingDistcounts * interpolInc;
%     downcrossingDistcounts(1,1:20) = ...
%         ones(1,20) * downcrossingDistcounts(20);
%     
%     figure(2); clf;
%     set(gcf, 'WindowStyle','normal');
%     set(gcf,'Position', [600 100 1000 500]);
%     subplot(2,1,1);
%     plot(totalMorphPL);
%     subplot(2,1,2);
%     plot(downcrossingDistcounts);
% end
% 
% %%
% 
