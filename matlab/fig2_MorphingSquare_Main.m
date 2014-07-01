%%%% plain demo that when a RNN is driven by different signals, the induced
%%%% internal signals will inhabit different subspaces of the signal space.


% set figure window to 1 x 2 panels


set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment control
randstate = 3; newNets = 1; newSystemScalings = 1;
linearMorphing = 1;
unitaryMorphing = 0;

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
TychonovAlphaReadout = 0.01;


%%% C learning and testing
alpha = 4;
CtestLength = 200;
SplotLength = 50;

%%% morphing
morphRange = [0 1];
morphTime = 30; morphWashout = 190; 
morphPlotLength = 15;
morphPlotPoints = 15;
tN = 9; % plot triangle mesh-N, set to integer of form 5 + 4n
minMu = -0.5; maxMu = 1.5; 


%%% Setting patterns


patterns = [53 54 10 36];



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
pattHandles; % if same patterns as in article are wanted, run the
% hierarchical architecture first to fix those patterns, and 
% comment out the pattHandles command here

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

%%
% % compute morph mixtures and plotting positions in square. We enumerate
%   everything bottom-up starting from left bottom tile.

% normalized plot positions
plotPositions = cell(1, tN);
delta = 1/(tN-1);
for i = 1:tN % filling rows bottom-up
    y = (i-1) * delta;
    xRow = 0:delta:(delta * (tN - 1)) ;
    plotPositions{1,i} = [xRow; y * ones(1,tN)];
end

% corresponding mixture vectors
mixVecs = cell(tN, tN);
%minMu = -1/2; maxMu = 1+1/2; % for tN = 5

rowMus = (maxMu - minMu)*(tN-1:-1:0)/(tN-1) + minMu;
colMus = (maxMu - minMu)*(tN-1:-1:0)/(tN-1) + minMu;
for i = 1:tN
    for j = 1:tN
    % the first two entries in a mixVec relate to the first two patterns,
    % the second two entries to the last two patterns
        mixVecs{i,j} = [rowMus(i) * [colMus(j), 1-colMus(j)],...
            (1 - rowMus(i))*[colMus(j), 1-colMus(j)]];
            end
end

%% linear morphing
C1 = Cs{1,1}; C2 = Cs{1,2}; C3 = Cs{1,3}; C4 = Cs{1,4};
x0 = rand(Netsize,1);

plots = cell(tN, tN);
for i = 1:tN
    for j = 1:tN
        mixVec = mixVecs{i,j};
        Cmix = mixVec(1)*C1 + mixVec(2)*C2 + mixVec(3)*C3 + mixVec(4)*C4;
        yColl = zeros(1,morphTime);
        x = x0;
        % washout
        for n = 1:morphWashout
            x = Cmix * tanh(W * x + Wbias);
        end
        % collect x
        for n = 1:morphTime
            x = Cmix * tanh(W * x + Wbias);
            yColl(1,n) = Wout * x;
        end
        plots{i,j} = yColl;
    end
    
end
%%

figure(1); clf;
fs = 18;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [850 100 800 800]);
panelWidth = 1/(tN + 1);
panelHight = 1/(tN + 1);
for i = 1:tN
    for j = 1:tN 
        thisPositionNormalized = plotPositions{1,i}(:,j);
        panelx = (j-1)*(1/tN);
        panely = (i-1)*(1/tN);
        subplot('Position', [panelx, panely, panelWidth, panelHight]);
        thisdata = plots{i,j}(1,1:morphPlotPoints);
        delayPL = [thisdata(1,1:end-1); thisdata(1,2:end)];
%         plot(thisdata(1,1:end-1), thisdata(1,2:end), '.',...
%          'MarkerSize',20);
     plot(thisdata, 'k', 'LineWidth',3);
%         set(gca, 'XTickLabel',[],'YTickLabel',[],...
%             'XLim', [-1,1], 'YLim', [-1,1],...
%             'Color', 1 - 1*(mixVecs{1,i}(:,j)').^2);
        set(gca, 'XTick',[],'YTick',[],...
             'YLim', [-1,1], 'XLim',[1 morphPlotPoints]);
         n = (tN - 5) / 4;
             if (i == n+2 && j == n+2) || (i == n+2 && j == 3*n+4) || ...
                     (i == 3*n+4 && j == n+2) || (i == 3*n+4 && j == 3*n+4)
                 %set(gca, 'Color',0.9 * [1 1 1]);
                 set(gca, 'LineWidth', 5);
             end
         
    end
end


% %%
% figure(2); clf;
% fs = 18;
% set(gcf, 'WindowStyle','normal');
% set(gcf,'Position', [900 100 800 800], 'Color','k');
% panelWidth = 1/(tN + 1);
% panelHight = 1/(tN + 1);
% for i = 1:tN
%     for j = 1:tN - i + 1
%         thisPositionNormalized = plotPositions{1,i}(:,j);
%         panelx = (i-1)*(1/tN)/2 + (j-1)*(1/tN);
%         panely = (i-1)*(1/tN);
%         subplot('Position', [panelx, panely, panelWidth, panelHight]);
%         thisdata = plots{i,j};
%         plot(thisdata(1,1:morphPlotLength), ...
%             'LineWidth',3, 'Color',mixVecs{1,i}(:,j)');
%         set(gca, 'XTickLabel',[],'YTickLabel',[],...
%              'YLim', [-1,1], 'XLim',[1 morphPlotLength],...
%              'Color', 1 - (mixVecs{1,i}(:,j)').^2);
%     end
% end




