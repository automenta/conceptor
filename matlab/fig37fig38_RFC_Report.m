%%%% demo of Kohonen-replacement c for C.
%%%% c is adapted inline during training of W

%G F scaling ueber SR, nicht fronorm

set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment control
randstate = 1; newNets = 1; newSystemScalings = 1;


%%% Setting system params
N = 100; % network size for W
K = 500; % dim for F
SR = 1.4;
WinScaling = 1.2; % scaling of pattern feeding weights
BiasScaling = 0.2; % size of bias

%%% loading
LRc = .5;
cAdaptLength = 2000;
washoutLength = 200;
learnLength = 400;
signalPlotLength = 20;

testType = 1; % 1: use precomputed c    2: adaptive cueing

if testType == 1
    aperture = 8;
    %%% loading    
    TychonovAlphaReadout = 1;
    TychonovAlphaG = 0.01;
    CtestWashout = 200;
    CtestLength = 500;
elseif testType == 2
    aperture = 8;
    %%% loading
    TychonovAlphaReadout = 1;
    TychonovAlphaG = 0.01;
    CtestWashout = 200;
    CtestCuelength = 800; % for CtestType = 2
    CtestAdaptlength = 10000; % for CtestType = 2
    measureRL = 500; % for CtestType = 2
    CtestCueLr = 0.5; % for CtestType = 2
    CtestAdaptLr = 0.3; % for CtestType = 2
end

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


I = eye(N);

% Set pattern handles
pattHandles;

% Create raw weights
if newNets
    WinRaw = randn(N, 1);
    WbiasRaw = randn(N, 1);
    FRaw = randn(K,N);
    GstarRaw = randn(N,K);
    GF = GstarRaw * FRaw;
    sr = max(abs(eig(GF)));
    FRaw = FRaw / sqrt(sr);
    GstarRaw = GstarRaw / sqrt(sr);
end

% Scale raw weights and initialize weights
if newSystemScalings
    Win = WinScaling * WinRaw;
    F = FRaw * sqrt(SR);
    Gstar = GstarRaw * sqrt(SR);
    b = BiasScaling * WbiasRaw;
end



% % learn equi weights

% harvest data from network externally driven by patterns
Np = length(patterns);
allTrainr = zeros(N, Np * learnLength);
allTrainczOld = zeros(K, Np * learnLength);
allTrainp = zeros(1, Np * learnLength);
allTraint = zeros(N, Np * learnLength);

% pCollectors = cell(1,Np);
% czCollectors = cell(1,Np);
cCollectors = cell(1,Np);
Cs = cell(1, Np);
train_rPL = cell(1,Np);
train_pPL = cell(1,Np);
% collect data from driving native reservoir with different drivers
for p = 1:Np
    patt = patts{patterns(p)}; % current pattern generator
    rCollector = zeros(N, learnLength );
    czOldCollector = zeros(K, learnLength );
    pCollector = zeros(1, learnLength );
    cCollector = zeros(K, cAdaptLength );
    tCollector = zeros(N, learnLength );
    z = zeros(K, 1);
    c = ones(K,1);
    cz = zeros(K, 1);
    for n = 1:(washoutLength + cAdaptLength + learnLength)
        u = patt(n); % pattern input
        czOld = cz;
        t = Gstar * cz + Win * u;
        r = tanh(t + b);
        z = F * r;
        cz = c .* z;
        if n <= cAdaptLength + washoutLength && ...
                n > washoutLength
            c = c + LRc * ((cz - c.*cz).*cz - aperture^(-2) * c);
            cCollector(:,n - washoutLength) = c;
        end
        if n == cAdaptLength + washoutLength
            Cs{1,p} = c;
        end
        if n > washoutLength + cAdaptLength
            rCollector(:, n - washoutLength - cAdaptLength) = r;
            czOldCollector(:, n - washoutLength - cAdaptLength ) = czOld;
            pCollector(1, n - washoutLength - cAdaptLength) = u;
            tCollector(:, n - washoutLength - cAdaptLength ) = t;
        end
    end
    %     xCollectors{1,p} = xCollector;
    %     zCollectors{1,p} = zCollector;
    %     pCollectors{1,p} = pCollector;
    cCollectors{1,p} = cCollector;
    
    train_rPL{1,p} = rCollector(1:5,1:signalPlotLength);
    %train_zPL{1,p} = zCollector(1:5,1:signalPlotLength);
    train_pPL{1,p} = pCollector(1,1:signalPlotLength);
    
    allTrainr(:, (p-1)*learnLength+1:p*learnLength) = ...
        rCollector;
    allTrainczOld(:, (p-1)*learnLength+1:p*learnLength) = ...
        czOldCollector;
    allTrainp(1, (p-1)*learnLength+1:p*learnLength) = ...
        pCollector;
    allTraint(:, (p-1)*learnLength+1:p*learnLength) = ...
        tCollector;
    
end

%%% compute readout

Wout = (inv(allTrainr * allTrainr' + ...
    TychonovAlphaReadout * eye(N)) ...
    * allTrainr * allTrainp')';
% training error
NRMSE_readout = nrmse(Wout*allTrainr, allTrainp);
disp(sprintf('NRMSE readout: %g', NRMSE_readout));

%%% compute G
Gtargs = allTraint;
Gargs = allTrainczOld;
G = (inv(Gargs * Gargs' + ...
    TychonovAlphaG * eye(K)) ...
    * Gargs * Gtargs')';
NRMSE_G = nrmse(G*Gargs, Gtargs);
disp(sprintf('NRMSE G: %g    size: %g',...
    mean(NRMSE_G), mean(mean(abs(G)))));




% test
if testType == 1
    r_CTestPL = zeros(N, CtestLength, Np);
    cz_CTestPL = zeros(5, CtestLength, Np);
    p_CTestPL = zeros(1, CtestLength, Np);
    SVr_CTestPL = zeros(N, Np);
    for p = 1:Np
        c = Cs{1, p};
        cz = .5 * randn(K,1);
        for n = 1:CtestWashout
            r = tanh(G * cz  + b);
            cz = c .* (F * r);
        end
        for n = 1:CtestLength
            r = tanh(G * cz  + b);
            cz = c .* (F * r);
            r_CTestPL(:,n,p) = r;
            cz_CTestPL(:,n,p) = cz(1:5,1);
            p_CTestPL(:,n,p) = Wout * r;
        end
        % conceptor singval spectrum of x
        Rr = r_CTestPL(:,:,p) * r_CTestPL(:,:,p)' / CtestLength;
        Cr = Rr * inv(Rr + aperture^(-2) * I);
        [U S V] = svd(Cr);
        SVr_CTestPL(:, p) = diag(S);
    end
elseif testType == 2
    p_CTest_postCuePL = zeros(1, measureRL, Np);
    p_CTest_postAdaptPL = zeros(1, measureRL, Np);
    r_CTest_postCuePL = zeros(N, measureRL, Np);
    r_CTest_postAdaptPL = zeros(N, measureRL, Np);
    SVr_postCue_CTestPL = zeros(N, Np);
    SVr_postAdapt_CTestPL = zeros(N, Np);
    cCuePL = zeros(K, CtestCuelength, Np);
    cAdaptPL = zeros(K, CtestAdaptlength, Np);
    for p = 1:Np
        patt = patts{patterns(p)};
        cz = .5 * randn(K,1);
        c = ones(K,1);
        for n = 1:CtestWashout
            cz = F * tanh(Gstar * cz + Win * patt(n) + b);
        end
        % cue
        for n = 1:CtestCuelength
            r = tanh(Gstar * cz + Win * patt(n+CtestWashout) + b);
            cz = c .* (F * r) ;
            c = c + CtestCueLr * ((cz - c.*cz).*cz - aperture^(-2) * c);
            cCuePL(:,n,p) = c;
        end
        czPostCue = cz;
        % measure post-cue
        for n = 1:measureRL
            r = tanh(G * cz + b);
            cz = c .* (F * r) ;
            p_CTest_postCuePL(:,n,p) = Wout * r;
            r_CTest_postCuePL(:,n,p) = r;
        end
        % conceptor singval spectrum of x
        Rr = r_CTest_postCuePL(:,:,p) * r_CTest_postCuePL(:,:,p)'...
            / measureRL;
        Cr = Rr * inv(Rr + aperture^(-2) * I);
        [U S V] = svd(Cr);
        SVr_postCue_CTestPL(:, p) = diag(S);
        
        cz = czPostCue;
        % free-running autoconceptive adaptation
        for n = 1:CtestAdaptlength
            r = tanh(G * cz + b);
            cz = c .* (F * r);
            c = c + CtestAdaptLr * ((cz - c.*cz).*cz - aperture^(-2) * c);
            cAdaptPL(:,n,p) = c;
        end
        % measure post-adapt
        for n = 1:measureRL
            r = tanh(G * cz + b);
            cz = c .* (F * r) ;
            p_CTest_postAdaptPL(:,n,p) = Wout * r;
            r_CTest_postAdaptPL(:,n,p) = r;
        end
        % conceptor singval spectrum of x
        Rr = r_CTest_postAdaptPL(:,:,p) * r_CTest_postAdaptPL(:,:,p)'...
            / measureRL;
        Cr = Rr * inv(Rr + aperture^(-2) * I);
        [U S V] = svd(Cr);
        SVr_postAdapt_CTestPL(:, p) = diag(S);
    end
end



%%% plotting


if testType == 1
    test_pAligned_PL = cell(1,Np);
    NRMSEsAligned = zeros(1,Np);
    
    for p = 1:Np
        intRate = 20;
        thisDriver = train_pPL{1,p};
        thisOut = p_CTestPL(1,:,p);
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
        %    coarseMaxInd = ceil(maxInd / intRate);
        %     test_zAligned_PL{1,p} = ...
        %         z_CTestPL(:,coarseMaxInd:coarseMaxInd+signalPlotLength-1,p);
        NRMSEsAligned(1,p) = ...
            nrmse(test_pAligned_PL{1,p},train_pPL{1,p});
    end
    
    %meanNRMSE = mean(NRMSEsAligned)
    %%
    figure(2); clf;
    fs = 14;
    set(gcf,'DefaultAxesColorOrder',[0  0.4 0.65 0.8]'*[1 1 1]);
    set(gcf, 'WindowStyle','normal');
    
    set(gcf,'Position', [600 400 750 500]);
    for p = 1:Np
        subplot(Np,4,(p-1)*4+1);
        hold on;
        plot(ones(1,K),'k--','LineWidth',1);
        plot((sort(Cs{1,p},'descend')),'LineWidth',2);
        hold off;
        set(gca,'YLim',[0,1.2], 'FontSize',fs);
        set(gca,'FontSize',fs, 'Box', 'on');
        if p == 1
            title('c spectrum','FontSize',fs);
        end
        if p ~= Np
            set(gca, 'XTickLabel',[]);
        end
        
        subplot(Np,4,(p-1)*4+2);
        
        hold on;
        plot(ones(1,N),'k--','LineWidth',1);
        plot(SVr_CTestPL(:, p), 'Color',0.5 * [1 1 1], ...
            'LineWidth',2);
        hold off;
        set(gca,'YLim',[0,1.2], 'FontSize',fs, ...
            'Box', 'on','YTickLabel',[], 'XLim',[0 20]);
        if p == 1
            title('virtual C spectrum','FontSize',fs);
        end
        if p ~= Np
            set(gca, 'XTickLabel',[]);
        end
        
        subplot(Np,4,(p-1)*4+3);
        plot(test_pAligned_PL{1,p}, 'LineWidth',6,'Color',0.85*[1 1 1]); hold on;
        plot(train_pPL{1,p},'LineWidth',1); hold off;
        if p == 1
            title('driver and y','FontSize',fs);
        end
        if p ~= Np
            set(gca, 'XTickLabel',[]);
        end
        set(gca, 'YLim',[-1,1], 'FontSize',fs);
        rectangle('Position', [0.5,-0.95,9.5,0.5],'FaceColor','w',...
            'LineWidth',1);
        text(1,-0.7,num2str(NRMSEsAligned(1,p),2),...
            'Color','k','FontSize',fs, 'FontWeight', 'bold');
        
        
        subplot(Np,4,(p-1)*4+4);
        plot(cCollectors{1,p}(1:50,:)');
        set(gca, 'YLim',[0 1.2], 'FontSize', fs);
        if p < 4
            set(gca, 'XTick',[]);
        end
        if p == 1
            title('c adaptation');
        end
        
    end
    
%     
%     figure(6); clf;
%     % set(gcf, 'WindowStyle','normal');
%     % set(gcf,'Position', [1500 100 150 500]);
%     for p = 1:Np
%         subplot(Np,1,p);
%         plot(r_CTestPL(1:3,end-signalPlotLength+1:end,p)','LineWidth',2);
%         if p == 1
%             title('r test');
%         end
%         
%     end
    
elseif testType == 2
    test_postCue_pAligned_PL = cell(1,Np);
    NRMSEs_postCue_Aligned = zeros(1,Np);
    
    for p = 1:Np
        intRate = 20;
        thisDriver = train_pPL{1,p};
        thisOut = p_CTest_postCuePL(1,:,p);
        thisDriverInt = interp1((1:signalPlotLength)',thisDriver',...
            (1:(1/intRate):signalPlotLength)', 'spline')';
        thisOutInt = interp1((1:measureRL)', thisOut',...
            (1:(1/intRate):measureRL)', 'spline')';
        
        L = size(thisOutInt,2); M = size(thisDriverInt,2);
        phasematches = zeros(1,L - M);
        for phaseshift = 1:(L - M)
            phasematches(1,phaseshift) = ...
                norm(thisDriverInt - ...
                thisOutInt(1,phaseshift:phaseshift+M-1));
        end
        [maxVal maxInd] = max(-phasematches);
        test_postCue_pAligned_PL{1,p} = ...
            thisOutInt(1,maxInd:intRate:...
            (maxInd+intRate*signalPlotLength-1));
        NRMSEs_postCue_Aligned(1,p) = ...
            nrmse(test_postCue_pAligned_PL{1,p},train_pPL{1,p});
    end
    
    test_postAdapt_pAligned_PL = cell(1,Np);
    NRMSEs_postAdapt_Aligned = zeros(1,Np);
    for p = 1:Np
        intRate = 20;
        thisDriver = train_pPL{1,p};
        thisOut = p_CTest_postAdaptPL(1,:,p);
        thisDriverInt = interp1((1:signalPlotLength)',thisDriver',...
            (1:(1/intRate):signalPlotLength)', 'spline')';
        thisOutInt = interp1((1:measureRL)', thisOut',...
            (1:(1/intRate):measureRL)', 'spline')';
        
        L = size(thisOutInt,2); M = size(thisDriverInt,2);
        phasematches = zeros(1,L - M);
        for phaseshift = 1:(L - M)
            phasematches(1,phaseshift) = ...
                norm(thisDriverInt - ...
                thisOutInt(1,phaseshift:phaseshift+M-1));
        end
        [maxVal maxInd] = max(-phasematches);
        test_postAdapt_pAligned_PL{1,p} = ...
            thisOutInt(1,maxInd:intRate:...
            (maxInd+intRate*signalPlotLength-1));
        NRMSEs_postAdapt_Aligned(1,p) = ...
            nrmse(test_postAdapt_pAligned_PL{1,p},train_pPL{1,p});
    end
    %%
    figure(1); clf;
    fs = 14;
    set(gcf,'DefaultAxesColorOrder',[0  0.4 0.65 0.8]'*[1 1 1]);
    set(gcf, 'WindowStyle','normal');
    set(gcf,'Position', [600 400 1000 500]);
    for p = 1:Np
        
        subplot(Np,5,(p-1)*5+1);
        hold on;
        plot(ones(1,K),'k--','LineWidth',1);
        plot(sort(cCuePL(:,end,p),'descend'),'k','LineWidth',2);
        plot(sort(cAdaptPL(:,end,p),'descend'),...
            'Color', 0.6 * [1 1 1],'LineWidth',2);
        hold off;
        set(gca,'YLim',[0,1.2], 'FontSize',fs, 'Box', 'on');
        if p == 1
            title('c spectrum','FontSize',fs);
        end
        if p ~= Np
            set(gca, 'XTickLabel',[]);
        end
        
        subplot(Np,5,(p-1)*5+2);
        hold on;
        plot(ones(1,N),'k--','LineWidth',1);
        plot(SVr_postCue_CTestPL(1:20, p),'k','LineWidth',2);
        plot(SVr_postAdapt_CTestPL(1:20, p), ...
            'Color', 0.6 * [1 1 1],'LineWidth',2);
        set(gca,'YLim',[0,1.2], 'FontSize',fs, ...
            'Box', 'on','YTickLabel',[], 'XLim',[0 20]);
        if p == 1
            title('virtual C spectrum','FontSize',fs);
        end
        if p ~= Np
            set(gca, 'XTickLabel',[]);
        end
        
        
        subplot(Np,5,(p-1)*5+3);
        hold on;
        plot(test_postCue_pAligned_PL{1,p}, ...
            'LineWidth',6,'Color',0.3*[1 1 1]);
        plot(test_postAdapt_pAligned_PL{1,p}, ...
            'LineWidth',6,'Color',0.85*[1 1 1]);
        plot(train_pPL{1,p},'LineWidth',1);
        hold off;
        if p == 1
            title('p and y','FontSize',fs);
        end
        if p ~= Np
            set(gca, 'XTickLabel',[]);
        end
        set(gca, 'YLim',[-1,1], 'FontSize',fs, ...
            'Box', 'on');
        rectangle('Position', [0.5,0.45,8,0.5],'FaceColor','w',...
            'LineWidth',1);
        text(1,0.7,num2str(NRMSEs_postCue_Aligned(1,p),2),...
            'Color','k','FontSize',fs, 'FontWeight', 'bold');
        rectangle('Position', [0.5,-0.95,8,0.5],'FaceColor','w',...
            'LineWidth',1);
        text(1,-0.7,num2str(NRMSEs_postAdapt_Aligned(1,p),2),...
            'Color','k','FontSize',fs, 'FontWeight', 'bold');
        
        
        
        subplot(Np,5,(p-1)*5+4);
        plot(cCuePL(1:50,:,p)');
        set(gca, 'YLim',[0 1.2], 'FontSize', fs, ...
            'XLim',[0, CtestCuelength]);
        if p < 4
            set(gca, 'XTick',[]);
        end
        if p == 1
            title('c adapt cue');
        end
        
        subplot(Np,5,(p-1)*5+5);
        plot(cAdaptPL(1:50,:,p)');
        set(gca, 'YLim',[0 1.2], 'FontSize', fs, ...
            'YTickLabel',[]);
        if p < 4
            set(gca, 'XTick',[]);
        end
        if p == 1
            title('c adapt free');
        end
        
    end
    
    
end

