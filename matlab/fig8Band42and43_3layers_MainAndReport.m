%%%% signal noise filtering application, two-stage with selfRate,
%%%%  bottom-up and top-down. Testing with cycling through patterns.

set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment control
randstate = 1;
%randstate = randstate + 1;
newNets = 1; newSystemScalings = 1;
relearn = 1;
relearnFilter = 1; % comparison linear filter
plotThumbs = 0;
plot4MainArticle = 1;

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

aperture = 8;

TychonovAlphaReadout = 0.1;
TychonovAlphaG = 0.1;
TychonovAlphaD = 0.1;
CtestWashout = 200;
CtestLength = 500;

%%% linear predictor for comparison
filterLength = 2600;
filterTychonovAlpha = 1;
filterTrainN = 500; % how many training examples per pattern

%%% testing 
morphing = 1; % whether we run the morphing experiment
recomputeFilterOut = 1; % concerns linear reference filter

if morphing
%     TrunLength = 8000; % for adapting alphas, per pattern
%     alphaLR = 0.0025;
%     trustsr = 0.99; % smoothing rate for trusts
%     truststeep12 = .125;
%     truststeep23 = .125;
%     SNR = 20;
%     drift = .1;
%     aperture = 8;
%     LRctest = .5;
    
    TrunLength = 4000; % for adapting alphas, per pattern
    alphaLR = 0.004;
    trustsr = 0.99; % smoothing rate for trusts
    truststeep12 = 8;
    truststeep23 = 8;
    SNR = .5;
    drift = .01;
    LRctest = .5;
else
    TrunLength = 4000; % for adapting alphas, per pattern
    alphaLR = 0.002;
    trustsr = 0.99; % smoothing rate for trusts
    truststeep12 = 8;
    truststeep23 = 8;
    SNR = 0.5; % signal-to-noise ratio
    drift = .01;
    LRctest = .5;
end

sr = 0.95; % smoothing rate for plotting nrmses
smoothNeighbors = 1; % half window size for local averaging of
% aligned nrmses.
% set to 0 for no smoothing

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

if relearn
    
    % Set pattern handles
    pattHandles;
    
    % Create raw weights
    if newNets
        WinRaw = randn(N, 1);
        WbiasRaw = randn(N, 1);
        FRaw = randn(K,N);
        GstarRaw = randn(N,K);
        GF = GstarRaw * FRaw;
        specrad = max(abs(eig(GF)));
        FRaw = FRaw / sqrt(specrad);
        GstarRaw = GstarRaw / sqrt(specrad);
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
    allTrainrOld = zeros(N, Np * learnLength);
    allTraincPhiOld = zeros(K, Np * learnLength);
    allTraincPhi = zeros(K, Np * learnLength);
    allTrainp = zeros(1, Np * learnLength);
    allTrainz = zeros(N, Np * learnLength);
    
    % harvest from network driven by random input
    allTrainrRand = zeros(N, Np * learnLength);
    allTrainpRand = zeros(1, Np * learnLength);
    allTrainzRand = zeros(N, Np * learnLength);
    allTraincPhiOldRand = zeros(K, Np * learnLength);
    
    cCollectors = cell(1,Np);
    Cs = zeros(K, Np);
    ZRaw = zeros(K, Np);
    train_rPL = cell(1,Np);
    train_pPL = cell(1,Np);
    % collect data from driving native reservoir with different drivers
    for p = 1:Np
        
        
        rRandCollector = zeros(N, learnLength );
        pRandCollector = zeros(1, learnLength );
        zRandCollector = zeros(N, learnLength );
        cPhiOldRandCollector = zeros(K, learnLength );
        cPhiRand = zeros(K, 1);
        r = zeros(N,1);
        for n = 1:(washoutLength + cAdaptLength + learnLength)
            
            
            % random drive
            uRand = 2*rand - 1;
            cPhiRandOld = cPhiRand;
            zRand = Gstar * cPhiRand;
            rRand = tanh(zRand  + Win * uRand + b);
            cPhiRand = F * rRand;  % cRand always is the identity
            
            
            if n > washoutLength + cAdaptLength
                rRandCollector(:, n - washoutLength - cAdaptLength) = rRand;
                pRandCollector(:, n - washoutLength - cAdaptLength) = uRand;
                zRandCollector(:, n - washoutLength - cAdaptLength) = zRand;
                cPhiOldRandCollector(:, n - washoutLength - cAdaptLength) = cPhiRandOld;
            end
        end
        
        allTrainrRand(:, (p-1)*learnLength+1:p*learnLength) = ...
            rRandCollector;
        allTrainpRand(:, (p-1)*learnLength+1:p*learnLength) = ...
            pRandCollector;
        allTrainzRand(:, (p-1)*learnLength+1:p*learnLength) = ...
            zRandCollector;
        allTraincPhiOldRand(:, (p-1)*learnLength+1:p*learnLength) = ...
            cPhiOldRandCollector;
        
    end
    
    Gtargs = allTrainzRand;
    Gargs = allTraincPhiOldRand;
    G = (inv(Gargs * Gargs' + ...
        TychonovAlphaG * eye(K)) ...
        * Gargs * Gtargs')';
    NRMSE_G = nrmse(G*Gargs, Gtargs);
    disp(sprintf('NRMSE G: %g    size: %g',...
        mean(NRMSE_G), mean(mean(abs(G)))));
   
    
    
    for p = 1:Np
        patt = patts{patterns(p)}; % current pattern generator
        rCollector = zeros(N, learnLength );
        rOldCollector = zeros(N, learnLength );
        cPhiOldCollector = zeros(K, learnLength );
        cPhiCollector = zeros(K, learnLength );
        PhiCollector = zeros(K, learnLength );
        pCollector = zeros(1, learnLength );
        cCollector = zeros(K, cAdaptLength );
        zCollector = zeros(N, learnLength );
        
        
        c = ones(K,1);
        cPhi = zeros(K, 1);
        r = zeros(N,1);
        for n = 1:(washoutLength + cAdaptLength + learnLength)
            u = patt(n); % pattern input
            cPhiOld = cPhi; rOld = r;
            z = G * cPhi ;
            r = tanh(z  + Win * u + b);
            Phi = F * r;
            cPhi = c .* Phi;
            
            
            if n <= cAdaptLength + washoutLength && ...
                    n > washoutLength
                c = c + LRc * ...
                    ((cPhi - c.*cPhi).*cPhi - aperture^(-2) * c);
                cCollector(:,n - washoutLength) = c;
            end
            if n == cAdaptLength + washoutLength
                Cs(:,p) = c;
            end
            if n > washoutLength + cAdaptLength
                rCollector(:, n - washoutLength - cAdaptLength) = r;
                rOldCollector(:, n - washoutLength - cAdaptLength) = rOld;
                cPhiOldCollector(:, n - washoutLength - cAdaptLength ) = cPhiOld;
                cPhiCollector(:, n - washoutLength - cAdaptLength ) = cPhi;
                PhiCollector(:, n - washoutLength - cAdaptLength ) = Phi;
                pCollector(1, n - washoutLength - cAdaptLength) = u;
                zCollector(:, n - washoutLength - cAdaptLength ) = z;
                
                
            end
        end
        cCollectors{1,p} = cCollector;
        ZRaw(:,p) = mean(cPhiOldCollector.^2,2);
        
        train_rPL{1,p} = rCollector(1:5,1:signalPlotLength);
        train_pPL{1,p} = pCollector(1,1:signalPlotLength);
        
        allTrainr(:, (p-1)*learnLength+1:p*learnLength) = ...
            rCollector;
        allTrainrOld(:, (p-1)*learnLength+1:p*learnLength) = ...
            rOldCollector;
        allTraincPhiOld(:, (p-1)*learnLength+1:p*learnLength) = ...
            cPhiOldCollector;
        allTraincPhi(:, (p-1)*learnLength+1:p*learnLength) = ...
            cPhiCollector;
        allTrainp(1, (p-1)*learnLength+1:p*learnLength) = ...
            pCollector;
        allTrainz(:, (p-1)*learnLength+1:p*learnLength) = ...
            zCollector;
        
        
    end
    
    % normalize Z
    normsZ = sqrt(sum(ZRaw.^2)); meanNormsZ = mean(normsZ);
    Z = ZRaw * diag(1./normsZ) * meanNormsZ;
    %Z = ZRaw;
    
    %%% compute readout
    Wout = (inv(allTrainr * allTrainr' + ...
        TychonovAlphaReadout * eye(N)) ...
        * allTrainr * allTrainp')';
    % training error
    NRMSE_readout = nrmse(Wout*allTrainr, allTrainp);
    disp(sprintf('NRMSE readout: %g  size: %g', NRMSE_readout, ...
        mean(mean(abs(Wout)))));
    
    
    
    %%% compute D
    Dtargs = allTrainp;
    Dargs = allTraincPhiOld;
    D = (inv(Dargs * Dargs' + ...
        TychonovAlphaD * eye(K)) ...
        * Dargs * Dtargs')';
    NRMSE_D = nrmse(D*Dargs, Dtargs);
    disp(sprintf('NRMSE D: %g    size: %g',...
        mean(NRMSE_D), mean(mean(abs(D)))));
    
    
    %%% testing quality of post-adaptation c's
    r_CTestPL = zeros(N, CtestLength, Np);
    cPhi_CTestPL = zeros(5, CtestLength, Np);
    p_CTestPL = zeros(1, CtestLength, Np);
    SVr_CTestPL = zeros(N, Np);
    for p = 1:Np
        c = Cs(:, p);
        cPhi = .5 * randn(K,1);
        r = .1 * randn(N,1);
        for n = 1:CtestWashout
            r = tanh(G * cPhi  + Win * D * cPhi + b);
            cPhi = c .* (F * r);
        end
        for n = 1:CtestLength
            r = tanh(G * cPhi + Win * D * cPhi + b);
            cPhi = c .* (F * r);
            r_CTestPL(:,n,p) = r;
            cPhi_CTestPL(:,n,p) = cPhi(1:5,1);
            p_CTestPL(:,n,p) = Wout * r;
        end
        % conceptor singval spectrum of x
        Rr = r_CTestPL(:,:,p) * r_CTestPL(:,:,p)' / CtestLength;
        Cr = Rr * inv(Rr + aperture^(-2) * I);
        [U S V] = svd(Cr);
        SVr_CTestPL(:, p) = diag(S);
    end
    
end

%%% learning linear filter for comparison
if relearnFilter
    filterTrainData = zeros(filterLength+1, Np * filterTrainN);
    for p = 1:Np
        patt = patts{patterns(p)};
        for d = 1:filterTrainN
            filterTrainData(:, (p-1)*p + d) = patt(d + (1:filterLength+1))';
        end
    end
    
    filterArgs = filterTrainData(1:filterLength,:);
    filterTargs = filterTrainData(end,:);
    
    filter = (inv(filterArgs * filterArgs' + ...
        filterTychonovAlpha * eye(filterLength)) ...
        * filterArgs * filterTargs')';
    NRMSE_filter = nrmse(filter * filterArgs, filterTargs);
    disp(sprintf('NRMSE filter: %g    size: %g',...
        mean(NRMSE_filter), mean(mean(abs(filter)))));
    clear filterTrainData filterArgs filterTargs;
end



%%

% compute template correlation vecs

Nt = size(Z, 2); % number of pattern templates

alpha1PL = zeros(Nt, TrunLength * Np);
alpha2PL = zeros(Nt, TrunLength * Np);
alpha3PL = zeros(Nt, TrunLength * Np);
trust1PL = zeros(1, TrunLength * Np);
trust2PL = zeros(1, TrunLength * Np);
trust3PL = zeros(1, TrunLength * Np);
trust12PL = zeros(1, TrunLength * Np);
trust23PL = zeros(1, TrunLength * Np);

u_cAlphaTestPL = zeros(1,TrunLength * Np);
noise_cAlphaTestPL = zeros(1,TrunLength * Np);

y1PL = zeros(1,TrunLength * Np);
y2PL = zeros(1,TrunLength * Np);
y3PL = zeros(1,TrunLength * Np);


% initializations
alphas1 = ones(1,Nt);
alphas1 = alphas1 / norm(alphas1);
alphas2 = alphas1;
alphas3 = alphas1;
t1 = Z * alphas1.^2';
t2 = Z * alphas2.^2';
t3 = Z * alphas3.^2';
c1 = t1 ./ (t1 + aperture^(-2));
c2 = t2 ./ (t2 + aperture^(-2));
c3 = t3 ./ (t3 + aperture^(-2));
c1Int = c1;
c2Int = c2;
c3Int = c3;
cPhi1 = zeros(K,1);
cPhi2 = zeros(K,1);
cPhi3 = zeros(K,1);
y1var = 1;
y2var = 1;
y3var = 1;
y1mean = 0;
y2mean = 0;
y3mean = 0;
trust12 = 0.5;
trust23 = 0.5;
discrepancy1 = 0.5;
discrepancy2 = 0.5;
discrepancy3 = 0.5;


noiseLevel = sqrt(  var(allTrainp) / SNR);

% run patterns
if morphing
    % create morphable pattern source for 5-period patterns
    patt5PeriodMorphable = @(n, morph) ...
        morph * patts{10}(n) + (1-morph)*patts{36}(n);
    morphRamp = (0:(1/(TrunLength-1)):1) ;
    phase = 0; % helper for sine morph
else
    patterns = [53  10 54 36 ];
end
for p = 1:Np
    if not(morphing)
        patt = patts{patterns(p)};
    end
    PLshift = (p-1)*TrunLength;
    for n = 1:TrunLength
        if morphing
            mR = morphRamp(n);
            if p == 1
                currentPeriod = (1-mR) *  8.8342522 + mR * 9.8342522;
                phase = phase + 2 * pi / currentPeriod;
                u = sin(phase);
            elseif p == 2
                currentPeriod = mR*8.8342522 + (1-mR) *  9.8342522;
                phase = phase + 2 * pi / currentPeriod;
                u = sin(phase);
            elseif p == 3
                u = patt5PeriodMorphable(n, (1-mR));
            else
                u = patt5PeriodMorphable(n, mR);
            end
        else
            u = patt(n);
        end
        noise = noiseLevel * randn;
        inext = (u + noise);
        inaut = D * cPhi1;
        r1 = tanh(G * cPhi1  + Win * ((1 ) * inext) + b);
        cPhi1 = c1.*(F * r1);
        y1 = Wout * r1;
        y1mean = trustsr * y1mean + (1-trustsr) * y1;
        y1var = trustsr * y1var + (1-trustsr) * (y1-y1mean)^2;
        discrepancy1 = trustsr * discrepancy1 + ...
            (1-trustsr) * (inaut - inext)^2 / y1var;
        c1Int = c1Int + LRctest * ...
                    ((cPhi1 - c1Int.*cPhi1).*cPhi1 - aperture^(-2) * c1Int);
        
        inaut = D * cPhi2;
        inext = y1;
        r2 = tanh(G * cPhi2  + Win * ((1 - trust12) * inext + ...
            trust12 * inaut) + b);
        cPhi2 = c2.* (F * r2);
        y2 = Wout * r2;
        y2mean = trustsr * y2mean + (1-trustsr) * y2;
        y2var = trustsr * y2var + (1-trustsr) * (y2-y2mean)^2;
        discrepancy2 = trustsr * discrepancy2 + ...
            (1-trustsr) * (inaut - inext)^2 / y2var;
        c2Int = c2Int + LRctest * ...
                    ((cPhi2 - c2Int.*cPhi2).*cPhi2 - aperture^(-2) * c2Int);
                
        inaut = D * cPhi3;
        inext = y2;
        r3 = tanh(G * cPhi3  + Win * ((1 - trust23) * inext + ...
            trust23 * inaut) + b);
        cPhi3 = c3.* (F * r3);
        y3 = Wout * r3;
        y3mean = trustsr * y3mean + (1-trustsr) * y3;
        y3var = trustsr * y3var + (1-trustsr) * (y3-y3mean)^2;        
        discrepancy3 = trustsr * discrepancy3 + ...
            (1-trustsr) * (inaut - inext)^2 / y3var;
         
       
        trust12 = 1 / (1 + ...
            (discrepancy2 / discrepancy1)^truststeep12);
           
        trust23 = 1 / (1 + ...
            (discrepancy3 / discrepancy2)^truststeep23);
        
        t1 = Z * alphas1.^2' ;
        alphadots1 =  4 * (cPhi1.^2 - t1)' *...
            Z * diag(alphas1) + drift * (0.5 - alphas1);
        alphas1 = alphas1 + alphaLR * alphadots1;
        alphas1 = alphas1 / sum(alphas1);
        
        t2 = Z * alphas2.^2' ; 
        alphadots2 =  4 * (cPhi2.^2 - t2)' *...
            Z * diag(alphas2) + drift * (0.5 - alphas2);
        alphas2 = alphas2 + alphaLR * alphadots2;
        alphas2 = alphas2 / sum(alphas2);
        
        t3 = Z * alphas3.^2' ; 
        alphadots3 =  4 * (cPhi3.^2 - t3)' *...
            Z * diag(alphas3) + drift * (0.5 - alphas3);
        alphas3 = alphas3 + alphaLR * alphadots3;
        alphas3 = alphas3 / sum(alphas3);

        c3 = t3 ./ (t3 + aperture^(-2));        
        c2 = trust23 * c3 + (1-trust23) * c2Int;
        c1 = trust12 * c2 + (1-trust12) * c1Int;
        
    
        
        % collect diagnostics
        alpha1PL(:,n+PLshift) = alphas1;
        alpha2PL(:,n+PLshift) = alphas2;
        alpha3PL(:,n+PLshift) = alphas3;
        
        y1PL(n+PLshift) = y1;
        y2PL(n+PLshift) = y2;
        y3PL(n+PLshift) = y3;
        
        trust1PL(n+PLshift) = discrepancy1;
        trust2PL(n+PLshift) = discrepancy2;
        trust3PL(n+PLshift) = discrepancy3;
        trust12PL(n+PLshift) = trust12;
        trust23PL(n+PLshift) = trust23;
        
        u_cAlphaTestPL(n+PLshift) = u;
        noise_cAlphaTestPL(n+PLshift) = noise;
        
    end
    
    
end

% compute comparison filter output
if recomputeFilterOut
    noisyu4filter =  u_cAlphaTestPL + noise_cAlphaTestPL;
    noisyu4filter = [zeros(1,filterLength) noisyu4filter];
    filterOut = zeros(1, TrunLength * Np);
    for n = 1:TrunLength * Np
        filterOut(1,n) = noisyu4filter(1,n:n+filterLength-1) * filter';
    end
end

%% plotting and diagnostics

% smoothing nrmse and trust
se_noise = noise_cAlphaTestPL.^2;
se_y1 = (y1PL - u_cAlphaTestPL).^2;
se_y2 = (y2PL - u_cAlphaTestPL).^2;
se_y3 = (y3PL - u_cAlphaTestPL).^2;

noisyu = u_cAlphaTestPL + noise_cAlphaTestPL;
noiseEst = noisyu - y2PL ;

se_filter = (filterOut - u_cAlphaTestPL).^2;

se_noise_smoothed = zeros(1,TrunLength * Np);
se_y1_smoothed = zeros(1,TrunLength * Np);
se_y2_smoothed = zeros(1,TrunLength * Np);
se_y3_smoothed = zeros(1,TrunLength * Np);
trust1_smoothed = zeros(1,TrunLength * Np);
trust2_smoothed = zeros(1,TrunLength * Np);
trust3_smoothed = zeros(1,TrunLength * Np);

se_filter_smoothed = zeros(1,TrunLength * Np);
u_smoothed = zeros(1,TrunLength * Np);
varu_smoothed = zeros(1,TrunLength * Np);

se_noise_val = 0.5;
se_y1_val = 0.5; se_y2_val = 0.5; se_y3_val = 0.5;
se_filter_val = 0.5; u_val = 0.5; varu_val = 1;
trust1_val = 0.2; trust2_val = 0.2; trust3_val = 0.2;
for n = 1:TrunLength * Np
    se_noise_val = sr * se_noise_val + (1-sr) * se_noise(n);
    se_y1_val = sr * se_y1_val + (1-sr) * se_y1(n);
    se_y2_val = sr * se_y2_val + (1-sr) * se_y2(n);
    se_y3_val = sr * se_y3_val + (1-sr) * se_y3(n);
    trust1_val = sr * trust1_val + (1-sr) * trust1PL(n);
    trust2_val = sr * trust2_val + (1-sr) * trust2PL(n);
    trust3_val = sr * trust3_val + (1-sr) * trust3PL(n);
    se_filter_val = sr * se_filter_val + (1-sr) * se_filter(n);
    varu_val = sr * varu_val + (1-sr) * (u_cAlphaTestPL(n) - u_val)^2;
    u_val = sr * u_val + (1-sr) * u_cAlphaTestPL(n);
    
    se_noise_smoothed(1,n) = se_noise_val;
    se_y1_smoothed(1,n) = se_y1_val;
    se_y2_smoothed(1,n) = se_y2_val;
    se_y3_smoothed(1,n) = se_y3_val;
    trust1_smoothed(1,n) = trust1_val;
    trust2_smoothed(1,n) = trust2_val;
    trust3_smoothed(1,n) = trust3_val;
    
    se_filter_smoothed(1,n) = se_filter_val;
    u_smoothed(1,n) = u_val;
    varu_smoothed(1,n) = varu_val;
end
nrmse_noise_PL = sqrt(se_noise_smoothed ./ varu_smoothed);
nrmse_y1_PL = sqrt(se_y1_smoothed ./ varu_smoothed);
nrmse_y2_PL = sqrt(se_y2_smoothed ./ varu_smoothed);
nrmse_y3_PL = sqrt(se_y3_smoothed ./ varu_smoothed);
nrmse_filter_PL = sqrt(se_filter_smoothed ./ varu_smoothed);

% aligned nrmse for yi
sigWindowLength = 20; refWindowAddonLength = 10;
alignPlotPoints = ...
    sigWindowLength:sigWindowLength:(TrunLength * Np - sigWindowLength);
nrmse_y1_aligned_PL = zeros(1,size(alignPlotPoints,2));
nrmse_y2_aligned_PL = zeros(1,size(alignPlotPoints,2));
nrmse_y3_aligned_PL = zeros(1,size(alignPlotPoints,2));
intRate = 5;
for n = 1:size(alignPlotPoints,2)
    y1signal =...
        y1PL(1,alignPlotPoints(n) - sigWindowLength/2 + 1:...
        alignPlotPoints(n) + sigWindowLength/2);
    y2signal =...
        y2PL(1,alignPlotPoints(n) - sigWindowLength/2 + 1:...
        alignPlotPoints(n) + sigWindowLength/2);
    y3signal =...
        y3PL(1,alignPlotPoints(n) - sigWindowLength/2 + 1:...
        alignPlotPoints(n) + sigWindowLength/2);
    reference = ...
        u_cAlphaTestPL(1,alignPlotPoints(n) - sigWindowLength/2 -...
        refWindowAddonLength + 1:...
        alignPlotPoints(n) + sigWindowLength/2 +...
        refWindowAddonLength);
    nrmse_y1_aligned_PL(n) = ...
        NRMSEaligned(reference, y1signal,...
        varu_smoothed(alignPlotPoints(n)), intRate);
    nrmse_y2_aligned_PL(n) = ...
        NRMSEaligned(reference, y2signal, ...
        varu_smoothed(alignPlotPoints(n)), intRate);
    nrmse_y3_aligned_PL(n) = ...
        NRMSEaligned(reference, y3signal, ...
        varu_smoothed(alignPlotPoints(n)), intRate);
end
%%
% smoothing aligned nrmses

nrmse_y1_alignedSmoothed_PL = nrmse_y1_aligned_PL;
nrmse_y2_alignedSmoothed_PL = nrmse_y2_aligned_PL;
nrmse_y3_alignedSmoothed_PL = nrmse_y3_aligned_PL;
for n = smoothNeighbors+1:size(alignPlotPoints,2)-smoothNeighbors
    nrmse_y1_alignedSmoothed_PL(n) = ...
        mean(nrmse_y1_aligned_PL(n-smoothNeighbors:n+smoothNeighbors));
    nrmse_y2_alignedSmoothed_PL(n) = ...
        mean(nrmse_y2_aligned_PL(n-smoothNeighbors:n+smoothNeighbors));
    nrmse_y3_alignedSmoothed_PL(n) = ...
        mean(nrmse_y3_aligned_PL(n-smoothNeighbors:n+smoothNeighbors));
    
end
%%

figure(14); clf;
%set(gcf,'DefaultAxesColorOrder',[0  0.4 0.65 0.8]'*[1 1 1]);
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [800 500 900 700]);
fs = 18;

subplot(5,1,1);
hold on;
plot(ones(1,TrunLength * Np), 'k--');
plot((alpha3PL'),  'LineWidth', 2);
hold off;
for x = 1:Np-1
    line([x*TrunLength,x*TrunLength], [0 1.2], 'Color',[0 0 0]);
end
%title('alphaSquare');
set(gca, 'YLim',[0 1.2], 'XLim',[0 TrunLength * Np],'XTick', [],...
    'Box', 'on', 'FontSize', fs);


subplot(5,1,2);
hold on;
plot(ones(1,TrunLength * Np), 'k--');
plot((alpha2PL'),  'LineWidth', 2);
hold off;
for x = 1:Np-1
    line([x*TrunLength,x*TrunLength], [0 1.2], 'Color',[0 0 0]);
end
%title('alphaSquare');
set(gca, 'YLim',[0 1.2], 'XLim',[0 TrunLength * Np], 'XTick', [],...
    'Box', 'on', 'FontSize', fs);

subplot(5,1,3);
hold on;
plot(ones(1,TrunLength * Np), 'k--');
plot((alpha1PL'),  'LineWidth', 2);
hold off;
for x = 1:Np-1
    line([x*TrunLength,x*TrunLength], [0 1.2], 'Color',[0 0 0]);
end
%title('alphaSquare');
set(gca, 'YLim',[0 1.2],'XLim',[0 TrunLength * Np], 'XTick', [],...
    'Box', 'on', 'FontSize', fs);

subplot(5,1,4);
steep = 2;
hold on;
plot(trust12PL, 'b', 'LineWidth', 2);
plot(trust23PL, 'g', 'LineWidth', 2);
hold off;
for x = 1:Np-1
    line([x*TrunLength,x*TrunLength], [0 1], 'Color',[0 0 0]);
end
%title('alphaSquare');
set(gca, 'YLim',[0 1], 'XLim',[0 TrunLength * Np],'XTick', [],...
    'Box', 'on', 'FontSize', fs);

subplot(5,1,5);
hold on;
plot(log10(nrmse_y1_PL), 'b', 'LineWidth', 1.5);
plot(log10(nrmse_y2_PL), 'g', 'LineWidth', 1.5);
plot(log10(nrmse_y3_PL), 'r', 'LineWidth', 1.5);
plot(log10(nrmse_filter_PL), 'k', 'LineWidth', 1);
plot(zeros(1,TrunLength * Np), 'k--');
plot(alignPlotPoints, log10(nrmse_y3_alignedSmoothed_PL), ...
    'r', 'LineWidth', 1);
for x = 1:Np-1
    line([x*TrunLength,x*TrunLength], [-2 .5], 'Color',[0 0 0]);
end
hold off;
set(gca, 'YLim',[-2 .5], 'XTick',  TrunLength*[0 1 2 3 4],...
    'XLim',[0 TrunLength * Np], 'Box', 'on', 'FontSize', fs);
%title('log10 nrmse noise(k) p1(b) p2(r)');
hold off;
%%
if plot4MainArticle
    figure(15); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [800 500 450 900]);
fs = 20;

subplot(6,2,[1 2]);
hold on;
plot(ones(1,TrunLength * 2), 'k--');
plot((alpha3PL(:,1:8000)'),  'LineWidth', 2);
line([TrunLength,TrunLength], [0 1.2], 'Color',[0 0 0]);
hold off;
%title('alphaSquare');
set(gca, 'YLim',[0 1.2], 'XLim',[0 TrunLength * 2],'XTick', [],...
    'Box', 'on', 'FontSize', fs);
title('layer 3 hypotheses');


subplot(6,2,[3 4]);
hold on;
plot(ones(1,TrunLength * 2), 'k--');
plot((alpha2PL(:,1:8000)'),  'LineWidth', 2);
line([TrunLength,TrunLength], [0 1.2], 'Color',[0 0 0]);
hold off;
%title('alphaSquare');
set(gca, 'YLim',[0 1.2], 'XLim',[0 TrunLength * 2], 'XTick', [],...
    'Box', 'on', 'FontSize', fs);
title('layer 2 hypotheses');

subplot(6,2,[5 6]);
hold on;
plot(ones(1,TrunLength * 2), 'k--');
plot((alpha1PL(:,1:8000)'),  'LineWidth', 2);
line([TrunLength,TrunLength], [0 1.2], 'Color',[0 0 0]);
hold off;
%title('alphaSquare');
set(gca, 'YLim',[0 1.2],'XLim',[0 TrunLength * 2], 'XTick', [],...
    'Box', 'on', 'FontSize', fs);
title('layer 1 hypotheses');

subplot(6,2,[7 8]);
steep = 2;
hold on;
plot(trust12PL(:,1:8000), 'b', 'LineWidth', 2);
plot(trust23PL(:,1:8000), 'g', 'LineWidth', 2);

line([TrunLength,TrunLength], [0 1.2], 'Color',[0 0 0]);
hold off;
%title('alphaSquare');
set(gca, 'YLim',[0 1], 'XLim',[0 TrunLength * 2],'XTick', [],...
    'Box', 'on', 'FontSize', fs);
title('trust variables');

subplot(6,2,[9 10]);
hold on;
plot(log10(nrmse_y1_PL(:,1:8000)), 'b', 'LineWidth', 2);
%plot(log10(nrmse_y2_PL(:,1:8000)), 'g', 'LineWidth', 1.5);
%plot(log10(nrmse_y3_PL(:,1:8000)), 'r', 'LineWidth', 1.5);
plot(log10(nrmse_filter_PL(:,1:8000)), 'k', 'LineWidth', 2);
plot(zeros(1,TrunLength * 2), 'k--');
plot(alignPlotPoints, log10(nrmse_y2_alignedSmoothed_PL), ...
    'g', 'LineWidth', 2);
plot(alignPlotPoints, log10(nrmse_y3_alignedSmoothed_PL), ...
    'r', 'LineWidth', 2);
line([TrunLength,TrunLength], [0 1.2], 'Color',[0 0 0]);
hold off;
set(gca, 'YLim',[-2 .5], 'XTick',  TrunLength*[0 1 2 ],...
    'XLim',[0 TrunLength * 2], 'Box', 'on', 'FontSize', fs);
%title('log10 nrmse noise(k) p1(b) p2(r)');
hold off;
title('log10 NRMSEs');

subplot(6,2,11);
PLshift = TrunLength;
    hold on;
    plot(u_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift ) + ...
        noise_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift),...
        'r','LineWidth',2);
    %     plot(y1PL(PLshift - signalPlotLength+1:PLshift), ...
    %         'b','LineWidth',2);
    
    
    plot(y3PL(PLshift - signalPlotLength+1:PLshift), ...
        'Color',0.9* [.6 .6 1],'LineWidth',6);
    plot(u_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift ) ,...
        'k','LineWidth',1.5);
    plot(ones(1,signalPlotLength), 'k--');
    plot(-ones(1,signalPlotLength), 'k--');
    hold off;
    set(gca, 'YLim',[-2 2], 'Ytick',[ -1 0 1 ], ...
        'XLim',[0 signalPlotLength],...
        'Box', 'on', 'FontSize', fs);
    title('pattern samples');
    
    subplot(6,2,12);
PLshift = 2*TrunLength;
    hold on;
    plot(u_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift ) + ...
        noise_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift),...
        'r','LineWidth',2);
    %     plot(y1PL(PLshift - signalPlotLength+1:PLshift), ...
    %         'b','LineWidth',2);
    
    
    plot(y3PL(PLshift - signalPlotLength+1:PLshift), ...
        'Color',0.9* [.6 .6 1],'LineWidth',6);
    plot(u_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift ) ,...
        'k','LineWidth',1.5);
    plot(ones(1,signalPlotLength), 'k--');
    plot(-ones(1,signalPlotLength), 'k--');
    hold off;
    set(gca, 'YLim',[-2 2], 'Ytick',[ ], ...
        'XLim',[0 signalPlotLength],...
        'Box', 'on', 'FontSize', fs);
    
        set(gca, 'YTick',[]);
    


end
%%

figure(6); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [800 50 1000 160]);
fs = 18;
for p = 1:Np
    PLshift = p*TrunLength;
    subplot(1,4,p);
    hold on;
    plot(u_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift ) + ...
        noise_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift),...
        'r','LineWidth',2);
    %     plot(y1PL(PLshift - signalPlotLength+1:PLshift), ...
    %         'b','LineWidth',2);
    
    
    plot(y3PL(PLshift - signalPlotLength+1:PLshift), ...
        'Color',0.9* [.6 .6 1],'LineWidth',6);
    plot(u_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift ) ,...
        'k','LineWidth',1.5);
    plot(ones(1,signalPlotLength), 'k--');
    plot(-ones(1,signalPlotLength), 'k--');
    hold off;
    set(gca, 'YLim',[-2 2], 'Ytick',[-2 -1 0 1 2], ...
        'XLim',[0 signalPlotLength],...
        'Box', 'on', 'FontSize', fs);
    if p > 1
        set(gca, 'YTick',[]);
    end
end
%%
%%% plotting
if relearn
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
    figure(1); clf;
    fs = 18;
    set(gcf,'DefaultAxesColorOrder',[0  0.4 0.65 0.8]'*[1 1 1]);
    set(gcf, 'WindowStyle','normal');
    set(gcf,'Position', [600 400 1000 500]);
    for p = 1:Np
        subplot(Np,4,(p-1)*4+1);
        hold on;
        plot(ones(1,K),'k--','LineWidth',1);
        plot((sort(Cs(:,p),'descend')),'LineWidth',2);
        hold off;
        set(gca,'YLim',[0,1.2], 'FontSize',fs);
        set(gca,'FontSize',fs, 'Box', 'on');
        if p == 1
            title('c1 spectrum','FontSize',fs);
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
            title('c1 adaptation');
        end
        
    end
end
%%
%% plotting thumbnails
if plotThumbs
figure(11); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [800 50 100 100]);
PLshift = TrunLength;
hold on;
plot(u_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift ) + ...
    noise_cAlphaTestPL(PLshift - signalPlotLength+1:PLshift),...
    'k','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(12); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [950 50 100 100]);
PLshift = TrunLength;
hold on;
 plot(y1PL(PLshift - signalPlotLength+1:PLshift), ...
         'b','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(13); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1100 50 100 100]);
PLshift = TrunLength;
hold on;
 plot(y2PL(PLshift - signalPlotLength+1:PLshift), ...
         'g','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(14); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1250 50 100 100]);
PLshift = TrunLength;
hold on;
 plot(y3PL(PLshift - signalPlotLength+1:PLshift), ...
         'r','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(15); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1300 50 100 100]);
patt = patts{patterns(1)};
hold on;
 plot(patt(1:20), ...
         'r','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(16); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1350 50 100 100]);
patt = patts{patterns(2)};
hold on;
 plot(patt(1:20), ...
         'r','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(17); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1300 100 100 100]);
patt = patts{patterns(3)};
hold on;
 plot(patt(1:20), ...
         'r','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(18); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1350 100 100 100]);
patt = patts{patterns(4)};
hold on;
 plot(patt(1:20), ...
         'r','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');
%%
figure(19); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1300 50 100 100]);
patt = patts{patterns(1)};
hold on;
 plot(patt(1:20) + noiseLevel * randn(1,20), ...
         'k','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(20); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1350 50 100 100]);
patt = patts{patterns(2)};
hold on;
 plot(patt(1:20)+ noiseLevel * randn(1,20), ...
         'k','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(21); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1300 100 100 100]);
patt = patts{patterns(3)};
hold on;
 plot(patt(1:20)+ noiseLevel * randn(1,20), ...
         'k','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(22); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1350 100 100 100]);
patt = patts{patterns(4)};
hold on;
 plot(patt(1:20)+ noiseLevel * randn(1,20), ...
         'k','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');
%%
figure(23); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1300 50 100 100]);
patt = patts{patterns(1)};
hold on;
 plot(patt(1:20), ...
         'g','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(24); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1350 50 100 100]);
patt = patts{patterns(2)};
hold on;
 plot(patt(1:20), ...
         'r','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(25); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1300 100 100 100]);
patt = patts{patterns(3)};
hold on;
 plot(patt(1:20), ...
         'b','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');

figure(26); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [1350 100 100 100]);
patt = patts{patterns(4)};
hold on;
 plot(patt(1:20), ...
         'c','LineWidth',4);
plot(ones(1,signalPlotLength), 'k--');
plot(-ones(1,signalPlotLength), 'k--');
hold off;
set(gca, 'YLim',[-2 2], 'Ytick',[], 'Xtick',[], ...
    'XLim',[0 signalPlotLength],...
    'Box', 'on');
end

