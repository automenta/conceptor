%%% testing autoadaptation stability along morph route. In testing we cue
%%% by running reservoir with cue-corresponding conceptor

%addpath('./ESN_Toolbox');
set(0,'DefaultFigureWindowStyle','docked');

%%% Experiment control
randstate = 4; newNets = 1; newSystemScalings = 1;

%%% Setting system params
N = 50; SR = 1.5;
WinScaling = 1.5; % scaling of pattern feeding weights
biasScaling = .8;


%%% Initial C adaptation in driven runs
trainWashoutLength = 100;
learnWLength = 50;

%%% D learning
incrementalLoad = 0;
TychonovAlphaD = .0001;

%%% pattern readout learning
TychonovAlphaWout = 0.01;

%%% C learning and testing
aperture = 100;
initialWashout = 100;
cueLength = 20;
adaptLengthes = [1 1000 10000 1000000];

CadaptRateAfterCue = 0.01; % C adaptation rate

NpLoad = 10; % nr of pattern intermediate morphs that are loaded
NpTest = 20;

figNr = 9;


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

% Set pattern handles for the two extreme patterns
pattHandles;
patt1 = patts{10}; patt2 = patts{36};
pattsLoad = cell(1,NpLoad);
mixesLoad = 0:1/(NpLoad - 1):1;
for p = 1:NpLoad
    pattsLoad{1,p} = ...
        @(n) (1-mixesLoad(p)) * patt1(n) + ...
        mixesLoad(p) * patt2(n);
end


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
% 60 = 18+perturb  61 = spike3  62 = spike4 63 = spike5
% 64 = spike6 65 = rand4  66 = rand5  67 = rand6 68 = rand7
% 69 = rand8 70 = rand4  71 = rand5  72 = rand6 73 = rand7
% 74 = rand8




%%% pattern loading

% init collectors needed for computing Wout
allTrainArgs = zeros(N, NpLoad * learnWLength);
allTrainYTargs = zeros(1, NpLoad * learnWLength);

allTrainOldArgs = zeros(N, NpLoad * learnWLength);
allTrainDTargs = zeros(N, NpLoad * learnWLength);

Cs = zeros(N,N,NpLoad);



for p = 1:NpLoad
    XCue = zeros(N,learnWLength);
    XOldCue = zeros(N,learnWLength);
    pTemplate = zeros(learnWLength,1);
    patt = pattsLoad{1,p};
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
    % store C
    R = XCue * XCue' / learnWLength;
    Cs(:,:,p) = R * inv(R + aperture^(-2) * eye(N));
    
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

% % learn readouts
Wout = (inv(allTrainArgs * allTrainArgs' + TychonovAlphaWout * I) ...
    * allTrainArgs * allTrainYTargs')';
NRMSEWout = nrmse(allTrainYTargs, Wout * allTrainArgs);
disp(sprintf('NRMSE Wout = %0.2g  D = %0.2g', NRMSEWout, NRMSED));

%%% test retrieval
CsAdapted = zeros(N,N,NpTest, length(adaptLengthes));
pRegenerated = zeros(5,NpTest);
pOriginal = zeros(5,NpTest);
for trial = 1:length(adaptLengthes)
    mixestest = 0:1/(NpTest-1):1;
    for p = 1:NpTest
        patt = @(n) (1-mixestest(p)) * patt1(n) + ...
            mixestest(p) * patt2(n);
        x = zeros(N,1);
        C = 0*(1-mixestest(p))*Cs(:,:,1) + mixestest(p)*Cs(:,:,end);
        % cue
        for n = 1:cueLength
            u = patt(n) ;
            x =  tanh(W * x + Win * u + bias ) ;
            C = C + CadaptRateAfterCue * ((x - C*x)*x' - ...
                aperture^(-2)*C);
        end
        
        
        % run with autoadaptation
        for n = 1:adaptLengthes(trial)
            x = tanh(W *  x + D * x + bias );
            z = C * x;
            C = C + CadaptRateAfterCue * ((z - C*z)*z' - ...
                aperture^(-2)*C);
            if n > adaptLengthes(trial) - 5 
               pRegenerated(n - adaptLengthes(trial)+5,p) = Wout * x;
            pOriginal(n - adaptLengthes(trial)+5,p) = ...
                patt(n);
            end
        end
        % store final C
        CsAdapted(:,:,p, trial) = C;
    end
    
end

 
%%
% % compute comparison matrix
% compMat = zeros(NpTest, NpLoad);
% for iload = 1:NpLoad
%     for itest = 1:NpTest
%         compMat(itest, iload) = ...
%             norm(Cs(:,:,iload) - CsAdapted(:,:,itest), 'fro');
%     end
% end

compMats = zeros(NpTest, NpTest, length(adaptLengthes));
for trial = 1:length(adaptLengthes)
    for iload = 1:NpTest
        for itest = 1:NpTest
            compMats(itest, iload,trial) = ...
                norm(CsAdapted(:,:,iload,trial) - ...
                CsAdapted(:,:,itest,trial), 'fro');
        end
    end
end

%%

figure(figNr); clf;
fs = 18; fstext = 18;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [700 600 1000 200]);
for i = 1:length(adaptLengthes)
    subplot(1,length(adaptLengthes),i);
    plotmatrix(compMats(:,:,i), 'c');
    if i > 1
        set(gca,'YTick',[]);
    else
        set(gca,'YTick',[10 20]);
    end
    set(gca,'FontSize',16);
    title(sprintf('n = %g',adaptLengthes(i)));
end

%%
figure(figNr + 1);
clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [900 250 600 300]);
for i = 1:NpTest
    subplot(4,5,i);
    plot(pRegenerated(:,i)); hold on;
    plot(pOriginal(:,i), 'r'); hold off;
    set(gca, 'XTick',[], 'YTick',[]);
end

nrmseTest = nrmse(reshape(pRegenerated,1,100), reshape(pOriginal,1,100))

figure(figNr + 2);
clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [900 50 200 100]);
for i = 1:2
    subplot(1,2,i);
    if i == 1
    plot(patt1(1:5));
    else
        plot(patt2(1:5));
    end
    set(gca, 'XTick',[], 'YTick',[]);
end
%%
%save autoAdaptationCenterManifold CsAdapted;
