
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

#%%%% plain demo that when a RNN is driven by different signals, the induced
#%%%% internal signals will inhabit different subspaces of the signal space.
#% set figure window to 1 x 2 panels
set(0., 'DefaultFigureWindowStyle', 'docked')
#%%% Experiment control
randstate = 8.
newNets = 1.
newSystemScalings = 1.
linearMorphing = 1.
#%%% Setting system params
Netsize = 100.
#% network size
NetSR = 1.5
#% spectral radius
NetinpScaling = 1.5
#% scaling of pattern feeding weights
BiasScaling = 0.2
#% size of bias
#%%% loading learning
TychonovAlpha = .0001
#% regularizer for  W training
washoutLength = 500.
learnLength = 1000.
signalPlotLength = 20.
#%%% pattern readout learning
TychonovAlphaReadout = 0.00001
#%%% C learning and testing
alpha = 1.
CtestLength = 200.
CtestWashout = 100.
SplotLength = 50.
#% %%% Autoadapt testing
#% cueLength = 50; postCueLength = 300;
#% deviationPlotInterval = 100;
#% TalphaAuto = 0.02;
#% startAlpha = .02; % starting value for cueing phase
#% TautoLR = 0.02;
#% TcueLR = 0.02;
#% SNR_cue = Inf; SNR_freeRun = Inf; % can be Inf for zero noise
#%%% Setting patterns
patterns = np.array(np.hstack((53., 54., 10., 36.)))
#%patterns = [23 6];
#%patterns = [1 2 21 20 22 8 19 6  16 9 10 11 12];
#% 1 = sine10  2 = sine15  3 = sine20  4 = spike20
#% 5 = spike10 6 = spike7  7 = 0   8 = 1
#% 9 = rand5; 10 = rand5  11 = rand6 12 = rand7
#% 13 = rand8 14 = sine10range01 15 = sine10rangept5pt9
#% 16 = rand3 17 = rand9 18 = rand10 19 = 0.8 20 = sineroot27
#% 21 = sineroot19 22 = sineroot50 23 = sineroot75
#% 24 = sineroot10 25 = sineroot110 26 = sineroot75tenth
#% 27 = sineroots20plus40  28 = sineroot75third
#% 29 = sineroot243  30 = sineroot150  31 = sineroot200
#% 32 = sine10.587352723 33 = sine10.387352723
#% 34 = rand7  35 = sine12  36 = 10+perturb  37 = sine11
#% 38 = sine10.17352723  39 = sine5 40 = sine6
#% 41 = sine7 42 = sine8  43 = sine9 44 = sine12
#% 45 = sine13  46 = sine14  47 = sine10.8342522
#% 48 = sine11.8342522  49 = sine12.8342522  50 = sine13.1900453
#% 51 = sine7.1900453  52 = sine7.8342522  53 = sine8.8342522
#% 54 = sine9.8342522 55 = sine5.19004  56 = sine5.8045
#% 57 = sine6.49004 58 = sine6.9004 59 = sine13.9004
#% 60 = 18+perturb
#%%% Initializations
plt.randn('state', randstate)
np.random.rand('twister', randstate)
#% Create raw weights
if newNets:
    if Netsize<=20.:
        Netconnectivity = 1.
    else:
        Netconnectivity = 10./Netsize
        
    
    WstarRaw = generate_internal_weights(Netsize, Netconnectivity)
    WinRaw = plt.randn(Netsize, 1.)
    WbiasRaw = plt.randn(Netsize, 1.)


#% Scale raw weights and initialize weights
if newSystemScalings:
    Wstar = np.dot(NetSR, WstarRaw)
    Win = np.dot(NetinpScaling, WinRaw)
    Wbias = np.dot(BiasScaling, WbiasRaw)


#% Set pattern handles
pattHandles
I = np.eye(Netsize)
#% % learn equi weights
#% harvest data from network externally driven by patterns
Np = length(patterns)
allTrainArgs = np.zeros(Netsize, np.dot(Np, learnLength))
allTrainOldArgs = np.zeros(Netsize, np.dot(Np, learnLength))
allTrainTargs = np.zeros(Netsize, np.dot(Np, learnLength))
allTrainOuts = np.zeros(1., np.dot(Np, learnLength))
readoutWeights = cell(1., Np)
patternCollectors = cell(1., Np)
xCollectorsCentered = cell(1., Np)
xCollectors = cell(1., Np)
SRCollectors = cell(1., Np)
URCollectors = cell(1., Np)
patternRs = cell(1., Np)
train_xPL = cell(1., Np)
train_pPL = cell(1., Np)
startXs = np.zeros(Netsize, Np)
#% collect data from driving native reservoir with different drivers
for p in np.arange(1., (Np)+1):
    patt = patts.cell_getattr(patterns[int(p)-1])
    #% current pattern generator
    xCollector = np.zeros(Netsize, learnLength)
    xOldCollector = np.zeros(Netsize, learnLength)
    pCollector = np.zeros(1., learnLength)
    x = np.zeros(Netsize, 1.)
    for n in np.arange(1., (washoutLength+learnLength)+1):
        u = patt[int(n)-1]
        #% pattern input
        xOld = x
        x = np.tanh((np.dot(Wstar, x)+np.dot(Win, u)+Wbias))
        if n > washoutLength:
            xCollector[:,int((n-washoutLength))-1] = x
            xOldCollector[:,int((n-washoutLength))-1] = xOld
            pCollector[0,int((n-washoutLength))-1] = u
        
        
        
    xCollectorCentered = xCollector-matcompat.repmat(np.mean(xCollector, 2.), 1., learnLength)
    xCollectorsCentered.cell[0,int(p)-1] = xCollectorCentered
    xCollectors.cell[0,int(p)-1] = xCollector
    R = matdiv(np.dot(xCollector, xCollector.conj().T), learnLength)
    [Ux, Sx, Vx] = plt.svd(R)
    SRCollectors.cell[0,int(p)-1] = Sx
    URCollectors.cell[0,int(p)-1] = Ux
    patternRs.cell[int(p)-1] = R
    startXs[:,int(p)-1] = x
    train_xPL.cell[0,int(p)-1] = xCollector[0:5.,0:signalPlotLength]
    train_pPL.cell[0,int(p)-1] = pCollector[0,0:signalPlotLength]
    patternCollectors.cell[0,int(p)-1] = pCollector
    allTrainArgs[:,int(np.dot(p-1., learnLength)+1.)-1:np.dot(p, learnLength)] = xCollector
    allTrainOldArgs[:,int(np.dot(p-1., learnLength)+1.)-1:np.dot(p, learnLength)] = xOldCollector
    allTrainOuts[0,int(np.dot(p-1., learnLength)+1.)-1:np.dot(p, learnLength)] = pCollector
    allTrainTargs[:,int(np.dot(p-1., learnLength)+1.)-1:np.dot(p, learnLength)] = np.dot(Win, pCollector)
    
#%%% compute readout
Wout = np.dot(np.dot(linalg.inv((np.dot(allTrainArgs, allTrainArgs.conj().T)+np.dot(TychonovAlphaReadout, np.eye(Netsize)))), allTrainArgs), allTrainOuts.conj().T).conj().T
#% training error
NRMSE_readout = nrmse(np.dot(Wout, allTrainArgs), allTrainOuts)
np.disp(sprintf('NRMSE readout: %g', NRMSE_readout))
#%%% compute W
Wtargets = atanh(allTrainArgs)-matcompat.repmat(Wbias, 1., np.dot(Np, learnLength))
W = np.dot(np.dot(linalg.inv((np.dot(allTrainOldArgs, allTrainOldArgs.conj().T)+np.dot(TychonovAlpha, np.eye(Netsize)))), allTrainOldArgs), Wtargets.conj().T).conj().T
#% training errors per neuron
NRMSE_W = nrmse(np.dot(W, allTrainOldArgs), Wtargets)
np.disp(sprintf('mean NRMSE W: %g', np.mean(NRMSE_W)))
#%%% run loaded reservoir to observe a messy output. Do this with starting
#%%% from four states originally obtained in the four driving conditions
#%%
#% figure(10); clf;
#% % initialize network state
#% for p = 1:4
#%     x = startXs(:,p);
#%     messyOutPL = zeros(1,CtestLength);
#%     % run
#%     for n = 1:CtestLength
#%         x = tanh(W*x + Wbias);
#%         y = Wout * x;
#%         messyOutPL(1,n) = y;
#%     end
#%     subplot(2,2,p);
#%     plot(messyOutPL(1,end-19:end));
#% end
#% % compute projectors
Cs = cell(4., Np)
for p in np.arange(1., (Np)+1):
    R = patternRs.cell[int(p)-1]
    [U, S, V] = plt.svd(R)
    Snew = np.dot(S, linalg.inv((S+np.dot(matixpower(alpha, -2.), np.eye(Netsize)))))
    C = np.dot(np.dot(U, Snew), U.conj().T)
    Cs.cell[0,int(p)-1] = C
    Cs.cell[1,int(p)-1] = U
    Cs.cell[2,int(p)-1] = np.diag(Snew)
    Cs.cell[3,int(p)-1] = np.diag(S)
    
bestPhis = 100.*np.array(np.hstack((1., 1., 1., 1.)))
factors = np.dot(10.**(1./4.), np.array(np.hstack((1., 1., 1., 1.))))
halfPlotNumber = 17.
exponents = np.arange(-halfPlotNumber, (halfPlotNumber)+1)
Nphis = 2.*halfPlotNumber+1.
allPhis = np.zeros(4., Nphis)
attenuationPL = np.zeros(4., Nphis)
for i in np.arange(1., (Nphis)+1):
    allPhis[:,int(i)-1] = bestPhis.conj().T*factors.conj().T**exponents[int(i)-1]
    
allNRMSEs = np.zeros(4., Nphis)
allAttenuations = np.zeros(4., Nphis)
allDiffs = np.zeros(4., Nphis)
allQuotas = np.zeros(4., Nphis)
allZengys = np.zeros(4., Nphis)
CnormPL = np.zeros(4., Nphis)
for k in np.arange(1., (Nphis)+1):
    #%%% run all patterns with conceptor aperture-adapted by phi, compute
    
#%% best apertures based on norm gradient
normGrads = CnormPL[:,1:]-CnormPL[:,0:0-1.]
normGrads = np.array(np.hstack((normGrads[:,0], normGrads)))
plt.figure(2.)
plt.clf
set(plt.gcf, 'WindowStyle', 'normal')
set(plt.gcf, 'Position', np.array(np.hstack((900., 200., 600., 400.))))
fs = 18.
for p in np.arange(1., (Np)+1):
    plt.subplot(2., 2., p)
    line(np.log10(allPhis[int(p)-1,:]), normGrads[int(p)-1,:], 'Color', 'k', 'LineWidth', 2.)
    set(plt.gca, 'FontSize', fs, 'XTickLabel', np.array([]), 'XLim', np.array(np.hstack((-2.5, 6.5))))
    if p == 3.:
        plt.xlabel('log10 aperture', 'FontSize', fs)
        plt.ylabel('norm^2 gradient', 'FontSize', fs)
    
    
    if p > 2.:
        set(plt.gca, 'YLim', np.array(np.hstack((-.5, 1.5))))
    
    
    ax1 = plt.gca
    ax2 = plt.axes('Position', plt.get(ax1, 'Position'), 'XAxisLocation', 'top', 'YAxisLocation', 'right', 'Color', 'none', 'YLim', np.array(np.hstack((np.floor(matcompat.max(np.log10(allNRMSEs[int(p)-1,:]))), 1.))), 'XLim', np.array(np.hstack((-2.5, 6.5))), 'XColor', 'k', 'YColor', np.dot(0.5, np.array(np.hstack((1., 1., 1.)))), 'FontSize', fs, 'Box', 'on')
    line(np.log10(allPhis[int(p)-1,:]), np.log10(allNRMSEs[int(p)-1,:]), 'Color', np.dot(0.6, np.array(np.hstack((1., 1., 1.)))), 'LineWidth', 6., 'Parent', ax2)
    if p == 4.:
        plt.ylabel('log10 NRMSE')
    
    
    
#%%
allDZengys = np.array(np.hstack((allZengys[:,1], -allZengys[:,0], np.dot(0.5, allZengys[:,2:]-allZengys[:,0:0-2.]), allZengys[:,int(0)-1], -allZengys[:,int((0-1.))-1])))
#%%
plt.figure(1.)
plt.clf
set(plt.gcf, 'WindowStyle', 'normal')
set(plt.gcf, 'Position', np.array(np.hstack((900., 200., 600., 400.))))
fs = 18.
for p in np.arange(1., (Np)+1):
    plt.subplot(2., 2., p)
    line(np.log10(allPhis[int(p)-1,:]), np.log10((allDiffs[int(p)-1,:]/allZengys[int(p)-1,:])), 'Color', 'k', 'LineWidth', 2.)
    set(plt.gca, 'FontSize', fs, 'XTickLabel', np.array([]), 'XLim', np.array(np.hstack((-2.5, 6.5))))
    if p == 3.:
        plt.xlabel('log10 aperture', 'FontSize', fs)
        plt.ylabel('log10 attenuation', 'FontSize', fs)
    
    
    ax1 = plt.gca
    ax2 = plt.axes('Position', plt.get(ax1, 'Position'), 'XAxisLocation', 'top', 'YAxisLocation', 'right', 'Color', 'none', 'YLim', np.array(np.hstack((np.floor(matcompat.max(np.log10(allNRMSEs[int(p)-1,:]))), 1.))), 'XLim', np.array(np.hstack((-2.5, 6.5))), 'XColor', 'k', 'YColor', np.dot(0.5, np.array(np.hstack((1., 1., 1.)))), 'FontSize', fs, 'Box', 'on')
    line(np.log10(allPhis[int(p)-1,:]), np.log10(allNRMSEs[int(p)-1,:]), 'Color', np.dot(0.6, np.array(np.hstack((1., 1., 1.)))), 'LineWidth', 6., 'Parent', ax2)
    if p == 4.:
        plt.ylabel('log10 NRMSE')
    
    
    
#%%
#%
#% figure(5); clf;
#% plot(log10(allPhis'));
#% figure(1); clf;
#% fs = 18;
#% set(gcf,'DefaultAxesColorOrder',[0  0.4 0.65 0.8]'*[1 1 1]);
#% set(gcf, 'WindowStyle','normal');
#%
#% set(gcf,'Position', [600 400 1000 500]);
#% for p = 1:Np
#%     subplot(Np,4,(p-1)*4+1);
#%     plot(test_pAligned_PL{1,p}, 'LineWidth',6,'Color',0.85*[1 1 1]); hold on;
#%     plot(train_pPL{1,p},'LineWidth',1); hold off;
#%     if p == 1
#%         title('driver and y','FontSize',fs);
#%     end
#%     if p ~= Np
#%         set(gca, 'XTickLabel',[]);
#%     end
#%     set(gca, 'YLim',[-1,1], 'FontSize',fs);
#%     rectangle('Position', [0.5,-0.95,8,0.5],'FaceColor','w',...
#%         'LineWidth',1);
#%     text(1,-0.7,num2str(NRMSEsAligned(1,p),2),...
#%         'Color','k','FontSize',fs, 'FontWeight', 'bold');
#%
#%     subplot(Np,4,(p-1)*4+2);
#%     plot(train_xPL{1,p}(1:3,:)','LineWidth',2);
#%
#%     if p == 1
#%         title('reservoir states','FontSize',fs);
#%     end
#%     if p ~= Np
#%         set(gca, 'XTickLabel',[]);
#%     end
#%     set(gca,'YLim',[-1,1],'YTickLabel',[], 'FontSize',fs);
#%
#%     subplot(Np,4,(p-1)*4+3);
#%     %diagNormalized = sDiagCollectors{1,p} / sum(sDiagCollectors{1,p});
#%     plot(log10(diag(SRCollectors{1,p})),'LineWidth',2);
#%
#%     set(gca,'YLim',[-20,10], 'FontSize',fs);
#%     if p == 1
#%         title('log10 PC energy','FontSize',fs);
#%     end
#%     subplot(Np,4,(p-1)*4+4);
#%     plot(diag(SRCollectors{1,p}(1:10,1:10)),'LineWidth',2);
#%     if p == 1
#%         title('leading PC energy','FontSize',fs);
#%     end
#%     set(gca,'YLim',[0,40], 'FontSize',fs);
#% end
#%