
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
linearMorphing = 0.
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
TychonovAlphaReadout = 0.01
#%%% C learning and testing
alpha = 10.
CtestLength = 200.
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
#%%% in order to get same patterns as in the hierarchical
#%architecture demo, run that demo first and do not call pattHandles here
#%again. 
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
    train_xPL.cell[0,int(p)-1] = xCollector[:,0:signalPlotLength]
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
plt.figure(10.)
plt.clf
#% initialize network state
for p in np.arange(1., 5.0):
    x = startXs[:,int(p)-1]
    messyOutPL = np.zeros(1., CtestLength)
    #% run
    for n in np.arange(1., (CtestLength)+1):
        x = np.tanh((np.dot(W, x)+Wbias))
        y = np.dot(Wout, x)
        messyOutPL[0,int(n)-1] = y
        
    plt.subplot(2., 2., p)
    plt.plot(messyOutPL[0,int(0-19.)-1:])
    
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
    
#% % test with C
x_CTestPL = np.zeros(5., CtestLength, Np)
p_CTestPL = np.zeros(1., CtestLength, Np)
for p in np.arange(1., (Np)+1):
    C = Cs.cell[0,int(p)-1]
    x = startXs[:,int(p)-1]
    x = np.dot(0.5, plt.randn(Netsize, 1.))
    for n in np.arange(1., (CtestLength)+1):
        x = np.tanh((np.dot(W, x)+Wbias))
        x = np.dot(C, x)
        x_CTestPL[:,int(n)-1,int(p)-1] = x[0:5.,0]
        p_CTestPL[:,int(n)-1,int(p)-1] = np.dot(Wout, x)
        
    
#%%% plotting
test_pAligned_PL = cell(1., Np)
test_xAligned_PL = cell(1., Np)
NRMSEsAligned = np.zeros(1., Np)
MSEsAligned = np.zeros(1., Np)
for p in np.arange(1., (Np)+1):
    intRate = 20.
    thisDriver = train_pPL.cell[0,int(p)-1]
    thisOut = p_CTestPL[0,:,int(p)-1]
    thisDriverInt = interp1(np.arange(1., (signalPlotLength)+1).conj().T, thisDriver.conj().T, np.arange(1., (signalPlotLength)+(1./intRate), 1./intRate).conj().T, 'spline').conj().T
    thisOutInt = interp1(np.arange(1., (CtestLength)+1).conj().T, thisOut.conj().T, np.arange(1., (CtestLength)+(1./intRate), 1./intRate).conj().T, 'spline').conj().T
    L = matcompat.size(thisOutInt, 2.)
    M = matcompat.size(thisDriverInt, 2.)
    phasematches = np.zeros(1., (L-M))
    for phaseshift in np.arange(1., (L-M)+1):
        phasematches[0,int(phaseshift)-1] = linalg.norm((thisDriverInt-thisOutInt[0,int(phaseshift)-1:phaseshift+M-1.]))
        
    [maxVal, maxInd] = matcompat.max((-phasematches))
    test_pAligned_PL.cell[0,int(p)-1] = thisOutInt[0,int(maxInd)-1:maxInd+np.dot(intRate, signalPlotLength)-1.:intRate]
    coarseMaxInd = np.ceil(matdiv(maxInd, intRate))
    test_xAligned_PL.cell[0,int(p)-1] = x_CTestPL[:,int(coarseMaxInd)-1:coarseMaxInd+signalPlotLength-1.,int(p)-1]
    NRMSEsAligned[0,int(p)-1] = nrmse(test_pAligned_PL.cell[0,int(p)-1], train_pPL.cell[0,int(p)-1])
    MSEsAligned[0,int(p)-1] = np.mean(((test_pAligned_PL.cell[0,int(p)-1]-train_pPL.cell[0,int(p)-1])**2.))
    
meanNRMSE = np.mean(NRMSEsAligned)
#%%
plt.figure(2.)
plt.clf
fs = 24.
fsNRMSE = 18.
#%set(gcf,'DefaultAxesColorOrder',[0  0.4 0.65 0.8]'*[1 1 1]);
set(plt.gcf, 'WindowStyle', 'normal')
set(plt.gcf, 'Position', np.array(np.hstack((600., 400., 1000., 500.))))
pick1 = 71.
pick2 = 80.
col1 = np.dot(0.6, np.array(np.hstack((1., 1., 1.))))
col2 = np.dot(0.3, np.array(np.hstack((1., 1., 1.))))
for p in np.arange(1., (Np)+1):
    plt.subplot(Np, 4., ((p-1.)*4.+1.))
    plt.plot(test_pAligned_PL.cell[0,int(p)-1], 'LineWidth', 10., 'Color', np.dot(0.75, np.array(np.hstack((1., 1., 1.)))))
    plt.hold(on)
    plt.plot(train_pPL.cell[0,int(p)-1], 'k', 'LineWidth', 1.5)
    plt.hold(off)
    if p == 1.:
        plt.title('p and y', 'FontSize', fs)
    
    
    if p != Np:
        set(plt.gca, 'XTickLabel', np.array([]))
    
    
    set(plt.gca, 'YLim', np.array(np.hstack((-1., 1.))), 'FontSize', fs)
    #%     rectangle('Position', [0.5,-0.95,8,0.5],'FaceColor','w',...
    #%         'LineWidth',1);
    #%      text(1,-0.7,num2str(NRMSEsAligned(1,p),2),...
    #%         'Color','k','FontSize',fsNRMSE, 'FontWeight', 'bold');
    plt.subplot(Np, 4., ((p-1.)*4.+2.))
    plt.hold(on)
    plt.plot(train_xPL.cell[0,int(p)-1,int(pick1)-1,:]().conj().T, 'Color', col1, 'LineWidth', 3.)
    plt.plot(train_xPL.cell[0,int(p)-1,int(pick2)-1,:]().conj().T, 'Color', col2, 'LineWidth', 3.)
    plt.hold(off)
    if p == 1.:
        plt.title('two neurons', 'FontSize', fs)
    
    
    if p != Np:
        set(plt.gca, 'XTickLabel', np.array([]))
    
    
    set(plt.gca, 'YLim', np.array(np.hstack((-1., 1.))), 'FontSize', fs, 'Box', 'on')
    plt.subplot(Np, 4., ((p-1.)*4.+3.))
    #%diagNormalized = sDiagCollectors{1,p} / sum(sDiagCollectors{1,p});
    plt.hold(on)
    plt.plot(np.log10(np.diag(SRCollectors.cell[0,int(p)-1])), 'k', 'LineWidth', 3.)
    plt.plot(np.zeros(1., 100.), 'k--')
    plt.hold(off)
    set(plt.gca, 'YLim', np.array(np.hstack((-17., 5.))), 'YTick', np.array(np.hstack((-10., 0.))), 'FontSize', fs, 'Box', 'on')
    if p == 1.:
        plt.title('log10 \sigma', 'FontSize', fs)
    
    
    if p<4.:
        set(plt.gca, 'XTick', np.array([]))
    
    
    plt.subplot(Np, 4., ((p-1.)*4.+4.))
    plt.hold(on)
    plt.plot(np.zeros(1., 100.), 'k--')
    plt.plot(np.ones(1., 100.), 'k--')
    plt.plot(Cs.cell[2,int(p)-1], 'k', 'LineWidth', 3.)
    plt.hold(off)
    if p == 1.:
        plt.title('s', 'FontSize', fs)
    
    
    set(plt.gca, 'YLim', np.array(np.hstack((-0.1, 1.1))), 'YTick', np.array(np.hstack((0., 1.))), 'FontSize', fs, 'Box', 'on')
    if p<4.:
        set(plt.gca, 'XTick', np.array([]))
    
    
    
#%%
#% pick1 = 11; pick2 = 16;
#% pick1 = 12; pick2 = 14;
#% pick1 = 16; pick2 = 32;
#% pick1 = 16; pick2 = 51; 71 80
#%pick1 = randi([1 100]); 
#%pick2 = 18;
#%pick2 = randi([1 100]); 
col1 = np.dot(0.5, np.array(np.hstack((1., 1., 1.))))
col2 = np.dot(0.7, np.array(np.hstack((1., 1., 1.))))
col3 = np.dot(0.3, np.array(np.hstack((1., 1., 1.))))
col4 = np.array(np.hstack((0.6, 0., 0.4)))
L = 100.
fs = 16.
trace1 = allTrainArgs[int(np.array(np.hstack((pick1, pick2))))-1,0:L]
trace2 = allTrainArgs[int(np.array(np.hstack((pick1, pick2))))-1,1000:1000.+L]
trace3 = allTrainArgs[int(np.array(np.hstack((pick1, pick2))))-1,2000:2000.+L]
trace4 = allTrainArgs[int(np.array(np.hstack((pick1, pick2))))-1,3000:3000.+L]
R1 = matdiv(np.dot(trace1, trace1.conj().T), L)
[U1, S1, V1] = plt.svd(R1)
R2 = matdiv(np.dot(trace2, trace2.conj().T), L)
[U2, S2, V2] = plt.svd(R2)
R3 = matdiv(np.dot(trace3, trace3.conj().T), L)
[U3, S3, V3] = plt.svd(R3)
R4 = matdiv(np.dot(trace4, trace4.conj().T), L)
[U4, S4, V4] = plt.svd(R4)
cycleData
C2 = np.dot(R2, linalg.inv((R2+np.dot(matixpower(a, -2.), np.eye(2.)))))
[U2c, S2c, V2c] = plt.svd(C2)
C3 = np.dot(R3, linalg.inv((R3+np.dot(matixpower(a, -2.), np.eye(2.)))))
[U3c, S3c, V3c] = plt.svd(C3)
C4 = np.dot(R4, linalg.inv((R4+np.dot(matixpower(a, -2.), np.eye(2.)))))
[U4c, S4c, V4c] = plt.svd(C4)
E1c = np.dot(C1, cycleData)
E2c = np.dot(C2, cycleData)
E3c = np.dot(C3, cycleData)
E4c = np.dot(C4, cycleData)
plt.subplot(1., 3., 2.)
plt.hold(on)
plt.plot(np.array(np.hstack((-rg, rg))), np.array(np.hstack((0., 0.))), 'k--')
plt.plot(np.array(np.hstack((0., 0.))), np.array(np.hstack((-rg, rg))), 'k--')
plt.plot(cycleData[0,:], cycleData[1,:], 'k', 'LineWidth', 1.5)
plt.plot(E1(1., :), E1(2., :), 'Color', col1, 'LineWidth', 1.5)
plt.plot(E1c[0,:], E1c[1,:], 'Color', col3, 'LineWidth', 4.)
plt.plot(np.dot(S1[0,0], np.array(np.hstack((0., U1[0,0])))), np.dot(S1[0,0], np.array(np.hstack((0., U1[1,0])))), 'Color', col1, 'LineWidth', 1.5)
plt.plot(np.dot(S1[1,1], np.array(np.hstack((0., U1[0,1])))), np.dot(S1[1,1], np.array(np.hstack((0., U1[1,1])))), 'Color', col1, 'LineWidth', 1.5)
plt.plot(np.dot(S1c(1., 1.), np.array(np.hstack((0., U1c(1., 1.))))), np.dot(S1c(1., 1.), np.array(np.hstack((0., U1c(2., 1.))))), 'Color', col3, 'LineWidth', 4.)
plt.plot(np.dot(S1c(2., 2.), np.array(np.hstack((0., U1c(1., 2.))))), np.dot(S1c(2., 2.), np.array(np.hstack((0., U1c(2., 2.))))), 'Color', col3, 'LineWidth', 4.)
#% plot(E2(1,:), E2(2,:), 'r', 'LineWidth',2);
#% plot(E2c(1,:), E2c(2,:), 'r', 'LineWidth',4);
#% plot(S2(1,1) * [0,U2(1,1)], S2(1,1) * [0,U2(2,1)],'r', 'LineWidth',1);
#% plot(S2(2,2) * [0,U2(1,2)], S2(2,2) * [0,U2(2,2)],'r', 'LineWidth',1);
plt.hold(off)
set(plt.gca, 'XLim', np.array(np.hstack((-rg, rg))), 'YLim', np.array(np.hstack((-rg, rg))), 'XTick', np.array(np.hstack((-1., 0., 1.))), 'YTick', np.array(np.hstack((-1., 0., 1.))), 'FontSize', fs, 'Box', 'on')
plt.axis(square)
plt.subplot(1., 3., 3.)
ms = 20.
lw = 1.5
plt.hold(on)
plt.plot(np.array(np.hstack((-rg, rg))), np.array(np.hstack((0., 0.))), 'k--')
plt.plot(np.array(np.hstack((0., 0.))), np.array(np.hstack((-rg, rg))), 'k--')
plt.plot(cycleData[0,:], cycleData[1,:], 'k', 'LineWidth', 1.5)
plt.plot(E1c[0,:], E1c[1,:], 'Color', col1, 'LineWidth', lw)
plt.plot(E2c[0,:], E2c[1,:], 'Color', col3, 'LineWidth', lw)
plt.plot(np.dot(S1c(1., 1.), np.array(np.hstack((0., U1c(1., 1.))))), np.dot(S1c(1., 1.), np.array(np.hstack((0., U1c(2., 1.))))), 'Color', col1, 'LineWidth', lw)
plt.plot(np.dot(S1c(2., 2.), np.array(np.hstack((0., U1c(1., 2.))))), np.dot(S1c(2., 2.), np.array(np.hstack((0., U1c(2., 2.))))), 'Color', col1, 'LineWidth', lw)
plt.plot(np.dot(S2c[0,0], np.array(np.hstack((0., U2c[0,0])))), np.dot(S2c[0,0], np.array(np.hstack((0., U2c[1,0])))), 'Color', col3, 'LineWidth', lw)
plt.plot(np.dot(S2c[1,1], np.array(np.hstack((0., U2c[0,1])))), np.dot(S2c[1,1], np.array(np.hstack((0., U2c[1,1])))), 'Color', col3, 'LineWidth', lw)
#% plot(E3c(1,:), E3c(2,:), '.', 'Color',col3,'LineWidth',2);
#% plot(E4c(1,:), E4c(2,:), '.', 'Color',col4,'LineWidth',2);
#% plot(S3c(1,1) * [0,U3c(1,1)], S3c(1,1) * [0,U3c(2,1)],'Color',col3, ...
#%     'LineWidth',2);
#% plot(S3c(2,2) * [0,U3c(1,2)], S3c(2,2) * [0,U3c(2,2)],'Color',col3, ...
#%     'LineWidth',2);
#% plot(S4c(1,1) * [0,U4c(1,1)], S4c(1,1) * [0,U4c(2,1)],'Color',col4, ...
#%     'LineWidth',2);
#% plot(S4c(2,2) * [0,U4c(1,2)], S4c(2,2) * [0,U4c(2,2)],'Color',col4,...
#%     'LineWidth',2);
plt.hold(off)
set(plt.gca, 'XLim', np.array(np.hstack((-rg, rg))), 'YLim', np.array(np.hstack((-rg, rg))), 'XTick', np.array(np.hstack((-1., 0., 1.))), 'YTick', np.array(np.hstack((-1., 0., 1.))), 'FontSize', fs, 'Box', 'on')
plt.axis(square)
#%%
#% %%
#% figure(10); clf;
#% fs = 24;
#% set(gcf,'DefaultAxesColorOrder',[0  ]'*[1 1 1]);
#% set(gcf, 'WindowStyle','normal');
#% set(gcf,'Position', [1000 400 120 80]);
#% plot(train_pPL{1,3}','LineWidth',2);
#% set(gca,'YLim',[-1,1],'YTickLabel',[],'XTickLabel',[]);
#% figure(11); clf;
#% fs = 24;
#% set(gcf,'DefaultAxesColorOrder',[0  ]'*[1 1 1]);
#% set(gcf, 'WindowStyle','normal');
#% set(gcf,'Position', [1000 500 200 80]);
#% plot(train_xPL{1,1}(1,:),'LineWidth',2);
#% set(gca,'YLim',[-1,1],'YTickLabel',[],'XTickLabel',[]);
#%
#% figure(12); clf;
#% fs = 24;
#% set(gcf,'DefaultAxesColorOrder',[0.5  ]'*[1 1 1]);
#% set(gcf, 'WindowStyle','normal');
#% set(gcf,'Position', [1000 600 200 80]);
#% plot(train_xPL{1,1}(2,:),'LineWidth',8);
#% set(gca,'YLim',[-1,1],'YTickLabel',[],'XTickLabel',[]);
#%
#% figure(13); clf;
#% fs = 24;
#% set(gcf,'DefaultAxesColorOrder',[0  ]'*[1 1 1]);
#% set(gcf, 'WindowStyle','normal');
#% set(gcf,'Position', [1000 700 200 80]);
#% plot(train_xPL{1,1}(3,:),'LineWidth',2);
#% set(gca,'YLim',[-1,1],'YTickLabel',[],'XTickLabel',[]);
#%
#% figure(14); clf;
#% fs = 24;
#% set(gcf,'DefaultAxesColorOrder',[0  ]'*[1 1 1]);
#% set(gcf, 'WindowStyle','normal');
#% set(gcf,'Position', [1000 800 200 80]);
#% plot(train_xPL{1,1}(4,:),'LineWidth',2);
#% set(gca,'YLim',[-1,1],'YTickLabel',[],'XTickLabel',[]);
#%%  energy similarities between driven response spaces
#% 
#% similarityMatrixC = zeros(Np, Np);
#% for i = 1:Np
#%     for j = i:Np
#%         similarity = ...
#%             norm((diag(sqrt(Cs{3, i})) * Cs{2,i}' * ...
#%             Cs{2,j}*diag(sqrt(Cs{3, j}))),'fro')^2 / ...
#%             (norm(Cs{1,i},'fro') * norm(Cs{1,j},'fro'));
#%         
#%         similarityMatrixC(i,j) = similarity;
#%         similarityMatrixC(j,i) = similarity;
#%     end
#% end
#% 
#% similarityMatrixR = zeros(Np, Np);
#% 
#% for i = 1:Np
#%     for j = i:Np
#%         similarity = ...
#%             norm((sqrt(SRCollectors{1, i}) * URCollectors{1,i}' * ...
#%             URCollectors{1,j}* sqrt(SRCollectors{1, j})),'fro')^2 / ...
#%             (norm(patternRs{i},'fro') * norm(patternRs{j},'fro'));
#%         
#%         similarityMatrixR(i,j) = similarity;
#%         similarityMatrixR(j,i) = similarity;
#%     end
#% end
#% %%
#% figure(22); clf;
#% fs = 24; fs1 = 24;
#% set(gcf, 'WindowStyle','normal');
#% set(gcf,'Position', [900 400 500 500]);
#% plotmat(similarityMatrixC, 0, 1, 'g');
#% for i = 1:Np
#%     for j = i:Np
#%         if similarityMatrixC(i,j) > 0.995
#%             text(i-0.1,j,num2str(similarityMatrixC(i,j),2),...
#%                 'FontSize',fs1);
#%         elseif similarityMatrixC(i,j) < 0.5
#%             text(i-0.3,j,num2str(similarityMatrixC(i,j),2),...
#%                 'Color','w','FontSize',fs1);
#%         else
#%             text(i-0.3,j,num2str(similarityMatrixC(i,j),2),...
#%                 'FontSize',fs1);
#%             
#%         end
#%     end
#% end
#% set(gca,'YTick',[1 2 3 4], 'XTick',[1 2 3 4],'FontSize',fs);
#% title(['C based similarities, \alpha = ', num2str(alpha)],...
#%     'FontSize', fs);
#% 
#% 
#% %%
#% figure(3); clf;
#% fs = 24; fs1 = 24;
#% set(gcf, 'WindowStyle','normal');
#% set(gcf,'Position', [1100 300 500 500]);
#% plotmat(similarityMatrixR, 0, 1, 'g');
#% for i = 1:Np
#%     for j = i:Np
#%         if similarityMatrixR(i,j) > 0.995
#%             text(i-0.1,j,num2str(similarityMatrixR(i,j),2),...
#%                 'FontSize',fs1);
#%         elseif similarityMatrixR(i,j) < 0.5
#%             text(i-0.3,j,num2str(similarityMatrixR(i,j),2),...
#%                 'Color','w','FontSize',fs1);
#%         else
#%             text(i-0.3,j,num2str(similarityMatrixR(i,j),2),...
#%                 'FontSize',fs1);
#%             
#%         end
#%     end
#% end
#% set(gca,'YTick',[1 2 3 4], 'XTick',[1 2 3 4],'FontSize',fs);
#% title('R based similarities', 'FontSize', fs);
#% 
#% 
#% 
#% figure(4); clf;
#% for p = 1:Np
#%     subplot(Np,2,(p-1)*2+1);
#%     plot(x_CTestPL(:,end - signalPlotLength+1:end,p)');
#%     if p == 1
#%         title('C controlled x');
#%     end
#%     subplot(Np,2,p*2);
#%     plot(patternCollectors{1,p}(:,end - signalPlotLength+1:end)', 'g'); hold on;
#%     plot(p_CTestPL(:,end - signalPlotLength+1:end,p)');
#%     hold off;
#%     if p == 1
#%         title('C controlled p');
#%     end
#%     
#% end
#% 
#% %%
#% % plotting comparisons for different alpha
#% sPL1 = zeros(5,Netsize);
#% sPL2 = zeros(5,Netsize);
#% alphas = [1 10 100 1000 10000];
#% for i = 1:5
#%     R1 =  patternRs{1};
#%     C1 =  R1 * inv(R1 + alphas(i)^(-2) * I);
#%     [U1 S1 V1] = svd(C1);
#%     sPL1(i,:) = diag(S1)';
#%     R2 =  patternRs{3};
#%     C2 =  R2 * inv(R2 + alphas(i)^(-2) * I);
#%     [U2 S2 V2] = svd(C2);
#%     sPL2(i,:) = diag(S2)';    
#% end
#% %%
#% figure(5); clf;
#% set(gcf, 'WindowStyle','normal');
#% set(gcf,'Position', [800 300 800 200]);
#% set(gcf,'DefaultAxesColorOrder',[0 0.2 0.4 0.6 0.7]'*[1 1 1]);
#% fs = 18;
#% subplot(1,2,1);
#% plot(sPL1', 'LineWidth',2);
#% title('sine (pattern 1)', 'FontSize',fs);
#% set(gca,'YTick',[0 1], 'YLim', [0 1.1], 'FontSize',fs);
#% subplot(1,2,2);
#% plot(sPL2', 'LineWidth',2);
#% title('10-periodic random (pattern 3)', 'FontSize',fs);
#% set(gca,'YTick',[0 1], 'YLim', [0 1.1], 'FontSize',fs);
#% legend('\alpha = 1', '\alpha = 10', ...
#%     '\alpha = 100', '\alpha = 1000', '\alpha = 10000' );