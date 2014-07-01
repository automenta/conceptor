
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
TychonovAlpha = .01
#% regularizer for  W training
washoutLength = 500.
learnLength = 1000.
signalPlotLength = 20.
#%%% pattern readout learning
TychonovAlphaReadout = 0.01
#%%% C learning and testing
alpha = 1000.
CtestLength = 200.
SplotLength = 50.
#%%% morphing
morphRange = np.array(np.hstack((-2., 3.)))
morphTime = 95.
morphWashout = 500.
preMorphRecordLength = 0.
delayMorphTime = 500.
delayPlotPoints = 25.
tN = 8.
#%%% Setting patterns
patterns = np.array(np.hstack((53., 54., 10., 36.)))
#%patterns = [54 48  18 60];
#%patterns = [23 6];
#%patterns = [1 2 21 20 22 8 19 6  16 9 10 11 12];
#% 1 = sine10  2 = sine15  3 = sine20  4 = spike20
#% 5 = spike10 6 = spike7  7 = 0   8 = 1
#% 9 = rand4; 10 = rand5  11 = rand6 12 = rand7
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
    
#%% linear morphing
if linearMorphing:
    ms = np.arange(morphRange[0], (morphRange[1])+(matdiv(morphRange[1]-morphRange[0], morphTime)), matdiv(morphRange[1]-morphRange[0], morphTime))
    morphPL = np.zeros(1., morphTime)
    #% sinewave morphing
    C1 = Cs.cell[0,2]
    C2 = Cs.cell[0,3]
    x = plt.randn(Netsize, 1.)
    #% washing out
    m = ms[0]
    for i in np.arange(1., (morphWashout)+1):
        x = np.dot(np.dot(1.-m, C1)+np.dot(m, C2), np.tanh((np.dot(W, x)+Wbias)))
        
    #% morphing and recording
    preMorphPL = np.zeros(1., preMorphRecordLength)
    m = ms[0]
    for i in np.arange(1., (preMorphRecordLength)+1):
        x = np.dot(np.dot(1.-m, C1)+np.dot(m, C2), np.tanh((np.dot(W, x)+Wbias)))
        preMorphPL[0,int(i)-1] = np.dot(Wout, x)
        
    for i in np.arange(1., (morphTime)+1):
        m = ms[int(i)-1]
        x = np.dot(np.dot(1.-m, C1)+np.dot(m, C2), np.tanh((np.dot(W, x)+Wbias)))
        morphPL[0,int(i)-1] = np.dot(Wout, x)
        
    #% post morphem
    postMorphRecordLenght = preMorphRecordLength
    postMorphPL = np.zeros(1., postMorphRecordLenght)
    m = ms[int(0)-1]
    for i in np.arange(1., (postMorphRecordLenght)+1):
        x = np.dot(np.dot(1.-m, C1)+np.dot(m, C2), np.tanh((np.dot(W, x)+Wbias)))
        postMorphPL[0,int(i)-1] = np.dot(Wout, x)
        
    #% % transform to period length plotlist
    L = preMorphRecordLength+morphTime+postMorphRecordLenght
    totalMorphPL = np.array(np.hstack((preMorphPL, morphPL, postMorphPL)))
    learnPoint1 = preMorphRecordLength+np.dot(morphTime, matdiv(-morphRange[0], morphRange[1]-morphRange[0]))
    learnPoint2 = preMorphRecordLength+np.dot(morphTime, matdiv(-(morphRange[0]-1.), morphRange[1]-morphRange[0]))
    #% delay plot fingerprints computations
    delayplotMs = np.arange(morphRange[0], (morphRange[1])+(matdiv(morphRange[1]-morphRange[0], tN-1.)), matdiv(morphRange[1]-morphRange[0], tN-1.))
    delayData = np.zeros(tN, delayMorphTime)
    x0 = np.random.rand(Netsize, 1.)
    for i in np.arange(1., (tN)+1):
        x = x0
        Cmix = np.dot(1.-delayplotMs[int(i)-1], C1)+np.dot(delayplotMs[int(i)-1], C2)
        for n in np.arange(1., (morphWashout)+1):
            x = np.dot(Cmix, np.tanh((np.dot(W, x)+Wbias)))
            
        #% collect x
        for n in np.arange(1., (delayMorphTime)+1):
            x = np.dot(Cmix, np.tanh((np.dot(W, x)+Wbias)))
            delayData[int(i)-1,int(n)-1] = np.dot(Wout, x)
            
        
    fingerPrintPoints = preMorphRecordLength+matdiv(np.dot(np.arange(0., (tN-1.)+1), morphTime), tN-1.)
    fingerPrintPoints[0] = 1.
    #%%   
    plt.figure(1.)
    plt.clf
    fs = 18.
    set(plt.gcf, 'WindowStyle', 'normal')
    set(plt.gcf, 'Position', np.array(np.hstack((700., 400., 800., 266.))))
    for i in np.arange(1., (tN)+1):
        panelWidth = np.dot(1./(tN+1.), 1.-0.08)
        panelHight = 1./3.2
        panelx = np.dot(np.dot(1.-0.08, i-1.), 1./tN)+matdiv(np.dot(np.dot(1.-0.08, i-1.), 1./tN-panelWidth), tN)+0.04
        panely = 1./2.+1.5/10.
        plt.subplot('Position', np.array(np.hstack((panelx, panely, panelWidth, panelHight))))
        thisdata = delayData[int(i)-1,0:delayPlotPoints+1.]
        plt.plot(delayData[int(i)-1,0:0-1.], delayData[int(i)-1,1:], 'k.', 'MarkerSize', 1.)
        plt.hold(on)
        plt.plot(thisdata[0,0:0-1.], thisdata[0,1:], 'k.', 'MarkerSize', 20.)
        plt.hold(off)
        set(plt.gca, 'XTickLabel', np.array([]), 'YTickLabel', np.array([]), 'XLim', np.array(np.hstack((-1.4, 1.4))), 'YLim', np.array(np.hstack((-1.4, 1.4))), 'Box', 'on')
        
    plt.subplot('Position', np.array(np.hstack((0.04, 0.15, 1.-0.08, 1./2.-0.05))))
    plt.plot(totalMorphPL, 'k-', 'LineWidth', 2.)
    plt.hold(on)
    plt.plot(np.array(np.hstack((learnPoint1, learnPoint2))), np.array(np.hstack((-1., -1.))), 'k.', 'MarkerSize', 35.)
    plt.plot(fingerPrintPoints, np.dot(1.1, np.ones(1., tN)), 'kv', 'MarkerSize', 10., 'MarkerFaceColor', 'k')
    plt.hold(off)
    set(plt.gca, 'YLim', np.array(np.hstack((-1.2, 1.2))), 'XLim', np.array(np.hstack((1., 95.))), 'FontSize', fs)
    #%     figure(2); clf;
    #%     set(gcf,'DefaultAxesColorOrder',...
    #%         [0 0.2 0.4 0.6 0.7 0.8 0.9]'*[1 1 1]);
    #%     set(gcf, 'WindowStyle','normal');
    #%     set(gcf,'Position', [700 200 200 200]);
    #%     
    #%     pattPL = zeros(7,5);
    #%     for i = 1:7
    #%         pattPL(i,:) = totalMorphPL(1, ((i-1)*15+1):((i-1)*15+5));
    #%     end
    #%     plot(pattPL', 'LineWidth',4); 
    #%     set(gca,'XTick',[1 2 3 4 5], 'FontSize',fs);
    #%     


#% Cend = ((1-m)*C1 + m*C2);
#% [Uend Send Vend] = svd(Cend);
#% figure(8); clf;
#% plot(diag(Send));
#% Zend = Uend' * Vend;
#% Zend(1:5,1:5)