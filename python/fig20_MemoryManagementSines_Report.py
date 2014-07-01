
import numpy as np
import scipy
import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

#%%%% demo: incremental loading of patterns into reservoir
#% set figure window to 1 x 1 panels
#set(0., 'DefaultFigureWindowStyle', 'docked')
#%%% Experiment basic setup
randstate = 1.
newNets = 1.
newSystemScalings = 1.
learnType = 2.
#% set to 1 if every new pattern is to be
#% entirely loaded into "virgin" reservoir
#% space (batch offline); set to 2 if new
#% patterns exploit already existing pattern
#% generation(batch offline); set to 3 if new
#% patterns exploit already existing pattern
#% generation (online adaptive using LMS);
#%%% Setting system params
Netsize = 100.
#% network size
NetSR = 1.5
#% spectral radius
NetinpScaling = 1.5
#% scaling of pattern feeding weights
BiasScaling = 0.25
#% size of bias
#%%% incremental loading learning (batch offline)
washoutLength = 200.
learnLength = 1000.
#%%% incremental loading learning (online adaptive)
adaptLength = 1500.
adaptRate = 0.02
errPlotSmoothRate = 0.01
#% between 0 and 1, smaller = more smoothing
#%%% pattern readout learning
TychonovAlphaReadout = 0.001
#%%% C testing
testLength = 400.
testWashout = 400.
#%%% plotting
nPlotSingVals = 100.
#% how many singular values are plotted
signalPlotLength = 20.
#%%% Setting patterns
#%patterns = [51 52 53 54 32 33]; aperture = 10;
#%patterns = [1 2 9 11 12 1 2 9 44 39 40 13  34 16 17 36]; aperture = 1000;
#%patterns = [1 3 9 11 12 14 35 19 8 18 37 ]; aperture = 1000;
#%patterns = [1 3 9 11 12  ]; aperture = 1000;
#%patterns = [39 40 41 42 43 1 37 44 45 46 2]; aperture = 10;
#%patterns = [55 56 57 58 51 52 53 54 32 33 38 47 48 49 50 59]; 
patts = cell(1., 16.)
for i in np.arange(1., 17.0):
    patts.cell[int(i)-1] = lambda n: np.sin(matdiv(np.dot(2.*np.pi, n), np.dot(np.sqrt(2.)*3., matixpower(1.1, i))))
    
pattperms = randperm(16.)
#%pattperms = 1:16;
aperture = 3.
#%patterns = [51 52 53 54]; aperture = 10;
#% 1 = sine10  2 = sine15  3 = sine20  4 = spike20
#% 5 = spike10 6 = spike7  7 = 0   8 = 1
#% 9 = rand4; 10 = rand5  11 = rand6 12 = rand7
#% 13 = rand8 14 = sine10range01 15 = sine10rangept5pt9
#% 16 = rand3 17 = rand9 18 = rand10 19 = 0.8 20 = sineroot27
#% 21 = sineroot19 22 = sineroot50 23 = sineroot75
#% 24 = sineroot10 25 = sineroot110 26 = sineroot75tenth
#% 27 = sineroots20plus40  28 = sineroot75third
#% 29 = sineroot243  30 = sineroot150  31 = sineroot200
#% 32 = sine10pt587352723 33 = sine10pt10.387352723
#% 34 = rand7  35 = sine12  36 = rand5  37 = sine11
#% 38 = sine10pt17352723  39 = sine5 40 = sine6
#% 41 = sine7 42 = sine8  43 = sine9 44 = sine12
#% 45 = sine13  46 = sine14  47 = sine10.8342522
#% 48 = sine11.8522311  49 = sine12.5223223  50 = sine13.1900453
#% 51 = sine7.1900453  52 = sine7.9004531  53 = sine8.4900453
#% 54 = sine9.1900453 55 = sine5.19004  56 = sine5.8045
#% 57 = sine6.49004 58 = sine6.9004 59 = sine13.9004
#%%% Initializations
plt.randn('state', randstate)
np.random.rand('twister', randstate)
I = np.eye(Netsize)
Np = length(patts)
#% Create raw weights
if newNets:
    if Netsize<=20.:
        Netconnectivity = 1.
    else:
        Netconnectivity = 10./Netsize
        
    
    WRaw = generate_internal_weights(Netsize, Netconnectivity)
    WinRaw = plt.randn(Netsize, 1.)
    WbiasRaw = plt.randn(Netsize, 1.)


#% Scale raw weights and initialize weights
if newSystemScalings:
    W = np.dot(NetSR, WRaw)
    Win = np.dot(NetinpScaling, WinRaw)
    Wbias = np.dot(BiasScaling, WbiasRaw)


#% Set pattern handles
#% pattHandles;
#%% incremental learning
if learnType == 1.:
    patternCollectors = cell(1., Np)
    pPL = cell(1., Np)
    sizesCall = np.zeros(1., Np)
    Calls = cell(1., Np)
    sizesCap = np.zeros(1., Np)
    startxs = np.zeros(Netsize, Np)
    Call = np.zeros(Netsize, Netsize)
    D = np.zeros(Netsize, Netsize)
    nativeCs = cell(1., Np)
    for p in np.arange(1., (Np)+1):
        patt = patts.cell[int(pattperms[int(p)-1])-1]
        #% current pattern generator
        #% drive reservoir in "virgin" subspace with current pattern
        xOldCollector = np.zeros(Netsize, learnLength)
        pCollector = np.zeros(1., learnLength)
        x = np.zeros(Netsize, 1.)
        G = I-Call
        for n in np.arange(1., (washoutLength+learnLength)+1):
            u = patt[int(n)-1]
            #% pattern input
            xOld = x
            x = np.dot(G, np.tanh((np.dot(W, x)+np.dot(Win, u)+Wbias)))
            if n > washoutLength:
                xOldCollector[:,int((n-washoutLength))-1] = xOld
                pCollector[0,int((n-washoutLength))-1] = u
            
            
            
        patternCollectors.cell[0,int(p)-1] = pCollector
        pPL.cell[0,int(p)-1] = pCollector[0,0:signalPlotLength]
        R = matdiv(np.dot(xOldCollector, xOldCollector.conj().T), learnLength+1.)
        Cnative = np.dot(R, linalg.inv((R+I)))
        nativeCs.cell[0,int(p)-1] = Cnative
        startxs[:,int(p)-1] = x
        #% compute D increment
        Dtargs = np.dot(Win, pCollector)
        F = NOT(Call)
        Dargs = xOldCollector
        Dinc = matdiv(np.dot(np.dot(linalg.pinv((matdiv(np.dot(Dargs, Dargs.conj().T), learnLength)+np.dot(matixpower(aperture, -2.), I))), Dargs), Dtargs.conj().T), learnLength).conj().T
        #% update D and Call
        D = D+Dinc
        Cap = PHI(Cnative, aperture)
        Call = OR(Call, Cap)
        Calls.cell[0,int(p)-1] = Call
        [Ux, Sx, Vx] = plt.svd(Call)
        sizesCall[0,int(p)-1] = np.mean(np.diag(Sx))
        [Ux, Sx, Vx] = plt.svd(Cap)
        sizesCap[0,int(p)-1] = np.mean(np.diag(Sx))
        
elif learnType == 2.:
    #% extension exploiting redundancies
    patternCollectors = cell(1., Np)
    pPL = cell(1., Np)
    sizesCall = np.zeros(1., Np)
    sizesCap = np.zeros(1., Np)
    Calls = cell(1., Np)
    startxs = np.zeros(Netsize, Np)
    Call = np.zeros(Netsize, Netsize)
    D = np.zeros(Netsize, Netsize)
    nativeCs = cell(1., Np)
    for p in np.arange(1., (Np)+1):
        patt = patts.cell[int(pattperms[int(p)-1])-1]
        #% current pattern generator
        #% drive reservoir with current pattern
        xOldCollector = np.zeros(Netsize, learnLength)
        pCollector = np.zeros(1., learnLength)
        x = np.zeros(Netsize, 1.)
        for n in np.arange(1., (washoutLength+learnLength)+1):
            u = patt[int(n)-1]
            #% pattern input
            xOld = x
            x = np.tanh((np.dot(W, x)+np.dot(Win, u)+Wbias))
            if n > washoutLength:
                xOldCollector[:,int((n-washoutLength))-1] = xOld
                pCollector[0,int((n-washoutLength))-1] = u
            
            
            
        patternCollectors.cell[0,int(p)-1] = pCollector
        pPL.cell[0,int(p)-1] = pCollector[0,0:signalPlotLength]
        R = matdiv(np.dot(xOldCollector, xOldCollector.conj().T), learnLength+1.)
        Cnative = np.dot(R, linalg.inv((R+I)))
        nativeCs.cell[0,int(p)-1] = Cnative
        startxs[:,int(p)-1] = x
        #% compute D increment
        Dtargs = np.dot(Win, pCollector)-np.dot(D, xOldCollector)
        F = NOT(Call)
        Dargs = np.dot(F, xOldCollector)
        Dinc = matdiv(np.dot(np.dot(linalg.pinv((matdiv(np.dot(Dargs, Dargs.conj().T), learnLength)+np.dot(matixpower(aperture, -2.), I))), Dargs), Dtargs.conj().T), learnLength).conj().T
        #% update D and Call
        D = D+Dinc
        Cap = PHI(Cnative, aperture)
        Call = OR(Call, Cap)
        Calls.cell[0,int(p)-1] = Call
        [Ux, Sx, Vx] = plt.svd(Call)
        sizesCall[0,int(p)-1] = np.mean(np.diag(Sx))
        [Ux, Sx, Vx] = plt.svd(Cap)
        sizesCap[0,int(p)-1] = np.mean(np.diag(Sx))
        
    
elif learnType == 3.:
    #%(online adaptive)
    errPL = np.zeros(Np, adaptLength)
    xAdaptPL = np.zeros(3., adaptLength, Np)
    patternCollectors = cell(1., Np)
    pPL = cell(1., Np)
    sizesCall = np.zeros(1., Np)
    sizesCap = np.zeros(1., Np)
    Calls = cell(1., Np)
    startxs = np.zeros(Netsize, Np)
    Call = np.zeros(Netsize, Netsize)
    D = np.zeros(Netsize, Netsize)
    nativeCs = cell(1., Np)
    for p in np.arange(1., (Np)+1):
        patt = patts.cell[int(pattperms[int(p)-1])-1]
        #% current pattern generator
        xOldCollector = np.zeros(Netsize, adaptLength)
        pCollector = np.zeros(1., adaptLength)
        x = np.zeros(Netsize, 1.)
        F = NOT(Call)
        err = 1.
        for i in np.arange(1., (washoutLength+adaptLength)+1):
            u = patt[int(i)-1]
            xOld = x
            x = np.tanh((np.dot(W, x)+np.dot(Win, u)+Wbias))
            if i > washoutLength:
                xAdaptPL[:,int((i-washoutLength))-1,int(p)-1] = x[0:3.,0]
                xOldCollector[:,int((i-washoutLength))-1] = xOld
                pCollector[0,int((i-washoutLength))-1] = u
                thisErr = np.mean(((np.dot(D, xOld)-np.dot(Win, u))**2.))
                err = np.dot(1.-errPlotSmoothRate, err)+np.dot(errPlotSmoothRate, thisErr)
                errPL[int(p)-1,int((i-washoutLength))-1] = err
                D = D+np.dot(np.dot(adaptRate, np.dot(np.dot(Win, u)-np.dot(D, xOld), xOld.conj().T)-np.dot(matixpower(aperture, -1./2.), D)), F)
            
            
            
        pPL.cell[0,int(p)-1] = pCollector[0:signalPlotLength]
        R = matdiv(np.dot(xOldCollector, xOldCollector.conj().T), adaptLength+1.)
        Cnative = np.dot(R, linalg.inv((R+I)))
        nativeCs.cell[0,int(p)-1] = Cnative
        Cap = PHI(Cnative, aperture)
        startxs[:,int(p)-1] = x
        Call = OR(Call, Cap)
        Calls.cell[0,int(p)-1] = Call
        [Ux, Sx, Vx] = plt.svd(Call)
        sizesCall[0,int(p)-1] = np.mean(np.diag(Sx))
        [Ux, Sx, Vx] = plt.svd(Cap)
        sizesCap[0,int(p)-1] = np.mean(np.diag(Sx))
        
    

#%% learn readouts
#% drive network again with all patterns, each run shielded by corresponding
#% C.
allTrainArgs = np.zeros(Netsize, np.dot(Np, learnLength))
allTrainOuts = np.zeros(1., np.dot(Np, learnLength))
for p in np.arange(1., (Np)+1):
    patt = patts.cell[int(pattperms[int(p)-1])-1]
    #% current pattern generator
    C = PHI(nativeCs.cell[0,int(p)-1], aperture)
    xCollector = np.zeros(Netsize, learnLength)
    pCollector = np.zeros(1., learnLength)
    x = startxs[:,int(p)-1]
    for n in np.arange(1., (washoutLength+learnLength)+1):
        u = patt[int(n)-1]
        #% pattern input
        x = np.dot(C, np.tanh((np.dot(W, x)+np.dot(Win, u)+Wbias)))
        #%x = tanh(Wstar * x + Win * u + Wbias);
        if n > washoutLength:
            xCollector[:,int((n-washoutLength))-1] = x
            pCollector[0,int((n-washoutLength))-1] = u
        
        
        
    allTrainArgs[:,int(np.dot(p-1., learnLength)+1.)-1:np.dot(p, learnLength)] = xCollector
    allTrainOuts[0,int(np.dot(p-1., learnLength)+1.)-1:np.dot(p, learnLength)] = pCollector
    
Wout = np.dot(np.dot(linalg.inv((np.dot(allTrainArgs, allTrainArgs.conj().T)+np.dot(TychonovAlphaReadout, np.eye(Netsize)))), allTrainArgs), allTrainOuts.conj().T).conj().T
#% training error
NRMSE_readout = nrmse(np.dot(Wout, allTrainArgs), allTrainOuts)
np.disp(sprintf('NRMSE readout relearn: %g', NRMSE_readout))
#% % test with C
x_TestPL = np.zeros(5., testLength, Np)
p_TestPL = np.zeros(1., testLength, Np)
for p in np.arange(1., (Np)+1):
    C = PHI(nativeCs.cell[0,int(p)-1], aperture)
    #%x = startxs(:,p);
    x = plt.randn(Netsize, 1.)
    for n in np.arange(1., (testWashout+testLength)+1):
        x = np.dot(C, np.tanh((np.dot(W, x)+np.dot(D, x)+Wbias)))
        if n > testWashout:
            x_TestPL[:,int((n-testWashout))-1,int(p)-1] = x[0:5.,0]
            p_TestPL[:,int((n-testWashout))-1,int(p)-1] = np.dot(Wout, x)
        
        
        
    
#%% plot
#% optimally align C-reconstructed readouts with drivers for nice plots
test_pAligned_PL = cell(1., Np)
test_xAligned_PL = cell(1., Np)
NRMSEsAligned = np.zeros(1., Np)
for p in np.arange(1., (Np)+1):
    intRate = 20.
    thisDriver = pPL.cell[0,int(p)-1]
    thisOut = p_TestPL[0,:,int(p)-1]
    thisDriverInt = interp1(np.arange(1., (signalPlotLength)+1).conj().T, thisDriver.conj().T, np.arange(1., (signalPlotLength)+(1./intRate), 1./intRate).conj().T, 'spline').conj().T
    thisOutInt = interp1(np.arange(1., (testLength)+1).conj().T, thisOut.conj().T, np.arange(1., (testLength)+(1./intRate), 1./intRate).conj().T, 'spline').conj().T
    L = matcompat.size(thisOutInt, 2.)
    M = matcompat.size(thisDriverInt, 2.)
    phasematches = np.zeros(1., (L-M))
    for phaseshift in np.arange(1., (L-M)+1):
        phasematches[0,int(phaseshift)-1] = linalg.norm((thisDriverInt-thisOutInt[0,int(phaseshift)-1:phaseshift+M-1.]))
        
    [maxVal, maxInd] = matcompat.max((-phasematches))
    test_pAligned_PL.cell[0,int(p)-1] = thisOutInt[0,int(maxInd)-1:maxInd+np.dot(intRate, signalPlotLength)-1.:intRate]
    coarseMaxInd = np.ceil(matdiv(maxInd, intRate))
    test_xAligned_PL.cell[0,int(p)-1] = x_TestPL[:,int(coarseMaxInd)-1:coarseMaxInd+signalPlotLength-1.,int(p)-1]
    NRMSEsAligned[0,int(p)-1] = nrmse(test_pAligned_PL.cell[0,int(p)-1], pPL.cell[0,int(p)-1])
    
#%%
#% figure(1); clf;
#% fs = 18; fstext = 18;
#% % set(gcf,'DefaultAxesColorOrder',[0  0.4 0.65 0.8]'*[1 1 1]);
#%  set(gcf, 'WindowStyle','normal');
#% 
#% set(gcf,'Position', [600 100 1000 800]);
#% for p = 1:Np
#%     if p <= 8
#%         thispanel = (p-1)*4+2;
#%     else
#%         thispanel = (p-9)*4+4;
#%     end
#%     subplot(8,4,thispanel);
#%     plot(test_pAligned_PL{1,p}, 'LineWidth',6,'Color',0.85*[1 1 1]); hold on;
#%     plot(pPL{1,p},'k','LineWidth',1); 
#%     rectangle('Position', [0.1,-0.95,7,0.7],'FaceColor','w',...
#%         'LineWidth',1);
#%      text(1,-0.6,num2str(NRMSEsAligned(1,p),2),...
#%         'Color','k','FontSize',fstext, 'FontWeight', 'bold');
#%     
#%     hold off;
#%     if p == 1 || p == 9
#%         title('driver and y','FontSize',fs);
#%     end
#%     if p ~= Np && p ~= 8
#%         set(gca, 'XTickLabel',[]);
#%     end
#%     set(gca, 'YLim',[-1,1], 'FontSize',fs, 'Box', 'on');
#%     
#%     
#%     if p <= 8
#%         thispanel = (p-1)*4+1;
#%     else
#%         thispanel = (p-9)*4+3;
#%     end
#%     subplot(8,4,thispanel);
#%     Call = Calls{1, p};
#%     [Ux Sx Vx] = svd(Call);
#%     diagSx = diag(Sx);
#%     hold on;
#%     %area(diagSx(1:nPlotSingVals,1),'LineWidth',4,'Color',0.35*[1 1 1] );
#%     area(diagSx(1:nPlotSingVals,1),'FaceColor', 0.7*[1 1 1]);
#%     if p == 1
#%     rectangle('Position', [1,0.02,28,0.4],'FaceColor','w',...
#%         'LineWidth',1);
#%     else
#%         rectangle('Position', [1,0.02,28,0.4],'FaceColor','w',...
#%         'LineWidth',1);
#%     end
#%     text(4,0.2,num2str(sizesCall(1,p),2), ...
#%         'Color','k','FontSize',fstext, 'FontWeight', 'bold');
#%     rectangle('Position', [57,0.4,35,0.4],'FaceColor','w',...
#%         'LineWidth',1);
#%      text(60,0.6,['j = ', num2str(p)],...
#%         'Color','k','FontSize',fstext, 'FontWeight', 'bold');
#% %     rectangle('Position', [50,0.02,28,0.4],'FaceColor','w',...
#% %         'LineWidth',1);
#% %     text(52,0.2,num2str(sum(sizesCap(1,1:p)),2),...
#% %         'Color','k','FontSize',fstext, 'FontWeight', 'bold');
#%     hold off;
#%     if p ~= Np && p ~= 8
#%         set(gca,'YLim',[0,1], 'Ytick', [0 1], ...
#%             'XLim', [0,nPlotSingVals], 'Xtick', [],'FontSize',fs);
#%     else
#%         set(gca,'YLim',[0,1], 'Ytick', [0 1], ...
#%             'XLim', [0,nPlotSingVals], 'FontSize',fs);
#%     end
#%     set(gca, 'Box', 'on');
#%     if p == 1 || p == 9
#%         title('used space','FontSize',fs);
#%     end
#% end
#%%
plt.figure(1.)
plt.clf
fs = 16.
fstext = 16.
#% set(gcf,'DefaultAxesColorOrder',[0  0.4 0.65 0.8]'*[1 1 1]);
set(plt.gcf, 'WindowStyle', 'normal')
set(plt.gcf, 'Position', np.array(np.hstack((100., 100., 1500., 600.))))
for p in np.arange(1., (Np)+1):
    if p<=8.:
        thispanel = (p-1.)*2.+1.
    else:
        thispanel = (p-9.)*2.+2.
        
    
    plt.subplot(4., 4., p)
    Call = Calls.cell[0,int(p)-1]
    [Ux, Sx, Vx] = plt.svd(Call)
    diagSx = np.diag(Sx)
    plt.hold(on)
    area((2.*diagSx[0:nPlotSingVals,0]-1.), (-1.), 'FaceColor', (1.*np.array(np.hstack((1., .6, .6)))))
    plt.plot(np.arange(5., 105.0, 5.), test_pAligned_PL.cell[0,int(p)-1], 'LineWidth', 10., 'Color', (1.*np.array(np.hstack((.5, 1., .5)))))
    plt.hold(on)
    plt.plot(np.arange(5., 105.0, 5.), pPL.cell[0,int(p)-1], 'k', 'LineWidth', 2.)
    rectangle('Position', np.array(np.hstack((72., -0.85, 24., 0.5))), 'FaceColor', 'w', 'LineWidth', 1.)
    plt.text(73., (-0.6), num2str(NRMSEsAligned[0,int(p)-1], 2.), 'Color', 'k', 'FontSize', fstext, 'FontWeight', 'bold')
    rectangle('Position', np.array(np.hstack((72., 0.35, 25., 0.5))), 'FaceColor', 'w', 'LineWidth', 1.)
    plt.text(75., 0.65, np.array(np.hstack(('j = ', num2str(p)))), 'Color', 'k', 'FontSize', fstext, 'FontWeight', 'bold')
    rectangle('Position', np.array(np.hstack((3., -0.85, 20., 0.5))), 'FaceColor', 'w', 'LineWidth', 1.)
    plt.text(6., (-0.6), num2str(sizesCall[0,int(p)-1], 2.), 'Color', 'k', 'FontSize', fstext, 'FontWeight', 'bold')
    plt.hold(off)
    if p<=12.:
        set(plt.gca, 'XTickLabel', np.array([]))
    else:
        set(plt.gca, 'Xtick', np.array(np.hstack((1., 50., 100.))), 'XTickLabel', np.array(np.hstack((cellarray(np.hstack(('1'))), cellarray(np.hstack(('10'))), cellarray(np.hstack(('20')))))))
        
    
    if not_rename(np.logical_or(np.logical_or(np.logical_or(p == 1., p == 5.), p == 9.), p == 13.)):
        set(plt.gca, 'YTickLabel', np.array([]))
    
    
    set(plt.gca, 'YLim', np.array(np.hstack((-1., 1.))), 'XLim', np.array(np.hstack((1., 100.))), 'FontSize', fs, 'Box', 'on')
    
#%%
if learnType == 3.:
    plt.figure(2.)
    plt.clf
    for p in np.arange(1., (Np)+1):
        plt.subplot(Np, 1., p)
        plt.plot(np.log10(errPL[int(p)-1,:]))
        if p == 1.:
            plt.title('log10 se')
        
        
        

