function MGseries = generateMGSequence2D(tau, ...
    incrementsperUnit, subsampleRate, samplelength, initWashoutLength) 
% generate a Mackey-Glass timeseries. File into 2-dim output series
% MGseries where the second component is just a delayed version of the
% first. The delay is tau measured in units of time of the original 
% delay differential
% equation time. 
% Each time the function is called it
% is initialized differently and will create another version of the
% timeseries. The sample rate used for the discrete approximation to 
% the delay differential equation is incrementsperUnit. Argument 
% subsampleRate is an integer n > 1 declaring only each n-th of the 
% computed values is registered into the output sequence. 
% Argument samplelength
% is the nr of generated datapoints, not length in real-valued time of the
% underlying delay differential equation. The output is a row vector,
% normalized to range [0,1].
%

genHistoryLength = tau * incrementsperUnit ;
seed = 1.2 * ones(genHistoryLength,1)+ ...
    0.2 * (rand(genHistoryLength,1)-0.5);
oldval = 1.2;
genHistory = seed;

MGseries = zeros(2,samplelength);

step = 0;
for n = 1: samplelength + initWashoutLength
    for i = 1:incrementsperUnit * subsampleRate
        step = step + 1;
        tauval = genHistory(mod(step,genHistoryLength)+1,1);
        newval = oldval + ...
            (0.2 * tauval/(1.0 + tauval^10) - 0.1 * oldval)/...
            incrementsperUnit;
        genHistory(mod(step,genHistoryLength)+1,1) = oldval;
        oldval = newval;
    end
    if n > initWashoutLength
        MGseries(:,n - initWashoutLength) = [newval; tauval];
    end
end


% normalize range
maxval = max(MGseries,[],2);
minval = min(MGseries,[],2);
MGseries = inv(diag(maxval - minval)) * ...
    (MGseries - repmat(minval,1,samplelength));
    

