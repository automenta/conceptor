function Henonseries = generateHenonSequence2D( ...
    samplelength, initWashoutLength)
% generate a Henon timeseries. Output Henonseries is 2-dim state
% sequence of Henon attractor.
% Each time the function is called it
% is initialized differently and will create another version of the
% timeseries.
% Argument samplelength
% is the nr of generated datapoints. The output is a matrix of size
% 2 x samplelength, each row is
% normalized to range [0,1].
%

% initialize Henon  state with a little random component
hs = [1.2677 -0.0278]' + 0.01 * randn(2,1);
a = 1.4; b =  0.3;
Henonseries = zeros(2,samplelength);
for n = 1:initWashoutLength
    hs = [hs(2) + 1 - a*hs(1)^2; b * hs(1)];
end
for n = 1: samplelength
    
    hs =  [hs(2) + 1 - a*hs(1)^2; b * hs(1)];
    
    Henonseries(:,n) = hs;
end

% normalize range

maxval = max(Henonseries,[],2);
minval = min(Henonseries,[],2);
Henonseries = inv(diag(maxval - minval)) * ...
    (Henonseries - repmat(minval,1,samplelength));