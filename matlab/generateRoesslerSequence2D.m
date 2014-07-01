function Roesslerseries = generateRoesslerSequence2D( ...
    incrementsperUnit, subsampleRate, samplelength, initWashoutLength) 
% generate a Roessler timeseries. Output Roesslerseries is 2-dim, with the
% y component of the 3 dim Roessler signal in its first and the z component
% in its second place. 
% Each time the function is called it
% is initialized differently and will create another version of the
% timeseries. Rossslerseris is 1-dim, giving the second component of the 
% 3-dim Roessler signal. The sample rate used for the discrete 
% approximation to 
% the  differential equation is incrementsperUnit. Argument 
% subsampleRate is an integer n > 1 declaring only each n-th of the 
% computed values is registered into the output sequence. 
% Argument samplelength
% is the nr of generated datapoints, not length in real-valued time of the
% underlying differential equation. The output is a row vector,
% normalized to range [0,1].
%

% initialize Roessler  state with a little random component
rs = [0.5943 -2.2038 0.0260]' + 0.01 * randn(3,1);
a = 0.2; b =   0.2; c = 8.0; 
delta = 1 / incrementsperUnit; % length of discrete approximation update interval
Roesslerseries = zeros(2,samplelength);
for n = 1:initWashoutLength
    rs = rs + delta * [-(rs(2) + rs(3)); ...
        rs(1) + a * rs(2); b + rs(1)*rs(3) - c * rs(3)];
end
for n = 1: samplelength
    for k = 1:subsampleRate
        rs = rs + delta *[-(rs(2) + rs(3)); ...
        rs(1) + a * rs(2); b + rs(1)*rs(3) - c * rs(3)];
    end
    Roesslerseries(:,n) = [rs(1); rs(2)];
end

% normalize range

maxval = max(Roesslerseries,[],2);
minval = min(Roesslerseries,[],2);
Roesslerseries = inv(diag(maxval - minval)) * ...
    (Roesslerseries - repmat(minval,1,samplelength));