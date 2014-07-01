function Lorenzseries = generateLorenzSequence2D( ...
    incrementsperUnit, subsampleRate, samplelength, initWashoutLength) 
% generate a Lorenz timeseries. Output Roesslerseries is 2-dim, with the
% x component of the 3 dim Lorenz signal in its first and the z component
% in its second place. 
% Each time the function is called it
% is initialized differently and will create another version of the
% timeseries. Lorenzseries is 1-dim, giving the first component of the 
% 3-dim Lorenz signal. The sample rate used for the discrete 
% approximation to 
% the  differential equation is incrementsperUnit. Argument 
% subsampleRate is an integer n > 1 declaring only each n-th of the 
% computed values is registered into the output sequence. 
% Argument samplelength
% is the nr of generated datapoints, not length in real-valued time of the
% underlying differential equation. The output is a row vector,
% normalized to range [0,1].
%

% initialize Lorenz  state with a little random component
ls = [10.036677794959058; 9.98674414052542; 
    29.024692318601613] + 0.01 * randn(3,1);
sigma = 10.0; b =   8.0/3; r = 28.0; 
delta = 1 / incrementsperUnit; % length of discrete approximation update interval
Lorenzseries = zeros(2,samplelength);
for n = 1:initWashoutLength
    ls = ls + delta * [sigma * (ls(2,1)-ls(1,1));...
            r * ls(1,1) - ls(2,1) - ls(1,1)*ls(3,1);... 
            ls(1,1) * ls(2,1) - b * ls(3,1)];
end
for n = 1: samplelength
    for k = 1:subsampleRate
        ls = ls + delta * [sigma * (ls(2,1)-ls(1,1));...
                r * ls(1,1) - ls(2,1) - ls(1,1)*ls(3,1);... 
                ls(1,1) * ls(2,1) - b * ls(3,1)];
    end
    Lorenzseries(:,n) = [ls(1); ls(3)];
end
% normalize range
maxval = max(Lorenzseries,[],2);
minval = min(Lorenzseries,[],2);
Lorenzseries = inv(diag(maxval - minval)) * ...
    (Lorenzseries - repmat(minval,1,samplelength));
