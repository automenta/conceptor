function [minM, maxM] = plotmatrix(M, colormapindicator)
% plots a grayscale image of matrix M
% Argument colormap is string 'g' or 'c'. When the first, linear
% grayscale is used, when the second, Matlab's standard colorscheme "Jet"
% is used.
%
% created HJaeger May 9 2008
% Rev 1 HJaeger Feb 14 2011 (adding color argument)

% normalize M to range 0 - 1
minM = min(min(M));
maxM = max(max(M));
if minM == maxM
    error('plotmatrix function is too stupid to plot matrix with all identical values');
end
M = (M - minM) / (maxM - minM);

if colormapindicator == 'g'
    colormap(gray(128));
elseif colormapindicator == 'c'
    colormap(jet(128));
else
    error('color argument must be string g or string c');
end
image(min(ceil(M * 128), 128));
%title(['Min = ' num2str(minM) '  Max = ' num2str(maxM)]);