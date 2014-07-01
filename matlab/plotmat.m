function [minM, maxM] = plotmat(M, lowest, highest, colormapindicator)
% Plots an image of real-valued matrix M.
% Arguments lowest, highest are reals with lowest < highest indicating the
% value range in M that is used. Values in M that are lower than lowest or
% higher than highest are truncated to lowest / highest.
% Argument colormap is string 'g', 'ginv', or 'c'. When the first, linear
% grayscale is used (white = maximal valus), when the second, inverse
% linear grayscale is used (white = minimal value); when the third, 
% Matlab's standard colorscheme "Jet"  is used.
%
% Outputs minM, maxM return extreme values in M
%
% created HJaeger May 9 2008
% Rev 1 HJaeger Feb 14 2011 (adding color argument)

% truncate M to range [lowest highest]
minM = min(min(M)); 
maxM = max(max(M));
M = min(M, highest);
M = max(M, lowest);
% scale/shift M to range [0 1]
M = (M - lowest) / (highest - lowest);

if strcmp(colormapindicator,'g')
    colormap(gray(128));
    image(min(ceil(M * 128), 128));
elseif strcmp(colormapindicator, 'ginv')
    colormap(gray(128));
    image(min(ceil(- M * 128 + 128), 128));
elseif strcmp(colormapindicator, 'c')
    colormap(jet(128));
    image(min(ceil(M * 128), 128));
else
    error('color argument must be string g, ginv or string c');
end
%title(['Min = ' num2str(minM) '  Max = ' num2str(maxM)]);