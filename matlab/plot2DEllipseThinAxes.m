
function [] = plot2DEllipseThinAxes(A, colorstring, linewidth, resolution)
% plots a 2D ellipse centered on 0 whose shape matrix is given by the 
% positive semidefinite matrix A. colorstring is a Matlab color symbol in string
% format. resolution is number of points used to draw ellipse.


circPoints = zeros(resolution,2);
for i = 1:resolution
    circPoints(i,:) = [cos(2*pi*i/resolution), sin(2*pi*i/resolution) ];
end
circPoints = [circPoints; circPoints(1,:)];
ellPoints = circPoints * A';

[U S Ut] = svd(A);

hold on;
[X Y] = vecLine(U(:,1) * S(1,1));
line(X, Y, 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');

[X Y] = vecLine(U(:,2) * S(2,2));
line(X, Y, 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');

line(ellPoints(:,1), ellPoints(:,2), ...
    'Color', colorstring, 'LineWidth', linewidth);
hold off;

function [X Y] = vecLine(x)
% x is 2-dim column vec
X = [0 x(1)]; Y = [0 x(2)];

