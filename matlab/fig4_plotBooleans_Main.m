% 26.8.2012-2 playing with 2D ellipses, check OR, AND, neg

dim = 2;
newData = 1; 
I = eye(dim);

if newData
    X = randn(dim,dim); 
    R = X * X' / dim; 
    A = R * inv(R + I);
    [Ua Sa Va] = svd(A);
    Sa(1,1) = 0.95; Sa(2,2) = 0.2;
    A = Ua * Sa * Ua';

    Y = randn(dim,dim); 
    Q = Y * Y' / dim; 
    B = Q * inv(Q + I);
    [Ub Sb Vb] = svd(B);
    Sb(1,1) = 0.8; Sb(2,2) = 0.3;
    B = Ub * Sb * Ub';
end




AandB = AND(A, B);
AorB = OR(A, B);
notA = I - A;

 
%% simple plotting 
fs = 24; lw = 3; LW = 6;
figure(1); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [800 300 900 300]);
subplot(1,3,1);
hold on;
plot([-1 1],[0 0], 'k--');
plot([0 0],[-1 1], 'k--');
plot(cos(2 * pi * (0:200)/200), sin(2 * pi * (0:200)/200), 'k' );
plot2DEllipse(A, 'r', lw, 200);
plot2DEllipse(B, 'b', lw, 200);
plot2DEllipse(AorB, 'm', LW, 200);
hold off;
set(gca, 'Box', 'on', 'FontSize',fs, 'PlotBoxAspectRatio',[1 1 1], ...
    'YTick',[-1 0 1]);
subplot(1,3,2);
hold on;
plot([-1 1],[0 0], 'k--');
plot([0 0],[-1 1], 'k--');
plot(cos(2 * pi * (0:200)/200), sin(2 * pi * (0:200)/200), 'k' );
plot2DEllipse(A, 'r', lw, 200);
plot2DEllipse(B, 'b', lw, 200);
plot2DEllipse(AandB, 'm', LW, 200);
hold off;
set(gca, 'Box', 'on', 'FontSize',fs, 'PlotBoxAspectRatio',[1 1 1], ...
    'YTick',[-1 0 1]);
subplot(1,3,3);
hold on;
plot([-1 1],[0 0], 'k--');
plot([0 0],[-1 1], 'k--');
plot(cos(2 * pi * (0:200)/200), sin(2 * pi * (0:200)/200), 'k' );
plot2DEllipse(A, 'r', lw, 200);
plot2DEllipse(notA, 'm', LW, 200);
hold off;
set(gca, 'Box', 'on', 'FontSize',fs, 'PlotBoxAspectRatio',[1 1 1], ...
    'YTick',[-1 0 1]);