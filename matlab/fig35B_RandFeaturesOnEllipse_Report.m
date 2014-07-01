randstate = 0;
randstate = randstate + 1;
randn('state', randstate);

resolution = 200;
aperture = 2;
veci = 160;
I = eye(2);

X = randn(2,4);
R0 = X * X' / size(X,2);
[UR SR URt] = svd(R0);


%%

figure(1); clf;
set(gcf, 'WindowStyle','normal');
set(gcf,'Position', [900 200 400 400]);
aperture = 3.5;
b = 0.35;

        
        SR = diag([1.2 b ]);
        R = UR * SR * UR';
        
        S = zeros(2,resolution);
        for i = 1:resolution
            S(:,i) = [cos(2*pi*i/resolution), sin(2*pi*i/resolution) ]';
        end
        
%         % compute  C
%         s = diag(SR);
%         rootArgs = (aperture^2 * s - 4) ./ (4*aperture^2 * s);
%         nonNegInds = (rootArgs >= 0);
%         sC = zeros(2,1);
%         sC(nonNegInds) = 0.5 + sqrt(rootArgs(nonNegInds));
%         C = R* inv(R + aperture^(-2)* eye(2));
%         CS = C * S;
        
        
        
        % compute c
         F = S ;
        engys = sum(((UR * diag(sqrt(diag(SR))))' * F).^2,1 )  / 2;
        
        rootArgs = (aperture^2 * engys - 4) ./ (4*aperture^2 * engys);
        nonNegInds = (rootArgs >= 0);
        c = zeros(1,200);
        c(1,nonNegInds) = 0.5 + sqrt(rootArgs(1,nonNegInds));
       % c = engys ./ (engys + aperture^(-2));
        % c = engys / 4;
        Sc = S * diag(c);
        Sengys = S * diag(engys);
        
        C2 = F * diag(c) * F' / norm(F(1,:))^2;
        % C2 = 2*F * diag(c) * F' / resolution; % is identical
        
        C2S = C2 * S;
        
        
        textsize = 24;
        hold on;
        line(S(1,:)', S(2,:)', ...
            'Color', 'k', 'LineWidth', 1);
        
        [X Y] = vecLine(UR(:,1));
        line(X, Y, 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');
        [X Y] = vecLine(-UR(:,1));
        line(X, Y, 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');
        [X Y] = vecLine(UR(:,2));
        line(X, Y, 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');
        [X Y] = vecLine(-UR(:,2));
        line(X, Y, 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');
        
        plot2DEllipseThinAxes(R, 'k', 2, 200);
        
        line(Sc(1,:)', Sc(2,:)', ...
            'Color', 0.7*[0.5 1 0.5], 'LineWidth',1.5);
        line(Sengys(1,:)', Sengys(2,:)', ...
            'Color', 'm', 'LineWidth', 1.5);
        line(C2S(1,:)', C2S(2,:)', ...
            'Color', 'r', 'LineWidth', 2);
        
        hold off;
        set(gca, 'YTick', [-1 0 1],'XLim',[-1.2 1.2], ...
            'YLim', [-1.2 1.2], 'Box', 'on',...
            'XTick', [-1 0 1],'FontSize',textsize);
        
        
    
%%
