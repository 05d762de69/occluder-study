function APR_shape_completion_demo2()
% -------------------------------------------------------------------------
% Shape completion driven by 2-D KDE and multi-basis regression line
%
% C(t)=A1+t*v + [Σ a_k φ_k(t)] n   ,  n ⟂ v
% φ_k(t)=t^k (1-t)^k   (k = 1,2,3)     → allows 3 bumps
% -------------------------------------------------------------------------

    clear; clc; close all;

    % ---------------- USER-TUNABLE PARAMETERS ----------------------------
    Npts      = 15000;   % number of KDE-biased samples
    alphaPDF  = 2.5;    % sharpen exponent for KDE probabilities
    expPowerZ = 3;      % weight ∝ (intensity)^expPowerZ
    % ---------------------------------------------------------------------

    %% 0) Load data --------------------------------------------------------
    S = load('shapes_0056.mat','shapes');
    H = load('heatmap_0056.mat','normalized_heatmap');
    shape = S.shapes(1);

    silhouette = shape.silhouette;
    occluder   = shape.occluder;
    interPts   = shape.intersection_points;
    heatmap    = H.normalized_heatmap;                 % 227×227
    [Hrows,Hcols] = size(heatmap);

    %% 1) Bounding-box rescale (flipped-Y) ---------------------------------
    margin = 5;
    allX = [silhouette(:,1); occluder(:,1)];
    allY = [silhouette(:,2); occluder(:,2)];
    minX = floor(min(allX)-margin);   maxX = ceil(max(allX)+margin);
    minY = floor(min(allY)-margin);   maxY = ceil(max(allY)+margin);
    wBB  = maxX-minX+1;               hBB  = maxY-minY+1;
    mapXY = @(P)[ (P(:,1)-minX)*(Hcols/wBB), (maxY-P(:,2))*(Hrows/hBB) ];

    silhouette_227  = mapXY(silhouette);
    occluder_227    = mapXY(occluder);
    inter_227       = mapXY(interPts);

    A1 = inter_227(1,:);    % anchor 1
    A2 = inter_227(2,:);    % anchor 2

    %% 2) Occluder mask ----------------------------------------------------
    mask = poly2mask(occluder_227(:,1), occluder_227(:,2), Hrows, Hcols);

    %% 3) Build 2-D data set + weights inside occluder --------------------
    [xx,yy] = meshgrid(1:Hcols,1:Hrows);
    xf = xx(:);  yf = yy(:);  zf = heatmap(:);          % z = intensity
    ok = (zf>0) & mask(sub2ind([Hrows,Hcols],yf,xf));

    XY  = [xf(ok), yf(ok)];       % coordinates (n×2)
    w   = zf(ok).^expPowerZ;      % weights
    w   = w / sum(w);

    %% 4) KDE evaluation at each data point -------------------------------
    % MATLAB's mvksdensity supports weights (R2023b+).  If absent, remove w.
    KDEvals = mvksdensity(XY, XY, 'Weights', w);   % returns n×1 PDF estimates
    KDEvals = KDEvals / sum(KDEvals);              % normalise

    %% 5) Draw KDE-biased samples -----------------------------------------
    idx = randsample(size(XY,1), Npts, true, KDEvals.^alphaPDF);
    S2  = XY(idx,:);            % Npts×2 samples
    wS  = KDEvals(idx);         % their (unsharpened) densities
    wS  = wS / sum(wS);         % weights for regression

    %% 6) Multi-basis regression line  (3 bumps) ---------------------------
    v = A2 - A1;                L2 = dot(v,v);
    n = [-v(2), v(1)];          n = n / norm(n);

    % projection of each sample onto baseline [0..1]
    tS = max(0, min(1, sum((S2 - A1).*v, 2) / L2));
    dev= sum((S2 - (A1 + tS.*v)).*n, 2);

    % basis functions  φ1,φ2,φ3
    Phi = [tS.*(1-tS), ...
           (tS.^2).*(1-tS).^2, ...
           (tS.^3).*(1-tS).^3];      % n×3

    % weighted least-squares:   a = (Φᵀ W Φ)⁻¹ Φᵀ W dev
    W   = diag(wS);
    Acoef = (Phi' * W * Phi) \ (Phi' * W * dev);   % 3×1

    fprintf('Offset coefficients  a = [% .4f  % .4f  % .4f]\\n',Acoef);

    % build curve
    tf   = linspace(0,1,600)';
    base = A1 + tf.*v;
    phi1 = tf .* (1-tf);
    phi2 = (tf.^2) .* (1-tf).^2;
    phi3 = (tf.^3) .* (1-tf).^3;
    offset = Acoef(1)*phi1 + Acoef(2)*phi2 + Acoef(3)*phi3;
    curve  = base + offset .* n;

    %% 7) Visualisation ----------------------------------------------------
    figure('Name','KDE-driven multi-bump regression','Color','w');
    imagesc(heatmap); axis image xy; colormap jet; hold on;
    title('Anchor-to-anchor line pulled by 2-D KDE (multi-bump)');

    plot(occluder_227(:,1), occluder_227(:,2),'w-','LineWidth',2);
    scatter(S2(:,1), S2(:,2), 4, [.7 .7 .7],'filled');
    plot([A1(1) A2(1)], [A1(2) A2(2)], 'b--','LineWidth',1.3);
    plot(curve(:,1), curve(:,2), 'g-','LineWidth',2);
    plot(A1(1),A1(2),'ms','MarkerSize',9,'LineWidth',2);
    plot(A2(1),A2(2),'bs','MarkerSize',9,'LineWidth',2);
    legend({'occluder','samples','baseline','regression'},'Location','northoutside');
    hold off;
end
