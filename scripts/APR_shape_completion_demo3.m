function APR_shape_completion_demo3()
% -------------------------------------------------------------------------
% KDE-driven completion with a flexible, anchor-preserving regression line
%
% offset(t) = t(1–t) · spline(r(t))     (r = dev / [t(1–t)])
% curve     = A1 + t·v + offset(t)·n    (n ⟂ v, hits both anchors)
% -------------------------------------------------------------------------

    clear; clc; close all;

    % ---------------- USER-TUNABLE PARAMETERS ----------------------------
    Npts      = 15000;   % KDE-biased samples
    alphaPDF  = 5;    % sharpen exponent for KDE probabilities
    expPowerZ = 2;      % intensity exponent in KDE weights
    smoothPar = 0;    % csaps smoothing (0 = very wiggly, 1 = rigid)
    % --------------------------------------------------------------------

    %% 0) Load data -------------------------------------------------------
    S = load('shapes_0056.mat','shapes');
    H = load('heatmap_0056.mat','normalized_heatmap');
    shape = S.shapes(1);

    silhouette = shape.silhouette;
    occluder   = shape.occluder;
    interPts   = shape.intersection_points;
    heatmap    = H.normalized_heatmap;                 % 227×227
    [Hrows,Hcols] = size(heatmap);

    %% 1) Bounding-box rescale (flipped Y) --------------------------------
    margin = 5;
    allX = [silhouette(:,1); occluder(:,1)];
    allY = [silhouette(:,2); occluder(:,2)];
    minX = floor(min(allX)-margin);  maxX = ceil(max(allX)+margin);
    minY = floor(min(allY)-margin);  maxY = ceil(max(allY)+margin);
    wBB  = maxX-minX+1;              hBB  = maxY-minY+1;

    mapXY = @(P)[ (P(:,1)-minX)*(Hcols/wBB), (maxY-P(:,2))*(Hrows/hBB) ];
    occluder_227 = mapXY(occluder);
    inter_227    = mapXY(interPts);

    A1 = inter_227(1,:);   A2 = inter_227(2,:);     % anchors

    %% 2) Occluder mask ---------------------------------------------------
    mask = poly2mask(occluder_227(:,1), occluder_227(:,2), Hrows, Hcols);

    %% 3) 2-D KDE ---------------------------------------------------------
    [xx,yy] = meshgrid(1:Hcols,1:Hrows);
    xf = xx(:);  yf = yy(:);  zf = heatmap(:);
    idx = (zf>0) & mask(sub2ind([Hrows,Hcols],yf,xf));
    XY  = [xf(idx), yf(idx)];        % n×2 coordinates
    w   = zf(idx).^expPowerZ;        % weights
    w   = w / sum(w);

    pdfXY = mvksdensity(XY, XY, 'Weights', w);
    pdfXY = pdfXY / sum(pdfXY);

    sampIdx = randsample(size(XY,1), Npts, true, pdfXY.^alphaPDF);
    S2  = XY(sampIdx,:);         % Npts×2 samples
    wS  = pdfXY(sampIdx);        wS = wS / sum(wS);

    %% 4) Flexible spline offset (anchors preserved) ----------------------
    v = A2 - A1;      L2 = dot(v,v);
    n = [-v(2), v(1)] / norm(v);

    tS  = max(0, min(1, sum((S2 - A1).*v,2) / L2));
    dev = sum((S2 - (A1 + tS.*v)).*n, 2);

    inside = (tS>0 & tS<1);
    tS  = tS(inside);   dev = dev(inside);   wS = wS(inside);

    rVals = dev ./ (tS .* (1-tS));         % residuals
    [tSort, ord] = sort(tS);
    rSort = rVals(ord);   wSort = wS(ord);

    rspline = csaps(tSort, rSort, smoothPar, [], wSort);

    %% 5) Build final curve ----------------------------------------------
    tf   = linspace(0,1,600)';             % column vector
    base = A1 + tf.*v;                     % 600×2 baseline
    offset = (tf .* (1-tf)) .* fnval(rspline, tf);   % 600×1
    curve  = base + offset * n;            % outer product → 600×2

    %% 6) Visualisation ---------------------------------------------------
    figure('Name','Flexible anchored KDE curve','Color','w');
    imagesc(heatmap); axis image xy; colormap jet; hold on;
    title(sprintf('Anchored flexible curve   smoothPar = %.2f', smoothPar));

    plot(occluder_227(:,1), occluder_227(:,2),'w-','LineWidth',2);
    scatter(S2(:,1), S2(:,2), 4, [.7 .7 .7],'filled');
    plot([A1(1) A2(1)], [A1(2) A2(2)], 'b--','LineWidth',1.3);
    plot(curve(:,1), curve(:,2), 'g-','LineWidth',2);
    plot(A1(1),A1(2),'ms','MarkerSize',9,'LineWidth',2);
    plot(A2(1),A2(2),'bs','MarkerSize',9,'LineWidth',2);
    legend({'occluder','samples','baseline','flex-anchored'},'Location','northoutside');
    hold off;
end
