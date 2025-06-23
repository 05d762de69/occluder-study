function APR_probabilisticCompletion(...
    occludedImgFile, heatmapMat, anchorPoints, numComponents, outPlotFile)
% probabilisticCompletion
%   1) Loads normalized_heatmap from 'heatmapMat' ([H x W]).
%   2) Fits a GMM (with 'numComponents'), draws samples.
%   3) Builds a spline from anchorPoints(1,:) => anchorPoints(2,:).
%   4) Overlays the final curve on the heatmap, plus an overlay 
%      on the actual occluded image if it matches [H x W].
%
% Inputs:
%  - occludedImgFile: e.g. 'silOccl_0044.png' at [H x W]
%  - heatmapMat     : e.g. 'heatmap_0044.mat' with 'normalized_heatmap'
%  - anchorPoints   : [2 x 2], anchors in pixel coords => [ [x1,y1]; [x2,y2] ]
%  - numComponents  : # of mixture components for GMM
%  - outPlotFile    : optional, final PNG
%
% Example:
%   anchorPts=[30,120; 180,120];
%   probabilisticCompletion('silOccl_0044.png','heatmap_0044.mat',...
%       anchorPts,5,'myCurve.png');

    if nargin < 4
        error('Usage: probabilisticCompletion(occludedImgFile, heatmapMat, anchorPoints, numComponents, [outPlotFile])');
    end
    if nargin < 5
        outPlotFile = '';
    end

    % 1) Load heatmap
    data = load(heatmapMat,'normalized_heatmap');
    heatmap = data.normalized_heatmap;
    [H, W] = size(heatmap);

    fprintf('Loaded heatmap => [%dx%d]\n', H,W);

    % 2) Weighted points
    [Xgrid, Ygrid] = meshgrid(1:W,1:H);
    Xgrid = Xgrid(:); 
    Ygrid = Ygrid(:);
    pvals = heatmap(:);

    valid = (pvals>1e-8);
    Xgrid = Xgrid(valid);
    Ygrid = Ygrid(valid);
    pvals = pvals(valid);

    sumP = sum(pvals);
    if sumP<1e-12
        error('Heatmap near-zero sum?');
    end
    pvals = pvals/sumP;

    points = [Xgrid,Ygrid];

    % 3) Fit GMM
    opts = statset('MaxIter',500);
    gm = fitgmdist(points,numComponents,...
        'Weights',pvals,...
        'CovarianceType','full',...
        'RegularizationValue',1e-5,...
        'Options',opts);
    fprintf('Fitted GMM with %d components.\n',numComponents);

    % 4) Sample, in-bounds
    numSamples=2000;
    Sxy=random(gm,numSamples);
    inB=(Sxy(:,1)>=1 & Sxy(:,1)<=W & Sxy(:,2)>=1 & Sxy(:,2)<=H);
    Sxy=Sxy(inB,:);

    % 5) Fit param-spline from anchorPoints(1,:) -> anchorPoints(2,:)
    anchorA=anchorPoints(1,:);
    anchorB=anchorPoints(2,:);

    subsetN=min(500,size(Sxy,1));
    ridx=randperm(size(Sxy,1),subsetN);
    midPts=Sxy(ridx,:);

    allPts=[anchorA;midPts;anchorB];
    t=linspace(0,1,size(allPts,1));
    px=spline(t,allPts(:,1));
    py=spline(t,allPts(:,2));

    tFine=linspace(0,1,200);
    xC=ppval(px,tFine);
    yC=ppval(py,tFine);

    % Show heatmap
    figure('Name','Probabilistic Completion on Heatmap');
    imagesc(heatmap); axis image xy; hold on;
    colormap('hot'); colorbar;
    scatter(Sxy(:,1),Sxy(:,2),10,'b','filled','MarkerFaceAlpha',0.2);
    plot(anchorA(1),anchorA(2),'go','MarkerSize',8,'LineWidth',2);
    plot(anchorB(1),anchorB(2),'go','MarkerSize',8,'LineWidth',2);
    plot(xC,yC,'r-','LineWidth',2);
    title(sprintf('Prob. Completion: GMM(%d)',numComponents));

    % Overplot on occluded image
    if exist(occludedImgFile,'file')
        occludedImg=imread(occludedImgFile);
        if size(occludedImg,1)==H && size(occludedImg,2)==W
            figure('Name','Overlay on occluded image');
            imshow(occludedImg); hold on;
            scatter(Sxy(:,1),Sxy(:,2),10,'b','MarkerFaceAlpha',0.1);
            plot(anchorA(1),anchorA(2),'go','MarkerSize',8,'LineWidth',2);
            plot(anchorB(1),anchorB(2),'go','MarkerSize',8,'LineWidth',2);
            plot(xC,yC,'r-','LineWidth',2);
            title('Probabilistic curve on real occluded image');
        else
            warning('Occluded image dimension mismatch, skipping overlay.');
        end
    else
        warning('Occluded image file not found => skipping overlay plot.');
    end

    if ~isempty(outPlotFile)
        exportgraphics(gca,outPlotFile,'Resolution',150);
        fprintf('Saved final figure => %s\n',outPlotFile);
    end
end
