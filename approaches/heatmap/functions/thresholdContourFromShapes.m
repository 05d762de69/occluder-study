function thresholdContourFromShapes(shapesFile, shapeIndex, heatmapMat, thresholdPercent, occluderMask, sigma)
% thresholdContourFromShapes  
%   Builds an occluded image from shapes(...) at [227 x 227],
%   loads normalized_heatmap (also [227 x 227]) from .mat,
%   then does threshold-based contour extraction 
%   **only within the occluder region**, provided as a binary mask.
%
% Inputs:
%   shapesFile      : Path to shapes.mat containing variable 'shapes'.
%   shapeIndex      : Index into shapes(...)
%   heatmapMat      : .mat file with 'normalized_heatmap' [227 x 227]
%   thresholdPercent: e.g. 95 => top 5% of heatmap inside occluder
%   occluderMask    : logical matrix [227 x 227], true = inside occluder
%   sigma           : (optional) Gaussian smoothing sigma (e.g. 1.5)
%
% Example:
%   thresholdContourFromShapes('shapes.mat', 1, 'heatmap.mat', 95, myMask, 1.5)

    if nargin < 5
        error('Must provide shapesFile, shapeIndex, heatmapMat, thresholdPercent, occluderMask.');
    end
    if nargin < 6 || isempty(sigma)
        sigma = 0;
    end

    H = 227; W = 227;

    %% 1) Load shapes and get silhouette
    S = load(shapesFile, 'shapes');
    if ~isfield(S, 'shapes') || isempty(S.shapes)
        error('No "shapes" found in %s.', shapesFile);
    end
    if shapeIndex > numel(S.shapes)
        error('shapeIndex=%d but shapes has only %d items.', shapeIndex, numel(S.shapes));
    end

    silhouette = S.shapes(shapeIndex).silhouette;

    %% 2) Create the occluded image
    occluder = S.shapes(shapeIndex).occluder;
    occludedImg = createOccludedImage(silhouette, occluder, H, W);

    %% 3) Load the heatmap
    D = load(heatmapMat, 'normalized_heatmap');
    if ~isfield(D, 'normalized_heatmap')
        error('Missing "normalized_heatmap" in %s', heatmapMat);
    end
    heatmap = D.normalized_heatmap;

    if ~isequal(size(heatmap), [H W])
        error('Heatmap must be size [227 x 227].');
    end
    if ~isequal(size(occluderMask), [H W])
        error('Occluder mask must be size [227 x 227].');
    end

    %% 4) Smooth if desired
    if sigma > 0
        fprintf('Smoothing heatmap with sigma=%.2f...\n', sigma);
        heatmap = imgaussfilt(heatmap, sigma);
    end

    %% 5) Thresholding
    occValues = heatmap(occluderMask);
    numOcc = nnz(occluderMask);
    if numOcc == 0
        warning('Empty occluder mask!');
        return;
    end

    thresholdVal = prctile(occValues, thresholdPercent);
    binMask = heatmap >= thresholdVal;

    %% 6) Only keep values inside occluder
    binMask(~occluderMask) = false;

    %% 7) Extract and plot boundaries
    boundaries = bwboundaries(binMask);
    fprintf('Found %d contours at %.1f%% threshold.\n', numel(boundaries), thresholdPercent);

    figure('Name','Threshold Contour Overlay');
    imshow(occludedImg); hold on;
    for i = 1:numel(boundaries)
        b = boundaries{i};
        plot(b(:,2), b(:,1), 'r-', 'LineWidth', 2);
    end
    title(sprintf('Threshold %.1f%%, \\sigma = %.1f', thresholdPercent, sigma));
end
