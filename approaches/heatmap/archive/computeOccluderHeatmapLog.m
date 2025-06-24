function computeOccluderHeatmapLog(occludedImg, images, occluderMask, activations_fn, matFile)
% computeOccluderHeatmapLogMat  
%   Builds a log-transformed occluder sensitivity heatmap, then saves the result 
%   in a .mat file (rather than a PNG). 
%
%   computeOccluderHeatmapLogMat(occludedImg, images, occluderMask, activations_fn, matFile)
%
%   Inputs:
%     occludedImg   : [H x W x 3], baseline occluded image
%     images        : [H x W x 3 x N], random completions
%     occluderMask  : [H x W], logical. true inside occluder, false outside
%     activations_fn: function handle for final-layer activations, e.g.
%                     @(img) squeeze(activations(net, img, 'prob'))
%     matFile       : name of the .mat file to save (string/char), e.g. 'occluderHeatmap.mat'
%
%   The log-space distance measure is:
%        d = || log(a_i + epsVal) - log(a_occluded + epsVal) ||_2
%
%   Then for each image, we accumulate that distance into 'heatmap' for the 
%   pixels that differ from occludedImg inside occluderMask. We normalize 
%   by the number of changes => 'normalized_heatmap'.
%
%   Finally, we do:
%        save(matFile, 'normalized_heatmap', '-v7.3');
%
%   so you can later do load(matFile) => run activecontour, etc.
%
%   Example:
%     occluderMask = poly2mask(occluderPoly(:,1), occluderPoly(:,2), 227, 227);
%     fn = @(x) squeeze(activations(net, x, 'prob'));
%     computeOccluderHeatmapLogMat(occludedImg, images, occluderMask, fn, 'myHeatmap.mat');
%
%   Afterward, you'd do:
%     data = load('myHeatmap.mat');
%     energyMap = data.normalized_heatmap; 
%     % Then call runActiveContourOnHeatmap('myHeatmap.mat', occludedImg, occluderMask) 
%     % or handle it directly in memory.
%
%   Author: Your Name
%   Date:   2025-03-31

    if nargin < 4
        error('Usage: computeOccluderHeatmapLogMat(occludedImg, images, occluderMask, activations_fn, matFile)');
    end
    if nargin < 5 || isempty(matFile)
        error('You must specify a .mat file to save the heatmap, e.g. "myHeatmap.mat".');
    end

    [H, W, C, numSamples] = size(images);
    if C ~= 3
        error('Expected images to be [H x W x 3 x N].');
    end
    if ~isequal(size(occluderMask), [H W])
        error('occluderMask must be size [H x W].');
    end

    %% 1) Baseline activation
    a_occluded = activations_fn(occludedImg);
    a_occluded = squeeze(a_occluded);

    %% Binarize the occluded image for difference detection
    bw_occluded = imbinarize(rgb2gray(occludedImg));

    heatmap  = zeros(H, W);
    weights  = zeros(H, W);

    epsVal = 1e-8;  % small epsilon to avoid log(0)

    fprintf('Computing log-space occluder heatmap for %d images...\n', numSamples);
    for i = 1:numSamples
        img_i = images(:,:,:,i);

        % (A) Activation
        a_i = activations_fn(img_i);
        a_i = squeeze(a_i);

        % (B) log-space distance
        log_occ = log(a_occluded + epsVal);
        log_i   = log(a_i + epsVal);
        d = norm(log_i - log_occ, 2);

        % (C) Binarize & find differences in the occluder region
        bw_img = imbinarize(rgb2gray(img_i));
        diffMaskAll = (bw_img ~= bw_occluded);   % changed pixels
        diffMask    = diffMaskAll & occluderMask;

        % (D) Accumulate
        heatmap = heatmap + (diffMask * d);
        weights = weights + diffMask;
    end

    %% 2) Normalize
    epsilon = 1e-6;
    normalized_heatmap = heatmap ./ (weights + epsilon);
    %% 3) Save to .mat
    fprintf('Saving "normalized_heatmap" to %s\n', matFile);
    save(matFile, 'normalized_heatmap', '-v7.3');

    fprintf('Done. You can later load("%s") => use "normalized_heatmap".\n', matFile);
end
