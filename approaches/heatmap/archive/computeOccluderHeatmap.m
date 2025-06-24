function computeOccluderHeatmap(occludedImg, images, occluderMask, activations_fn, outFile)
% computeOccluderHeatmapBinary  Builds a heatmap of pixel importance 
%                               by binarizing the occluder region comparisons 
%                               and restricting differences to occluderMask.
%
%   computeOccluderHeatmapBinary(occludedImg, images, occluderMask, activations_fn, outFile)
%
%   Inputs:
%     occludedImg   : the baseline occluded image ([H x W x 3])
%     images        : a 4D array [H x W x 3 x N] with random completions
%     occluderMask  : [H x W], logical. true inside occluder, false outside
%     activations_fn: function handle for final-layer activations
%     outFile       : optional, name of PNG to save the final heatmap
%
%   The difference mask is computed by:
%       bw_occluded = imbinarize(rgb2gray(occludedImg))
%       bw_img      = imbinarize(rgb2gray(images(:,:,:,i)))
%     Then we take diffMask = (bw_img ~= bw_occluded) & occluderMask.
%
%   The rest is as before: accumulate Euclidean distance to the baseline 
%   activation whenever we see a changed pixel inside the occluder region.
%
%   Example:
%     occluderMask = poly2mask(occluderPolygon(:,1), occluderPolygon(:,2), 227, 227);
%     activations_fn = @(x) squeeze(activations(net, x, 'fc8'));
%     computeOccluderHeatmapBinary(occludedImg, images, occluderMask, activations_fn, 'heatmap.png');

    if nargin < 4
        error('Usage: computeOccluderHeatmapBinary(occludedImg, images, occluderMask, activations_fn, [outFile])');
    end
    if nargin < 5
        outFile = '';
    end

    [H, W, C, numSamples] = size(images);
    if C ~= 3
        error('Expected images array to be [H x W x 3 x N].');
    end
    if ~isequal(size(occluderMask), [H W])
        error('occluderMask must be size [H x W].');
    end

    % 1) Baseline activation
    a_occluded = activations_fn(occludedImg);
    a_occluded = squeeze(a_occluded);

    % Also binarize the occluded image for difference checking
    bw_occluded = imbinarize(rgb2gray(occludedImg));

    heatmap  = zeros(H, W);
    weights  = zeros(H, W);
    epsilon  = 1e-6;

    fprintf('Processing %d images...\n', numSamples);
    for i = 1:numSamples
        img_i = images(:,:,:,i);

        % (a) Activation
        a_i = activations_fn(img_i);
        a_i = squeeze(a_i);
        
        % (b) Distance from occluded baseline
        d = norm(a_i - a_occluded, 2);

        % (c) Binarize & compute difference within occluder
        bw_img = imbinarize(rgb2gray(img_i));

        % True where this image differs from baseline in binarized sense
        diffMaskAll = (bw_img ~= bw_occluded);

        % Restrict to occluder region
        diffMask = diffMaskAll & occluderMask;

        % (d) Accumulate distance for changed pixels
        heatmap = heatmap + (diffMask * d);
        weights = weights + diffMask;
    end

    % 2) Normalize
    normalized_heatmap = heatmap ./ (weights + epsilon);

    % 3) Display
    figure('Name','Occluder Sensitivity Heatmap (Binary / Masked)');
    imagesc(normalized_heatmap);
    axis image off;
    colorbar;
    title('Activation Sensitivity (binary diff, restricted to occluder)');

    % Overlay
    figure('Name','Overlay on Occluded Image');
    imshow(occludedImg);
    hold on;
    hOverlay = imagesc(normalized_heatmap);
    colormap('hot');
    set(hOverlay, 'AlphaData', mat2gray(normalized_heatmap));
    colorbar;
    title('Overlay of Sensitivity Heatmap (occluder only)');

    % 4) (Optional) Save
    if ~isempty(outFile)
        figure('Visible','off');
        imagesc(normalized_heatmap);
        axis image off; colormap('hot'); colorbar;
        title('Occluder Sensitivity Heatmap (Binary Diff)');
        exportgraphics(gca, outFile, 'BackgroundColor','white');
        close(gcf);
        fprintf('Saved heatmap to "%s".\n', outFile);
    end
end
