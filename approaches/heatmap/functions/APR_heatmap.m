function APR_heatmap(netFile, occludedFile, completionDir, outHeatmapMat, outPng)
% APR_heatmap  Builds and overlays a log-transformed occluder sensitivity heatmap.
%
%   APR_heatmap(netFile, occludedFile, completionDir, outHeatmapMat, outPng)
%
%   Steps:
%     1) Load trained network from netFile => trainedNet
%     2) Read occludedFile => occludedImg (resized to [227 x 227])
%     3) Collect all random completions from completionDir => images [227x227x3xN]
%     4) For each completion, compute final-layer activation => distance to baseline
%        in log-space
%     5) For each pixel that changed from baseline => accumulate distance
%     6) Normalize => normalized_heatmap => save to outHeatmapMat
%     7) Overlay heatmap on occludedImg => outPng
%
%   Example:
%     APR_heatmap('trainedNet.mat', ...
%                 'silOccl_0058.png', ...
%                 '/path/to/completions', ...
%                 'myLogHeatmap.mat', ...
%                 'overlay.png');
%
%   If you only want the .mat file and no PNG, pass '' for outPng. 
%
%   Requirements:
%     - The final-layer name can be changed below (layerName).
%     - The images are resized to [227 x 227] for typical AlexNet input.
%
%   Author: Your Name
%   Date:   2025-04-01

    % ========== 1) Load trained network ==========
    load(netFile, 'trainedNet');
    if ~exist('trainedNet','var')
        error('No variable "trainedNet" in %s.', netFile);
    end
    layerName = 'classoutput';  % or 'fc8', etc. Adjust if needed.
    activations_fn = @(img) squeeze(activations(trainedNet, img, layerName));

    % ========== 2) Read occludedFile => occludedImg (227x227) ==========
    H = 227; W = 227;
    occludedImg = imread(occludedFile);
    occludedImg = imresize(occludedImg, [H W]);

    % Convert to uint8 if needed (some networks prefer double, but for 
    % consistent binarization we can do uint8)
    occludedImg = im2uint8(occludedImg);

    % ========== 3) Gather completions from completionDir ==========
    files = dir(fullfile(completionDir, '*.png'));
    numImages = numel(files);
    if numImages < 1
        error('No .png images found in %s.', completionDir);
    end
    fprintf('Found %d completions in %s\n', numImages, completionDir);

    images = zeros(H, W, 3, numImages, 'uint8');
    for i = 1:numImages
        tmp = imread(fullfile(files(i).folder, files(i).name));
        tmp = imresize(tmp, [H W]);
        images(:,:,:,i) = im2uint8(tmp);
    end

    % ========== 4) Baseline activation in log-space ==========
    a_occ = activations_fn(occludedImg);
    epsVal = 1e-8;
    log_occ = log(a_occ + epsVal);

    % ========== 5) Build heatmap by comparing activations & binarized differences ==========
    bw_occ = imbinarize(rgb2gray(occludedImg));  % baseline binary
    heatmap  = zeros(H, W);
    weights  = zeros(H, W);

    for i = 1:numImages
        img_i = images(:,:,:,i);
        a_i   = activations_fn(img_i);
        log_i = log(a_i + epsVal);

        d = norm(log_i - log_occ, 2);  % log-space distance

        bw_i = imbinarize(rgb2gray(img_i));
        diffMask = (bw_i ~= bw_occ);

        heatmap  = heatmap + double(diffMask) * d;
        weights  = weights + double(diffMask);
    end

    normalized_heatmap = heatmap ./ (weights + 1e-6);

    % ========== 6) Save to outHeatmapMat ==========
    if ~isempty(outHeatmapMat)
        save(outHeatmapMat, 'normalized_heatmap', '-v7.3');
        fprintf('Saved log-based heatmap to %s\n', outHeatmapMat);
    end

    % ========== 7) Overlay on occludedImg & save outPng ==========
    figure('Name','APR Heatmap Overlay');
    imshow(occludedImg); hold on;
    hOverlay = imagesc(normalized_heatmap);
    colormap('hot'); colorbar;
    set(hOverlay, 'AlphaData', mat2gray(normalized_heatmap));
    title('Log-Space Activation Distance Heatmap Overlay');
    axis image off;

    if ~isempty(outPng)
        f = getframe(gca);
        imwrite(f.cdata, outPng);
        fprintf('Saved overlay to %s\n', outPng);
    end
end
