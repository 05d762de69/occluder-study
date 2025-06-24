function runActiveContourOnHeatmap(heatmapFile, occludedImgFile, occluderPolygon)
% runActiveContourStrictAll  Hard-restrict activecontour to the occluder region
%                            by cropping, zeroing outside polygon, and eroding the initial mask.
%
%   runActiveContourStrictAll(heatmapFile, occludedImgFile, occluderPolygon)
%
%   Inputs:
%     heatmapFile     : Path to the heatmap image (e.g., .png) or .mat with normalized_heatmap
%     occludedImgFile : Path to the occluded image for final visualization ([H x W x 3] when read)
%     occluderPolygon : Nx2 array of [x,y] polygon coords from shapes.mat
%                       describing the occluder boundary in [1..W], [1..H].
%
%   Steps:
%     1) Read the heatmap => energyMap ([H x W], e.g. 227x227). 
%        If mismatch with the occluded image size, we can resize or error out.
%     2) Read occludedImg => confirm or resize to match [H x W].
%     3) Create occluderMask = poly2mask(occluderPolygon).
%     4) Crop bounding box => rowRange, colRange => subEnergy, subMask
%     5) Zero out outside polygon => subEnergy(~subMask)=0
%     6) Erode subMask => initMask = imerode(subMask, strel('disk',N))
%     7) activecontour(subEnergy, initMask, ... 'Chan-Vese')
%     8) Reembed in finalMask => display over occludedImg & heatmap
%
%   Example:
%     poly = shapes(1).occluder;
%     runActiveContourStrictAll('heatmap_log.png','silOccl_0058.png', poly);

    if nargin < 3
        error('Usage: runActiveContourStrictAll(heatmapFile, occludedImgFile, occluderPolygon)');
    end

    %% 1) Read or load the heatmap => energyMap
    try
        heatmapImg = imread(heatmapFile);
        heatmapGray = im2double(rgb2gray(heatmapImg));
        energyMap   = mat2gray(heatmapGray);
    catch
        warning('Could not imread(%s). Trying load(...) for normalized_heatmap.', heatmapFile);
        data = load(heatmapFile, 'normalized_heatmap');
        energyMap = mat2gray(data.normalized_heatmap);
        heatmapImg = [];
    end
    [H, W] = size(energyMap);

    %% 2) Read the occluded image
    occludedImg = imread(occludedImgFile);
    occludedImg = im2double(occludedImg);
    if ~isequal(size(occludedImg,1),H) || ~isequal(size(occludedImg,2),W)
        % If mismatch, either resize or error out. We'll do a quick resize here for convenience:
        fprintf('Resizing occludedImg from [%dx%d] to [%dx%d]\n',...
            size(occludedImg,1), size(occludedImg,2), H, W);
        occludedImg = imresize(occludedImg, [H, W], 'bicubic');
    end

    %% 3) Create occluderMask from polygon => [H x W]
    occluderMask = poly2mask( ...
        occluderPolygon(:,1), ...
        occluderPolygon(:,2), ...
        H, W);

    %% 4) Crop bounding box => subEnergy, subMask
    [rr, cc] = find(occluderMask);
    rowRange = min(rr):max(rr);
    colRange = min(cc):max(cc);

    subEnergy = energyMap(rowRange, colRange);
    subMask   = occluderMask(rowRange, colRange);

    %% 5) Zero out outside polygon
    subEnergy(~subMask) = 0;

    %% 6) Erode the subMask => initMask
    % This ensures the snake starts well inside the polygon
    initMask = imerode(subMask, strel('disk',8));  % adjust disk size if needed

    %% 7) activecontour => 'Chan-Vese'
    method   = 'Chan-Vese';
    maxIters = 300;
    fprintf('Running activecontour(...) with method=%s, up to %d iters.\n', method, maxIters);
    subFinal = activecontour(subEnergy, initMask, maxIters, method);

    %% 8) Re-embed subFinal in finalMask => [H x W]
    finalMask = false(size(energyMap));
    finalMask(rowRange, colRange) = subFinal;

    %% Display
    figure('Name','Active Contour - Strict Cropping/Zero/Erode');
    subplot(1,2,1);
    if ~isempty(heatmapImg) && all(size(heatmapImg,1:2)==[H W])
        imshow(heatmapImg);  % reference
    else
        imshow(energyMap);
    end
    hold on;
    visboundaries(finalMask, 'Color','r','LineWidth',2);
    title('Final Mask on Heatmap');

    subplot(1,2,2);
    imshow(occludedImg);
    hold on;
    visboundaries(finalMask, 'Color','r','LineWidth',2);
    title('Final Mask on Occluded Image');
end
