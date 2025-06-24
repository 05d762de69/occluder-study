function finalMask = runActiveContourStrictHeatmap(energyMap, occluderMask)
% runActiveContourStrictHeatmap  Combines zeroing outside polygon, morphological 
%                                erosion of init mask, and bounding-box cropping 
%                                so the snake sees only the heatmap inside 
%                                the occluder region.
%
%   finalMask = runActiveContourStrictHeatmap(energyMap, occluderMask)
%
%   Inputs:
%     energyMap     : [H x W] numeric array (the "heatmap" from your log approach)
%     occluderMask  : [H x W] logical array, true inside the occluder polygon
%
%   Steps:
%     1) bounding box => rowRange, colRange
%     2) subEnergy = energyMap(rowRange,colRange)
%     3) subMask   = occluderMask(rowRange,colRange)
%     4) zero out outside subMask => subEnergy(~subMask) = 0
%     5) erode subMask => initMask
%     6) optionally scale or invert subEnergy if needed
%     7) activecontour(subEnergy, initMask, method='Chan-Vese')
%     8) re-embed subFinal => finalMask
%
%   Returns:
%     finalMask : [H x W] logical array, the final segmentation
%
%   Usage Example:
%     data = load('myHeatmap.mat','normalized_heatmap');  % your log-based heatmap
%     energyMap = data.normalized_heatmap; 
%     % load occluderMask => [227x227] 
%     finalMask = runActiveContourStrictHeatmap(energyMap, occluderMask);
%
%   Author: Your Name
%   Date:   2025-03-31

    [H, W] = size(energyMap);
    if ~isequal(size(occluderMask), [H, W])
        error('Mask size mismatch with heatmap. mask=[%dx%d], map=[%dx%d].',...
            size(occluderMask,1), size(occluderMask,2), H, W);
    end

    %% 1) bounding box => rowRange, colRange
    [rr, cc] = find(occluderMask);
    rowRange = min(rr):max(rr);
    colRange = min(cc):max(cc);

    %% 2) subEnergy, subMask
    subEnergy = energyMap(rowRange, colRange);
    subMask   = occluderMask(rowRange, colRange);

    %% 3) zero out outside subMask
    subEnergy(~subMask) = 0;

    %% 4) erode subMask => initMask
    initMask = imerode(subMask, strel('disk',8));  % tweak disk size if needed

    %% 5) optionally scale or invert subEnergy
    % If subEnergy is too small in magnitude, try:
    % subEnergy = subEnergy * 255;
    % or invert if interior is bright but you want the region-based approach:
    % subEnergy = 1 - subEnergy;
   % subEnergy = imgaussfilt(subEnergy, 2); % smoothing, etc.

    %% 6) activecontour => 'Chan-Vese'
    method   = 'Chan-Vese';
    maxIters = 300;
    fprintf('Running activecontour(...) with method=%s on subregion.\n', method);

    subFinal = activecontour(subEnergy, initMask, maxIters, method);

    %% 7) re-embed subFinal
    finalMask = false(H,W);
    finalMask(rowRange, colRange) = subFinal;

    %% display
    figure('Name','Active Contour - Heatmap Inside Occluder Only');
    subplot(1,2,1);
    imshow(mat2gray(energyMap));  % show entire map
    hold on;
    visboundaries(finalMask, 'Color','r','LineWidth',2);
    title('Final Mask on Full Heatmap');

    subplot(1,2,2);
    subVis = mat2gray(subEnergy);
    imshow(subVis);
    hold on;
    visboundaries(subFinal, 'Color','g','LineWidth',2);
    title('Subregion Zoom');
end
