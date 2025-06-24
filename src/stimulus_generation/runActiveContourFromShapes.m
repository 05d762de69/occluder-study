function runActiveContourFromShapes(heatmapMatFile, shapesFile, shapeIndex)
% runActiveContourFromShapes  Reconstructs an occluded image from shapes,
%                             loads normalized_heatmap from a .mat file,
%                             creates occluderMask, and runs activecontour.
%
%   runActiveContourFromShapes(heatmapMatFile, shapesFile, shapeIndex)
%
%   Inputs:
%     heatmapMatFile : Path to a .mat containing "normalized_heatmap" (the final 
%                      sensitivity heatmap array). For example, "myHeatmap.mat".
%     shapesFile     : Path to shapes.mat, which has shapes(...) with
%                      .silhouette, .occluder, etc.
%     shapeIndex     : Which shape in shapes(...) you want to use (e.g. 1)
%
%   Steps:
%     1) Load shapesFile => shapes
%     2) Extract silhouette & occluder from shapes(shapeIndex)
%     3) Build the occluded image [227x227x3] from these polygons
%     4) Load the .mat => get "normalized_heatmap" (size [227x227])
%     5) Convert occluder => occluderMask = poly2mask(...)
%     6) Optionally set initMask = occluderMask
%     7) activecontour(normalized_heatmap, initMask, ... 'Chan-Vese')
%     8) Display final mask on top of the occluded image
%
%   Example usage:
%     runActiveContourFromShapes('myHeatmap.mat','shapes.mat',1);
%
%   Author: Your Name
%   Date:   2025-03-31

    if nargin < 3
        error('Usage: runActiveContourFromShapes(heatmapMatFile, shapesFile, shapeIndex)');
    end

    %% 1) Load shapes
    S = load(shapesFile,'shapes');
    if ~isfield(S,'shapes') || isempty(S.shapes)
        error('No shapes found in %s.', shapesFile);
    end
    if shapeIndex > numel(S.shapes)
        error('Requested shapeIndex=%d but shapes has only %d entries.', shapeIndex, numel(S.shapes));
    end

    shapeData   = S.shapes(shapeIndex);
    silhouette  = shapeData.silhouette;  % Nx2
    occluder    = shapeData.occluder;    % Mx2

    %% 2) Build the occluded image at 227x227
    H = 227; W = 227;
    occludedImg = createOccludedImage(silhouette, occluder, H, W);

    %% 3) Load normalized_heatmap from .mat
    data = load(heatmapMatFile,'normalized_heatmap');
    if ~isfield(data,'normalized_heatmap')
        error('No "normalized_heatmap" found in %s.', heatmapMatFile);
    end
    energyMap = data.normalized_heatmap;  % [227 x 227]

    [hMap, wMap] = size(energyMap);
    if hMap~=H || wMap~=W
        error('normalized_heatmap is [%dx%d], but occludedImg is [227x227].', hMap,wMap);
    end

    %% 4) Create occluderMask from occluder polygon => [227x227]
    %    We assume the polygons are in [x,y] coords inside the same 227x227 space.
    occluderMask = poly2mask(occluder(:,1), occluder(:,2), H, W);

    %% 5) Use the occluderMask as the initial region for activecontour
    initMask  = occluderMask;
    maxIters  = 300;
    method    = 'Chan-Vese';
    fprintf('Running activecontour(...) with method="%s", up to %d iters.\n', method, maxIters);
    finalMask = activecontour(energyMap, initMask, maxIters, method);

    %% 6) Display the final contour on both the heatmap & the occluded image
    figure('Name','Active Contour from Shapes + Heatmap .mat');
    subplot(1,2,1);
    imshow(mat2gray(energyMap));  % just a grayscale of your final heatmap
    hold on;
    visboundaries(finalMask, 'Color','r','LineWidth',2);
    title('Contour over Normalized Heatmap');

    subplot(1,2,2);
    imshow(occludedImg);  % The in-memory occluded image
    hold on;
    visboundaries(finalMask, 'Color','r','LineWidth',2);
    title('Contour Overlaid on Occluded Image');
end