function runShortestPathContour(shapesFile, shapeIndex, heatmapMatFile)
% runShortestPathContour  Demonstrates a shortest-path approach on a 2D cost grid
%                         to connect the two intersection points within the occluder.
%
%   runShortestPathContour(shapesFile, shapeIndex, heatmapMatFile)
%
%   Inputs:
%     shapesFile      : e.g. 'shapes.mat' containing 'shapes'
%     shapeIndex      : which shape in shapes(...) to use
%     heatmapMatFile  : path to .mat containing normalized_heatmap (size [227x227])
%
%   Steps:
%     1) Load shapes => silhouette, occluder
%     2) Find intersection points => start_pt, end_pt
%     3) Build occludedImg => [227 x 227 x 3]
%     4) Load normalized_heatmap => [227 x 227]
%     5) Build occluderMask from occluder => poly2mask
%     6) Convert start_pt, end_pt to [row, col] or [y, x]
%     7) shortestPathOnHeatmap(...) => path
%     8) Plot path on occludedImg
%
%   Example:
%     runShortestPathContour('shapes.mat',1,'myHeatmap.mat');
%
%   Requirements:
%     - The function shortestPathOnHeatmap in your path
%     - The function createOccludedImage in your path
%
%   Author: Your Name
%   Date:   2025-03-31

    if nargin < 3
        error('Usage: runShortestPathContour(shapesFile, shapeIndex, heatmapMatFile)');
    end

    H = 227; W = 227;  % standard size

    %% 1) Load shapes => silhouette, occluder
    S = load(shapesFile,'shapes');
    if ~isfield(S,'shapes') || isempty(S.shapes)
        error('No shapes found in %s.', shapesFile);
    end
    if shapeIndex > numel(S.shapes)
        error('shapeIndex=%d > #shapes=%d.', shapeIndex, numel(S.shapes));
    end
    shapeData   = S.shapes(shapeIndex);
    silhouette  = shapeData.silhouette;  % Nx2
    occluder    = shapeData.occluder;    % Mx2

    %% 2) Find intersection points => start_pt, end_pt
    intersection_points = find_intersection_points(silhouette, occluder);
    if size(intersection_points,1)<2
        error('Not enough intersection points found between silhouette & occluder');
    end
    start_pt = intersection_points(1,:);
    end_pt   = intersection_points(2,:);
    fprintf('Intersection points => start_pt=(%.1f,%.1f), end_pt=(%.1f,%.1f)\n',...
        start_pt(1),start_pt(2),end_pt(1),end_pt(2));

    %% 3) Build occludedImg => [227x227x3]
    occludedImg = createOccludedImage(silhouette, occluder, H, W);

    %% 4) Load normalized_heatmap => [227x227]
    data = load(heatmapMatFile,'normalized_heatmap');
    if ~isfield(data,'normalized_heatmap')
        error('No "normalized_heatmap" found in %s.', heatmapMatFile);
    end
    heatmap = data.normalized_heatmap;
    if ~isequal(size(heatmap), [H W])
        error('Heatmap size mismatch => got [%dx%d], expected [227x227].', ...
            size(heatmap,1), size(heatmap,2));
    end

    %% 5) Build occluderMask => [227x227]
    occluderMask = poly2mask(occluder(:,1), occluder(:,2), H, W);

    %% 6) Convert start_pt, end_pt => row,col
    % We interpret the polygon coords as (x,y). 
    % In images, row=y, col=x => (row= y, col= x)
    start_rc = [round(start_pt(2)), round(start_pt(1))]; % (row, col)
    end_rc   = [round(end_pt(2)),   round(end_pt(1))];

    %% 7) shortest path => pathRC
    pathRC = shortestPathOnHeatmap(heatmap, occluderMask, start_rc, end_rc);

    %% 8) Plot path on occludedImg
    figure('Name','Shortest Path Contour');
    imshow(occludedImg); hold on;
    % pathRC is Nx2 => rows in pathRC(:,1), cols in pathRC(:,2)
    plot(pathRC(:,2), pathRC(:,1), 'r-', 'LineWidth',2); % (col,row)
    plot(start_rc(2),start_rc(1),'go','MarkerSize',8,'LineWidth',2);
    plot(end_rc(2),  end_rc(1),'go','MarkerSize',8,'LineWidth',2);
    title('Shortest Path from Intersection Points (Dijkstra)');

end

%% Helper function
function occludedImg = createOccludedImage(silhouettePts, occluderPts, H, W)
% createOccludedImage  Renders silhouette+occluder polygons into [H x W x 3] image.
    fig = figure('Visible','off');
    patch(silhouettePts(:,1), silhouettePts(:,2), 'k','FaceAlpha',1,'LineWidth',2);
    hold on;
    patch(occluderPts(:,1), occluderPts(:,2), 'k',...
        'EdgeColor',[0.5 0.5 0.5], 'FaceColor',[0.5 0.5 0.5],'LineWidth',1);
    axis off; axis equal;

    frameData = getframe(gca);
    rawImg = frame2im(frameData);
    close(fig);

    occludedImg = imresize(rawImg,[H W]);
end

function pts = find_intersection_points(silhouette, occluder)
% find_intersection_points  Basic placeholder for your intersection routine 
%   that returns 2 intersection points in [x,y].
    [xi, yi] = polyxpoly(silhouette(:,1), silhouette(:,2), ...
                         occluder(:,1), occluder(:,2));
    if length(xi) < 2
        pts = [];
    else
        pts = [xi(1), yi(1); xi(2), yi(2)];
    end
end
