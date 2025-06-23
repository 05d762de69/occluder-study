function generateAndRunShortestPath(animals, silhouette_index, H, W, numImages)
% generateAndRunShortestPath  Minimal approach that:
%   1) Picks silhouette & occluder from animals(...),
%   2) Draws them => occluded.png,
%   3) Finds intersection points in the same coordinate system used by polyxpoly,
%   4) Builds completions (resized to [H x W]),
%   5) Creates & saves a log-based heatmap => myHeatmap.mat,
%   6) Calls shortestPathOnHeatmap using the newly found intersection row,col.
%
%  Inputs:
%   animals         : your array of structures with fields .c (silhouette), .o (occluder)
%   silhouette_index: which animal struct to pick
%   H, W            : final image size, e.g. 227,227 (for the network)
%   numImages       : how many random completions exist
%
%  Requirements:
%    - polyxpoly
%    - computeOccluderHeatmapLogMat or similar
%    - shortestPathOnHeatmap
%
%  Example usage:
%    >> generateAndRunShortestPath(animals, 35, 227, 227, 1000);

    %% 1) Pick silhouette & occluder
    silhouette = animals(silhouette_index).c;  % Nx2
    occluder   = animals(silhouette_index).o;  % Mx2

    %% 2) Draw them => occluded.png
    outFile = sprintf('occluded_%04d.png', silhouette_index);
    figA = figure('Visible','off');
    patch(silhouette(:,1), silhouette(:,2), 'k','FaceAlpha',1,'LineWidth',2);
    hold on;
    patch(occluder(:,1), occluder(:,2), 'k',...
          'EdgeColor',[0.5 0.5 0.5], 'FaceColor',[0.5 0.5 0.5],'LineWidth',1);
    axis off; axis equal;
    frameDataA = getframe(gca);
    imgA = frame2im(frameDataA);
    imwrite(imgA, outFile);
    close(figA);
    fprintf('Saved occluded silhouette+occluder => %s\n', outFile);

    %% 3) Intersection points
    [xi, yi] = polyxpoly(silhouette(:,1), silhouette(:,2),...
                         occluder(:,1),   occluder(:,2));
    if length(xi)<2
        error('No valid intersection points found');
    end
    start_pt = [xi(1), yi(1)];
    end_pt   = [xi(2), yi(2)];
    fprintf('Intersection: start_pt=(%.1f,%.1f), end_pt=(%.1f,%.1f)\n',...
        start_pt(1),start_pt(2), end_pt(1),end_pt(2));

    %% 4) Load the newly saved occluded.png & resize => [H x W]
    occludedImg = imresize(imread(outFile), [H, W]);
    % We'll interpret row=1..H, col=1..W in that image.

    %% 5) Build random completions => images array
    images = zeros(H, W, 3, numImages, 'uint8');
    for i=1:numImages
        fn = sprintf('completion_%04d_%04d.png', silhouette_index, i);
        tmp = imread(fn);
        tmp = imresize(tmp, [H, W]);
        images(:,:,:,i) = tmp;
    end

    %% 6) Build an occluderMask in the same "drawing" coordinates
    % But here's the tricky part: if poly2mask wants x=column,y=row,
    % we have no direct 'scaled' approach. Let's do a basic approach:
    % We'll also just "draw" the occluder => occluder.png => threshold => mask
    % or we replicate your patch approach as an offscreen figure:
    maskFile = 'temp_occluder.png';
    figB = figure('Visible','off');
    patch(occluder(:,1), occluder(:,2), 'w','FaceAlpha',1,'LineWidth',2);
    axis off; axis equal;
    fdB = getframe(gca);
    maskImg = frame2im(fdB);
    imwrite(maskImg, maskFile);
    close(figB);
    maskResized = imresize(imread(maskFile), [H W]);
    % Now let's treat white=1 => occluder, everything else=0
    occluderMask = imbinarize(rgb2gray(maskResized), 0.5);

    %% 7) Generate a log-based heatmap => "myHeatmap.mat"
    % We'll define a function handle for the activation:
    % (You must define 'trainedNet' + 'layerName' outside or pass it in)
    layerName='classoutput';  % or whichever
    activations_fn = @(img) squeeze(activations(trainedNet, img, layerName));

    % We'll do a version that saves to "myHeatmap.mat" => normalized_heatmap
    computeOccluderHeatmapLog(occludedImg, images, occluderMask, activations_fn, 'myHeatmap.mat');

    %% 8) Shortest path:
    %  a) load normalized_heatmap => [H x W]
    data = load('myHeatmap.mat','normalized_heatmap');
    heatmap = data.normalized_heatmap;

    %  b) convert start_pt, end_pt => row,col in [1..H], [1..W]
    % Here is the mismatch: we have intersection points in original coords,
    % but we displayed them into an image of size [someWidth x someHeight].
    % We also did getframe => partial or bigger. So let's do a naive approach:
    % We'll do bounding box => or we treat the bounding box from 
    % the min and max of silhouette+occluder => "some approach".
    % 
    % For minimal approach, let's guess a direct proportion:
    [minx, maxx] = bounds([silhouette(:,1); occluder(:,1)]);
    [miny, maxy] = bounds([silhouette(:,2); occluder(:,2)]);

    row_start = round( (start_pt(2)-miny)/(maxy-miny)*(H-1)+1 );
    col_start = round( (start_pt(1)-minx)/(maxx-minx)*(W-1)+1 );
    row_end   = round( (end_pt(2)  -miny)/(maxy-miny)*(H-1)+1 );
    col_end   = round( (end_pt(1)  -minx)/(maxx-minx)*(W-1)+1 );

    % clamp to [1..H/W]
    row_start = max(min(row_start,H),1);
    col_start = max(min(col_start,W),1);
    row_end   = max(min(row_end,H),1);
    col_end   = max(min(col_end,W),1);

    % c) run shortestPathOnHeatmap
    pathRC = shortestPathOnHeatmap(heatmap, occluderMask,...
        [row_start, col_start], [row_end, col_end]);

    % d) Plot path on occludedImg
    figure('Name','Shortest Path from Intersection');
    imshow(occludedImg); hold on;
    plot(pathRC(:,2), pathRC(:,1),'r-','LineWidth',2);
    plot(col_start, row_start,'go','MarkerSize',8,'LineWidth',2);
    plot(col_end,   row_end,'go','MarkerSize',8,'LineWidth',2);
    title('Dijkstra Path in Resized Coordinates');

end
