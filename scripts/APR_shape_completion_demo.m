function APR_shape_completion_demo()
    % shape_completion_importance_sampling_sharpened_demo
    %
    % 1) Loads silhouette, occluder, intersection points, and heatmap.
    % 2) Rescales them into [1..width] x [1..height] with bounding box + flipped Y.
    % 3) Creates an occluder mask, fits a GMM in that region (with replicates, 
    %    regularization, etc.) to encourage tighter components.
    % 4) Iteratively samples points:
    %    - For each step, draw M samples from the GMM
    %    - Keep only those within 'distanceRadius' of the last chosen pixel
    %    - Sharpen the PDF by exponent alpha>1 => pdf^alpha
    %    - Sample one point from that sharpened distribution
    % 5) Forces the final anchor to be intersection point #2
    % 6) Fits a 2D spline through all chosen points, from anchor1 to anchor2
    %
    % Author: (Your Name)
    % Date:   (Today's Date)

    clear; clc; close all;

    %% --- Load shape data
    shapesFile = 'shapes_0046.mat';
    S = load(shapesFile, 'shapes');
    shape = S.shapes(1);
    silhouette = shape.silhouette;
    occluder   = shape.occluder;
    intersection_points = shape.intersection_points;

    fprintf('Loaded shapes_0046 => silhouette[%dx2], occluder[%dx2].\n', ...
        size(silhouette,1), size(occluder,1));

    %% --- Load heatmap
    heatmapFile = 'heatmap_0046.mat';
    H = load(heatmapFile, 'normalized_heatmap');
    heatmap = H.normalized_heatmap;  % e.g. 227x227
    [height, width] = size(heatmap);
    fprintf('Loaded heatmap => size=[%dx%d]\n', height, width);

    %% --- Bounding Box + Flip Y (as in APR_runProbabilisticCompletionDemo)
    allX = [silhouette(:,1); occluder(:,1)];
    allY = [silhouette(:,2); occluder(:,2)];
    margin = 5;
    minX = floor(min(allX) - margin);
    maxX = ceil(max(allX) + margin);
    minY = floor(min(allY) - margin);
    maxY = ceil(max(allY) + margin);

    widthBB  = maxX - minX + 1;
    heightBB = maxY - minY + 1;

    % Rescale silhouette, occluder, intersections
    silhouette_227 = [ ...
        (silhouette(:,1) - minX) * (width  / widthBB), ...
        (maxY - silhouette(:,2)) * (height / heightBB)];
    occluder_227   = [ ...
        (occluder(:,1)   - minX) * (width  / widthBB), ...
        (maxY - occluder(:,2))   * (height / heightBB)];
    intersection_227 = [ ...
        (intersection_points(:,1) - minX) * (width  / widthBB), ...
        (maxY - intersection_points(:,2)) * (height / heightBB)];

    anchor1 = intersection_227(1,:);
    anchor2 = intersection_227(2,:);

    %% --- Create Occluder Mask
    occluder_mask = poly2mask(occluder_227(:,1), occluder_227(:,2), height, width);

    %% --- Prepare Data for GMM (only inside occluder + positive heatmap)
    [xx, yy]  = meshgrid(1:width, 1:height);
    X_all     = [xx(:), yy(:)];
    heat_vals = heatmap(:);
    linInd    = sub2ind([height,width], yy(:), xx(:));
    inMask    = occluder_mask(linInd);

    valid     = (heat_vals > 0) & inMask;
    X_valid   = X_all(valid,:);
    w_valid   = heat_vals(valid);

    w_valid = w_valid.^3;  % try ^2, ^3, or even exp(w_valid)


    % Normalize weights (optional)
    w_valid = w_valid / sum(w_valid);

    %% --- Fit GMM 
    numComponents = 5;  
    fprintf('Fitting GMM with %d components in occluded region...\n', numComponents);

    gmModel = fitgmdist(X_valid, numComponents, ...
        'Weights', w_valid, ...
        'CovarianceType','full', ...
        'SharedCovariance', false, ...
        'RegularizationValue',1e-3, ...  % bigger => narrower lumps
        'Replicates',5, ...             % multiple tries => better local solution
        'Options', statset('MaxIter',1000));

    fprintf('GMM fit done.\n');

    %% --- Iterative Sampling with Sharpened PDF
    % Start at anchor1, end at anchor2, pick intermediate points by drawing M samples,
    % filtering by distance, then weighting by pdf^alpha

    N = 25;  % total chain length
    chosenPts = zeros(N, 2);
    chosenPts(1,:) = anchor1;
    chosenPts(end,:) = anchor2;

    M = 10000;         % how many GMM samples each iteration
    distanceRadius = 15;  
    alpha = 3.0;       % sharpen exponent (pdf^alpha)

    for i = 2:(N-1)
        prevPt = chosenPts(i-1,:);
        
        % Draw M samples from the GMM
        samples = random(gmModel, M);

        % Keep only in [1..width, 1..height]
        inBounds = (samples(:,1)>=1 & samples(:,1)<=width & ...
                    samples(:,2)>=1 & samples(:,2)<=height);
        samples = samples(inBounds,:);
        if isempty(samples)
            % fallback if no in-bounds samples
            chosenPts(i,:) = prevPt;
            continue;
        end

        % Distance constraint
        dists = sqrt(sum((samples - prevPt).^2,2));
        keep = (dists <= distanceRadius);
        closeSamples = samples(keep,:);

        if isempty(closeSamples)
            % fallback if none are within that radius
            chosenPts(i,:) = prevPt;
            continue;
        end

        % Evaluate GMM pdf
        pdfVals = pdf(gmModel, closeSamples);

        % Sharpen
        pdfVals = pdfVals.^alpha;
        pdfVals = pdfVals / sum(pdfVals);

        % Importance-sample exactly 1 point
        cdfVals = cumsum(pdfVals);
        r = rand();
        idxPick = find(cdfVals >= r, 1, 'first');
        chosenPts(i,:) = closeSamples(idxPick,:);
    end


    %% --- Fit a 2D Spline from anchor1 -> anchor2
    t = linspace(0,1,N);
    x_spline = csape(t, chosenPts(:,1), 'not-a-knot');
    y_spline = csape(t, chosenPts(:,2), 'not-a-knot');

    tFine = linspace(0,1,200);
    x_curve = fnval(x_spline, tFine);
    y_curve = fnval(y_spline, tFine);

    %% --- Visualization
    figure('Name','High-Probability Completion','Color','w');
    imagesc(heatmap);
    axis image xy;  
    colormap jet;  
    hold on;

    title('(1) GMM Fit + (2) Sharpened Sampling + (3) Spline from Anchor1 to Anchor2');
    
    % Plot occluder boundary in white
    plot(occluder_227(:,1), occluder_227(:,2), 'w-','LineWidth',2);

    % Plot chosen points
    for i = 1:N
        plot(chosenPts(i,1), chosenPts(i,2), 'ro','MarkerSize',6,'LineWidth',1.5);
        drawnow; pause(0.3);
    end

    % Plot final spline
    plot(x_curve, y_curve, 'g-','LineWidth',2);

    % Mark anchors
    plot(anchor1(1), anchor1(2), 'ms','LineWidth',2,'MarkerSize',8);
    plot(anchor2(1), anchor2(2), 'bs','LineWidth',2,'MarkerSize',8);

    hold off;
end
