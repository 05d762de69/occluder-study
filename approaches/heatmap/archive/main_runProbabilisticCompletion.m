% APR_runProbabilisticCompletionDemo
%
% Updated with:
%   - Occluder mask
%   - Importance sampling based on heatmap probabilities
%   - B-spline fit for smooth natural curve
%
% Author: Your Name
% Date: 2025-04-14

    clear; clc; close all;

    %% Load shape data
    shapesFile = '/Users/I743312/Documents/MATLAB/occluder-study/data/processed/stimuli_files/shapes_0046.mat';
    S = load(shapesFile, 'shapes');
    shape = S.shapes(1);
    silhouette = shape.silhouette;
    occluder = shape.occluder;
    intersection_points = shape.intersection_points;

    fprintf('Loaded shapes_0046 => silhouette[%dx2], occluder[%dx2].\n', ...
        size(silhouette,1), size(occluder,1));

    %% Load heatmap
    heatmapFile = '/Users/I743312/Documents/MATLAB/occluder-study/approaches/heatmap/data/processed/heatmaps/animal_shapes/heatmap_0046.mat';
    H = load(heatmapFile, 'normalized_heatmap');
    heatmap = H.normalized_heatmap;
    [height, width] = size(heatmap);
    fprintf('Loaded heatmap => size=[%dx%d]\n', height, width);

    %% Compute occluder mask in heatmap space
    allX = [silhouette(:,1); occluder(:,1)];
    allY = [silhouette(:,2); occluder(:,2)];
    margin = 5;
    minX = floor(min(allX) - margin);
    maxX = ceil(max(allX) + margin);
    minY = floor(min(allY) - margin);
    maxY = ceil(max(allY) + margin);
    widthBB = maxX - minX + 1;
    heightBB = maxY - minY + 1;

    % Rescale occluder polygon to [227 x 227], Y flipped
    occluder_shifted = [ ...
        (occluder(:,1)-minX) * (width / widthBB), ...
        (maxY - occluder(:,2)) * (height / heightBB)];
    occluder_mask = poly2mask(occluder_shifted(:,1), occluder_shifted(:,2), height, width);

    %% Prepare data for GMM
    [X, Y] = meshgrid(1:width, 1:height);
    X = X(:); Y = Y(:); probs = heatmap(:);
    valid = (probs > 1e-12);
    coords = [X(valid), Y(valid)];
    weights = probs(valid) / sum(probs(valid));

    %% Fit GMM
    numComponents = 7;
    fprintf('Fitting GMM with %d components...\n', numComponents);
    gm = fitgmdist(coords, numComponents, ...
        'Weights', weights, ...
        'CovarianceType','full', ...
        'RegularizationValue',1e-5, ...
        'Options', statset('MaxIter',500));
    fprintf('GMM fit done.\n');

    %% Sample and filter inside occluder
    samples = random(gm, 10000);
    samples = samples(samples(:,1) >= 1 & samples(:,1) <= width & ...
                      samples(:,2) >= 1 & samples(:,2) <= height, :);
    idx = sub2ind([height, width], round(samples(:,2)), round(samples(:,1)));
    inside = occluder_mask(idx);
    samples = samples(inside,:);

    %% Importance sampling: sample proportionally to heatmap
    sample_probs = interp2(reshape(X, height, width), reshape(Y, height, width), ...
                          heatmap, samples(:,1), samples(:,2), 'linear', 0);
    sample_probs(sample_probs < 0) = 0;
    sample_probs = sample_probs / sum(sample_probs);
    N_spline = 100;  % number of intermediate spline control points
    chosen_idx = randsample(size(samples,1), min(N_spline, size(samples,1)), true, sample_probs);
    spline_pts = samples(chosen_idx,:);

    %% Add rescaled/flipped anchors
    rescaleXY = @(pt) [ ...
        (pt(:,1)-minX) * (width / widthBB), ...
        (maxY - pt(:,2)) * (height / heightBB)];
    anchor1 = rescaleXY(intersection_points(1,:));
    anchor2 = rescaleXY(intersection_points(2,:));
    spline_pts = [anchor1; spline_pts; anchor2];

       %% Sort points along anchor vector (for consistent spline)
    vec = anchor2 - anchor1;
    projection = (spline_pts - anchor1) * vec' / norm(vec);
    [~, sortIdx] = sort(projection);
    spline_pts = spline_pts(sortIdx,:);

    %% Force interpolation using cubic spline through anchors
    t = linspace(0, 1, size(spline_pts,1));
    x_spline = csape(t, spline_pts(:,1), 'clamped');
    y_spline = csape(t, spline_pts(:,2), 'clamped');

    tFine = linspace(0, 1, 300);
    x_curve = fnval(x_spline, tFine);
    y_curve = fnval(y_spline, tFine);

    %% Visualization
    figure('Name','B-Spline Completion (Forced Anchors)');
    imagesc(heatmap); axis image xy; colormap('hot'); colorbar;
    hold on;
    scatter(samples(:,1), samples(:,2), 6, 'b', 'filled', 'MarkerFaceAlpha', 0.2);
    plot(anchor1(1), anchor1(2), 'go', 'MarkerSize', 8, 'LineWidth', 2);
    plot(anchor2(1), anchor2(2), 'go', 'MarkerSize', 8, 'LineWidth', 2);
    plot(x_curve, y_curve, 'r-', 'LineWidth', 2);
    title(sprintf('GMM(%d) + Interpolated B-Spline (Anchors Enforced)', numComponents));

