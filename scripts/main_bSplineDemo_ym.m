%% ========================================================
%  bSplineOccluderFit Demo Script
%
%  This script:
%   1) Loads silhouettes and occluders from a nested data structure.
%   2) Randomly selects a silhouette (and associated occluder) from one class.
%   3) Finds two intersection points on the silhouette/occluder boundary.
%   4) Extracts a random fraction of the selected segment.
%   5) Deforms that extracted segment so it:
%         - fixes its endpoints to the two intersection points,
%         - remains inside the occluder,
%         - maintains smooth curvature,
%         - stays close (Procrustes-like) to its original shape.
%   6) Saves:
%         (2a) The silhouette with occluder,
%         (2b) The silhouette alone,
%         (1c) The new shapes (aligned segments) repeated nImages times.
%
%  Requirements:
%   - MATLAB Optimization Toolbox (fmincon).
%   - The functions in 'directory' must be on the MATLAB path.
%   - The 'stim' data file must exist at the specified path.
%
%  Author: Hannes Schätzle
%  Date:   2025-03-14
%% ========================================================

clear; clc; close all;

%% 1. Load Data
load('/Users/I743312/Documents/MATLAB/CNN Project/data/naturalregulargestaltstim01.mat', 'stim');
directory = 'functions'; % Adjust path to your function folder
addpath(genpath(directory));

% Optional for reproducibility
rng(1);

% Access nested structure: stim.natural.animal
animals = stim.natural.animal;
unique_animals = unique({animals.animal});

%% 2. Pick Silhouette & Occluder
silhouette_class = unique_animals{randi(length(unique_animals))};
silhouette_indices = find(strcmp({animals.animal}, silhouette_class));
silhouette_index = silhouette_indices(randi(length(silhouette_indices)));
silhouette = animals(silhouette_index).c;
occluder = animals(silhouette_index).o;

%% Create output directories for the three image types
mainOutDir = 'data/generated_images';
folders = {'with_occluder','no_occluder','random_segments'};
for f = 1:numel(folders)
    outPath = fullfile(mainOutDir, folders{f});
    if ~exist(outPath, 'dir')
        mkdir(outPath);
    end
end
withOccluderDir    = fullfile(mainOutDir, 'with_occluder');
noOccluderDir      = fullfile(mainOutDir, 'no_occluder');
randomSegmentsDir  = fullfile(mainOutDir, 'random_segments');

%% (2a) Silhouette + Occluder
figA = figure('Visible','off');
patch(silhouette(:,1), silhouette(:,2), 'k', 'FaceAlpha',1,'LineWidth',2);
hold on;
patch(occluder(:,1), occluder(:,2), 'k', ...
      'EdgeColor',[0.5 0.5 0.5], ...
      'FaceColor',[0.5 0.5 0.5], ...
      'LineWidth',1);
axis off; axis equal;
frameDataA = getframe(gca);
imgA = frame2im(frameDataA);
% e.g. "silOccl_0012.png" if silhouette_index is 12
imgA_name = fullfile(withOccluderDir, sprintf('silOccl_%04d.png', silhouette_index));
imwrite(imgA, imgA_name);
close(figA);

%% (2b) Silhouette alone (no occluder)
figB = figure('Visible','off');
patch(silhouette(:,1), silhouette(:,2), 'k', 'FaceAlpha',1,'LineWidth',2);
axis off; axis equal;
frameDataB = getframe(gca);
imgB = frame2im(frameDataB);
% e.g. "sil_0012.png"
imgB_name = fullfile(noOccluderDir, sprintf('sil_%04d.png', silhouette_index));
imwrite(imgB, imgB_name);
close(figB);

%% 3. Intersection Points
intersection_points = find_intersection_points(silhouette, occluder);
if isempty(intersection_points)
    error('No valid intersection points found between silhouette and occluder.');
end
start_pt = intersection_points(1,:);
end_pt   = intersection_points(2,:);

%% 4. Extraction parameters
fraction = 0.40;  % e.g., 40% of alt_shape’s length

%% 5. B-spline Optimization settings
numControlPoints = 16;
alphaShape       = 1.5;
betaCurvature    = 0.1;
maxIters         = 200;

% We will generate nImages "new shapes" with random segments
nImages = 100;

for i = 1:nImages
    close all;

    %% 6. Pick a Different Class for the Segment
    other_classes = setdiff(unique_animals, silhouette_class);
    segment_class = other_classes{randi(length(other_classes))};
    segment_indices = find(strcmp({animals.animal}, segment_class));
    segment_index = segment_indices(randi(length(segment_indices)));
    alt_shape = animals(segment_index).imc;

    %% 7. Extract a Random Fraction of the Selected Segment
    [random_segment, ~] = extract_random_segment(alt_shape, fraction);

    %% Deform the segment with bSplineOccluderFit
    aligned_segment = bSplineOccluderFit( ...
        random_segment, start_pt, end_pt, ...
        occluder, numControlPoints, alphaShape, betaCurvature, maxIters);

    %% Visualization / Combining with portion of silhouette inside occluder
    in = inpolygon(silhouette(:,1), silhouette(:,2), occluder(:,1), occluder(:,2));
    id = find(in);
    contour1 = [silhouette(id(end)+1:end,:); silhouette(1:id(1)-1,:)];

    rmse = sqrt(sum((aligned_segment(1,:)-contour1([1 end],:)).^2,2));
    if find(rmse==min(rmse))==1
        aligned_segment = flipud(aligned_segment);
        new_shape = [aligned_segment; contour1];
    else
        new_shape = [aligned_segment; contour1];
    end

    %% 8. Plot the "new shape" for saving
    figC = figure('Visible','off');
    patch(new_shape(:,1), new_shape(:,2), 'k', 'FaceAlpha',1, 'LineWidth',2);
    axis equal; axis off;
    frameDataC = getframe(gca);
    imgC = frame2im(frameDataC);

    % e.g. "completion_0012_0001.png" if silhouette_index=12, i=1
    imgC_name = fullfile(randomSegmentsDir, ...
        sprintf('completion_%04d_%04d.png', silhouette_index, i));
    imwrite(imgC, imgC_name);
    close(figC);

    % Optionally store info about each shape
    nshapes(i).c = new_shape;         % final shape
    nshapes(i).o = occluder;          % occluder
    nshapes(i).aligned_segment = aligned_segment;
    nshapes(i).rand_segment = random_segment;
     %% 7. Save info about each shape
    outFile = 'shapes.mat';
    save(outFile, 'new_shape', 'occluder', 'aligned_segment', 'random_segment', '-v7.3');
end

