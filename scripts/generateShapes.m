%% ========================================================
%  generateShapes Script
%
%  This script:
%   1) Loads silhouettes & occluders from your data (stim.natural.animal).
%   2) Picks a random silhouette/occluder from one class.
%   3) Saves "silhouette + occluder" and "silhouette alone" images.
%   4) Finds intersection points.
%   5) Loops nImages times:
%       - picks a random segment from a different class,
%       - calls bSplineOccluderFit (which now has nSamples=100),
%       - merges the result with the partial silhouette,
%       - saves the final "new shape" image,
%       - stores everything in shapes(i).
%   6) Finally saves all shapes to "shapes.mat".
%
%  Requirements:
%   - bSplineOccluderFit with nSamples=100 enforced
%   - The 'stim' data structure
%   - The "functions" folder on the path
%
%  Author: Hannes Schaetzle
%  Date:   2025-03-17
%% ========================================================

clear; clc; close all;

%% 1. Load Data
load('/Users/I743312/Documents/MATLAB/CNN Project/data/naturalregulargestaltstim01.mat', 'stim');
directory = 'functions'; % Adjust path to your function folder
addpath(genpath(directory));

rng(93);  % for reproducibility

animals = stim.natural.animal;
%novel_shapes = stim.regularity;
unique_animals = unique({animals.animal});
%unique_novel = unique({novel_shapes.name})

%% 2. Pick Silhouette & Occluder
%silhouette_class = unique_novel{randi(length(unique_novel))};
%silhouette_indices = find(strcmp({novel_shapes.name}, silhouette_class));
%silhouette_index = silhouette_indices(randi(length(silhouette_indices)));
%silhouette = novel_shapes(silhouette_index).c;
%occluder   = novel_shapes(silhouette_index).o;

silhouette_class = unique_animals{randi(length(unique_animals))};
silhouette_indices = find(strcmp({animals.animal}, silhouette_class));
silhouette_index = silhouette_indices(randi(length(silhouette_indices)));
silhouette = animals(silhouette_index).c;
occluder   = animals(silhouette_index).o;


%% 3. Create Output Folders & Save "Silhouette + Occluder" / "Silhouette Alone"
mainOutDir = 'data/generated_images';
folders = {'with_occluder','no_occluder','random_segments'};
for f = 1:numel(folders)
    outPath = fullfile(mainOutDir, folders{f});
    if ~exist(outPath, 'dir'), mkdir(outPath); end
end
withOccluderDir   = fullfile(mainOutDir, 'with_occluder');
noOccluderDir     = fullfile(mainOutDir, 'no_occluder');
randomSegmentsDir = fullfile(mainOutDir, 'random_segments');

% (2a) Silhouette + Occluder
figA = figure('Visible','off');
patch(silhouette(:,1), silhouette(:,2), 'k', 'FaceAlpha',1,'LineWidth',2); 
hold on;
patch(occluder(:,1), occluder(:,2), 'k', ...
      'EdgeColor',[0.5 0.5 0.5], 'FaceColor',[0.5 0.5 0.5], 'LineWidth',1);
axis off; axis equal;
frameDataA = getframe(gca);
imgA = frame2im(frameDataA);
imgA_name = fullfile(withOccluderDir, sprintf('silOccl_%04d.png', silhouette_index));
imwrite(imgA, imgA_name);
close(figA);

% (2b) Silhouette alone
figB = figure('Visible','off');
patch(silhouette(:,1), silhouette(:,2), 'k', 'FaceAlpha',1,'LineWidth',2);
axis off; axis equal;
frameDataB = getframe(gca);
imgB = frame2im(frameDataB);
imgB_name = fullfile(noOccluderDir, sprintf('sil_%04d.png', silhouette_index));
imwrite(imgB, imgB_name);
close(figB);

%% 4. Find Intersection Points
intersection_points = find_intersection_points(silhouette, occluder);
if isempty(intersection_points)
    error('No valid intersection points found between silhouette and occluder.');
end
start_pt = intersection_points(1,:);
end_pt   = intersection_points(2,:);

%% 5. Random Segment Extraction & B-spline Settings
fraction       = 0.40;   % fraction of alt_shape
numControlPts  = 16; 
alphaShape     = 1.5;
betaCurvature  = 0.1;
maxIters       = 200;
nImages        = 1000;

%% Initialize struct array to store shapes
shapes(nImages) = struct( ...
    'silhouette', [], 'occluder', [], ...
    'random_segment', [], 'aligned_segment', [], 'new_shape', [], ...
    'imgFile_withOccluder', imgA_name, 'imgFile_silhouetteOnly', imgB_name, ...
    'imgFile_completion', '' ...
);

for i = 1:nImages
    close all;

    %% (a) Pick a Different Class for the Segment
    other_classes = setdiff(unique_animals, silhouette_class);
    segment_class = other_classes{randi(length(other_classes))};
    segment_indices = find(strcmp({animals.animal}, segment_class));
    segment_index = segment_indices(randi(length(segment_indices)));
    alt_shape = animals(segment_index).imc;

    %other_classes = setdiff(unique_novel, silhouette_class);
    %segment_class = other_classes{randi(length(other_classes))};
    %segment_indices = find(strcmp({novel_shapes.name}, segment_class));
    %segment_index = segment_indices(randi(length(segment_indices)));
    %alt_shape = novel_shapes(segment_index).imc;

    %% (b) Extract a Random Fraction
    [random_segment, ~] = extract_random_segment(alt_shape, fraction);

    %% (c) Deform with B-spline (which returns 100 points in aligned_segment)
    aligned_segment = bSplineOccluderFit( ...
        random_segment, start_pt, end_pt, ...
        occluder, numControlPts, alphaShape, betaCurvature, maxIters);

    %% (d) Combine with partial silhouette inside occluder
    in = inpolygon(silhouette(:,1), silhouette(:,2), occluder(:,1), occluder(:,2));
    id = find(in);
    contour1 = [silhouette(id(end)+1:end,:); silhouette(1:id(1)-1,:)];

    rmse = sqrt(sum((aligned_segment(1,:)-contour1([1 end],:)).^2,2));
    if find(rmse==min(rmse)) == 1
        aligned_segment = flipud(aligned_segment);
    end
    new_shape = [aligned_segment; contour1];

    %% (e) Save an image of this new shape
    figC = figure('Visible','off');
    patch(new_shape(:,1), new_shape(:,2), 'k', 'FaceAlpha',1, 'LineWidth',2);
    axis equal; axis off;
    frameDataC = getframe(gca);
    imgC = frame2im(frameDataC);
    imgC_name = fullfile(randomSegmentsDir, sprintf('completion_%04d_%04d.png', silhouette_index, i));
    imwrite(imgC, imgC_name);
    close(figC);

    %% (f) Store in shapes(i)
    shapes(i).silhouette      = silhouette;  % original silhouette
    shapes(i).occluder        = occluder;    % same occluder each time 
    shapes(i).random_segment  = random_segment;
    shapes(i).aligned_segment = aligned_segment;  % now always [100x2]
    shapes(i).new_shape       = new_shape;
    shapes(i).imgFile_completion = imgC_name;
end

%% 6. Save All in shapes.mat
save('shapes.mat', 'shapes', '-v7.3');
