%% ========================================================
%  APR_generateShapes Script
%
%  This script ensures all generated images (occluded, silhouette, random 
%  completions) are EXACTLY [H x W], using the same bounding box for each
%  shape so there is no pixel shift.
%
%  Steps:
%   1) Load silhouettes & occluders from your data (stim.natural.animal).
%   2) Pick a random silhouette/occluder from one class.
%   3) Compute a single bounding box from silhouette + occluder.
%   4) Save "silhouette + occluder" and "silhouette alone" images at final [H x W].
%   5) Find intersection points (for bSpline).
%   6) For i=1..nImages:
%       - pick a random segment from a different class,
%       - bSplineOccluderFit => aligned_segment
%       - merge partial silhouette => new_shape
%       - save final "completion" image in [H x W], same bounding box
%   7) Save everything in shapes.mat
%
%  Requirements:
%   - bSplineOccluderFit with nSamples=100
%   - 'stim' data from "naturalregulargestaltstim01.mat"
%   - "functions" folder on path
%
%  Author: Hannes Schaetzle (adapted for single bounding box)
%  Date:   2025-03-17
%% ========================================================

clear; clc; close all;

%% 1) Load data
load('/Users/I743312/Documents/MATLAB/CNN Project/data/naturalregulargestaltstim01.mat', 'stim');
addpath(genpath('functions'));  % folder with bSplineOccluderFit, etc.

rng(1000083);  % reproducibility

%novel = stim.natural.novel;
%unique_novel = unique([novel.dorig]);

animals = stim.natural.animal;
unique_animals = unique({animals.animal});

%% 2) Pick silhouette + occluder from random class
sil_class = unique_animals{randi(numel(unique_animals))};
sil_idx = find(strcmp({animals.animal}, sil_class));
silhouette_index = sil_idx(randi(numel(sil_idx)));
silhouette = animals(silhouette_index).c;
occluder   = animals(silhouette_index).o;

%sil_class = unique_novel(randi(numel(unique_novel)));
%sil_idx = find([novel.dorig] == sil_class);
%silhouette_index = sil_idx(randi(numel(sil_idx)));
%silhouette = novel(silhouette_index).c;
%occluder   = novel(silhouette_index).o;

%% Desired final output size
H = 600;  % e.g. 600
W = 900;  % e.g. 900

%% 3) Compute bounding box for silhouette+occluder
allX = [silhouette(:,1); occluder(:,1)];
allY = [silhouette(:,2); occluder(:,2)];
margin = 5;
minX = floor(min(allX) - margin);
maxX = ceil(max(allX) + margin);
minY = floor(min(allY) - margin);
maxY = ceil(max(allY) + margin);
widthBB  = maxX - minX + 1;
heightBB = maxY - minY + 1;

fprintf('Bounding box => width=%d, height=%d\n', widthBB, heightBB);

%% 4) Create output folders
mainOutDir = 'data/generated_images';
folders = {'with_occluder','no_occluder','random_segments'};
for f = 1:numel(folders)
    outPath = fullfile(mainOutDir, folders{f});
    if ~exist(outPath, 'dir'), mkdir(outPath); end
end
withOccluderDir   = fullfile(mainOutDir, 'with_occluder');
noOccluderDir     = fullfile(mainOutDir, 'no_occluder');
randomSegmentsDir = fullfile(mainOutDir, 'random_segments');

imgA_name = fullfile(withOccluderDir, sprintf('silOccl_%04d.png', silhouette_index));
imgB_name = fullfile(noOccluderDir,   sprintf('sil_%04d.png', silhouette_index));

%% 4a) Save silhouette+occluder => [H x W] with single bounding box
drawAndSave({silhouette, occluder}, {[0 0 0],[0.5 0.5 0.5]}, ...
    minX, minY, widthBB, heightBB, ...
    W, H, imgA_name);

%% 4b) Save silhouette alone => same bounding box => [H x W]
drawAndSave({silhouette}, {[0 0 0]}, ...
    minX, minY, widthBB, heightBB, ...
    W, H, imgB_name);

%% 5) Intersection points
intersection_points = find_intersection_points(silhouette, occluder);
if isempty(intersection_points)
    error('No valid intersection points found between silhouette & occluder.');
end
start_pt = intersection_points(1,:);
end_pt   = intersection_points(2,:);

%% 6) bSpline + random segments
fraction       = 0.40;
numControlPts  = 16; 
alphaShape     = 1.5;
betaCurvature  = 0.1;
maxIters       = 200;
nImages        = 1000;

shapes(nImages) = struct(...
    'silhouette', silhouette, 'occluder', occluder,...
    'random_segment', [], 'aligned_segment', [], 'new_shape', [],...
    'imgFile_withOccluder', imgA_name, 'imgFile_silhouetteOnly', imgB_name,...
    'imgFile_completion', '' );

for i = 1:nImages
    close all;

    %% (a) Different class for random segment
    other_classes = setdiff(unique_animals, sil_class);
    seg_class = other_classes{randi(numel(other_classes))};
    seg_idx = find(strcmp({animals.animal}, seg_class));
    seg_index = seg_idx(randi(numel(seg_idx)));
    alt_shape = animals(seg_index).imc;

    %other_classes = setdiff(unique_novel, sil_class);
    %seg_class = other_classes(randi(numel(other_classes)));
    %seg_idx = find([novel.dorig] == seg_class);
    %seg_index = seg_idx(randi(numel(seg_idx)));
    %alt_shape = novel(seg_index).imc;

    %% (b) Extract fraction
    [random_segment, ~] = extract_random_segment(alt_shape, fraction);

    %% (c) bSpline
    aligned_segment = bSplineOccluderFit(...
        random_segment, start_pt, end_pt,...
        occluder, numControlPts, alphaShape, betaCurvature, maxIters);

    %% (d) Combine partial silhouette
    in = inpolygon(silhouette(:,1), silhouette(:,2), occluder(:,1), occluder(:,2));
    idx = find(in);
    contour1 = [silhouette(idx(end)+1:end,:); silhouette(1:idx(1)-1,:)];

    rmse = sqrt(sum((aligned_segment(1,:)-contour1([1 end],:)).^2,2));
    if find(rmse==min(rmse))==1
        aligned_segment = flipud(aligned_segment);
    end
    new_shape = [aligned_segment; contour1];

    %% (e) Save bounding-box => [H x W]
    outFile = fullfile(randomSegmentsDir, ...
        sprintf('completion_%04d_%04d.png', silhouette_index, i));
    drawAndSave({new_shape}, {[0 0 0]}, ...
        minX, minY, widthBB, heightBB, ...
        W, H, outFile);

    %% (f) Store data
    shapes(i).intersection_points = intersection_points;
    shapes(i).silhouette = silhouette;
    shapes(i).occluder = occluder;
    shapes(i).random_segment  = random_segment;
    shapes(i).aligned_segment = aligned_segment;
    shapes(i).new_shape       = new_shape;
    shapes(i).imgFile_completion = outFile;
end

%% 7) Save shapes
save('shapes.mat','shapes','-v7.3');
fprintf('All images forced to bounding box => final size [%dx%d].\n',H,W);


%% ================= HELPER FUNCTION =================
function drawAndSave(polygons, colors, minX, minY, wBB, hBB, W, H, outFile)
% drawAndSave  Plots polygons in the EXACT same bounding box,
% forcing final output to [H x W].
%
% Inputs:
%   polygons : cell array of Nx2 coords (silhouette, occluder, etc.)
%   colors   : cell array of [r g b], same length as polygons
%   minX, minY, wBB, hBB : bounding box in original coordinates
%   W, H     : final image size
%   outFile  : path to .png

fig = figure('Visible','off','Units','pixels',...
    'Position',[100,100,wBB,hBB],'Color','w');
ax  = axes('Parent',fig,'Units','normalized','Position',[0 0 1 1]);
hold(ax,'on'); axis(ax,'off','equal');
xlim(ax,[0 wBB]); ylim(ax,[0 hBB]);

% shift each polygon so minX->0, minY->0
for k=1:numel(polygons)
    coords = polygons{k};
    coordsShifted = [coords(:,1)-minX, coords(:,2)-minY];
    patch(ax, coordsShifted(:,1), coordsShifted(:,2),...
        colors{k}, 'EdgeColor','none');
end

drawnow;
frameData = getframe(ax);
img = frame2im(frameData);

% Force final to [H x W]
img = imresize(img, [H,W]);
imwrite(img, outFile);

close(fig);
end
