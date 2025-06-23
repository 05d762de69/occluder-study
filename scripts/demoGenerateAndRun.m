%% demoUseExistingImages.m
%
% This script:
%  1) Loads the existing occluded PNG and random-segment completions 
%     from /generated_images/with_occluder and /random_segments
%  2) Finds intersection points from silhouette + occluder data
%  3) Resizes everything to [227x227]
%  4) Builds an occluderMask by patch -> getframe, or poly2mask if you have the polygon
%  5) Calls computeOccluderHeatmapLogMat => myHeatmap.mat
%  6) Loads myHeatmap.mat => does shortestPathOnHeatmap 
%  7) Overlays path on the occluded [227x227] image
%
% "We don't re-generate the images" => we just re-use them from disk.

clear; clc; close all;

%% 1) Path config
baseDir = '/Users/I743312/Documents/MATLAB/CNN Project/data/generated_images_horses';
withOccluderDir   = fullfile(baseDir, 'with_occluder'); 
randomSegmentsDir = fullfile(baseDir, 'random_segments');
noOccluderDir     = fullfile(baseDir, 'no_occluder');

% We have e.g. "silOccl_0058.png" under with_occluder
silIndex = 40;   % or whichever index you used
occludedFile = fullfile(withOccluderDir, sprintf('silOccl_%04d.png', silIndex));

% We have random completions: "completion_0058_%04d.png"
numImages = 1000; % how many completions
H = 227; W = 227; % final size for AlexNet

%% 2) Silhouette + occluder data => from your shapes or animals
%   If you have them in some .mat, load them. 
%   Or if you have a separate "shapes" structure for index=58
%   We'll suppose:
%       silhouette = shapes(58).c
%       occluder   = shapes(58).o
%   We'll do a placeholder:

% (Placeholder) load them from a separate .mat if needed:
load('/Users/I743312/Documents/MATLAB/CNN Project/data/shapes_horses.mat','shapes');
silhouette = shapes(1).silhouette;   % Nx2
occluder   = shapes(1).occluder;   % Mx2
% Or if you have them in a struct called "animals" with index=...
% silhouette = []; % fill in from your data
% occluder   = [];

if isempty(silhouette) || isempty(occluder)
    error('You must fill silhouette, occluder from your shapes data for index=58');
end

%% 3) Find intersection points (in original coordinate system)
[xi, yi] = polyxpoly(silhouette(:,1), silhouette(:,2), ...
                     occluder(:,1),   occluder(:,2));
if length(xi)<2
    error('No valid intersections found for silhouette_index=%d.', silIndex);
end
start_pt = [xi(1), yi(1)];
end_pt   = [xi(2), yi(2)];

fprintf('Intersection points => (%.1f,%.1f), (%.1f,%.1f)\n',...
    start_pt(1), start_pt(2), end_pt(1), end_pt(2));

%% 4) Load the existing occluded PNG => [H x W]
occludedImg = imread(occludedFile);
occludedImg = imresize(occludedImg, [H W]);

%% 5) Create the 4D array of random completions
images = zeros(H, W, 3, numImages, 'uint8');
for i = 1:numImages
    fComp = fullfile(randomSegmentsDir, sprintf('completion_%04d_%04d.png', silIndex, i));
    tmp = imread(fComp);
    tmp = imresize(tmp,[H W]);
    images(:,:,:,i) = tmp;
end

%% 6) Build occluderMask by rasterizing the occluder polygon

H = 227; W = 227;

% Rescale occluder into image coordinates
minx = min(silhouette(:,1)); maxx = max(silhouette(:,1));
miny = min(silhouette(:,2)); maxy = max(silhouette(:,2));

scaled_x = (occluder(:,1) - minx) / (maxx - minx) * (W - 1) + 1;
scaled_y = (occluder(:,2) - miny) / (maxy - miny) * (H - 1) + 1;

% Use patch to render the occluder only
fig = figure('Visible','off');
axis off; axis equal;
set(gca, 'Units','pixels', 'Position', [1 1 W H]);

patch(scaled_x, scaled_y, [0.5 0.5 0.5], 'EdgeColor','none');

frame = getframe(gca);
imgGray = rgb2gray(frame2im(frame));
close(fig);

% Use the mean of scaled coordinates to pick a center pixel
center_col = round(mean(scaled_x));
center_row = round(mean(scaled_y));

% Ensure they're within bounds
center_col = min(max(center_col,1), W);
center_row = min(max(center_row,1), H);

% Threshold based on the gray occluder intensity
grayVal = imgGray(center_row, center_col);
occluderMask = imgGray == grayVal;
occluderMask = logical(imresize(occluderMask, [H W]));  % Final mask



%% 7) Compute the log-based heatmap => say we name it "myHeatmap.mat"
% We'll define an activation function for your network:
layerName = 'classoutput';
activations_fn = @(img) squeeze(activations(trainedNet, img, layerName));

computeOccluderHeatmapLog(occludedImg, images, occluderMask, activations_fn, 'myHeatmap.mat');

%% 8) Load the heatmap => do shortest path
data = load('myHeatmap.mat','normalized_heatmap');
heatmap = data.normalized_heatmap; % [H x W]

% convert intersection points => row,col in [1..H/W] 
row_start = round((start_pt(2)-miny)/(maxy-miny)*(H-1)+1);
col_start = round((start_pt(1)-minx)/(maxx-minx)*(W-1)+1);
row_end   = round((end_pt(2)  -miny)/(maxy-miny)*(H-1)+1);
col_end   = round((end_pt(1)  -minx)/(maxx-minx)*(W-1)+1);

% clamp
row_start = max(min(row_start,H),1);
col_start = max(min(col_start,W),1);
row_end   = max(min(row_end,H),1);
col_end   = max(min(col_end,W),1);

% path
pathRC = shortestPathOnHeatmap(heatmap, occluderMask, [row_start,col_start], [row_end,col_end]);

%% 9) Show final
figure('Name','Final Shortest Path');
imshow(occludedImg);
hold on;
plot(pathRC(:,2), pathRC(:,1), 'r-','LineWidth',2);
plot(col_start,row_start,'go','MarkerSize',8,'LineWidth',2);
plot(col_end,  row_end,  'go','MarkerSize',8,'LineWidth',2);
title('Minimal-Cost Path from Intersection #1->#2');

%% Debug
load('heatmap_log.mat', 'normalized_heatmap');
occludedImg = imread('/Users/I743312/Documents/MATLAB/CNN Project/data/generated_images_horses/with_occluder/silOccl_0040.png');
occludedImg = imresize(occludedImg, [227 227]);

overlayHeatmapOnImage(occludedImg, normalized_heatmap);
