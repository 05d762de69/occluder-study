%% getOccluderHeatmap.m
%
% This script demonstrates a fully consistent pipeline:
%  1) Load shapes & trained network.
%  2) Scale silhouette + occluder to [1..227] coords.
%  3) Find intersection points in scaled coords.
%  4) Build the occluded image [227x227x3].
%  5) Build occluderMask [227x227].
%  6) Create a 4D array of random completions (also scaled).
%  7) Compute heatmap => "Heatmap_chicken.mat" via computeOccluderHeatmapLogMat.
%  8) runShortestPathContour(...) => minimal-cost path from intersection #1->#2
%
% Requirements:
%   - computeOccluderHeatmapLogMat.m (saves normalized_heatmap to mat)
%   - runShortestPathContour.m (finds 2 intersection points, calls shortestPathOnHeatmap)
%   - shapes(...) data, networkFile, etc.
%
% Usage:
%   >> run('getOccluderHeatmap.m');

clear; clc; close all;

%% 1) Load your network + shapes
networkFile   = '/Users/I743312/Documents/MATLAB/CNN Project/data/trainedNet.mat';
shapesFile    = '/Users/I743312/Documents/MATLAB/CNN Project/data/shapes_chicken.mat';
load(networkFile, 'trainedNet');
S = load(shapesFile, 'shapes');

if ~isfield(S,'shapes') || isempty(S.shapes)
    error('No shapes found in %s', shapesFile);
end
shapeData     = S.shapes(1);

%% 2) Scale silhouette + occluder to [1..227]
H = 227; W = 227;  % final size
silhouette     = shapeData.silhouette;  % Nx2, original coords
occluder       = shapeData.occluder;    % Mx2

% We'll do rescale for X coords => [1..W], Y coords => [1..H]
silhouette_scaled(:,1) = rescale(silhouette(:,1), 1, W);
silhouette_scaled(:,2) = rescale(silhouette(:,2), 1, H);

occluder_scaled(:,1)   = rescale(occluder(:,1),   1, W);
occluder_scaled(:,2)   = rescale(occluder(:,2),   1, H);

%% 3) Find intersection points in scaled coords
[xi, yi] = polyxpoly(silhouette_scaled(:,1), silhouette_scaled(:,2), ...
                     occluder_scaled(:,1),   occluder_scaled(:,2));
if length(xi)<2
    error('Not enough intersection points found in scaled coords');
end
start_pt = [xi(1), yi(1)]; 
end_pt   = [xi(2), yi(2)];

fprintf('Scaled intersection points => start_pt=(%.2f,%.2f), end_pt=(%.2f,%.2f)\n',...
    start_pt(1), start_pt(2), end_pt(1), end_pt(2));

%% 4) Build the occluded image [227x227x3]
occludedImg = createOccludedImage(silhouette_scaled, occluder_scaled, H, W);

%% 5) Build occluderMask [227x227]
occluderMask = poly2mask(occluder_scaled(:,1), occluder_scaled(:,2), H, W);

%% Show partial check
figure('Name','Check Occluded Image + Intersection');
imshow(occludedImg); hold on;
plot(start_pt(1),start_pt(2),'go','MarkerSize',10,'LineWidth',2);
plot(end_pt(1),  end_pt(2),'go','MarkerSize',10,'LineWidth',2);
title('Scaled Silhouette + Occluder + Intersection Points');

%% 6) Create the 4D array of random completions (scaled)
numImages = 1000;  % or however many you have
images = zeros(H, W, 3, numImages, 'uint8');

for i = 1:numImages
    fname = sprintf('/Users/I743312/Documents/MATLAB/CNN Project/data/generated_images_bird/random_segments/completion_0014_%04d.png', i);
    tmp   = imread(fname);
    % Since that image might be in original coords, we rely on 
    % the same approach: just imresize => [227x227].
    tmp = imresize(tmp, [H, W]);
    images(:,:,:,i) = tmp;
end

%% 7) Compute the log-based heatmap => 'Heatmap_chicken.mat'
% Use the function that saves normalized_heatmap in a mat file:
activations_fn = @(img) squeeze(activations(trainedNet, img, 'classoutput'));

% We rely on computeOccluderHeatmapLogMat version:
computeOccluderHeatmapLog(occludedImg, images, occluderMask, activations_fn, 'Heatmap_chicken.mat');

%% 8) Run the shortest-path approach from intersection #1->#2
% The function runShortestPathContour does:
%  (a) load shapes => but we want scaled shapes => let's see
% Actually, let's just skip that shape re-load step and do it ourselves:
% We'll directly do the path with:
%   - the just-saved 'Heatmap_chicken.mat'
%   - the start/end points in row,col
%   - the occluderMask

% Or we can adapt runShortestPathContour so it doesn't reload shapes 
% but uses our scaled data. For clarity, let's do it inline:

% 8a) load the final heatmap
data = load('Heatmap_chicken.mat','normalized_heatmap');
heatmap = data.normalized_heatmap; % [227x227]

% 8b) Convert start_pt, end_pt => row,col
start_rc = [ round(start_pt(2)), round(start_pt(1)) ];
end_rc   = [ round(end_pt(2)),   round(end_pt(1)) ];

% 8c) shortest path
pathRC = shortestPathOnHeatmap(heatmap, occluderMask, start_rc, end_rc);

% 8d) plot path on the occludedImg
figure('Name','Shortest Path on Heatmap');
imshow(occludedImg); hold on;
plot(pathRC(:,2), pathRC(:,1), 'r-', 'LineWidth',2);
plot(start_rc(2),start_rc(1),'go','MarkerSize',8,'LineWidth',2);
plot(end_rc(2),  end_rc(1),'go','MarkerSize',8,'LineWidth',2);
title('Dijkstra Path from intersection #1->#2, scaled coords');
