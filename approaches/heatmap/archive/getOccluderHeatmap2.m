%% getOccluderHeatmap.m
%
% Example script showing how to call 'computeOccluderHeatmap'
% (based on the pseudocode approach from Yaniv).
%
% Requirements:
%  - 'computeOccluderHeatmap.m' on your MATLAB path
%  - 'trainedNet.mat' containing a variable 'trainedNet' (your AlexNet or similar)
%  - An 'occludedImg' array (HxWx3), e.g. from shapes or a file
%  - A 'images' array (HxWx3xN) holding N completions with random segments
%
% Usage:
%   >> run('runOccluderHeatmap.m');
%   (then check the displayed figure or the saved 'heatmap.png')
%
% Author: Your Name
% Date: 2025-03-26

clear; clc; close all;

%% 1) Load your trained network
networkFile = '/Users/I743312/Documents/MATLAB/CNN Project/data/trainedNet.mat';
shapesFile = '/Users/I743312/Documents/MATLAB/CNN Project/data/shapes_horses.mat';
S = load(shapesFile, 'shapes');
load(networkFile, 'trainedNet');




% Create a function handle for final-layer activations
layerName = 'classoutput';
activations_fn = @(img) activations(trainedNet, img, layerName);

%% 2) Load occluded image & 4D array of completions
% Here, we assume you have these variables in a .mat file or from your workspace.
% For example:

% Suppose occluderMask is [H x W], true inside the occluder, false outside
H = 227; W = 227;
occluder = S.shapes(1).occluder;
occluder_scaled(:,1)   = rescale(occluder(:,1),   1, W);
occluder_scaled(:,2)   = rescale(occluder(:,2),   1, H);

occluderMask = poly2mask(occluder_scaled(:,1), occluder_scaled(:,2), H, W);

occludedImg = imread('/Users/I743312/Documents/MATLAB/CNN Project/data/generated_images_horses/with_occluder/silOccl_0040.png');  % replace XXXX
occludedImg = imresize(occludedImg, [227 227]);  % fix size for AlexNet

numImages = 1000;  % however many you have
images = zeros(227, 227, 3, numImages, 'uint8');

for i = 1:numImages
    fname = sprintf('/Users/I743312/Documents/MATLAB/CNN Project/data/generated_images_horses/random_segments/completion_0040_%04d.png', i); 
    tmp = imread(fname);
    tmp = imresize(tmp, [227 227]); 
    images(:,:,:,i) = tmp;
end


%% 3) Call computeOccluderHeatmap
outputFile = 'heatmap.mat';  % name of PNG to save the final heatmap
fprintf('Generating heatmap -> %s\n', outputFile);

%computeOccluderHeatmapLog(occludedImg, images, occluderMask, activations_fn, outputFile);
computeOccluderHeatmap(occludedImg, images, occluderMask, activations_fn, 'heatmap_overlay.mat');

%% 4) Calculate Contour based on heatmap

%runActiveContourOnHeatmap('/Users/I743312/Documents/MATLAB/CNN Project/heatmap_overlay.png','/Users/I743312/Documents/MATLAB/CNN Project/data/generated_images/with_occluder/silOccl_0058.png',occluderMask);
thresholdContourFromShapes(shapesFile, 1, 'heatmap_log.mat', 95, occluderMask, 1);
runActiveContourStrictHeatmap(normalized_heatmap, occluderMask);