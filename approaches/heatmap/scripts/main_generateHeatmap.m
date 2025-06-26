%% main_generateHeatmap.m
%
% Example usage of APR_heatmap.m
%
% 1) Adjust the file paths as needed
% 2) Run the script
% 3) See the overlay figure & saved output
%
% Author: Your Name
% Date:   2025-04-01

clear; clc; close all;

%% Paths
netFile        = '/Users/I743312/Documents/MATLAB/occluder-study/data/models/trainedNet_20250624.mat';
occludedFile   = '/Users/I743312/Documents/MATLAB/CNN Project/data/generated_images/with_occluder/silOccl_0056.png';
completionDir  = '/Users/I743312/Documents/MATLAB/CNN Project/data/generated_images/random_segments';
outHeatmapMat  = 'heatmap_0056.mat';
outPng         = 'heatmap_0056_overlay.png';

%% Call the function
APR_heatmap(netFile, occludedFile, completionDir, outHeatmapMat, outPng);

fprintf('\nAll done. Check %s and %s!\n', outHeatmapMat, outPng);
