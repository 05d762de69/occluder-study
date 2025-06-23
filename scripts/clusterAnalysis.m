%% ========================================================
% clusterAnalysis.m
%% ========================================================

clear; clc; close all;

%% 1) File Paths
distFile   = '/Users/I743312/Documents/MATLAB/CNN Project/data/alexNetResponses.mat'; 
shapesFile = '/Users/I743312/Documents/MATLAB/CNN Project/data/shapes.mat';

%% 2) Clustering
% Suppose feats_randomSegments is [54 x 10000], shapes is 10000-element array
kRange = 2:10;
k = findBestKBySilhouette(distFile,kRange);
[labels, protos] = clusterCompletions(distFile,shapesFile,k);
% => labels is [10000 x 1], protos is n-element struct
%% 3) Determine Cluster with Smallest Average Distance to withOccluder
[closestC, avgDVec, bestShape] = findClosestClusterByAvgDistMatrix(distFile, labels);

fprintf('Closest cluster is %d (avg dist=%.3f)\n', closestC, avgDVec(closestC));
fprintf('Best shape in that cluster => %d\n', bestShape);

finalShape = shapes.(bestShape).new_shape;
figure;
patch(finalShape(:,1), finalShape(:,2),'k'); axis equal off;
title('Best shape from closest cluster');
