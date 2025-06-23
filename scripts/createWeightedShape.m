%% ========================================================
% createWeightedShape.m
%
%  Part 3 of your pipeline:
%    1) Calls addGaussianWeightsMedianRule(...) to compute a Gaussian-based 
%       weight transform from distMatrix (alexNetResponses.mat),
%       and appends 'weightsGauss' into shapes.mat.
%    2) Renames 'weightsGauss' -> shapes(i).weight
%    3) Calls createWeightedSegment(...) to produce the final weighted line
%    4) Combines that weighted_segment with the partial silhouette (occluder)
%       to form a single "weighted_shape".
%    5) Plots ONLY this final weighted_shape (no other elements).
%
%  Requirements:
%    - 'addGaussianWeightsMedianRule.m' on your MATLAB path
%    - 'createWeightedSegment.m' on your MATLAB path
%    - 'alexNetResponses.mat' containing 'distMatrix'
%    - 'shapes.mat' containing your shapes from step 1
%        (each shapes(i) has .silhouette, .occluder, etc.)
%
%  Usage:
%    >> addWeights
%
%  Author: Hannes Schaetzle
%  Date:   2025-03-26
%% ========================================================

clear; clc; close all;

%% 1) File Paths
distFile   = '/Users/I743312/Documents/MATLAB/CNN Project/data/alexNetResponses.mat'; 
shapesFile = '/Users/I743312/Documents/MATLAB/CNN Project/data/shapes.mat';

%% 2) Add Gaussian Weights Based on Median Rule
fprintf('(1) Adding Gaussian Weights (Median Rule)\n');
addGaussianWeightsMedianRule(distFile, shapesFile);

%% 3) Create Weighted Segment (the average line)
createWeightedSegment(shapesFile);
% This appends "weighted_segment" ([100x2]) to shapes.mat

% Reload shapes + weighted_segment
data = load(shapesFile, 'shapes', 'weighted_segment');
if ~isfield(data, 'weighted_segment')
    error('weighted_segment not found in %s after createWeightedSegment.', shapesFile);
end
shapes           = data.shapes;
weighted_segment = data.weighted_segment;

%% 4) Combine Weighted Segment with Partial Silhouette => "weighted_shape"

if isempty(shapes)
    error('No shapes to reference in shapes.mat.');
end
silhouette = shapes(1).silhouette;
occluder   = shapes(1).occluder;

% Identify the portion of silhouette that lies inside occluder
in = inpolygon(silhouette(:,1), silhouette(:,2), occluder(:,1), occluder(:,2));
id = find(in);
contour1 = [silhouette(id(end)+1:end,:); silhouette(1:id(1)-1,:)];

% Optionally flip weighted_segment to align endpoints
rmseW = sqrt(sum((weighted_segment(1,:) - contour1([1,end],:)).^2, 2));
if find(rmseW == min(rmseW)) == 1
    weighted_segment = flipud(weighted_segment);
end

weighted_shape = [weighted_segment; contour1];

% Save it in shapes(1) or as a separate variable:
shapes(1).weighted_shape = weighted_shape;
save(shapesFile, 'shapes', '-append', '-v7.3');

%% 5) Plot final weighted_shape
fig = figure('Name','Final Weighted Shape','Color','w');
patch(weighted_shape(:,1), weighted_shape(:,2), 'k', 'FaceAlpha',1, 'LineWidth',2);
axis equal; axis off;
title('Weighted Shape (Merged Weighted Segment + Partial Silhouette)');