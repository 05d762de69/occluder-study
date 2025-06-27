% Example script to run dimensionality analysis using PCA, Kernel PCA, and MDS

clear; clc;

%% Load data
load('/Users/I743312/Documents/MATLAB/occluder-study/approaches/cluster_heatmap/data/processed/distance_matrix/distance_matrix_0058.mat', 'dist_matrix');

%% Run MDS on distance matrix
[n_dims_mds, stress, eigvals_mds] = analyzeDimensionalityMDS(dist_matrix, 20, true);

%% Save results
save('dimensionality_results_0058.mat', 'n_dims_mds', 'stress', 'eigvals_mds');
