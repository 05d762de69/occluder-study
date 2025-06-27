%% RUNMDSPIPELINE  Log-transform a distance matrix and run MDS.
%  --------------------------------------------------------------
%  1) Load or create an n×n matrix D.
%  2) Transform it with any log base you like.
%  3) Perform 2-D classical MDS.
%  4) Plot and (optionally) save the results.

clear; clc;

%% 1. Load your distance matrix (expects variable "D")
pathToMatrix = '/Users/I743312/Documents/MATLAB/occluder-study/approaches/cluster_heatmap/data/processed/distance_matrix/distance_matrix_0058.mat'; 
load(pathToMatrix,'dist_matrix');

%% 2. Log-transform (base-10 in this example)
Dlog = log_transform(dist_matrix,'Base',10);

%% 3. Multidimensional scaling to 2 D
[Y,eigVals,k] = perform_MDS(Dlog,0.95);

%% 4. Inspect eigenvalues
fprintf('\nMDS returned %d positive eigenvalues; dimensions kept: %d\n', ...
        sum(eigVals>1e-10), size(Y,2));

%% 5. Clustering in the full MDS space
[idx,info] = assignClusters(Y,k, 'kmeans');  

%% 6. Save everything
save('mds_results.mat','Y','eigVals','idx');
