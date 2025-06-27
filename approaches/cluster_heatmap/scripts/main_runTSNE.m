clear; clc;

%% Load the activations (not the distance matrix)
load('/Users/I743312/Documents/MATLAB/occluder-study/approaches/cluster_heatmap/data/processed/model_responses/activations_0058.mat', 'act_matrix');

%% Run t-SNE on the activations
fprintf('Running t-SNE on activations...\n');
rng(42); % For reproducibility

% Run t-SNE
Y = tsne(act_matrix, 'NumDimensions', 2, ...
         'Perplexity', 30, ...
         'Distance', 'euclidean');  % You can change this to 'cosine' or 'correlation'

%% Visualize the results
figure('Name', 't-SNE Visualization', 'Position', [100, 100, 800, 600]);
scatter(Y(:,1), Y(:,2), 10, 'filled');
xlabel('t-SNE Dimension 1');
ylabel('t-SNE Dimension 2');
title('t-SNE of Random Completions');
grid on;

%% Save the t-SNE coordinates
save('tsne_coordinates_0058.mat', 'Y');

fprintf('t-SNE completed. Results saved to tsne_coordinates_0058.mat\n');