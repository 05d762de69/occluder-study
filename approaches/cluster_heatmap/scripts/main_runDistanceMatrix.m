% Load the activations
load('data/processed/model_responses/activations_0058.mat', 'act_matrix');

% Compute distance matrix (Euclidean distance)
dist_matrix = computeDistanceMatrix(act_matrix);

% Save the distance matrix
save('distance_matrix_0058.mat', 'dist_matrix');