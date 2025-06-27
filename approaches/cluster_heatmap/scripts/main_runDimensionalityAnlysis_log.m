% Script to analyze distance distribution and try log transformation

clear; clc;

%% Load distance matrix
load('/Users/I743312/Documents/MATLAB/occluder-study/approaches/cluster_heatmap/data/processed/distance_matrix/distance_matrix_0058.mat', 'dist_matrix');

%% Analyze distance distribution
% Extract upper triangular distances
n = size(dist_matrix, 1);
idx = triu(true(n), 1);
distances = dist_matrix(idx);
distances = distances(:);

% Plot distance distribution
figure('Name', 'Distance Distribution Analysis', 'Position', [100, 100, 1200, 800]);

% Original distances
subplot(2,3,1);
histogram(distances, 100);
xlabel('Distance');
ylabel('Count');
title('Original Distance Distribution');

subplot(2,3,2);
histogram(distances, 100);
set(gca, 'YScale', 'log');
xlabel('Distance');
ylabel('Count (log scale)');
title('Original Distances (Log Y-axis)');

% Log-transformed distances
% Add small epsilon to avoid log(0) if any zeros exist
epsilon = min(distances(distances > 0)) * 0.01;
log_distances = log(distances + epsilon);

subplot(2,3,3);
histogram(log_distances, 100);
xlabel('Log(Distance)');
ylabel('Count');
title('Log-Transformed Distances');

% Statistics
subplot(2,3,4);
boxplot([distances, exp(log_distances)], {'Original', 'Back-transformed'});
ylabel('Distance');
title('Distance Distributions');

% Q-Q plots
subplot(2,3,5);
qqplot(distances);
title('Q-Q Plot: Original Distances');

subplot(2,3,6);
qqplot(log_distances);
title('Q-Q Plot: Log Distances');

%% Print statistics
fprintf('Distance Statistics:\n');
fprintf('Original distances:\n');
fprintf('  Min: %.4f\n', min(distances));
fprintf('  Max: %.4f\n', max(distances));
fprintf('  Mean: %.4f\n', mean(distances));
fprintf('  Median: %.4f\n', median(distances));
fprintf('  Std: %.4f\n', std(distances));
fprintf('  Skewness: %.4f\n', skewness(distances));
fprintf('\nLog-transformed distances:\n');
fprintf('  Skewness: %.4f\n', skewness(log_distances));

%% Create log-transformed distance matrix
dist_matrix_log = zeros(size(dist_matrix));
dist_matrix_log(idx) = log_distances;
dist_matrix_log = dist_matrix_log + dist_matrix_log';  % Make symmetric

%% Run MDS on both versions
fprintf('\n\nRunning MDS on original distances...\n');
[n_dims_orig, stress_orig, eigvals_orig] = analyzeDimensionalityMDS(dist_matrix, 20, false);

fprintf('\nRunning MDS on log-transformed distances...\n');
[n_dims_log, stress_log, eigvals_log] = analyzeDimensionalityMDS(dist_matrix_log, 20, false);

%% Compare results
figure('Name', 'MDS Comparison: Original vs Log', 'Position', [100, 100, 1200, 500]);

subplot(1,2,1);
plot(1:20, stress_orig, 'b-o', 'LineWidth', 2);
hold on;
plot(1:20, stress_log, 'r-s', 'LineWidth', 2);
plot([1 20], [0.1 0.1], 'k--', 'LineWidth', 1);
xlabel('Number of Dimensions');
ylabel('Stress');
title('Stress Comparison');
legend('Original', 'Log-transformed', 'Threshold', 'Location', 'best');
grid on;

subplot(1,2,2);
semilogy(eigvals_orig(1:min(50,length(eigvals_orig))), 'b-o', 'LineWidth', 2);
hold on;
semilogy(eigvals_log(1:min(50,length(eigvals_log))), 'r-s', 'LineWidth', 2);
xlabel('Dimension');
ylabel('Eigenvalue (log scale)');
title('Eigenvalue Comparison');
legend('Original', 'Log-transformed', 'Location', 'best');
grid on;

%% Save log-transformed matrix
save('distance_matrix_log_0058.mat', 'dist_matrix_log');

fprintf('\n\nSummary:\n');
fprintf('Original - Dimensions for stress < 0.1: ');
if isnan(n_dims_orig)
    fprintf('Not achieved\n');
else
    fprintf('%d\n', n_dims_orig);
end
fprintf('Log-transformed - Dimensions for stress < 0.1: ');
if isnan(n_dims_log)
    fprintf('Not achieved\n');
else
    fprintf('%d\n', n_dims_log);
end