function [n_dims, stress, eigvals] = analyzeDimensionalityMDS(dist_matrix, max_dims, plot_results)
% Analyze intrinsic dimensionality using MDS on distance matrix
%
% Inputs:
%   dist_matrix - NxN distance matrix
%   max_dims - maximum dimensions to test (default: 20)
%   plot_results - boolean, whether to create plots (default: true)
%
% Outputs:
%   n_dims - recommended number of dimensions based on stress threshold
%   stress - stress values for each dimension tested
%   eigvals - eigenvalues from classical MDS

    if nargin < 2 || isempty(max_dims)
        max_dims = 20;
    end
    if nargin < 3
        plot_results = true;
    end
    
    % Validate input
    if size(dist_matrix, 1) ~= size(dist_matrix, 2)
        error('Distance matrix must be square');
    end
    
    % Classical MDS
    [Y_mds, eigvals] = cmdscale(dist_matrix);
    
    % Limit max_dims to available dimensions
    max_dims = min(max_dims, size(Y_mds, 2));
    
    % Calculate stress for different dimensions
    stress = zeros(max_dims, 1);
    
    % Extract upper triangular distances (excluding diagonal) as column vector
    n = size(dist_matrix, 1);
    idx = triu(true(n), 1);
    D_original = dist_matrix(idx);
    D_original = double(D_original(:));  % Ensure column vector and double type
    
    fprintf('Distance matrix size: %d x %d\n', n, n);
    fprintf('Number of unique pairwise distances: %d\n', length(D_original));
    
    for d = 1:max_dims
        % Reconstruct distances from d dimensions
        D_reconstructed = pdist(Y_mds(:, 1:d));
        D_reconstructed = double(D_reconstructed(:));  % Convert to column vector and ensure double
        
        % Debug info
        if d == 1
            fprintf('D_original size: %d x %d\n', size(D_original, 1), size(D_original, 2));
            fprintf('D_reconstructed size: %d x %d\n', size(D_reconstructed, 1), size(D_reconstructed, 2));
        end
        
        % Calculate normalized stress using element-wise operations
        diff = D_original - D_reconstructed;
        stress(d) = sqrt(sum(diff.^2) / sum(D_original.^2));
    end
    
    % Find recommended dimensions (stress < 0.1)
    n_dims = find(stress < 0.1, 1);
    if isempty(n_dims)
        n_dims = NaN;
    end
    
    % Plot if requested
    if plot_results
        figure('Name', 'MDS Analysis', 'Position', [100, 100, 800, 400]);
        
        subplot(1,2,1);
        n_plot = min(50, length(eigvals));
        plot(eigvals(1:n_plot), 'o-', 'LineWidth', 2);
        xlabel('Dimension');
        ylabel('Eigenvalue');
        title('MDS Eigenvalues');
        grid on;
        
        subplot(1,2,2);
        plot(1:max_dims, stress, 'o-', 'LineWidth', 2);
        hold on;
        plot([1 max_dims], [0.1 0.1], 'r--', 'LineWidth', 1);
        xlabel('Number of Dimensions');
        ylabel('Stress');
        title('MDS Stress vs Dimensions');
        legend('Stress', 'Threshold (0.1)', 'Location', 'northeast');
        grid on;
    end
    
    % Print results
    fprintf('MDS Results:\n');
    fprintf('  Positive eigenvalues: %d\n', sum(eigvals > 0));
    if ~isnan(n_dims)
        fprintf('  Dimensions for stress < 0.1: %d\n', n_dims);
    else
        fprintf('  Dimensions for stress < 0.1: Not achieved within %d dims\n', max_dims);
    end
    fprintf('  Final stress (%d dims): %.4f\n', max_dims, stress(end));
end