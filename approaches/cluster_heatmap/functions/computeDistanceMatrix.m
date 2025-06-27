function dist_matrix = computeDistanceMatrix(activations, metric)
% Compute pairwise distance matrix from network activations
%
% Inputs:
%   activations - NxM matrix where N is number of samples (1000) and M is features (54)
%   metric - (optional) distance metric: 'euclidean' (default), 'cosine', 'correlation'
%
% Output:
%   dist_matrix - NxN symmetric distance matrix

    if nargin < 2
        metric = 'euclidean';
    end
    
    % Compute distance matrix using pdist and squareform
    dist_matrix = squareform(pdist(activations, metric));
    
end