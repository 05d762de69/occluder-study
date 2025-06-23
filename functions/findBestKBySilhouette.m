function bestK = findBestKBySilhouette(responsesFile, kRange)
% findBestKBySilhouetteNoPCA  Finds the best number of clusters (k) 
%                             by looping over kRange and computing silhouette scores 
%                             on raw feats_randomSegments.
%
%   bestK = findBestKBySilhouetteNoPCA(responsesFile, kRange)
%
%   Inputs:
%       responsesFile : path to .mat file containing feats_randomSegments (size [numFeatures x nImages])
%       kRange        : vector of candidate k-values, e.g. 2:10
%
%   Output:
%       bestK         : the k in kRange that yields the highest average silhouette score.
%
%   Example:
%       % Suppose alexNetResponses.mat has feats_randomSegments=[54 x 10000].
%       kRange = 2:10;
%       bestK = findBestKBySilhouetteNoPCA('alexNetResponses.mat', kRange);
%       fprintf('Best k was %d\n', bestK);
%       % Then call clusterCompletionsNoPCA(..., bestK) to finalize clustering.
%
%   Steps:
%     1) Load feats_randomSegments from responsesFile
%     2) For each k in kRange, run k-means => clusterLabels
%     3) Compute silhouette => average
%     4) Keep track of best average silhouette => bestK
%
%   Note:
%     This approach can be slow if feats is large, and if you do many k-values
%     with multiple 'Replicates'. Consider PCA or a smaller subset to estimate k.

    % 1) Load the feats
    data = load(responsesFile, 'features_randomSegments');
    if ~isfield(data, 'features_randomSegments')
        error('No variable "features_randomSegments" found in %s.', responsesFile);
    end
    feats = data.features_randomSegments;  % [numFeatures x nImages]
    fprintf('Loaded feats_randomSegments: [%d x %d]\n', size(feats,1), size(feats,2));

    nImages = size(feats,2);

    % For k-means + silhouette, each row = a sample => [nImages x numFeatures]
    featsForClust = feats';

    bestScore = -Inf;
    bestK     = NaN;

    fprintf('Testing k in [%s] via silhouette...\n', num2str(kRange));
    for k = kRange
        % --- k-means
        rng(1); % for reproducibility
        clusterLabels = kmeans(featsForClust, k, ...
            'Distance','sqEuclidean', ...
            'Replicates',5, ...
            'Display','off');

        % --- silhouette
        sVals = silhouette(featsForClust, clusterLabels);
        avgS  = mean(sVals);

        fprintf('  k=%2d => avg silhouette=%.4f\n', k, avgS);

        % Check if best
        if avgS > bestScore
            bestScore = avgS;
            bestK = k;
        end
    end

    fprintf('\nBest k by silhouette: %d (avg score=%.4f)\n', bestK, bestScore);
end
