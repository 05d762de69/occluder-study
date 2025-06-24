function [clusterLabels, clusterPrototypes] = clusterCompletionsPCA(...
    responsesFile, shapesFile, numComponents, k)
% clusterCompletionsPCA  Performs PCA and k-means clustering on random-segment features,
%                         then finds a prototype completion per cluster.
%
%   [clusterLabels, clusterPrototypes] = clusterCompletionsPCA(...
%       responsesFile, shapesFile, numComponents, k)
%
%   Inputs:
%     responsesFile : Path to alexNetResponses.mat containing feats_randomSegments
%                        e.g. feats_randomSegments is [numFeatures x nImages]
%     shapesFile    : Path to shapes.mat containing shape info for completions
%     numComponents : # of PCA components to keep (10~30 is typical)
%     k             : # of clusters to form in feature space
%
%   Outputs:
%     clusterLabels      : nImages x 1 vector of cluster labels (1..k)
%     clusterPrototypes  : struct array with .index, .centroid, .closestIdx
%
%   The function:
%     1) Loads feats_randomSegments from responsesFile
%     2) Applies PCA to reduce dimension to 'numComponents'
%     3) Runs k-means to produce 'k' clusters
%     4) For each cluster, finds the item closest to the centroid => prototype
%     5) Optionally, you can then retrieve shapes(...) from shapesFile to
%        see the actual aligned segments or images.
%
%   Example:
%     [labels, protos] = clusterCompletionsPCA('alexNetResponses.mat','shapes.mat',20,5);
%     % Then 'labels' tells you which cluster each completion belongs to,
%     % 'protos' gives you the prototypes' indices & centroid in feature space.

    %% 1) Load feats_randomSegments
    dataResp = load(responsesFile, 'features_randomSegments');
    if ~isfield(dataResp, 'features_randomSegments')
        error('No variable "features_randomSegments" found in %s.', responsesFile);
    end
    feats = dataResp.feats_randomSegments;  % [numFeatures x nImages]
    fprintf('Loaded features_randomSegments: size [%d x %d]\n', size(feats,1), size(feats,2));

    nImages = size(feats,2);

    %% 2) PCA dimensionality reduction
    % feats' => [nImages x numFeatures], so each row is a sample.
    % Keep 'numComponents' principal components.
    [coeff, score, latent, tsExplained] = pca(feats');
    featsReduced = score(:,1:numComponents);  % [nImages x numComponents]

    fprintf('Applied PCA to reduce to %d components (%.2f%% variance explained by first %d PCs)\n',...
        numComponents, sum(tsExplained(1:numComponents)), numComponents);

    %% 3) k-means Clustering in reduced space
    % featsReduced is [nImages x numComponents].
    % clusterLabels is nImages x 1
    rng(1); % for reproducibility
    clusterLabels = kmeans(featsReduced, k, 'Distance','sqEuclidean', ...
        'Replicates',5, 'Display','off');

    fprintf('Ran k-means with k=%d clusters.\n', k);

    %% 4) Find Cluster Prototypes
    % We'll compute the centroid of each cluster in the reduced space, then find
    % the completion whose features are closest to that centroid.
    clusterPrototypes(k) = struct('index',[],'centroid',[],'closestIdx',[]);
    for cIdx = 1:k
        % Indices belonging to cluster cIdx
        thisCluster = find(clusterLabels==cIdx);
        if isempty(thisCluster)
            warning('Cluster %d is empty! Check your data or k value.', cIdx);
            continue;
        end

        % 4a) Compute the centroid in reduced space
        clusterCentroid = mean(featsReduced(thisCluster,:),1);  % [1 x numComponents]

        % 4b) For each item in this cluster, compute distance to centroid
        dists = sum((featsReduced(thisCluster,:) - clusterCentroid).^2,2); 
        [~, minPos] = min(dists); % index of closest item

        closestGlobalIdx = thisCluster(minPos);

        % Store info
        clusterPrototypes(cIdx).index       = cIdx;
        clusterPrototypes(cIdx).centroid    = clusterCentroid;
        clusterPrototypes(cIdx).closestIdx  = closestGlobalIdx; % which item in the entire dataset
    end

    fprintf('Identified prototypes for each of the %d clusters.\n', k);

    %% Optional: you can load shapes, so you can find the actual shape or image
    % for each prototype.
    shapesData = load(shapesFile,'shapes');
    if isfield(shapesData,'shapes')
        % Suppose 'shapes' is Nx1, each shape corresponds to feats_randomSegments' columns
        % We can do a quick check #shapes == nImages
        if numel(shapesData.shapes)==nImages
            % For each cluster, print the shape index of the prototype
            for cIdx = 1:k
                idx = clusterPrototypes(cIdx).closestIdx;
                fprintf('  Cluster %d prototype => shapes(%d)\n', cIdx, idx);
            end
        else
            warning('shapes.mat has %d shapes, but feats has %d. Indices may mismatch.', ...
                numel(shapesData.shapes), nImages);
        end
    else
        warning('No "shapes" variable in %s - cannot link prototypes to shapes.', shapesFile);
    end

    fprintf('Done. You can use clusterLabels and clusterPrototypes in further analysis.\n');
end
