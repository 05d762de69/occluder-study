function [clusterLabels, clusterPrototypes] = clusterCompletions(...
    responsesFile, shapesFile, k)
% clusterCompletionsNoPCA  Clusters random-segment features (no PCA),
%                          then identifies a prototype for each cluster.
%
%   [clusterLabels, clusterPrototypes] = clusterCompletionsNoPCA(...
%       responsesFile, shapesFile, k)
%
%   Inputs:
%       responsesFile : Path to alexNetResponses.mat containing feats_randomSegments
%                       (size [numFeatures x nImages])
%       shapesFile    : Path to shapes.mat (optional linking to shapes array)
%       k             : Number of clusters
%
%   Outputs:
%       clusterLabels     : [nImages x 1] cluster labels for each image
%       clusterPrototypes : struct array with fields:
%                            .index    => cluster index (1..k)
%                            .centroid => mean in feature space
%                            .closestIdx => index of the item best representing the cluster
%
%   The function:
%    1) Loads feats_randomSegments (raw features, [numFeatures x nImages])
%    2) Runs k-means on feats' => [nImages x numFeatures]
%    3) Finds cluster prototypes by picking the item closest to the centroid
%    4) Optionally loads shapesFile to link prototypes to actual shape indices
%
%   Example:
%     [labels, protos] = clusterCompletionsNoPCA('alexNetResponses.mat','shapes.mat',5);
%     % => clusterLabels is 10000x1, protos is 5-element struct with .closestIdx, etc.
%
%   Tip:
%     If feats_randomSegments is large (e.g. [1000 x 10,000]),
%     you might want to do PCA first for speed/robustness.

    % 1) Load the raw features
    dataResp = load(responsesFile, 'features_randomSegments');
    if ~isfield(dataResp, 'features_randomSegments')
        error('No variable "features_randomSegments" found in %s.', responsesFile);
    end
    feats = dataResp.features_randomSegments;  % [numFeatures x nImages]
    fprintf('Loaded features_randomSegments: [%d x %d] (features x images)\n', ...
            size(feats,1), size(feats,2));

    nImages = size(feats,2);
    
    % For k-means, we want each row = a sample, so transpose feats => [nImages x numFeatures]
    featsForClust = feats';  % [nImages x numFeatures]

    % 2) K-means Clustering
    rng(1); % for reproducibility
    clusterLabels = kmeans(featsForClust, k, ...
        'Distance','sqEuclidean', ...
        'Replicates',5, ...
        'Display','off');

    fprintf('K-means done: %d clusters.\n', k);

    % 3) Find Cluster Prototypes
    clusterPrototypes(k) = struct('index',[],'centroid',[],'closestIdx',[]);
    for cIdx = 1:k
        idxThisCluster = find(clusterLabels==cIdx);
        if isempty(idxThisCluster)
            warning('Cluster %d is empty! Might consider smaller k.', cIdx);
            continue;
        end

        % Centroid in the same dimension as featsForClust
        clusterCentroid = mean(featsForClust(idxThisCluster,:),1);  % [1 x numFeatures]

        % Distances to each item in idxThisCluster
        dists = sum((featsForClust(idxThisCluster,:) - clusterCentroid).^2,2);
        [~, minPos] = min(dists);
        closestGlobalIdx = idxThisCluster(minPos);

        clusterPrototypes(cIdx).index      = cIdx;
        clusterPrototypes(cIdx).centroid   = clusterCentroid;
        clusterPrototypes(cIdx).closestIdx = closestGlobalIdx;  % image # from [1..nImages]
    end
    fprintf('Found prototypes for each of the %d clusters.\n', k);

    % 4) (Optional) Link with shapes from shapesFile
    shapesData = load(shapesFile, 'shapes');
    if isfield(shapesData, 'shapes')
        shapes = shapesData.shapes;
        if numel(shapes)~=nImages
            warning('shapes.mat has %d shapes, feats has %d images => mismatch.', ...
                numel(shapes), nImages);
        else
            for cIdx = 1:k
                cProt = clusterPrototypes(cIdx).closestIdx;
                fprintf('  Cluster %d prototype => shapes(%d)\n', cIdx, cProt);
            end
        end
    else
        warning('No "shapes" variable in %s => prototypes not linked to actual shapes.', ...
                shapesFile);
    end

    fprintf('Done. clusterLabels is [%dx1], clusterPrototypes is 1x%d struct.\n', nImages, k);
end
