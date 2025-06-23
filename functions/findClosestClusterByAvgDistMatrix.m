function [closestClusterIdx, clusterAvgDists, bestShapeInCluster] = ...
    findClosestClusterByAvgDistMatrix(distFile, clusterLabels)
% findClosestClusterByAvgDistMatrix  Chooses the cluster with the smallest average 
%                                    distance to the occluded image, using distMatrix 
%                                    from alexNetResponses.mat, then picks the best 
%                                    shape (min distance) in that cluster.
%
%   [closestClusterIdx, clusterAvgDists, bestShapeInCluster] = ...
%       findClosestClusterByAvgDistMatrix(distFile, clusterLabels)
%
%   Inputs:
%     distFile       : Path to alexNetResponses.mat, which must contain 'distMatrix'.
%                      distMatrix is [N x 1], the distance from each random segment 
%                      to the occluded image.
%     clusterLabels  : [N x 1], the cluster assignment for each shape (1..k).
%
%   Outputs:
%     closestClusterIdx : Index of the cluster (1..k) whose average distance is minimal.
%     clusterAvgDists   : [k x 1] average distance for each cluster.
%     bestShapeInCluster: The shape index within that cluster that has the minimal 
%                         distance to the occluded image.
%
%   Steps:
%    1) Load distMatrix from distFile, an Nx1 vector.
%    2) For each cluster c = 1..k:
%       - gather members: find(clusterLabels==c)
%       - compute avgDist(c) = mean(distMatrix(members))
%    3) Among c=1..k, pick c* with smallest avgDist => closestClusterIdx
%    4) In that cluster, pick the shape i with minimal distMatrix(i).
%
%   Example:
%     [cIdx, avgVec, bestShape] = findClosestClusterByAvgDistMatrix('alexNetResponses.mat', labels);
%     fprintf('Closest cluster: %d, avg dist=%.3f, best shape idx= %d\n', ...
%             cIdx, avgVec(cIdx), bestShape);
%     % Then if shapes is Nx1, do shapes(bestShape).new_shape -> patch(...);
%

    % 1) Load distMatrix
    data = load(distFile, 'distMatrix');
    if ~isfield(data, 'distMatrix')
        error('No variable "distMatrix" found in %s.', distFile);
    end
    distMatrix = data.distMatrix;  % Nx1
    fprintf('Loaded distMatrix of size [%d x 1]\n', length(distMatrix));

    nShapes = length(clusterLabels);
    if length(distMatrix)~=nShapes
        error('distMatrix has %d entries, but clusterLabels has %d.', length(distMatrix), nShapes);
    end

    k = max(clusterLabels);  % #clusters
    clusterAvgDists = zeros(k,1);

    % 2) For each cluster, compute average
    for cIdx = 1:k
        members = find(clusterLabels==cIdx);
        if isempty(members)
            warning('Cluster %d is empty, setting avgDist=Inf', cIdx);
            clusterAvgDists(cIdx) = Inf;
        else
            clusterAvgDists(cIdx) = mean(distMatrix(members));
        end
    end

    % 3) find cluster with min average
    [~, closestClusterIdx] = min(clusterAvgDists);

    % 4) in that cluster, pick shape with minimal distance
    clusterMembers = find(clusterLabels==closestClusterIdx);
    [~, localMinPos] = min(distMatrix(clusterMembers));
    bestShapeInCluster = clusterMembers(localMinPos);
end
