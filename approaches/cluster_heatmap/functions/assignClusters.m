function [idx,info] = assign_clusters(Y, k, method)
%ASSIGN_CLUSTERS  Cluster MDS coordinates and return labels.
%
%   idx  – n×1 vector of cluster labels (1…k)
%   info – struct with extra details for diagnostics
%
%   [idx,info] = ASSIGN_CLUSTERS(Y,k)         – k-means (default)
%   [idx,info] = ASSIGN_CLUSTERS(Y,k,'linkage') – Ward linkage
%
%   If k is empty or 0 the function picks k via silhouette maximisation.

    if nargin < 2, k = []; end
    if nargin < 3 || isempty(method), method = 'kmeans'; end

    % ---- choose k automatically if not given ---------------------------
    if isempty(k) || k==0
        kRange     = 2:10;                   % search range; tweak as needed
        silhMean   = zeros(size(kRange));
        for ii = 1:numel(kRange)
            tempIdx  = kmeans(Y,kRange(ii), 'Replicates',5,'Display','off');
            silhMean(ii) = mean(silhouette(Y,tempIdx));
        end
        [~,best]   = max(silhMean);
        k          = kRange(best);
    end

    % ---- clustering ----------------------------------------------------
    switch lower(method)
        case 'kmeans'
            [idx,C,sumd] = kmeans(Y,k,'Replicates',10,'Display','final');
            info.Centroids = C;
            info.SumD      = sumd;
        case 'linkage'
            Z       = linkage(Y,'ward');
            idx     = cluster(Z,'maxclust',k);
            info.Tree = Z;
        otherwise
            error('Unknown method "%s".',method);
    end
end
