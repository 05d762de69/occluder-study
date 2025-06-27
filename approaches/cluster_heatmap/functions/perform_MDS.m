function [Y,eigVals,k] = perform_mds(D, dims)
%PERFORM_MDS  Classical (metric) MDS with flexible dimensionality selection.
%
%   [Y,eigVals]   = PERFORM_MDS(D)          – keep just enough dims to
%                                             explain 95 % of the variance.
%
%   [Y,eigVals]   = PERFORM_MDS(D,0.90)     – keep dims up to 90 % variance.
%   [Y,eigVals]   = PERFORM_MDS(D,k)        – if k is an integer (k ≥ 1),
%                                             keep exactly k dims.
%   [Y,eigVals]   = PERFORM_MDS(D,'auto')   – keep **all** positive-eigen dims
%                                             (old behaviour).
%
%   [Y,eigVals,k] also returns the number of dimensions actually kept.
%
%   Examples
%   --------
%       % 95 % variance (default)
%       [Y,λ] = perform_mds(Dlog);
%
%       % Force exactly 3 dimensions
%       Y = perform_mds(Dlog,3);
%
%       % Keep all positive eigenvalues
%       Y = perform_mds(Dlog,'auto');
%
%   Notes
%   -----
%   * “Variance” here means the fraction of total *squared* distance that
%     the retained axes reproduce (classical MDS eigenvalues are in units
%     of squared distance).
%   * Tiny negatives (< 1 × 10⁻¹⁰) from round-off are treated as zero.

    if nargin < 2 || isempty(dims)
        dims = 0.95;                     % default: 95 % variance
    end
    validateattributes(D,{'numeric'},{'square'},mfilename,'D');

    % --- full eigen-spectrum first --------------------------------------
    [Yfull,eigVals] = cmdscale(D);       % all eigenvalues, no cap

    % --- decide how many axes to keep -----------------------------------
    if ischar(dims) || isstring(dims)        % 'auto' → keep all positive λ
        if strcmpi(dims,'auto')
            keepMask = eigVals > 1e-10;
        else
            error('Unknown string option "%s".',dims);
        end

    elseif dims >= 1                          % exact integer dimension count
        validateattributes(dims,{'numeric'},{'scalar','integer','>=',1},mfilename,'dims');
        keepMask           = false(size(eigVals));
        keepMask(1:min(dims,numel(eigVals))) = true;

    elseif dims > 0 && dims < 1               % dims = variance threshold
        varTarget          = dims;
        posIdx             = find(eigVals > 1e-10);
        if isempty(posIdx)
            error('No positive eigenvalues found – distance matrix is non-Euclidean.');
        end
        lambda             = eigVals(posIdx);
        cumVar             = cumsum(lambda) ./ sum(lambda);
        kNeeded            = find(cumVar >= varTarget,1,'first');
        keepMask           = false(size(eigVals));
        keepMask(posIdx(1:kNeeded)) = true;

    else
        error('dims must be ''auto'', an integer ≥ 1, or a scalar between 0 and 1.');
    end

    % --- trim output -----------------------------------------------------
    Y = Yfull(:, keepMask);
    k = sum(keepMask);                     % number of dimensions retained
end
