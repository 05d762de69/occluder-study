function createWeightedSegment(shapesFile)
% createWeightedSegment  Builds a single weighted average of all aligned_segment 
%                        polylines in shapes, using weightsGauss from the same file.
%
%  createWeightedSegment(shapesFile)
%    - shapesFile: path to shapes.mat containing a struct array "shapes"
%                  and a vector "weightsGauss"
%
%  The function:
%    1) Loads shapes, which must have shapes(i).aligned_segment of size [100×2].
%    2) Loads weightsGauss (Nx1) for N shapes.
%    3) Summation: sum_i(weightsGauss(i)*aligned_segment(i)) / sum(weightsGauss).
%    4) Appends the final [100×2] "weighted_segment" to shapesFile.
%
%  Example usage:
%    >> createWeightedSegment('shapes.mat');
%
%  Then you can retrieve weighted_segment from shapesFile.

    % 1) Load shapes + weightsGauss
    data = load(shapesFile, 'shapes', 'weightsGauss');
    if ~isfield(data, 'shapes')
        error('No variable "shapes" found in %s.', shapesFile);
    end
    if ~isfield(data, 'weightsGauss')
        error('No variable "weightsGauss" found in %s.', shapesFile);
    end

    shapes      = data.shapes;
    weightsGauss= data.weightsGauss;

    nShapes = numel(shapes);
    if nShapes < 1
        error('No shapes available in shapes.mat.');
    end
    if nShapes ~= numel(weightsGauss)
        warning('Mismatch: #shapes(%d) != length of weightsGauss(%d).', ...
                 nShapes, numel(weightsGauss));
    end

    % 2) Initialize accumulators
    sumOfWeightedSegments = zeros(100,2);
    sumOfWeights = 0;

    % 3) Loop over shapes
    for i = 1:nShapes
        seg = shapes(i).aligned_segment;
        if size(seg,1)~=100 || size(seg,2)~=2
            error('shapes(%d).aligned_segment must be [100×2], found [%dx%d].',...
                  i,size(seg,1),size(seg,2));
        end

        w_i = weightsGauss(i); % directly from the vector
        sumOfWeightedSegments = sumOfWeightedSegments + w_i * seg;
        sumOfWeights = sumOfWeights + w_i;
    end

    if sumOfWeights <= 0
        error('Sum of weights is non-positive; cannot create weighted average.');
    end

    % 4) Compute the final average
    weighted_segment = sumOfWeightedSegments ./ sumOfWeights;

    % 5) Save to shapesFile
    save(shapesFile, 'weighted_segment','-append','-v7.3');

    fprintf('Created "weighted_segment" of size [%dx%d] and appended to %s.\n', ...
            size(weighted_segment,1), size(weighted_segment,2), shapesFile);
end
