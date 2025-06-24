function addGaussianWeightsMedianRule(distFile, shapesFile)
% addGaussianWeightsMedianRule  Applies a Gaussian (RBF) weight transform to distMatrix,
%                               choosing sigma based on the median distance,
%                               and appends those weights to shapesFile.
%
%   addGaussianWeightsMedianRule(distFile, shapesFile)
%     - distFile:   Path to a .mat file containing 'distMatrix' (Nx1).
%     - shapesFile: Path to a .mat file containing
%                     'new_shape', 'occluder', 'aligned_segment', 'random_segment'
%
%   The function:
%     1) Loads distMatrix from distFile,
%     2) Sets sigma = median(distMatrix) / sqrt(2*ln(2)),
%     3) Computes weightsGauss(d) = exp( -d^2 / (2*sigma^2) ),
%     4) Appends 'weightsGauss' into shapesFile (using -append).
%
%   Example:
%     >> addGaussianWeightsMedianRule('alexNetResponses.mat','shapes.mat');
%
%   This sets sigma so that the median distance has w(dMed) = 0.5,
%   giving a balanced weighting around that midpoint.

    %% 1) Load distMatrix
    dataDist = load(distFile, 'distMatrix');
    if ~isfield(dataDist, 'distMatrix')
        error('No distMatrix found in %s.', distFile);
    end
    distVec = dataDist.distMatrix(:);  % Ensure it's a column vector

    %% 2) Compute sigma from the median distance
    dMed = median(distVec);
    % Sigmoid rule-of-thumb => w(dMed) = 0.5
    % => 0.5 = exp(-(dMed^2)/(2 sigma^2)) => sigma = dMed / sqrt(2 ln(2))
    sigma = dMed / sqrt(2*log(2));

    %% 3) Gaussian weighting
    weightsGauss = exp(-distVec.^2 ./ (2*sigma^2));

    %% 4) Load shapes data to ensure the file exists and variables are present
    shapesData = load(shapesFile, ...
        'new_shape', 'occluder', 'aligned_segment', 'random_segment');
    % We won't overwrite these variables, just read them so we know the file is valid.

    %% 5) Append weightsGauss to shapesFile
    save(shapesFile, 'weightsGauss', '-append', '-v7.3');

    %% Print summary
    fprintf('Gaussian weights appended to %s as "weightsGauss".\n', shapesFile);
    fprintf('Median distance: %.6f => sigma: %.6f\n', dMed, sigma);
    fprintf('Distance range:  [%.6f, %.6f]\n', min(distVec), max(distVec));
    fprintf('Weights range:   [%.6f, %.6f]\n', min(weightsGauss), max(weightsGauss));
end
