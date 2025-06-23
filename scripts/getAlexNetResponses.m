function getAlexNetResponses()
% getAlexNetResponses
%   Step 2 of pipeline:
%    - Loads the trained AlexNet (transfer-learned) from /data/trainedNet.mat
%    - Reads images from:
%         1) with_occluder/
%         2) no_occluder/
%         3) random_segments/
%    - Resizes them to [227×227×3] via augmentedImageDatastore
%    - Obtains the final-layer (fc8 or classoutput) outputs
%    - Computes pairwise Euclidean distances between feats_randomSegments (1c)
%      and feats_withOccluder (1a), storing them in a matrix.
%    - Saves everything in alexNetResponses.mat.
%
%  Requirements:
%    - The images from Step 1 must exist in generated_images/<subfolders>.
%    - The file /data/trainedNet.mat should contain a variable 'trainedNet'.
%
%  Example usage:
%    >> step2_getAlexNetResponses
%
%  This script will produce:
%    - features_withOccluder:    [K×N1]
%    - features_noOccluder:      [K×N2]
%    - features_randomSegments:  [K×N3]
%    - distMatrix:            [N3×N1] (Euclidean distances)
%

    %% 1. Load trained AlexNet
    load('/Users/I743312/Documents/MATLAB/CNN Project/data/trainedNet.mat', 'trainedNet');
    if ~exist('trainedNet','var')
        error('Variable ''trainedNet'' not found in /data/trainedNet.mat');
    end

    %% 2. Define paths to images
    mainDir = '/Users/I743312/Documents/MATLAB/CNN Project/data/generated_images';
    withOccluderDir   = fullfile(mainDir, 'with_occluder');
    noOccluderDir     = fullfile(mainDir, 'no_occluder');
    randomSegmentsDir = fullfile(mainDir, 'random_segments');

    % Check folders exist
    if ~exist(withOccluderDir, 'dir')
        error('Folder does not exist: %s', withOccluderDir);
    end
    if ~exist(noOccluderDir, 'dir')
        error('Folder does not exist: %s', noOccluderDir);
    end
    if ~exist(randomSegmentsDir, 'dir')
        error('Folder does not exist: %s', randomSegmentsDir);
    end

    %% 3. Create imageDatastore objects
    imds_withOccluder   = imageDatastore(withOccluderDir);
    imds_noOccluder     = imageDatastore(noOccluderDir);
    imds_randomSegments = imageDatastore(randomSegmentsDir);

    %% 4. Use augmentedImageDatastore to resize images
    inputSize = [227 227 3];
    aug_withOccluder   = augmentedImageDatastore(inputSize(1:2), imds_withOccluder,   'ColorPreprocessing','none');
    aug_noOccluder     = augmentedImageDatastore(inputSize(1:2), imds_noOccluder,     'ColorPreprocessing','none');
    aug_randomSegments = augmentedImageDatastore(inputSize(1:2), imds_randomSegments, 'ColorPreprocessing','none');

    %% 5. Retrieve activations
    %    Adjust layerName depending on how your network is structured.
    %    'fc_new' in the transfered AlexNet is 54-D (pre-softmax).
    %    'softmax_new' is the softmax probabilities (also 54-D).
    %    'classoutput' is the final classification output layer.
    layerName = 'classoutput';

    features_withOccluder = squeeze(activations(trainedNet, aug_withOccluder, layerName));

    features_noOccluder = squeeze(activations(trainedNet, aug_noOccluder, layerName));

    features_randomSegments = squeeze(activations(trainedNet, aug_randomSegments, layerName));

    %% 6. Compute distance

    distMatrix = pdist2(features_randomSegments', features_withOccluder', 'euclidean');

    %% 7. Save results
    outFile = '/Users/I743312/Documents/MATLAB/CNN Project/data/alexNetResponses.mat';
    save(outFile, 'features_withOccluder', 'features_noOccluder', 'features_randomSegments', 'distMatrix', '-v7.3');
end
