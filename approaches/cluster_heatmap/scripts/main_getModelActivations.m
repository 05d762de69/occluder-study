% Script to extract final layer activations for random segments

clear; clc;

%% Define paths
base_path = 'data/processed/stimuli_images/animal_shapes/';
category_folder = 'generated_images_0058';
model_path = 'data/models/trainedNet_20250624.mat';
random_segments_folder = fullfile(base_path, category_folder, 'random_segments');

%% Load the trained network
load(model_path, 'trainedNet');
net = trainedNet;

%% Get list of random segment images
image_files = dir(fullfile(random_segments_folder, '*.png'));
num_images = length(image_files);

% Initialize activation matrix (1000 images x 54 features)
act_matrix = zeros(num_images, 54);

%% Process each image
for i = 1:num_images
    % Read image
    img_path = fullfile(random_segments_folder, image_files(i).name);
    img = imread(img_path);
    
    % Preprocess image for the network
    inputSize = net.Layers(1).InputSize;
    
    % Resize if necessary
    if ~isequal(size(img, 1:2), inputSize(1:2))
        img = imresize(img, inputSize(1:2));
    end
    
    % Handle color channels
    if size(img, 3) == 1 && inputSize(3) == 3
        img = repmat(img, 1, 1, 3);
    elseif size(img, 3) == 3 && inputSize(3) == 1
        img = rgb2gray(img);
    end
    
    % Extract activations from the last fully connected layer
    layer_name = net.Layers(end-1).Name;
    act = activations(net, img, layer_name);
    act_matrix(i, :) = act(:)';
end

%% Save results
save('activations_0058.mat', 'act_matrix');