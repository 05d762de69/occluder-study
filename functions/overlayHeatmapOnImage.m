function overlayHeatmapOnImage(occludedImg, normalized_heatmap)
    % Convert heatmap to [0 1] scale
    normMap = mat2gray(normalized_heatmap);

    % Convert normalized map to colormap image (e.g., 'hot')
    cmap = hot(256);  % use 'hot' or other colormap
    heatRGB = ind2rgb(gray2ind(normMap, 256), cmap);

    % Resize occluded image if needed
    if size(occludedImg,3) == 1
        occludedImg = repmat(occludedImg, 1, 1, 3);  % make grayscale -> RGB
    end
    occludedImg = im2double(occludedImg);

    % Blend the two images
    alpha = normMap * 0.8;  % transparency mask based on heatmap
    overlay = occludedImg .* (1 - alpha) + heatRGB .* alpha;

    % Display the result
    figure('Name','Overlay Heatmap as RGB on Occluded Image');
    imshow(overlay);
    title('Occluded Image + RGB Heatmap Overlay');
end
