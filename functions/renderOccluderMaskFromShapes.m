function [occludedImg, occluderMask] = renderOccluderMaskFromShapes(silhouette, occluder, H, W)
    % Create invisible figure
    fig = figure('Visible', 'off', 'Position', [100, 100, W, H]);
    ax = axes('Parent', fig, 'Position', [0 0 1 1], 'Units', 'normalized');
    
    % Plot silhouette (black)
    patch(ax, silhouette(:,1), silhouette(:,2), 'k', 'EdgeColor', 'none');
    hold on;

    % Plot occluder (gray)
    patch(ax, occluder(:,1), occluder(:,2), [0.5 0.5 0.5], 'EdgeColor', 'none');
    
    % Fix view
    axis(ax, 'off'); axis(ax, 'equal');
    xlim(ax, [0 W]);
    ylim(ax, [0 H]);
    
    % Capture the frame (full W x H canvas)
    frame = getframe(ax);
    img = frame2im(frame);
    img = imresize(img, [H W]);

    close(fig);

    % Convert to grayscale and extract occluder
    grayImg = rgb2gray(img);
    occluderMask = grayImg > 110 & grayImg < 145;

    occludedImg = img;
end
