function occludedImg = createOccludedImage(silhouettePts, occluderPts, H, W)
% createOccludedImage  Renders silhouette + occluder polygons into a 
%                      [H x W x 3] image, with silhouette in black, 
%                      occluder in gray, etc. Adjust as needed.

    fig = figure('Visible','off');
    patch(silhouettePts(:,1), silhouettePts(:,2), 'k','FaceAlpha',1,'LineWidth',2);
    hold on;
    patch(occluderPts(:,1), occluderPts(:,2), 'k',...
        'EdgeColor',[0.5 0.5 0.5], 'FaceColor',[0.5 0.5 0.5],'LineWidth',1);
    axis off; axis equal;

    frameData = getframe(gca);
    rawImg = frame2im(frameData);
    close(fig);

    % rawImg could be some random size depending on your figure. Let's do 
    % a simple imresize to [H x W].
    occludedImg = imresize(rawImg,[H W]);
end