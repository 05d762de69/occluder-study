function [segment, segment_idx] = extract_random_segment(shape, fraction)
% EXTRACT_RANDOM_SEGMENT  Extracts a random sub-segment of the specified fraction
%                         of the total length from the given shape.
%
%   [segment, idx_range] = extract_random_segment(shape, fraction)
%     - shape:   Nx2 matrix of (x,y) coordinates
%     - fraction: scalar between 0 and 1
%   Returns:
%     - segment: sub-part of 'shape' that spans 'fraction' of its total length
%     - idx_range: the indices in 'shape' used for the sub-segment

    if fraction <= 0 || fraction > 1
        error('Fraction must be between 0 and 1');
    end

    % Compute arc lengths along shape
    dists = sqrt(sum(diff(shape).^2, 2));
    arc_length = [0; cumsum(dists)];
    total_length = arc_length(end);
    desired_length = fraction * total_length;

    % Random start along the valid arc range
    max_start = total_length - desired_length;
    start_dist = rand() * max_start;

    % Find the indices for the start and end
    start_idx = find(arc_length >= start_dist, 1, 'first');
    end_idx   = find(arc_length >= (start_dist + desired_length), 1, 'first');

    % Extract
    segment_idx = start_idx:end_idx;
    segment = shape(segment_idx, :);
end