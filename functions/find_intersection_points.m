function intersection_points = find_intersection_points(silhouette, occluder)
% FIND_INTERSECTION_POINTS  Finds intersection points between a silhouette and an occluder.
%
%   intersection_points = find_intersection_points(silhouette, occluder)
%   - silhouette: Nx2 matrix (x,y coordinates)
%   - occluder:   Mx2 matrix (x,y coordinates)
%   Returns a 2x2 matrix with exactly two intersection points:
%       [x1, y1; x2, y2]
%   or empty if fewer than two intersections were found.

    [x_intersect, y_intersect] = polyxpoly(silhouette(:,1), silhouette(:,2), ...
                                           occluder(:,1), occluder(:,2));
    if length(x_intersect) ~= 2
        warning('Expected exactly two intersection points; found %d.', length(x_intersect));
        intersection_points = [];
    else
        intersection_points = [x_intersect, y_intersect];
    end
end