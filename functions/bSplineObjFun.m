function [f] = bSplineObjFun(x, start_pt, end_pt, refPts, occluder, alpha, beta, numCtrlPoints)
% BSPLINEOBJFUN  Objective function for B-spline fitting with polygon penalty and curvature penalty.
%
%  x: Flattened interior control points.
%  start_pt, end_pt: fixed first/last control points.
%  refPts: Nx2 reference curve (Procrustes aligned).
%  occluder: Mx2 polygon for inpolygon checks.
%  alpha, beta: weights for shape closeness and curvature, respectively.
%  numCtrlPoints: total # of B-spline control points.

    interiorCount = numCtrlPoints - 2;
    ctrlPts = zeros(numCtrlPoints,2);
    ctrlPts(1,:) = start_pt;
    ctrlPts(end,:) = end_pt;
    ctrlPts(2:end-1,:) = reshape(x, interiorCount,2);

    % Sample the B-spline at same # of points as refPts
    nSamples = size(refPts,1);
    curvePts = sampleBSpline(ctrlPts, nSamples);

    % 1) "Shape closeness" cost
    dist_sq = sum((curvePts - refPts).^2,2);
    shape_cost = sum(dist_sq);

    % 2) Curvature penalty (approx: sum of squared second differences)
    d1 = diff(curvePts,1,1);  % first difference
    d2 = diff(d1,1,1);        % second difference
    curvature_cost = sum(sum(d2.^2,2));

    % 3) Out-of-bounds penalty
    penalty_factor = 1e8; 
    outside_cost = 0;

    % Check each sampled point
    for i = 1:nSamples
        if ~inpolygon(curvePts(i,1), curvePts(i,2), occluder(:,1), occluder(:,2))
            outside_cost = outside_cost + 1;
        end
    end
    % Check each control point (penalize more if out-of-bounds)
    for i = 1:numCtrlPoints
        if ~inpolygon(ctrlPts(i,1), ctrlPts(i,2), occluder(:,1), occluder(:,2))
            outside_cost = outside_cost + 3;
        end
    end
    out_of_bounds_penalty = penalty_factor * outside_cost;

    % Combine all
    f = alpha * shape_cost + beta * curvature_cost + out_of_bounds_penalty;
end