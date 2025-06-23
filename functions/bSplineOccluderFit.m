function aligned_segment = bSplineOccluderFit( ...
    original_segment, start_pt, end_pt, occluder, ...
    numCtrlPoints, alphaShape, betaCurv, maxIters)
% BSPLINEOCCLUDERFIT  Uses an optimization-based B-spline approach to deform a segment
%                     so it remains inside a polygonal occluder and has specified endpoints.
%
%   aligned_segment = bSplineOccluderFit(original_segment, start_pt, end_pt, occluder, ...
%                                        numCtrlPoints, alphaShape, betaCurv, maxIters)
%   Inputs:
%     original_segment : Nx2 coordinates of the source segment.
%     start_pt, end_pt : 1x2 vectors for the fixed endpoints.
%     occluder         : Mx2 polygon bounding the allowed region (assumed closed).
%     numCtrlPoints    : Number of B-spline control points (>= 2).
%     alphaShape       : Weight for "closeness to Procrustes reference" in objective.
%     betaCurv         : Weight for curvature penalty in objective.
%     maxIters         : Maximum iterations for fmincon.
%
%   Output:
%     aligned_segment  : Nx2 final coordinates for the B-spline fitted curve.
%
%   Method Outline:
%     1) Create a Procrustes-aligned reference from original_segment to (start_pt->end_pt).
%     2) Represent the curve by a cubic B-spline with numCtrlPoints control points.
%        - The first and last control points are fixed to start_pt and end_pt.
%        - The interior control points are the optimization variables.
%     3) The objective function includes:
%        - closeness to the Procrustes reference (alphaShape),
%        - curvature penalty (betaCurv),
%        - penalty if points go outside 'occluder'.
%     4) fmincon solves for the interior control points that minimize this objective.
%     5) The final curve is sampled at the same number of points as original_segment.
%
%   The polygon inpolygon check is used as a penalty. For complicated occluders or
%   narrow corridors, you may need stricter constraints or more advanced geometry.

    % --- 1) Generate a simple Procrustes reference for shape closeness
    procrustesRef = simpleProcrustesReference(original_segment, start_pt, end_pt);

    % Number of interior control points
    interiorCount = numCtrlPoints - 2;
    if interiorCount < 0
        error('numCtrlPoints must be >= 2 (to include start & end).');
    end

    % --- 2) Build initial guess for control points (simple linear interpolation)
    ctrlPts = zeros(numCtrlPoints,2);
    ctrlPts(1,:) = start_pt;
    ctrlPts(end,:)= end_pt;
    for i = 2:(numCtrlPoints-1)
        t = (i-1)/(numCtrlPoints-1);
        ctrlPts(i,:) = (1-t)*start_pt + t*end_pt;
    end
    x0 = reshape(ctrlPts(2:end-1,:), [], 1); % Flatten interior points

    % --- 3) Objective function for fmincon
    objFun = @(x) bSplineObjFun(x, start_pt, end_pt, procrustesRef, ...
                                occluder, alphaShape, betaCurv, numCtrlPoints);

    % Using penalty (in the objective) to handle out-of-bounds,
    % so no explicit linear/nonlinear constraints:
    lb = []; ub = [];

    % --- 4) Call fmincon ---
    opts = optimoptions('fmincon','Display','none','MaxIterations',maxIters, ...
                        'MaxFunctionEvaluations',1e5,'Algorithm','sqp');
    xOpt = fmincon(objFun, x0, [],[], [],[], lb, ub, [], opts);

    % Rebuild the final control points
    ctrlPts(2:end-1,:) = reshape(xOpt, interiorCount,2);

    % --- 5) Convert the B-spline into a dense set of points
    nSamples = 100;  % or 200, or however many you like
    aligned_segment = sampleBSpline(ctrlPts, nSamples);
end