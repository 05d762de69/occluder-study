function curvePts = sampleBSpline(ctrlPts, nSamples)
% SAMPLEBSPLINE  Samples a cubic spline from 'ctrlPts' at nSamples points.
%                Uses the built-in cscvn function if available.

    pp = cscvn(ctrlPts'); % Builds a piecewise cubic Hermite spline
    tvals = linspace(pp.breaks(1), pp.breaks(end), nSamples);
    sampled = fnval(pp, tvals)';  % Nx2
    curvePts = sampled;
end