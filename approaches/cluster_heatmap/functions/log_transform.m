function Dlog = log_transform(D, varargin)
%LOG_TRANSFORM  Log-transform a distance matrix but keep it valid.
%
%   Dlog = LOG_TRANSFORM(D)                 – natural-log of (1 + D)
%   Dlog = LOG_TRANSFORM(D,'Base',B)        – log base B of (1 + D)
%
%   ▸ Diagonal is forced to 0 afterwards.
%   ▸ Output is re-symmetrised for good measure.

    validateattributes(D,{'numeric'},{'square','nonnegative'},mfilename,'D');

    p = inputParser;
    addParameter(p,'Base',exp(1), @(x)isscalar(x)&&x>0);
    parse(p,varargin{:});
    B = p.Results.Base;

    % element-wise log(1 + D) – never negative, handles zeros safely
    Dlog = log1p(D)./log(B);

    % enforce perfect symmetry (eliminate round-off)
    Dlog = 0.5*(Dlog + Dlog.');

    % ensure the diagonal is exactly zero
    Dlog(1:size(Dlog,1)+1:end) = 0;
end