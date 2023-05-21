function w = uniform_weights(k,S,varargin)
%UNIFORM_WEIGHTS Return uniform quadrature weights
%   Inputs:
%   - k: kernel function (ignored)
%   - S: a d*n matrix of points
%   Optional input:
%   - total_mass: total measure of space (defaults to 1)
%   Outputs:
%   - w: an n*1 vector of uniform weights

total_mass = 1;
for i = 1:length(varargin)
    if isnumeric(varargin{i})
        total_mass = varargin{i};
        break;
    end
end
w = ones(size(S,2),1) * total_mass / size(S,2);
end