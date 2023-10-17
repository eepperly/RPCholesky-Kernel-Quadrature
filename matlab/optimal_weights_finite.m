function w = optimal_weights_finite(data,k,S,varargin)
%OPTIMAL_WEIGHTS_FINITE Return optimal quadrature weights for a finite
%dataset
%   Inputs:
%   - data: |X| * n data matrix
%   - k: kernel function
%   - S: a 1*n matrix of selected data indices
%   - g: |X|*1 vector representing the weight function (default:
%        ones(|X|,1)/|X|
%   Outputs:
%   - w: an n*1 vector of weights

if length(varargin) >= 1 && ~isempty(varargin{1})
    g = varargin{1};
else
    g = ones(size(data,1),1)/size(data,1);
end

K = k(data(S,:),data(S,:));
Tg = kernel_columns_finite(data,k,S)'*g;
w = (K + 10*eps*trace(K)*eye(size(K,1))) \ Tg;
end