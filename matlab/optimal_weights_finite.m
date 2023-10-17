function w = optimal_weights_finite(data,k,S,varargin)
N = size(data, 1);
n = length(S);

if length(varargin) >= 1 && ~isempty(varargin{1})
    g = varargin{1};
else
    g = ones(N,1)/N;
end

K = k(data(S,:),data(S,:));
Tg = kernel_columns_finite(data,k,S)'*g;
w = (K + 10*eps*trace(K)*eye(size(K,1))) \ Tg;
end