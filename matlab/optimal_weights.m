function w = optimal_weights(k,S,integrator,varargin)
%OPTIMAL_WEIGHTS Return optimal quadrature weights for an RKHS
%   Inputs:
%   - k: kernel function
%   - S: a d*n matrix of points
%   - integrator: function x -> integral k(x,y) g(y) dmu(y)
%   Outputs:
%   - w: an n*1 vector of weights

n = size(S,2);
K = kernel_matrix(k,S);
G = zeros(n,1);
for i = 1:n
    G(i) = integrator(S(:,i));
end

% Small regularization for numerical stability
w = (K + 10*eps*trace(K)*eye(size(K,1))) \ G;
end