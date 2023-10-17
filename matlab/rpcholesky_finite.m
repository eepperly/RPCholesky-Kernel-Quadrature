function S = rpcholesky_finite(data,n,k)
%RPCHOLESKY_FINITE Select points using randomly pivoted Cholesky from
%finite dataset using pivoted Cholesky algorithm
%   Inputs:
%   - data: data stored as a |X| * d array
%   - n: number of points to select
%   - k: kernel function
%   Outputs:
%   - S: a 1*n matrix of the indices of the selected points

N = size(data, 1);
F = zeros(N,n);
S = zeros(1,n);
d = ones(N,1);

%% Run algorithm
for i = 1:n
    S(i) = randsample(N,1,true,d);
    g = kernel_columns_finite(data,k,S(i));
    g = g - F(:,1:(i-1)) * F(S(i),1:(i-1))';
    F(:,i) = g / sqrt(g(S(i)));
    d = d - abs(F(:,i)).^2;
    d(S(1:i)) = 0;
end
end