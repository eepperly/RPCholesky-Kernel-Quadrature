function K = kernel_matrix(k,S)
%KERNEL_MATRIX Compute kernel matrix
%   Inputs:
%   - k: kernel function
%   - S: a d*n matrix of points
%   Outputs:
%   - K: n*n kernel matrix k(S,S)

K = zeros(size(S,2));
for i = 1:size(S,2)
    for j = i:size(S,2)
        K(i,j) = k(S(:,i),S(:,j));
        K(j,i) = K(i,j);
    end
end
end